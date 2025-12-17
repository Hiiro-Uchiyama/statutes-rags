"""
CriticAgent - 批判者役

解釈の問題点を指摘し、反論を生成する。
グループシンキング防止の核心となるエージェント。

【重要】
- 必ず1つ以上の反論を生成する（強制反論生成）
- 「問題なし」のみの回答は禁止
- 代替案の提示を必須とする
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent
from .citation import CitationRegistry, CitationStatus

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    批判者エージェント（グループシンキング防止機構）
    
    責務:
    - 解釈の問題点・例外を指摘
    - 必ず反論を生成（強制反論生成）
    - 代替解釈の提示
    - 引用の妥当性検証
    """
    
    # 強制反論生成のためのプロンプト
    MANDATORY_COUNTER_ARG_INSTRUCTION = """
【重要な制約 - 必ず守ること】
1. 「問題ありません」「同意します」のみの回答は禁止
2. 最低1つの反論または問題点を必ず提示すること
3. 代替となる解釈または選択肢を必ず1つ提示すること
4. 反論には可能な限り法的根拠（引用ID）を付けること

【反論の観点（少なくとも1つを検討すること）】
- 例外規定の見落とし
- 適用範囲の限定
- 条文解釈の別の可能性
- 前提条件の妥当性
- 見落とされている関連条文
"""
    
    def __init__(self, llm, config, citation_registry: CitationRegistry):
        """
        Args:
            llm: LLMインスタンス
            config: 設定オブジェクト
            citation_registry: 引用レジストリ
        """
        super().__init__(llm, config)
        self.citation_registry = citation_registry
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        批判・反論を生成
        
        Args:
            input_data: {
                "query": str,  # 質問文
                "interpretation": Dict,  # InterpreterAgentの解釈
                "choice_analysis": Dict,  # 選択肢分析
                "recommended_answer": str,  # 推奨回答
                "citation_ids": List[str],  # 利用可能な引用ID
            }
        
        Returns:
            {
                "counter_arguments": List[Dict],  # 反論リスト（最低1つ）
                "alternative_answer": str,  # 代替回答
                "alternative_reasoning": str,  # 代替の理由
                "validation_result": Dict,  # 検証結果
                "severity": str,  # 批判の重大度
            }
        """
        query = input_data.get("query", "")
        interpretation = input_data.get("interpretation", {})
        choice_analysis = input_data.get("choice_analysis", {})
        recommended_answer = input_data.get("recommended_answer", "a")
        citation_ids = input_data.get("citation_ids", [])
        
        if not query:
            return self._empty_result("Empty query")
        
        logger.info(f"CriticAgent executing for recommended answer: {recommended_answer}")
        
        # 反論を生成
        counter_arguments = self._generate_counter_arguments(
            query, interpretation, choice_analysis, recommended_answer
        )
        
        # 反論が空の場合、強制的に生成
        if not counter_arguments:
            logger.warning("No counter arguments generated, forcing generation")
            counter_arguments = self._force_generate_counter_arguments(
                recommended_answer, choice_analysis
            )
        
        # 代替回答を生成
        alternative_answer, alternative_reasoning = self._generate_alternative(
            recommended_answer, choice_analysis, counter_arguments
        )
        
        # 引用の妥当性を検証
        validation_result = self._validate_citations(
            interpretation, citation_ids
        )
        
        # 批判の重大度を評価
        severity = self._evaluate_severity(counter_arguments, validation_result)
        
        # 推論ステップを記録
        self.citation_registry.add_reasoning_step(
            agent="CriticAgent",
            action="critique",
            claim=f"選択肢{recommended_answer}への反論: {len(counter_arguments)}件, 代替案: {alternative_answer}",
            supporting_citations=[],  # 反論には別の引用を使う場合がある
            confidence=0.0,
            metadata={
                "counter_arguments_count": len(counter_arguments),
                "severity": severity,
                "alternative_answer": alternative_answer
            }
        )
        
        return {
            "counter_arguments": counter_arguments,
            "alternative_answer": alternative_answer,
            "alternative_reasoning": alternative_reasoning,
            "validation_result": validation_result,
            "severity": severity
        }
    
    def _generate_counter_arguments(self, query: str, interpretation: Dict,
                                    choice_analysis: Dict, recommended: str) -> List[Dict]:
        """反論を生成"""
        
        interp_text = interpretation.get("interpretation", "")
        recommended_analysis = choice_analysis.get(recommended, {})
        
        prompt = f"""あなたは法律議論の批判者です。以下の解釈に対して、問題点や反論を指摘してください。

{self.MANDATORY_COUNTER_ARG_INSTRUCTION}

【質問】
{query}

【現在の解釈】
{interp_text}

【推奨された回答】
選択肢{recommended}: {recommended_analysis.get('reason', '')}

【批判の出力形式】
反論1:
- 対象: （批判対象を特定）
- 問題点: （具体的な問題点）
- 根拠: （法的根拠があれば引用ID、なければ「論理的観点から」等）
- 重大度: high/medium/low

反論2:
- 対象: ...
- 問題点: ...
- 根拠: ...
- 重大度: ...

代替案:
- 推奨選択肢: （a/b/c/d のいずれか、現在の推奨と異なるもの）
- 理由: （代替を推奨する理由）
"""
        
        response = self._safe_llm_invoke(prompt)
        
        if not response:
            return []
        
        return self._parse_counter_arguments(response)
    
    def _parse_counter_arguments(self, response: str) -> List[Dict]:
        """反論をパース"""
        counter_args = []
        
        # 反論パターンを検索
        patterns = [
            r'反論\d*[:：]?\s*[-・]?\s*対象[:：]?\s*(.+?)[-・]\s*問題点[:：]?\s*(.+?)[-・]\s*根拠[:：]?\s*(.+?)[-・]\s*重大度[:：]?\s*(high|medium|low)',
            r'[-・]\s*対象[:：]?\s*(.+?)[-・]\s*問題点[:：]?\s*(.+?)[-・]\s*根拠[:：]?\s*(.+?)[-・]\s*重大度[:：]?\s*(high|medium|low)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) >= 4:
                    counter_args.append({
                        "target": match[0].strip()[:100],
                        "objection": match[1].strip()[:200],
                        "legal_basis": match[2].strip()[:100],
                        "severity": match[3].lower()
                    })
        
        # パターンマッチしない場合は、テキストから抽出を試みる
        if not counter_args:
            # 「問題点」「懸念」「反論」などのキーワードを含む文を抽出
            lines = response.split('\n')
            for line in lines:
                if any(kw in line for kw in ['問題', '懸念', '反論', '例外', '見落とし']):
                    counter_args.append({
                        "target": "解釈全体",
                        "objection": line.strip()[:200],
                        "legal_basis": "論理的観点から",
                        "severity": "medium"
                    })
                    if len(counter_args) >= 2:
                        break
        
        return counter_args
    
    def _force_generate_counter_arguments(self, recommended: str, 
                                          choice_analysis: Dict) -> List[Dict]:
        """反論が生成されなかった場合の強制生成"""
        
        # 他の選択肢から反論を構築
        other_choices = [c for c in ["a", "b", "c", "d"] if c != recommended]
        
        forced_args = []
        for choice in other_choices[:2]:
            analysis = choice_analysis.get(choice, {})
            if analysis.get("fit") == "fit":
                forced_args.append({
                    "target": f"選択肢{recommended}の排他的選択",
                    "objection": f"選択肢{choice}も条件を満たす可能性がある: {analysis.get('reason', '')[:100]}",
                    "legal_basis": "選択肢の比較検討",
                    "severity": "medium"
                })
        
        # それでも反論がない場合
        if not forced_args:
            forced_args.append({
                "target": "解釈の前提条件",
                "objection": "提示された条文のみでは、他の選択肢を完全に排除する根拠が不十分である可能性",
                "legal_basis": "論理的観点から",
                "severity": "low"
            })
        
        return forced_args
    
    def _generate_alternative(self, recommended: str, choice_analysis: Dict,
                             counter_arguments: List[Dict]) -> tuple:
        """代替回答を生成"""
        
        # 反論から代替を抽出
        for arg in counter_arguments:
            alt_match = re.search(r'選択肢([abcd])', arg.get("objection", ""))
            if alt_match and alt_match.group(1) != recommended:
                alt = alt_match.group(1)
                return alt, arg.get("objection", "")
        
        # 他の適合選択肢を探す
        for choice in ["a", "b", "c", "d"]:
            if choice != recommended:
                analysis = choice_analysis.get(choice, {})
                if analysis.get("fit") == "fit":
                    return choice, analysis.get("reason", "代替として検討可能")
        
        # デフォルト: 推奨と異なる選択肢
        alternatives = [c for c in ["a", "b", "c", "d"] if c != recommended]
        return alternatives[0] if alternatives else "b", "批判的検討のための代替案"
    
    def _validate_citations(self, interpretation: Dict, 
                           citation_ids: List[str]) -> Dict[str, Any]:
        """引用の妥当性を検証"""
        
        related_citations = interpretation.get("related_citations", [])
        
        # 引用IDの有効性チェック
        valid_citations = [cid for cid in related_citations if cid in citation_ids]
        invalid_citations = [cid for cid in related_citations if cid not in citation_ids]
        
        # 引用の内容と解釈の整合性チェック（簡易版）
        consistency_score = 1.0 if not invalid_citations else 0.7
        
        return {
            "citation_accuracy": len(invalid_citations) == 0,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "consistency_score": consistency_score,
            "missing_citations": []  # 将来: 見落とされた引用の検出
        }
    
    def _evaluate_severity(self, counter_arguments: List[Dict],
                          validation_result: Dict) -> str:
        """批判の重大度を評価"""
        
        # 高重大度の反論があるか
        high_severity = any(arg.get("severity") == "high" for arg in counter_arguments)
        
        # 引用に問題があるか
        citation_issues = not validation_result.get("citation_accuracy", True)
        
        if high_severity or citation_issues:
            return "high"
        elif counter_arguments:
            return "medium"
        else:
            return "low"
    
    def _empty_result(self, error: str) -> Dict[str, Any]:
        """空の結果を返す"""
        return {
            "counter_arguments": [{
                "target": "解釈全体",
                "objection": "批判の生成に失敗しました",
                "legal_basis": "",
                "severity": "low"
            }],
            "alternative_answer": "b",
            "alternative_reasoning": error,
            "validation_result": {"citation_accuracy": False},
            "severity": "low"
        }

