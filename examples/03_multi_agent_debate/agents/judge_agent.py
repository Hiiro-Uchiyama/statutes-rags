"""
JudgeAgent - 検証・サポート役

Interpreterの判断を条文と照合して検証し、
法令CoTを補強して最終判断を確定する。
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
from .citation import CitationRegistry

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """
    検証・サポートエージェント
    
    責務:
    - Interpreterの判断を条文と照合して検証
    - 法令CoTの整合性を確認
    - 矛盾があれば修正を提案
    - 最終的な回答と確信度を確定
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
        検証・サポートを実行
        
        Args:
            input_data: {
                "query": str,  # 質問文
                "interpretation": Dict,  # 解釈結果
                "choice_analysis": Dict,  # 選択肢分析
                "interpreter_answer": str,  # 解釈者の推奨
                "interpreter_confidence": float,  # 解釈者の確信度
                "citation_ids": List[str],  # 利用可能な引用
                "legal_cot": str,  # Interpreterの法令CoT
            }
        
        Returns:
            {
                "final_answer": str,  # 最終回答
                "confidence": float,  # 最終確信度
                "reasoning": str,  # 判断理由
                "citation_chain": List[str],  # 根拠となる引用チェーン
                "verified_cot": str,  # 検証済み法令CoT
                "judgment_summary": Dict,  # 判断サマリー
            }
        """
        query = input_data.get("query", "")
        interpretation = input_data.get("interpretation", {})
        choice_analysis = input_data.get("choice_analysis", {})
        interpreter_answer = input_data.get("interpreter_answer", "a")
        interpreter_confidence = input_data.get("interpreter_confidence", 0.5)
        citation_ids = input_data.get("citation_ids", [])
        legal_cot = input_data.get("legal_cot", "")
        
        if not query:
            return self._empty_result("Empty query")
        
        logger.info(f"JudgeAgent verifying: interpreter recommended '{interpreter_answer}'")
        
        # 検索された条文を取得
        citations_text = self._build_citations_context(citation_ids)
        
        # Interpreterの判断を検証
        verification = self._verify_interpretation(
            query, citations_text, choice_analysis, 
            interpreter_answer, legal_cot
        )
        
        # 検証結果に基づき最終判断
        if verification.get("verified", True):
            # 検証OK: Interpreterの判断を採用
            final_answer = interpreter_answer
            confidence = min(interpreter_confidence + 0.1, 0.95)  # 検証済みで確信度UP
            verified_cot = verification.get("enhanced_cot", legal_cot)
        else:
            # 検証NG: 修正提案を採用
            suggested = verification.get("suggested_answer", interpreter_answer)
            final_answer = suggested if suggested in ["a", "b", "c", "d"] else interpreter_answer
            confidence = verification.get("confidence", interpreter_confidence)
            verified_cot = verification.get("correction_reasoning", legal_cot)
            
            if final_answer != interpreter_answer:
                logger.info(f"JudgeAgent corrected: {interpreter_answer} -> {final_answer}")
        
        # 引用チェーンを構築
        citation_chain = self._build_citation_chain(
            final_answer, choice_analysis, citation_ids
        )
        
        # 推論ステップを記録
        self.citation_registry.add_reasoning_step(
            agent="JudgeAgent",
            action="verify",
            claim=f"最終回答: 選択肢{final_answer}（確信度: {confidence:.2f}）",
            supporting_citations=citation_chain,
            confidence=confidence,
            metadata={
                "interpreter_answer": interpreter_answer,
                "verified": verification.get("verified", True),
                "changed": final_answer != interpreter_answer
            }
        )
        
        # 判断サマリーを作成
        judgment_summary = {
            "interpreter_recommended": interpreter_answer,
            "final_decision": final_answer,
            "decision_changed": final_answer != interpreter_answer,
            "verification_passed": verification.get("verified", True),
            "verification_issues": verification.get("issues", []),
            "needs_discussion": verification.get("needs_discussion", False)
        }
        
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "reasoning": verification.get("reasoning", ""),
            "citation_chain": citation_chain,
            "verified_cot": verified_cot,
            "unresolved_concerns": [],
            "judgment_summary": judgment_summary
        }
    
    def _build_citations_context(self, citation_ids: List[str]) -> str:
        """検索された条文のコンテキストを構築"""
        context_parts = []
        
        for cid in citation_ids[:12]:  # 上位12件
            citation = self.citation_registry.get(cid)
            if citation:
                ref = citation.to_reference_string()
                text = citation.text[:500] if len(citation.text) > 500 else citation.text
                context_parts.append(f"[{cid}] {ref}\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _verify_interpretation(self, query: str, citations_text: str,
                               choice_analysis: Dict, interpreter_answer: str,
                               legal_cot: str) -> Dict[str, Any]:
        """Interpreterの思考を検証（妥当性判断に特化）"""
        
        prompt = f"""あなたはInterpreterの思考を検証する役割です。
Interpreterが考えた内容を読み、その思考過程が妥当かを判断してください。

【質問】
{query}

【検索された法令条文】
{citations_text}

【Interpreterの思考】
「{legal_cot}」

Interpreterの回答: {interpreter_answer}

【あなたの役割】
Interpreterの思考を読んで、以下を判断してください：

1. 思考の論理は通っているか？
2. 条文の引用は正しいか？
3. 質問タイプ（正しいもの/誤っているもの）の判断は合っているか？
4. 結論は思考と整合しているか？

もし問題があれば、何が問題かを指摘し、別の答えを提案してください。
問題がなければ、Interpreterの回答を承認してください。

【回答形式】
判定: （承認/要再検討）

私の検証:
（Interpreterの思考を読んで感じたこと、問題点や同意点を自然に書いてください）

最終回答: （a/b/c/d）
確信度: （0.0-1.0）
"""
        
        response = self._safe_llm_invoke(prompt)
        
        if not response:
            return {"verified": True, "reasoning": "検証処理に失敗"}
        
        return self._parse_verification(response, interpreter_answer)
    
    def _parse_verification(self, response: str, interpreter_answer: str) -> Dict[str, Any]:
        """検証結果をパース（簡素化版）"""
        result = {
            "verified": True,
            "reasoning": "",
            "issues": [],
            "suggested_answer": interpreter_answer,
            "confidence": 0.5,
            "enhanced_cot": "",
            "correction_reasoning": "",
            "needs_discussion": False
        }
        
        # 判定を抽出
        verify_match = re.search(r'判定[:：]\s*(承認|要再検討)', response)
        if verify_match:
            result["verified"] = verify_match.group(1) == "承認"
            result["needs_discussion"] = verify_match.group(1) == "要再検討"
        
        # 「私の検証」セクションを抽出
        thinking_match = re.search(r'私の検証[:：]?\s*\n?(.*?)(?=最終回答[:：]|$)', response, re.DOTALL)
        if thinking_match:
            result["reasoning"] = thinking_match.group(1).strip()[:500]
            result["enhanced_cot"] = result["reasoning"]
        
        # 最終回答を抽出
        answer_patterns = [
            r'最終回答[:：]\s*([abcd])',
            r'回答[:：]\s*([abcd])',
        ]
        for pattern in answer_patterns:
            answer_match = re.search(pattern, response, re.IGNORECASE)
            if answer_match:
                result["suggested_answer"] = answer_match.group(1).lower()
                break
        
        # 確信度を抽出
        conf_match = re.search(r'確信度[:：]\s*([\d.]+)', response)
        if conf_match:
            try:
                result["confidence"] = min(float(conf_match.group(1)), 1.0)
            except ValueError:
                pass
        
        # 問題点があれば記録
        if not result["verified"]:
            result["issues"].append(result["reasoning"][:200])
        
        return result
    
    def _build_citation_chain(self, final_answer: str,
                              choice_analysis: Dict,
                              citation_ids: List[str]) -> List[str]:
        """最終回答を支持する引用チェーンを構築"""
        
        chain = []
        
        # 選択肢分析から引用を取得
        if final_answer in choice_analysis:
            citations = choice_analysis[final_answer].get("citations", [])
            chain.extend([c for c in citations if c in citation_ids])
        
        # チェーンが空の場合、上位の引用を使用
        if not chain and citation_ids:
            chain = citation_ids[:3]
        
        return chain
    
    def _empty_result(self, error: str) -> Dict[str, Any]:
        """空の結果を返す"""
        return {
            "final_answer": "a",
            "confidence": 0.0,
            "reasoning": error,
            "citation_chain": [],
            "verified_cot": "",
            "unresolved_concerns": [error],
            "judgment_summary": {}
        }
