"""
InterpreterAgent - 法解釈役

検索された法令条文を解釈し、
各選択肢の適合性を分析する。
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


class InterpreterAgent(BaseAgent):
    """
    法解釈エージェント
    
    責務:
    - 検索された法令条文の解釈
    - 各選択肢への適用可能性の分析
    - 法的根拠に基づく推論の提示
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
        法令解釈を実行
        
        Args:
            input_data: {
                "query": str,  # 質問文
                "citation_ids": List[str],  # 検索された引用ID
                "choices": List[str],  # 選択肢
                "retrieval_cot": str,  # RetrieverからのCoT（検索意図）
            }
        
        Returns:
            {
                "interpretation": Dict,  # 解釈結果
                "choice_analysis": Dict[str, Dict],  # 選択肢ごとの分析
                "recommended_answer": str,  # 推奨回答
                "confidence": float,  # 確信度
                "supporting_citations": List[str],  # 根拠となる引用
                "legal_cot": str,  # 法令CoT（推論チェーン）
            }
        """
        query = input_data.get("query", "")
        citation_ids = input_data.get("citation_ids", [])
        choices = input_data.get("choices", [])
        retrieval_cot = input_data.get("retrieval_cot", "")
        
        if not query:
            return self._empty_result("Empty query")
        
        logger.info(f"InterpreterAgent executing for query: {query[:100]}...")
        
        # 引用された条文を取得
        citations = []
        for cid in citation_ids[:15]:  # 上位15件
            citation = self.citation_registry.get(cid)
            if citation:
                citations.append(citation)
        
        # 条文のコンテキストを構築
        context = self._build_citation_context(citations)
        
        # 統合解釈を生成（1回のLLM呼び出しで完結）
        result = self._generate_unified_interpretation(query, context, choices, retrieval_cot)
        
        recommended_answer = result.get("recommended_answer", "a")
        confidence = result.get("confidence", 0.5)
        choice_analysis = result.get("choice_analysis", {})
        legal_cot = result.get("legal_cot", "")
        
        # 根拠となる引用を特定
        supporting_citations = self._identify_supporting_citations(
            citation_ids, recommended_answer, choice_analysis
        )
        
        # 推論ステップを記録
        self.citation_registry.add_reasoning_step(
            agent="InterpreterAgent",
            action="interpret",
            claim=f"選択肢{recommended_answer}が最も適合（確信度: {confidence:.2f}）",
            supporting_citations=supporting_citations,
            confidence=confidence,
            metadata={
                "choice_analysis": choice_analysis,
                "legal_cot": legal_cot[:500]
            }
        )
        
        # 使用した引用のステータスを更新
        for cid in supporting_citations:
            self.citation_registry.update_status(cid, CitationStatus.VERIFIED)
        
        return {
            "interpretation": result.get("interpretation", {}),
            "choice_analysis": choice_analysis,
            "recommended_answer": recommended_answer,
            "confidence": confidence,
            "supporting_citations": supporting_citations,
            "legal_cot": legal_cot
        }
    
    def _build_citation_context(self, citations: List[Any]) -> str:
        """引用から解釈用コンテキストを構築"""
        context_parts = []
        
        for citation in citations:
            ref = citation.to_reference_string()
            text = citation.text[:600] if len(citation.text) > 600 else citation.text
            context_parts.append(f"[{citation.citation_id}] {ref}\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _generate_unified_interpretation(self, query: str, context: str, 
                                         choices: List[str], retrieval_cot: str = "") -> Dict[str, Any]:
        """統合解釈を1回のLLM呼び出しで生成（簡素化版）"""
        
        choices_text = "\n".join([f"{chr(ord('a') + i)}. {c}" for i, c in enumerate(choices[:4])])
        
        # RetrieverからのCoTがある場合は含める（思考の流れとして共有）
        retrieval_section = ""
        if retrieval_cot:
            retrieval_section = f"""
【Retrieverの思考】
{retrieval_cot}
"""
        
        prompt = f"""あなたは法律の専門家です。法令条文に基づいて質問に回答してください。
{retrieval_section}
【関連法令条文】
{context}

【質問】
{query}

【選択肢】
{choices_text}

まず、この質問が「正しいものを選ぶ」のか「誤っているものを選ぶ」のかを判断してください。

次に、各選択肢について条文と照らし合わせ、最も適切な回答を選んでください。

あなたの思考過程を自然に書いてください。後でJudgeがこの思考を検証します。

【回答形式】
質問タイプ: （正しいもの/誤っているもの）

私の考え:
（ここに自然な思考過程を記述。条文のどの部分がどの選択肢に関係するか、
なぜその答えを選んだかを、あなたの言葉で説明してください。）

回答: （a/b/c/d）
確信度: （0.0-1.0）
"""
        
        response = self._safe_llm_invoke(prompt)
        
        if not response:
            return {
                "interpretation": {},
                "choice_analysis": {},
                "recommended_answer": "a",
                "confidence": 0.3,
                "legal_cot": ""
            }
        
        return self._parse_unified_response(response)
    
    def _parse_unified_response(self, response: str) -> Dict[str, Any]:
        """統合応答をパース（簡素化版）"""
        result = {
            "interpretation": {"raw": response},
            "choice_analysis": {},
            "recommended_answer": "a",
            "confidence": 0.5,
            "legal_cot": ""
        }
        
        # 質問タイプを抽出
        type_match = re.search(r'質問タイプ[:：]\s*(.+?)(?=\n|$)', response)
        if type_match:
            result["interpretation"]["question_type"] = type_match.group(1).strip()
        
        # 「私の考え」セクションを法令CoTとして抽出
        thinking_match = re.search(r'私の考え[:：]?\s*\n?(.*?)(?=回答[:：]|$)', response, re.DOTALL)
        if thinking_match:
            result["legal_cot"] = thinking_match.group(1).strip()[:800]
        
        # 回答を抽出（複数パターン対応）
        answer_patterns = [
            r'回答[:：]\s*([abcd])',
            r'最終回答[:：]\s*([abcd])',
            r'答え[:：]\s*([abcd])',
            r'正解[:：]\s*([abcd])',
        ]
        for pattern in answer_patterns:
            answer_match = re.search(pattern, response, re.IGNORECASE)
            if answer_match:
                result["recommended_answer"] = answer_match.group(1).lower()
                break
        
        # 確信度を抽出
        conf_match = re.search(r'確信度[:：]\s*([\d.]+)', response)
        if conf_match:
            try:
                result["confidence"] = min(float(conf_match.group(1)), 1.0)
            except ValueError:
                pass
        
        # 引用IDを抽出（思考全体から）
        citations = re.findall(r'C\d{3}', response)
        result["interpretation"]["citations"] = list(set(citations))
        
        # 選択肢分析は簡略化（思考から推測）
        for choice in ["a", "b", "c", "d"]:
            result["choice_analysis"][choice] = {
                "fit": "selected" if choice == result["recommended_answer"] else "not_selected",
                "reason": "",
                "citations": []
            }
        
        return result
    
    def _identify_supporting_citations(self, citation_ids: List[str], 
                                       recommended: str,
                                       choice_analysis: Dict) -> List[str]:
        """推奨回答を支持する引用を特定"""
        supporting = []
        
        # 選択肢分析から引用を取得
        if recommended in choice_analysis:
            supporting.extend(choice_analysis[recommended].get("citations", []))
        
        # 有効な引用IDのみ残す
        valid_ids = [cid for cid in supporting if cid in citation_ids]
        
        # 引用がない場合は上位の引用を使用
        if not valid_ids and citation_ids:
            valid_ids = citation_ids[:3]
        
        return valid_ids
    
    def _empty_result(self, error: str) -> Dict[str, Any]:
        """空の結果を返す"""
        return {
            "interpretation": {"summary": "", "interpretation": ""},
            "choice_analysis": {},
            "recommended_answer": "a",
            "confidence": 0.0,
            "supporting_citations": [],
            "legal_cot": "",
            "error": error
        }
