"""
提案手法 v3: 選択肢ごとの検証 + 統合 + 再確認 + 最終判定

フロー:
1. RetrieverAgent - 初期検索
2. ChoiceVerifierAgent - 選択肢ごとに条文照合
3. IntegratorAgent - 検証結果を統合、法令CoT生成
4. RetrieverAgent - 再確認検索（不確実な部分を補強）
5. JudgeAgent - 法令CoTを基に最終判定
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_community.llms import Ollama

# エージェント
from agents.choice_verifier_agent import ChoiceVerifierAgent
from agents.integrator_agent import IntegratorAgent

logger = logging.getLogger(__name__)


@dataclass 
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 120
    num_ctx: int = 16000
    top_k: int = 30
    context_docs: int = 15
    enable_recheck: bool = True


class ProposedWorkflowV3:
    """提案手法v3のワークフロー"""
    
    def __init__(
        self,
        retriever,
        config: Optional[WorkflowConfig] = None
    ):
        self.retriever = retriever
        self.config = config or WorkflowConfig()
        
        # LLM
        self.llm = Ollama(
            model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx
        )
        
        # エージェント
        self.choice_verifier = ChoiceVerifierAgent(
            llm_model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx
        )
        self.integrator = IntegratorAgent(
            llm_model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx
        )
    
    def query(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """メインクエリ処理"""
        
        logger.info("=== Proposed Workflow v3 ===")
        
        # Step 1: 初期検索
        logger.info("Step 1: Initial Retrieval")
        initial_docs = self._retrieve(question, choices)
        initial_context = self._build_context(initial_docs)
        
        # Step 2: 選択肢ごとの検証
        logger.info("Step 2: Choice Verification")
        verification_result = self.choice_verifier.verify_all_choices(
            question=question,
            choices=choices,
            context=initial_context
        )
        
        # Step 3: 統合
        logger.info("Step 3: Integration")
        integration_result = self.integrator.integrate(
            question=question,
            choices=choices,
            verification_result=verification_result
        )
        
        # Step 4: 再確認検索（必要な場合）
        recheck_context = ""
        if self.config.enable_recheck and integration_result.recheck_queries:
            logger.info("Step 4: Recheck Retrieval")
            recheck_context = self._recheck_retrieval(integration_result.recheck_queries)
        
        # Step 5: 最終判定
        logger.info("Step 5: Final Judgment")
        final_result = self._final_judgment(
            question=question,
            choices=choices,
            verification_result=verification_result,
            integration_result=integration_result,
            recheck_context=recheck_context
        )
        
        return {
            "answer": final_result["answer"],
            "confidence": final_result["confidence"],
            "legal_cot": integration_result.legal_cot,
            "verification_summary": verification_result.get("summary", {}),
            "uncertain_points": integration_result.uncertain_points,
            "stages": {
                "initial_docs": len(initial_docs),
                "verifications": len(verification_result.get("verifications", [])),
                "provisional_answer": integration_result.provisional_answer,
                "recheck_queries": len(integration_result.recheck_queries),
                "final_answer": final_result["answer"]
            }
        }
    
    def _retrieve(self, question: str, choices: List[str]) -> List:
        """検索実行"""
        from app.utils.number_normalizer import normalize_article_numbers
        
        full_query = question + " " + " ".join(choices)
        normalized = normalize_article_numbers(full_query, to_kanji=True)
        
        return self.retriever.retrieve(normalized, top_k=self.config.top_k)
    
    def _build_context(self, docs: List, max_docs: int = None) -> str:
        """コンテキスト構築"""
        max_docs = max_docs or self.config.context_docs
        parts = []
        for i, doc in enumerate(docs[:max_docs]):
            law = doc.metadata.get("law_title", "")
            article = doc.metadata.get("article_title", "")
            text = doc.page_content[:600]
            parts.append(f"[{i+1}] {law} {article}\n{text}")
        return "\n\n".join(parts)
    
    def _recheck_retrieval(self, queries: List[str]) -> str:
        """再確認検索"""
        from app.utils.number_normalizer import normalize_article_numbers
        
        all_docs = []
        for query in queries[:3]:  # 最大3クエリ
            normalized = normalize_article_numbers(query, to_kanji=True)
            docs = self.retriever.retrieve(normalized, top_k=10)
            all_docs.extend(docs)
        
        # 重複除去
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return self._build_context(unique_docs, max_docs=10)
    
    def _final_judgment(
        self,
        question: str,
        choices: List[str],
        verification_result: Dict[str, Any],
        integration_result,
        recheck_context: str
    ) -> Dict[str, Any]:
        """最終判定"""
        
        # 検証サマリ
        verifications = verification_result.get("verifications", [])
        summary = verification_result.get("summary", {})
        
        # 質問タイプ
        question_type = verification_result.get("question_type", "unknown")
        
        # 暫定回答の確信度が高ければそのまま採用
        if integration_result.confidence >= 0.8:
            logger.info(f"High confidence ({integration_result.confidence}), using provisional answer: {integration_result.provisional_answer}")
            return {
                "answer": integration_result.provisional_answer,
                "confidence": integration_result.confidence,
                "response": "High confidence - using provisional answer"
            }
        
        # 検証結果に基づく判定
        matched = summary.get("matched_choices", [])
        unmatched = summary.get("unmatched_choices", [])
        
        if question_type == "incorrect":
            # 「誤っているもの」を選ぶ → 不一致を選ぶ
            if unmatched:
                # 不一致の中から最も確信度が高いものを選択
                best_unmatched = None
                best_conf = 0
                for v in verifications:
                    if v.choice_id in unmatched and v.confidence > best_conf:
                        best_unmatched = v.choice_id
                        best_conf = v.confidence
                if best_unmatched:
                    logger.info(f"Selecting unmatched choice for 'incorrect' question: {best_unmatched}")
                    return {
                        "answer": best_unmatched,
                        "confidence": best_conf,
                        "response": f"Selected unmatched choice: {best_unmatched}"
                    }
        elif question_type == "correct":
            # 「正しいもの」を選ぶ → 一致を選ぶ
            if matched:
                best_matched = None
                best_conf = 0
                for v in verifications:
                    if v.choice_id in matched and v.confidence > best_conf:
                        best_matched = v.choice_id
                        best_conf = v.confidence
                if best_matched:
                    logger.info(f"Selecting matched choice for 'correct' question: {best_matched}")
                    return {
                        "answer": best_matched,
                        "confidence": best_conf,
                        "response": f"Selected matched choice: {best_matched}"
                    }
        
        # フォールバック: 暫定回答を使用
        logger.info(f"Fallback to provisional answer: {integration_result.provisional_answer}")
        return {
            "answer": integration_result.provisional_answer,
            "confidence": integration_result.confidence,
            "response": "Fallback to provisional answer"
        }

        try:
            response = self.llm.invoke(prompt)
            answer = self._extract_answer(response)
            return {
                "answer": answer,
                "confidence": integration_result.confidence,
                "response": response
            }
        except Exception as e:
            logger.error(f"Final judgment failed: {e}")
            return {
                "answer": integration_result.provisional_answer,
                "confidence": integration_result.confidence,
                "response": f"Error: {str(e)}"
            }
    
    def _format_choices(self, choices: List[str]) -> str:
        """選択肢をフォーマット"""
        choice_ids = ['a', 'b', 'c', 'd']
        return "\n".join([f"{choice_ids[i]}. {c}" for i, c in enumerate(choices) if i < len(choice_ids)])
    
    def _extract_answer(self, response: str) -> str:
        """回答を抽出"""
        response_lower = response.lower().strip()
        for char in ['a', 'b', 'c', 'd']:
            if char in response_lower[:50]:
                return char
        return 'a'

