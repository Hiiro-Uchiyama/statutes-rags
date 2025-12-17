"""
Proposed Workflow - 提案手法のワークフロー

3役割エージェント + 法的根拠共有型CoT

【主要な貢献】
1. 法務ドメイン特化型3役割分離アーキテクチャ
   - RetrieverAgent: 法令検索・条文特定
   - InterpreterAgent: 法解釈・選択肢分析
   - JudgeAgent: 最終判断・確信度評価

2. 法的根拠共有型CoT（Citation-Grounded Chain-of-Thought）
   - CitationRegistryによる条文引用の一元管理
   - 推論チェーンの監査トレイル
   - 各エージェントの判断根拠を透明化
"""
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent
sys.path.insert(0, str(debate_dir))

from langchain_community.llms import Ollama

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from config import MultiAgentDebateConfig
from agents import (
    RetrieverAgent,
    InterpreterAgent,
    JudgeAgent,
    CitationRegistry,
)

logger = logging.getLogger(__name__)


class ProposedWorkflow:
    """
    提案手法のワークフロー
    
    3役割エージェント構成:
    1. RetrieverAgent: 法令検索・関連条文の特定
    2. InterpreterAgent: 法解釈・選択肢の適合性分析
    3. JudgeAgent: 最終判断・確信度評価
    
    【設計原則】
    - 法務実務のワークフロー（事実認定→法令検索→解釈→結論）に基づく役割設計
    - 各エージェントは単一責任を持つ
    - CitationRegistryを通じて法的根拠を共有
    """
    
    def __init__(self, config: MultiAgentDebateConfig, 
                 index_path: Optional[str] = None):
        """
        Args:
            config: 設定
            index_path: FAISSインデックスのパス
        """
        self.config = config
        
        # LLMの初期化
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = Ollama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=ollama_host,
            timeout=config.llm_timeout,
            num_ctx=config.llm_num_ctx
        )
        
        # Retrieverの初期化
        self.retriever = self._initialize_retriever(config, index_path)
        
        # CitationRegistryの初期化（法的根拠共有の核心）
        self.citation_registry = CitationRegistry()
        
        # 3役割エージェントの初期化
        self.retriever_agent = RetrieverAgent(
            self.llm, config, self.retriever, self.citation_registry
        )
        self.interpreter_agent = InterpreterAgent(
            self.llm, config, self.citation_registry
        )
        self.judge_agent = JudgeAgent(
            self.llm, config, self.citation_registry
        )
        
        logger.info("ProposedWorkflow initialized with 3-role agents")
    
    def _initialize_retriever(self, config: MultiAgentDebateConfig,
                              index_path: Optional[str] = None,
                              use_hybrid: bool = True):
        """Retrieverを初期化（Hybrid検索対応）"""
        try:
            # インデックスパスの決定
            if index_path:
                base_index_path = index_path
            elif config.vector_store_path:
                base_index_path = str(project_root / config.vector_store_path)
            else:
                base_index_path = str(project_root / "data" / "faiss_index_full")
            
            vector_index_path = f"{base_index_path}/vector"
            bm25_index_path = f"{base_index_path}/bm25"
            
            logger.info(f"Index base path: {base_index_path}")
            
            # Vector Retrieverの初期化
            if not Path(vector_index_path).exists():
                raise FileNotFoundError(f"Vector index not found: {vector_index_path}")
            
            vector_retriever = VectorRetriever(
                embedding_model=config.embedding_model,
                index_path=vector_index_path
            )
            vector_retriever.load_index()
            
            if vector_retriever.vector_store:
                doc_count = vector_retriever.vector_store.index.ntotal
                logger.info(f"Vector index loaded: {doc_count} documents")
            
            # BM25が存在し、Hybrid検索が有効な場合
            if use_hybrid and Path(bm25_index_path).exists():
                logger.info(f"BM25 index found: {bm25_index_path}")
                bm25_retriever = BM25Retriever(index_path=bm25_index_path)
                bm25_retriever.load_index()
                logger.info(f"BM25 index loaded: {len(bm25_retriever.documents)} documents")
                
                # HybridRetrieverを作成
                retriever = HybridRetriever(
                    vector_retriever=vector_retriever,
                    bm25_retriever=bm25_retriever,
                    fusion_method="rrf",
                    vector_weight=0.5,
                    bm25_weight=0.5,
                    rrf_k=60
                )
                logger.info("Hybrid retriever initialized (Vector + BM25)")
                return retriever
            else:
                logger.info("Using Vector retriever only (BM25 not available)")
                return vector_retriever
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise
    
    def query(self, question: str, choices: List[str] = None) -> Dict[str, Any]:
        """
        質問に対して3役割パイプラインを実行
        
        ワークフロー:
        1. RetrieverAgent: 関連法令を検索し、CitationRegistryに登録
        2. InterpreterAgent: 法令を解釈し、選択肢の適合性を分析
        3. JudgeAgent: 解釈を評価して最終判断
        
        Args:
            question: 質問文
            choices: 選択肢リスト（オプション）
        
        Returns:
            {
                "answer": str,  # 最終回答 (a/b/c/d)
                "confidence": float,  # 確信度 (0.0-1.0)
                "reasoning": str,  # 判断理由
                "metadata": Dict,  # メタデータ
            }
        """
        logger.info(f"ProposedWorkflow starting for: {question[:100]}...")
        
        # CitationRegistryをクリア
        self.citation_registry.clear()
        
        # 選択肢を抽出（提供されていない場合）
        if choices is None:
            choices = self._extract_choices(question)
        
        try:
            # Step 1: 法令検索（RetrieverAgent）
            retriever_result = self.retriever_agent.execute({
                "query": question,
                "choices": choices
            })
            citation_ids = retriever_result.get("citation_ids", [])
            
            if not citation_ids:
                logger.warning("No citations found")
                return self._fallback_result("No relevant laws found")
            
            # Step 2: 法解釈（InterpreterAgent）- Retrieverの思考を共有
            interpreter_result = self.interpreter_agent.execute({
                "query": question,
                "citation_ids": citation_ids,
                "choices": choices,
                "retrieval_cot": retriever_result.get("retrieval_cot", "")
            })
            
            interpreter_answer = interpreter_result.get("recommended_answer", "a")
            interpreter_confidence = interpreter_result.get("confidence", 0.5)
            legal_cot = interpreter_result.get("legal_cot", "")
            
            # Step 3: 検証（JudgeAgent）- Interpreterの思考を検証
            judge_result = self.judge_agent.execute({
                "query": question,
                "interpretation": interpreter_result.get("interpretation", {}),
                "choice_analysis": interpreter_result.get("choice_analysis", {}),
                "interpreter_answer": interpreter_answer,
                "interpreter_confidence": interpreter_confidence,
                "citation_ids": citation_ids,
                "legal_cot": legal_cot
            })
            
            # Step 4: 議論ラウンド（Judgeが要再検討の場合）
            discussion_round = 0
            max_discussion_rounds = 1  # 最大1回の追加議論
            
            judgment_summary = judge_result.get("judgment_summary", {})
            needs_discussion = judgment_summary.get("needs_discussion", False)
            
            if needs_discussion and discussion_round < max_discussion_rounds:
                logger.info("Discussion round triggered")
                discussion_round += 1
                
                # Judgeの指摘を含めてInterpreterに再考を依頼
                judge_reasoning = judge_result.get("reasoning", "")
                
                # 追加の検索（Judgeの指摘に基づく）
                additional_result = self.retriever_agent.execute({
                    "query": f"{question} {judge_reasoning[:200]}",
                    "choices": choices
                })
                additional_citations = additional_result.get("citation_ids", [])
                
                # 既存の引用と統合
                all_citation_ids = list(set(citation_ids + additional_citations))
                
                # Interpreterに再考を依頼（Judgeの思考を共有）
                reinterpret_result = self.interpreter_agent.execute({
                    "query": question,
                    "citation_ids": all_citation_ids,
                    "choices": choices,
                    "retrieval_cot": f"Judgeからの指摘: {judge_reasoning}\n追加検索結果あり。"
                })
                
                # 更新
                interpreter_answer = reinterpret_result.get("recommended_answer", interpreter_answer)
                interpreter_confidence = reinterpret_result.get("confidence", interpreter_confidence)
                legal_cot = reinterpret_result.get("legal_cot", legal_cot)
                citation_ids = all_citation_ids
                
                # 最終検証
                judge_result = self.judge_agent.execute({
                    "query": question,
                    "interpretation": reinterpret_result.get("interpretation", {}),
                    "choice_analysis": reinterpret_result.get("choice_analysis", {}),
                    "interpreter_answer": interpreter_answer,
                    "interpreter_confidence": interpreter_confidence,
                    "citation_ids": citation_ids,
                    "legal_cot": legal_cot
            })
            
            # 結果を構築
            final_answer = judge_result.get("final_answer", interpreter_answer)
            confidence = judge_result.get("confidence", interpreter_confidence)
            
            # メタデータ（詳細履歴を含む）
            metadata = {
                # 基本情報
                "citation_count": len(citation_ids),
                "citation_rate": self.citation_registry.calculate_citation_rate(),
                "interpreter_answer": interpreter_answer,
                "interpreter_confidence": interpreter_confidence,
                "discussion_rounds": discussion_round,
                
                # 詳細履歴（分析用）
                "agent_outputs": {
                    "retriever": {
                        "citation_ids": citation_ids,
                        "search_queries": retriever_result.get("search_queries", []),
                        "unique_laws": retriever_result.get("metadata", {}).get("unique_laws", 0),
                        "retrieval_cot": retriever_result.get("retrieval_cot", ""),
                        "extracted_keywords": retriever_result.get("extracted_keywords", {})
                    },
                    "interpreter": {
                        "interpretation_summary": interpreter_result.get("interpretation", {}).get("summary", ""),
                        "choice_analysis": interpreter_result.get("choice_analysis", {}),
                        "supporting_citations": interpreter_result.get("supporting_citations", []),
                        "legal_cot": interpreter_result.get("legal_cot", "")
                    },
                    "judge": {
                        "verified_cot": judge_result.get("verified_cot", ""),
                        "verification_passed": judge_result.get("judgment_summary", {}).get("verification_passed", True),
                        "reasoning": judge_result.get("reasoning", ""),
                        "citation_chain": judge_result.get("citation_chain", []),
                        "judgment_summary": judge_result.get("judgment_summary", {})
                    }
                },
                
                # 監査トレイル（CitationRegistry）
                "audit_trail": self.citation_registry.get_audit_summary()
            }
            
            logger.info(f"ProposedWorkflow completed: answer={final_answer}, confidence={confidence:.2f}")
            
            return {
                "answer": final_answer,
                "confidence": confidence,
                "reasoning": judge_result.get("reasoning", ""),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"ProposedWorkflow failed: {e}", exc_info=True)
            return self._fallback_result(str(e))
    
    def _extract_choices(self, question: str) -> List[str]:
        """質問文から選択肢を抽出"""
        import re
        
        choices = []
        pattern = r'([a-d])[\.\)]\s*(.+?)(?=[a-d][\.\)]|$)'
        matches = re.findall(pattern, question, re.DOTALL | re.IGNORECASE)
        
        for label, text in matches:
            choices.append(text.strip()[:200])
        
        while len(choices) < 4:
            choices.append(f"選択肢{chr(ord('a') + len(choices))}")
        
        return choices[:4]
    
    def _fallback_result(self, error: str) -> Dict[str, Any]:
        """フォールバック結果"""
        return {
            "answer": "a",
            "confidence": 0.0,
            "reasoning": f"Error: {error}",
            "metadata": {
                "error": error,
                "citation_count": 0,
                "audit_trail": {}
            }
        }


def create_proposed_workflow(
    config: Optional[MultiAgentDebateConfig] = None,
    index_path: Optional[str] = None
) -> ProposedWorkflow:
    """
    ProposedWorkflowを作成する便利関数
    
    Args:
        config: 設定（Noneの場合はデフォルト設定）
        index_path: FAISSインデックスのパス
    
    Returns:
        ProposedWorkflow インスタンス
    """
    if config is None:
        from config import load_config
        config = load_config()
    
    return ProposedWorkflow(config, index_path)
