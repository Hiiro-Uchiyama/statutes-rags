"""
Multi-Agent Debate Workflow

LangGraphを使用した議論ワークフローの実装。
"""
import os
import logging
from typing import Dict, Any, List, TypedDict
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent
sys.path.insert(0, str(debate_dir))

from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from config import MultiAgentDebateConfig
from agents import DebaterAgent, ModeratorAgent

logger = logging.getLogger(__name__)


class DebateState(TypedDict):
    """LangGraphの状態定義"""
    # 入力
    query: str
    
    # 検索結果
    documents: List[Any]
    
    # 議論状態
    round: int
    max_rounds: int
    
    # 各エージェントの主張
    debater_a_position: Dict[str, Any]
    debater_b_position: Dict[str, Any]
    
    # モデレーターの評価
    agreement_score: float
    should_continue: bool
    moderator_comment: str
    
    # 最終結果
    final_answer: str
    
    # メタデータ
    debate_history: List[Dict[str, Any]]


class DebateWorkflow:
    """Multi-Agent Debate ワークフロー"""
    
    def __init__(self, config: MultiAgentDebateConfig):
        """
        Args:
            config: Multi-Agent Debate設定
        """
        self.config = config
        
        # LLMの初期化
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = Ollama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=ollama_host,
            timeout=config.llm_timeout
        )
        
        # Retrieverの初期化
        self.retriever = self._initialize_retriever(config)
        
        # エージェントの初期化
        self.debater_a = DebaterAgent(self.llm, config, stance="affirmative")
        self.debater_b = DebaterAgent(self.llm, config, stance="critical")
        self.moderator = ModeratorAgent(self.llm, config)
        
        # グラフの構築
        self.graph = self._build_graph()
        
        logger.info("DebateWorkflow initialized")
    
    def _initialize_retriever(self, config: MultiAgentDebateConfig):
        """Retrieverを初期化"""
        try:
            # インデックスパスの構築
            from pathlib import Path
            # プロジェクトルートからの絶対パスに変換
            project_root = Path(__file__).parent.parent.parent
            base_path = Path(config.vector_store_path)
            if not base_path.is_absolute():
                base_path = project_root / base_path
            
            vector_index_path = str(base_path / "vector")
            bm25_index_path = str(base_path / "bm25")
            
            logger.info(f"Vector index path: {vector_index_path}, exists: {Path(vector_index_path).exists()}")
            logger.info(f"BM25 index path: {bm25_index_path}, exists: {Path(bm25_index_path).exists()}")
            
            # ハイブリッド検索を使用
            vector_retriever = VectorRetriever(
                embedding_model=config.embedding_model,
                index_path=vector_index_path
            )
            # 明示的にインデックスをロード
            if Path(vector_index_path).exists():
                logger.info(f"Loading vector index from {vector_index_path}")
                vector_retriever.load_index()
            
            bm25_retriever = BM25Retriever(
                index_path=bm25_index_path
            )
            # 明示的にインデックスをロード
            if Path(bm25_index_path).exists():
                logger.info(f"Loading BM25 index from {bm25_index_path}")
                bm25_retriever.load_index()
            
            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                vector_weight=0.6,
                bm25_weight=0.4
            )
            
            # インデックスが正しくロードされたか確認
            if vector_retriever.vector_store:
                doc_count = vector_retriever.vector_store.index.ntotal
                logger.info(f"Vector index loaded: {doc_count} documents")
            else:
                logger.warning("Vector index not loaded!")
                
            if bm25_retriever.bm25:
                doc_count = len(bm25_retriever.documents)
                logger.info(f"BM25 index loaded: {doc_count} documents")
            else:
                logger.warning("BM25 index not loaded!")
            
            logger.info("Retriever initialized (Hybrid)")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise
    
    def _build_graph(self) -> Any:
        """LangGraphのワークフローを構築"""
        workflow = StateGraph(DebateState)
        
        # ノードの追加
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("debate_round", self._debate_round_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # エントリーポイント
        workflow.set_entry_point("retrieve")
        
        # エッジの追加
        workflow.add_edge("retrieve", "debate_round")
        workflow.add_edge("debate_round", "evaluate")
        
        # 条件分岐: 継続 or 終了
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue_debate,
            {
                "continue": "debate_round",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        # コンパイル
        app = workflow.compile()
        return app
    
    def _retrieve_node(self, state: DebateState) -> DebateState:
        """検索ノード"""
        query = state["query"]
        
        logger.info(f"Retrieving documents for: {query}")
        
        try:
            documents = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
            state["documents"] = documents
            
            # 議論状態の初期化
            state["round"] = 1
            state["max_rounds"] = self.config.max_debate_rounds
            state["debate_history"] = []
            state["agreement_score"] = 0.0
            state["should_continue"] = True
            
            logger.info(f"Retrieved {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["documents"] = []
        
        return state
    
    def _debate_round_node(self, state: DebateState) -> DebateState:
        """議論ラウンドノード"""
        query = state["query"]
        documents = state["documents"]
        round_num = state["round"]
        
        logger.info(f"Starting debate round {round_num}")
        
        # Debater Aの主張を取得
        debater_a_input = {
            "query": query,
            "documents": documents,
            "round": round_num
        }
        
        # 2ラウンド目以降は相手の主張を含める
        if round_num > 1 and "debater_b_position" in state:
            debater_a_input["opponent_position"] = state["debater_b_position"].get("position", "")
        
        position_a = self.debater_a.execute(debater_a_input)
        state["debater_a_position"] = position_a
        
        # Debater Bの主張を取得
        debater_b_input = {
            "query": query,
            "documents": documents,
            "round": round_num
        }
        
        if round_num > 1 and "debater_a_position" in state:
            debater_b_input["opponent_position"] = position_a.get("position", "")
        
        position_b = self.debater_b.execute(debater_b_input)
        state["debater_b_position"] = position_b
        
        # 議論履歴に記録
        state["debate_history"].append({
            "round": round_num,
            "debater_a": position_a,
            "debater_b": position_b
        })
        
        logger.info(f"Debate round {round_num} completed")
        
        return state
    
    def _evaluate_node(self, state: DebateState) -> DebateState:
        """評価ノード"""
        logger.info("Evaluating debate round")
        
        moderator_input = {
            "query": state["query"],
            "documents": state["documents"],
            "debater_a_position": state["debater_a_position"],
            "debater_b_position": state["debater_b_position"],
            "round": state["round"],
            "max_rounds": state["max_rounds"]
        }
        
        result = self.moderator.execute(moderator_input)
        
        state["agreement_score"] = result["agreement_score"]
        state["should_continue"] = result["should_continue"]
        
        if result["should_continue"]:
            state["moderator_comment"] = result.get("moderator_comment", "")
            # ラウンドを進める
            state["round"] += 1
        else:
            state["final_answer"] = result.get("final_answer", "")
        
        logger.info(f"Agreement score: {state['agreement_score']:.2f}, Continue: {state['should_continue']}")
        
        return state
    
    def _finalize_node(self, state: DebateState) -> DebateState:
        """最終化ノード"""
        logger.info("Finalizing debate")
        
        # 最終回答がない場合は生成
        if not state.get("final_answer"):
            moderator_input = {
                "query": state["query"],
                "documents": state["documents"],
                "debater_a_position": state["debater_a_position"],
                "debater_b_position": state["debater_b_position"],
                "round": state["round"],
                "max_rounds": state["max_rounds"]
            }
            
            result = self.moderator.execute(moderator_input)
            state["final_answer"] = result.get("final_answer", "最終回答の生成に失敗しました。")
        
        return state
    
    def _should_continue_debate(self, state: DebateState) -> str:
        """継続判定（条件分岐用）"""
        if state["should_continue"]:
            return "continue"
        else:
            return "finalize"
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        質問に対して議論を実行
        
        Args:
            question: 法律質問
        
        Returns:
            {
                "answer": str,
                "metadata": {
                    "rounds": int,
                    "agreement_score": float,
                    "debate_history": List[Dict]
                }
            }
        """
        logger.info(f"Starting debate for query: {question}")
        
        # 初期状態
        initial_state = {
            "query": question,
            "documents": [],
            "round": 1,
            "max_rounds": self.config.max_debate_rounds,
            "debater_a_position": {},
            "debater_b_position": {},
            "agreement_score": 0.0,
            "should_continue": True,
            "moderator_comment": "",
            "final_answer": "",
            "debate_history": []
        }
        
        # グラフ実行
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "answer": final_state["final_answer"],
                "metadata": {
                    "rounds": final_state["round"],
                    "agreement_score": final_state["agreement_score"],
                    "debate_history": final_state["debate_history"],
                    "documents_count": len(final_state["documents"])
                }
            }
            
        except Exception as e:
            logger.error(f"Debate workflow failed: {e}", exc_info=True)
            return {
                "answer": f"議論の実行中にエラーが発生しました: {str(e)}",
                "metadata": {
                    "rounds": 0,
                    "agreement_score": 0.0,
                    "debate_history": [],
                    "error": str(e)
                }
            }

