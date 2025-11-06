"""
Agentic RAG Pipeline

LangGraphを使用して、複数のエージェントを協調させるパイプライン。
"""
import os
import logging
from typing import Dict, Any, List, TypedDict
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from config import AgenticRAGConfig
from agents import (
    ManagerAgent,
    RetrievalAgent,
    ReasoningAgent,
    ValidationAgent,
)

logger = logging.getLogger(__name__)


class AgenticRAGState(TypedDict):
    """LangGraphの状態定義"""
    # 入力
    query: str
    
    # クエリ分析結果
    complexity: str
    query_type: str
    
    # 検索結果
    documents: List[Any]
    retrieval_strategy: str
    
    # 推論結果
    reasoning: str
    legal_structure: Dict[str, Any]
    
    # 回答
    answer: str
    citations: List[Dict[str, Any]]
    
    # メタデータ
    iteration: int
    max_iterations: int
    confidence: float
    agents_used: List[str]
    
    # 制御フラグ
    needs_retry: bool
    is_valid: bool


class AgenticRAGPipeline:
    """Agentic RAGパイプライン"""
    
    def __init__(self, config: AgenticRAGConfig):
        """
        Args:
            config: Agentic RAG設定
        """
        self.config = config
        
        # LLMの初期化
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = OllamaLLM(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=ollama_host,
            timeout=config.llm_timeout
        )
        
        # Retrieverの初期化
        self.retrievers = self._initialize_retrievers(config)
        
        # エージェントの初期化
        self.manager = ManagerAgent(self.llm, config)
        self.retrieval = RetrievalAgent(self.llm, config, self.retrievers)
        self.reasoning = ReasoningAgent(self.llm, config)
        self.validation = ValidationAgent(self.llm, config)
        
        # グラフの構築
        self.graph = self._build_graph()
        
        logger.info("AgenticRAGPipeline initialized")
    
    def _initialize_retrievers(self, config: AgenticRAGConfig) -> Dict[str, Any]:
        """Retrieverを初期化"""
        # パスの解決（プロジェクトルートからの相対パス）
        vector_store_path = Path(config.vector_store_path)
        if not vector_store_path.is_absolute():
            vector_store_path = project_root / vector_store_path
        
        logger.info(f"Vector store path: {vector_store_path}")
        
        # Vector Retriever
        vector_retriever = VectorRetriever(
            embedding_model="intfloat/multilingual-e5-large",
            index_path=str(vector_store_path / "vector"),
            use_mmr=True,
            mmr_lambda=0.5
        )
        
        # BM25 Retriever
        bm25_retriever = BM25Retriever(
            index_path=str(vector_store_path / "bm25")
        )
        
        # Hybrid Retriever
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            vector_weight=0.5,
            bm25_weight=0.5
        )
        
        return {
            "vector": vector_retriever,
            "bm25": bm25_retriever,
            "hybrid": hybrid_retriever
        }
    
    def _build_graph(self) -> Any:
        """LangGraphワークフローを構築"""
        workflow = StateGraph(AgenticRAGState)
        
        # ノードの追加
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("generate_simple", self._generate_simple_node)
        
        # エントリーポイント
        workflow.set_entry_point("classify")
        
        # 複雑度別の分岐
        workflow.add_conditional_edges(
            "classify",
            self._route_by_complexity,
            {
                "simple": "generate_simple",
                "medium": "retrieve",
                "complex": "retrieve"
            }
        )
        
        # Simple: 直接生成 → 検証
        workflow.add_edge("generate_simple", "validate")
        
        # Medium/Complex: 検索 → 推論
        workflow.add_edge("retrieve", "reason")
        
        # 反復判定
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "continue": "retrieve",
                "validate": "validate"
            }
        )
        
        # 検証後の処理
        workflow.add_conditional_edges(
            "validate",
            self._validation_result,
            {
                "retry": "retrieve",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _classify_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """クエリ分類ノード"""
        query = state["query"]
        
        result = self.manager.execute({"query": query})
        
        state["complexity"] = result["complexity"]
        state["query_type"] = result["query_type"]
        state["agents_used"].append("manager")
        
        logger.info(f"Classified: complexity={result['complexity']}, type={result['query_type']}")
        
        return state
    
    def _retrieve_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """検索ノード"""
        query = state["query"]
        query_type = state.get("query_type", "lookup")
        
        result = self.retrieval.execute({
            "query": query,
            "query_type": query_type
        })
        
        state["documents"] = result["documents"]
        state["retrieval_strategy"] = result["strategy"]
        state["confidence"] = result.get("quality_score", 0.5)
        state["agents_used"].append("retrieval")
        
        logger.info(f"Retrieved {len(result['documents'])} documents using {result['strategy']}")
        
        return state
    
    def _reason_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """推論ノード"""
        if not self.config.enable_reasoning:
            # Reasoning無効時は簡易生成
            state["reasoning"] = self._generate_simple_answer(state)
            state["legal_structure"] = {}
            state["needs_retry"] = False  # 再試行不要
            return state
        
        result = self.reasoning.execute({
            "query": state["query"],
            "documents": state["documents"],
            "complexity": state["complexity"]
        })
        
        state["reasoning"] = result["reasoning"]
        state["legal_structure"] = result["legal_structure"]
        state["answer"] = result["reasoning"]  # 推論結果を回答として使用
        state["agents_used"].append("reasoning")
        state["needs_retry"] = False  # 再試行不要（デフォルト）
        
        logger.info("Reasoning completed")
        
        return state
    
    def _validate_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """検証ノード"""
        if not self.config.enable_validation:
            # Validation無効時はスキップ
            state["is_valid"] = True
            state["confidence"] = max(state.get("confidence", 0.5), 0.7)
            return state
        
        result = self.validation.execute({
            "query": state["query"],
            "answer": state.get("answer", state.get("reasoning", "")),
            "documents": state["documents"]
        })
        
        state["is_valid"] = result["is_valid"]
        state["confidence"] = max(state["confidence"], result["confidence"])
        state["agents_used"].append("validation")
        
        logger.info(f"Validation: valid={result['is_valid']}, confidence={result['confidence']:.3f}")
        
        return state
    
    def _generate_simple_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """簡易回答生成ノード（Simple質問用）"""
        answer = self._generate_simple_answer(state)
        state["answer"] = answer
        state["reasoning"] = answer
        
        logger.info("Simple answer generated")
        
        return state
    
    def _generate_simple_answer(self, state: AgenticRAGState) -> str:
        """簡易回答生成"""
        # 既存RAGと同様の簡易生成
        query = state["query"]
        
        # 簡易検索（まだ検索していない場合）
        if not state.get("documents"):
            result = self.retrieval.execute({
                "query": query,
                "query_type": state.get("query_type", "lookup")
            })
            state["documents"] = result["documents"]
        
        documents = state["documents"][:3]  # 上位3件のみ
        
        if not documents:
            return "関連する法令条文が見つかりませんでした。"
        
        context = self._format_context(documents)
        
        prompt = f"""以下の法令条文に基づいて、質問に簡潔に回答してください。

【法令条文】
{context}

【質問】
{query}

【回答】"""
        
        try:
            answer = self.llm.invoke(prompt)
            return answer.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "回答の生成中にエラーが発生しました。"
    
    def _format_context(self, documents: List[Any]) -> str:
        """コンテキスト整形（BaseAgentの_format_documentsと同じ形式）"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            law_title = meta.get("law_title", "不明")
            article = meta.get("article", "")
            paragraph = meta.get("paragraph", "")
            item = meta.get("item", "")
            
            header = f"[{i}] {law_title}"
            if article:
                header += f" 第{article}条"
            if paragraph:
                header += f" 第{paragraph}項"
            if item:
                header += f" 第{item}号"
            
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _route_by_complexity(self, state: AgenticRAGState) -> str:
        """複雑度に応じたルーティング"""
        complexity = state["complexity"]
        
        # simple: 簡易生成、medium/complex: 詳細検索と推論
        if complexity == "simple":
            return "simple"
        else:
            # medium と complex は同じフローを使用
            return complexity
    
    def _should_continue(self, state: AgenticRAGState) -> str:
        """反復を継続するか判定"""
        # 最大反復回数チェック
        if state["iteration"] >= state["max_iterations"]:
            logger.info("Max iterations reached")
            return "validate"
        
        # 信頼度チェック
        if state["confidence"] >= self.config.confidence_threshold:
            logger.info(f"Confidence threshold met: {state['confidence']:.3f}")
            return "validate"
        
        # 再試行不要
        if not state.get("needs_retry", False):
            return "validate"
        
        # 継続（iterationのインクリメントはretrieveノードで行う）
        logger.info(f"Continue iteration: {state['iteration']}/{state['max_iterations']}")
        return "continue"
    
    def _validation_result(self, state: AgenticRAGState) -> str:
        """検証結果に基づく分岐"""
        # 有効な回答が得られた、または最大反復回数に達した
        if state["is_valid"] or state["iteration"] >= state["max_iterations"]:
            return "end"
        
        # 再試行（iterationのインクリメントはretrieveノードで行う）
        if state["iteration"] < state["max_iterations"]:
            logger.info("Retry due to validation failure")
            return "retry"
        
        return "end"
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        質問に回答
        
        Args:
            question: ユーザーの質問
        
        Returns:
            {
                "answer": str,
                "citations": List[Dict],
                "metadata": Dict
            }
        """
        logger.info(f"Query: {question[:100]}...")
        
        # 初期状態
        initial_state = {
            "query": question,
            "complexity": "",
            "query_type": "",
            "documents": [],
            "retrieval_strategy": "",
            "reasoning": "",
            "legal_structure": {},
            "answer": "",
            "citations": [],
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "confidence": 0.0,
            "agents_used": [],
            "needs_retry": False,
            "is_valid": True
        }
        
        try:
            # グラフ実行
            result = self.graph.invoke(initial_state)
            
            # 引用情報の抽出
            citations = self._extract_citations(result.get("documents", []))
            
            return {
                "answer": result.get("answer", result.get("reasoning", "回答を生成できませんでした。")),
                "citations": citations,
                "metadata": {
                    "complexity": result.get("complexity", ""),
                    "query_type": result.get("query_type", ""),
                    "iterations": result.get("iteration", 0),
                    "confidence": result.get("confidence", 0.0),
                    "agents_used": result.get("agents_used", []),
                    "retrieval_strategy": result.get("retrieval_strategy", ""),
                    "is_valid": result.get("is_valid", True)
                }
            }
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "answer": f"エラーが発生しました: {str(e)}",
                "citations": [],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def _extract_citations(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """引用情報を抽出"""
        citations = []
        seen = set()
        
        for doc in documents[:5]:  # 上位5件のみ
            meta = doc.metadata
            law_title = meta.get("law_title", "")
            article = meta.get("article", "")
            
            key = (law_title, article)
            if key not in seen and law_title:
                citations.append({
                    "law_title": law_title,
                    "article": article,
                    "paragraph": meta.get("paragraph", ""),
                    "item": meta.get("item", "")
                })
                seen.add(key)
        
        return citations

