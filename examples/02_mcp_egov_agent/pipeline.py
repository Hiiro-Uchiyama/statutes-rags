"""
MCP e-Gov Agent パイプライン

ハイブリッド検索戦略を実装したRAGパイプライン。
"""
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retrieval.base import Document, BaseRetriever
from .config import MCPEgovConfig, load_config
from .agents import MCPEgovAgent

logger = logging.getLogger(__name__)


class MCPEgovPipeline:
    """
    MCP e-Gov Agent パイプライン
    
    e-Gov APIとローカルデータを組み合わせたハイブリッド検索パイプライン。
    """
    
    def __init__(
        self,
        config: Optional[MCPEgovConfig] = None,
        retriever: Optional[BaseRetriever] = None,
        llm = None,
        reranker = None
    ):
        """
        Args:
            config: MCP e-Gov Agent設定
            retriever: ローカル検索用のRetriever
            llm: LLMインスタンス（オプション）
            reranker: リランカー（オプション）
        """
        self.config = config or load_config(validate=False)
        self.retriever = retriever
        self.reranker = reranker
        
        # LLMの初期化
        if llm is None:
            ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.llm = Ollama(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                base_url=ollama_base_url,
                timeout=self.config.llm_timeout
            )
        else:
            self.llm = llm
        
        # MCPEgovAgentの初期化
        self.agent = MCPEgovAgent(
            llm=self.llm,
            config=self.config,
            local_retriever=retriever
        )
        
        # プロンプトテンプレートの設定
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""あなたは日本の法律に精通した法律アシスタントです。以下の法令条文に基づいて質問に答えてください。

【法令条文】
{context}

【質問】
{question}

【回答】
上記の法令条文に基づいて、正確かつ具体的に回答してください。必ず該当する法令名と条文番号を明記してください。"""
        )
        
        # LCELチェーンの構築
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
        logger.info("MCPEgovPipeline initialized")
    
    def retrieve_documents(
        self,
        query: str,
        force_api: bool = False,
        force_local: bool = False
    ) -> List[Document]:
        """
        ドキュメントを検索
        
        Args:
            query: 検索クエリ
            force_api: API強制使用
            force_local: ローカル強制使用
        
        Returns:
            ドキュメントのリスト
        """
        # エージェントで検索
        result = self.agent.execute({
            "query": query,
            "force_api": force_api,
            "force_local": force_local
        })
        
        documents = result.get("documents", [])
        source = result.get("source", "unknown")
        
        logger.info(f"Retrieved {len(documents)} documents from {source}")
        
        # リランキング
        if self.reranker and documents:
            documents = self.reranker.rerank(
                query,
                documents,
                top_n=self.config.rerank_top_n
            )
            logger.info(f"Reranked to {len(documents)} documents")
        
        return documents
    
    def format_context(self, documents: List[Document]) -> str:
        """
        ドキュメントをコンテキスト文字列に整形
        
        Args:
            documents: ドキュメントのリスト
        
        Returns:
            整形されたコンテキスト文字列
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            law_title = meta.get("law_title", "不明")
            article = meta.get("article", "")
            paragraph = meta.get("paragraph", "")
            item = meta.get("item", "")
            source = meta.get("source", "local")
            
            # ヘッダーの構築
            header = f"[{i}] {law_title}"
            if article:
                header += f" 第{article}条"
            if paragraph:
                header += f" 第{paragraph}項"
            if item:
                header += f" 第{item}号"
            
            # ソース情報を追加
            if source == "egov_api":
                header += " (最新データ)"
            
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def extract_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        引用情報を抽出
        
        Args:
            documents: ドキュメントのリスト
        
        Returns:
            引用情報のリスト
        """
        citations = []
        seen = set()
        
        for doc in documents:
            meta = doc.metadata
            law_title = meta.get("law_title", "")
            article = meta.get("article", "")
            paragraph = meta.get("paragraph", "")
            source = meta.get("source", "local")
            
            key = (law_title, article, paragraph)
            if key not in seen:
                citations.append({
                    "law_title": law_title,
                    "article": article,
                    "paragraph": paragraph if paragraph else None,
                    "item": meta.get("item", None),
                    "source": source
                })
                seen.add(key)
        
        return citations
    
    def query(
        self,
        question: str,
        force_api: bool = False,
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        質問に回答
        
        Args:
            question: 質問文
            force_api: API強制使用
            force_local: ローカル強制使用
        
        Returns:
            {
                "answer": str,  # 回答
                "citations": List[Dict],  # 引用情報
                "contexts": List[Dict],  # コンテキスト情報
                "source": str,  # データソース
                "metadata": Dict  # 追加メタデータ
            }
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # ドキュメント検索
            documents = self.retrieve_documents(
                question,
                force_api=force_api,
                force_local=force_local
            )
            
            if not documents:
                logger.warning("No documents found for the query")
                return {
                    "answer": "関連する法令条文が見つかりませんでした。",
                    "citations": [],
                    "contexts": [],
                    "source": "none",
                    "metadata": {}
                }
            
            # コンテキスト生成
            context = self.format_context(documents)
            logger.debug(f"Context length: {len(context)} characters")
            
            # LLM呼び出し
            try:
                answer = self.chain.invoke({
                    "context": context,
                    "question": question
                })
                logger.info("LLM response received successfully")
            
            except TimeoutError as e:
                logger.error(f"LLM request timeout: {e}")
                return {
                    "answer": f"LLMのリクエストがタイムアウトしました（{self.config.llm_timeout}秒）。",
                    "citations": [],
                    "contexts": [],
                    "source": "error",
                    "metadata": {"error": "timeout"}
                }
            
            except Exception as e:
                logger.error(f"LLM invocation error: {e}", exc_info=True)
                return {
                    "answer": f"LLMの呼び出し中にエラーが発生しました: {str(e)}",
                    "citations": [],
                    "contexts": [],
                    "source": "error",
                    "metadata": {"error": str(e)}
                }
            
            # 引用情報とコンテキストの抽出
            citations = self.extract_citations(documents)
            
            contexts = [
                {
                    "law_title": doc.metadata.get("law_title", ""),
                    "article": doc.metadata.get("article", ""),
                    "paragraph": doc.metadata.get("paragraph", ""),
                    "text": doc.page_content,
                    "score": doc.score,
                    "source": doc.metadata.get("source", "local")
                }
                for doc in documents
            ]
            
            # データソースの判定
            sources = [doc.metadata.get("source", "local") for doc in documents]
            if all(s == "egov_api" for s in sources):
                data_source = "api"
            elif any(s == "egov_api" for s in sources):
                data_source = "hybrid"
            else:
                data_source = "local"
            
            return {
                "answer": answer.strip(),
                "citations": citations,
                "contexts": contexts,
                "source": data_source,
                "metadata": {
                    "num_documents": len(documents),
                    "num_api_documents": sum(1 for s in sources if s == "egov_api"),
                    "num_local_documents": sum(1 for s in sources if s == "local")
                }
            }
        
        except Exception as e:
            logger.error(f"Unexpected error during query processing: {e}", exc_info=True)
            return {
                "answer": f"予期しないエラーが発生しました: {str(e)}",
                "citations": [],
                "contexts": [],
                "source": "error",
                "metadata": {"error": str(e)}
            }


def create_pipeline(
    config: Optional[MCPEgovConfig] = None,
    retriever: Optional[BaseRetriever] = None
) -> MCPEgovPipeline:
    """
    MCPEgovPipelineを作成する便利関数
    
    Args:
        config: MCP e-Gov Agent設定
        retriever: ローカル検索用のRetriever
    
    Returns:
        MCPEgovPipeline インスタンス
    """
    return MCPEgovPipeline(config=config, retriever=retriever)

