"""
RAGパイプライン: Retriever + Reranker + LLM
"""
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .base import BaseRetriever, BaseReranker, Document

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAGパイプライン"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: str = "ollama",
        llm_model: str = "gpt-oss:20b",
        temperature: float = 0.1,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 10,
        rerank_top_n: int = 5,
        max_context_length: int = 4000,  # 最大コンテキスト長（文字数）
        request_timeout: int = 60  # LLMリクエストのタイムアウト（秒）
    ):
        # パラメータのバリデーション
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if rerank_top_n <= 0:
            raise ValueError(f"rerank_top_n must be positive, got {rerank_top_n}")
        if max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive, got {max_context_length}")
        if request_timeout <= 0:
            raise ValueError(f"request_timeout must be positive, got {request_timeout}")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be in range [0.0, 2.0], got {temperature}")
        
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.max_context_length = max_context_length
        self.request_timeout = request_timeout
        
        if llm_provider == "ollama":
            ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.llm = Ollama(
                model=llm_model, 
                temperature=temperature, 
                base_url=ollama_base_url,
                timeout=request_timeout
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
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
        
        # LCEL (LangChain Expression Language) を使用
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """ドキュメントを検索"""
        documents = self.retriever.retrieve(query, top_k=self.top_k)
        
        if self.reranker and documents:
            documents = self.reranker.rerank(query, documents, top_n=self.rerank_top_n)
        
        return documents
    
    def format_context(self, documents: List[Document]) -> str:
        """ドキュメントをコンテキスト文字列に整形（最大長制限あり）"""
        context_parts = []
        total_length = 0
        
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
            
            context_part = f"{header}\n{doc.page_content}\n"
            
            # 最大コンテキスト長をチェック
            if total_length + len(context_part) > self.max_context_length:
                logger.warning(
                    f"Context length limit ({self.max_context_length}) reached. "
                    f"Truncating at document {i}/{len(documents)}"
                )
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def extract_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """引用情報を抽出"""
        citations = []
        seen = set()
        
        for doc in documents:
            meta = doc.metadata
            law_title = meta.get("law_title", "")
            article = meta.get("article", "")
            paragraph = meta.get("paragraph", "")
            
            key = (law_title, article, paragraph)
            if key not in seen:
                citations.append({
                    "law_title": law_title,
                    "article": article,
                    "paragraph": paragraph if paragraph else None,
                    "item": meta.get("item", None)
                })
                seen.add(key)
        
        return citations
    
    def query(self, question: str) -> Dict[str, Any]:
        """質問に回答"""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # ドキュメント検索
            documents = self.retrieve_documents(question)
            logger.info(f"Retrieved {len(documents)} documents")
            
            if not documents:
                logger.warning("No documents found for the query")
                return {
                    "answer": "関連する法令条文が見つかりませんでした。",
                    "citations": [],
                    "contexts": []
                }
            
            # コンテキスト生成
            context = self.format_context(documents)
            logger.debug(f"Context length: {len(context)} characters")
            
            # LLM呼び出し
            try:
                answer = self.chain.invoke({"context": context, "question": question})
                logger.info("LLM response received successfully")
            except TimeoutError as e:
                logger.error(f"LLM request timeout: {e}")
                return {
                    "answer": f"LLMのリクエストがタイムアウトしました（{self.request_timeout}秒）。",
                    "citations": [],
                    "contexts": [],
                    "error": "timeout"
                }
            except Exception as e:
                # その他のタイムアウトエラーもチェック（文字列ベースのフォールバック）
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    logger.error(f"LLM request timeout (fallback detection): {e}")
                    return {
                        "answer": f"LLMのリクエストがタイムアウトしました（{self.request_timeout}秒）。",
                        "citations": [],
                        "contexts": [],
                        "error": "timeout"
                    }
                else:
                    logger.error(f"LLM invocation error: {e}", exc_info=True)
                    return {
                        "answer": f"LLMの呼び出し中にエラーが発生しました: {str(e)}",
                        "citations": [],
                        "contexts": [],
                        "error": str(e)
                    }
            
            # 引用情報と文脈の抽出
            citations = self.extract_citations(documents)
            
            contexts = [
                {
                    "law_title": doc.metadata.get("law_title", ""),
                    "article": doc.metadata.get("article", ""),
                    "paragraph": doc.metadata.get("paragraph", ""),
                    "text": doc.page_content,
                    "score": doc.score
                }
                for doc in documents
            ]
            
            return {
                "answer": answer.strip(),
                "citations": citations,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"Unexpected error during query processing: {e}", exc_info=True)
            return {
                "answer": f"予期しないエラーが発生しました: {str(e)}",
                "citations": [],
                "contexts": [],
                "error": str(e)
            }
