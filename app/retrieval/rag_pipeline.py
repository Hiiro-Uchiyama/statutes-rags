"""
RAGパイプライン: Retriever + Reranker + LLM
"""
import os
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .base import BaseRetriever, BaseReranker, Document


class RAGPipeline:
    """RAGパイプライン"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: str = "ollama",
        llm_model: str = "qwen2.5:7b",
        temperature: float = 0.1,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 10,
        rerank_top_n: int = 5
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        
        if llm_provider == "ollama":
            ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.llm = Ollama(model=llm_model, temperature=temperature, base_url=ollama_base_url)
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
        """ドキュメントをコンテキスト文字列に整形"""
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
        documents = self.retrieve_documents(question)
        
        if not documents:
            return {
                "answer": "関連する法令条文が見つかりませんでした。",
                "citations": [],
                "contexts": []
            }
        
        context = self.format_context(documents)
        
        # LCELを使用してinvokeで実行
        answer = self.chain.invoke({"context": context, "question": question})
        
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
