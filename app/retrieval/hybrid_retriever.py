"""
HybridRetriever: BM25とベクトル検索の組み合わせ
"""
from typing import List, Dict, Any
from .base import BaseRetriever, Document
from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever(BaseRetriever):
    """BM25とベクトル検索のハイブリッド"""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """両方のRetrieverにドキュメントを追加"""
        self.vector_retriever.add_documents(documents)
        self.bm25_retriever.add_documents(documents)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """ハイブリッド検索: BM25とベクトル検索の結果を統合"""
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        
        score_map = {}
        
        for doc in vector_results:
            key = doc.page_content
            score_map[key] = {
                "doc": doc,
                "vector_score": doc.score,
                "bm25_score": 0.0
            }
        
        for doc in bm25_results:
            key = doc.page_content
            if key in score_map:
                score_map[key]["bm25_score"] = doc.score
            else:
                score_map[key] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "bm25_score": doc.score
                }
        
        normalized_results = []
        for key, data in score_map.items():
            combined_score = (
                self.vector_weight * data["vector_score"] +
                self.bm25_weight * data["bm25_score"]
            )
            doc = data["doc"]
            doc.score = combined_score
            normalized_results.append(doc)
        
        normalized_results.sort(key=lambda x: x.score, reverse=True)
        
        return normalized_results[:top_k]
    
    def save_index(self):
        """両方のインデックスを保存"""
        self.vector_retriever.save_index()
        self.bm25_retriever.save_index()
    
    def load_index(self):
        """両方のインデックスをロード"""
        self.vector_retriever.load_index()
        self.bm25_retriever.load_index()
