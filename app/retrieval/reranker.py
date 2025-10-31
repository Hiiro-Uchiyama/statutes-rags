"""
Cross-encoderベースのReranker実装
"""
from typing import List
from sentence_transformers import CrossEncoder
from .base import BaseReranker, Document


class CrossEncoderReranker(BaseReranker):
    """Cross-encoderを使ったReranker"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """ドキュメントを再ランキング"""
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc in documents]
        
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc.score = float(score)
        
        documents.sort(key=lambda x: x.score, reverse=True)
        
        return documents[:top_n]
