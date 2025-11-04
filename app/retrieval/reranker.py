"""
Cross-encoderベースのReranker実装
"""
import logging
from typing import List
import torch
from sentence_transformers import CrossEncoder
from .base import BaseReranker, Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Cross-encoderを使ったReranker"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        # デバイスの自動検出
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for reranker")
        
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """ドキュメントを再ランキング"""
        # パラメータバリデーション
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        if not query or not query.strip():
            logger.warning("Empty query provided for reranking, returning original documents")
            return documents[:top_n]
        if not documents:
            return []
        
        # top_nがドキュメント数より多い場合は調整
        top_n = min(top_n, len(documents))
        
        pairs = [[query, doc.page_content] for doc in documents]
        
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc.score = float(score)
        
        documents.sort(key=lambda x: x.score, reverse=True)
        
        return documents[:top_n]
