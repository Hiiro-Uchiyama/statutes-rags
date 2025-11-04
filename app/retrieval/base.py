"""
抽象Retrieverインターフェース
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """検索結果のドキュメント"""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = 0.0


class BaseRetriever(ABC):
    """Retrieverの基底クラス"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """クエリに対して関連ドキュメントを検索"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], **kwargs):
        """ドキュメントを追加
        
        Args:
            documents: 追加するドキュメントのリスト
            **kwargs: 実装固有のオプションパラメータ
        """
        pass


class BaseReranker(ABC):
    """Rerankerの基底クラス"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """ドキュメントを再ランキング"""
        pass
