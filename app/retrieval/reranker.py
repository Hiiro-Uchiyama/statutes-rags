"""
Reranker: Cross-Encoderによる検索結果の再ランキング
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """再ランキングされたドキュメント"""
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    final_rank: int


class Reranker:
    """Cross-Encoderベースの再ランキング"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda"
    ):
        """
        Args:
            model_name: Cross-Encoderモデル名
            device: 実行デバイス
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """モデルをロード"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Reranker loaded: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback scoring.")
            self.model = None
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Using fallback scoring.")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        ドキュメントを再ランキング
        
        Args:
            query: 検索クエリ
            documents: 元のドキュメントリスト
            top_k: 返す上位件数（Noneの場合は全件）
            
        Returns:
            再ランキングされたドキュメントリスト
        """
        if not documents:
            return []
        
        # ドキュメントのテキストを抽出
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif isinstance(doc, dict):
                texts.append(doc.get('text', doc.get('content', str(doc))))
            else:
                texts.append(str(doc))
        
        # スコア計算
        if self.model is not None:
            # Cross-Encoderでスコアリング
            pairs = [(query, text) for text in texts]
            scores = self.model.predict(pairs)
        else:
            # フォールバック: 単純なキーワードマッチング
            scores = self._fallback_scoring(query, texts)
        
        # ランキング結果を作成
        ranked_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            elif isinstance(doc, dict):
                content = doc.get('text', doc.get('content', str(doc)))
                metadata = doc.get('metadata', {})
            else:
                content = str(doc)
                metadata = {}
            
            ranked_docs.append(RankedDocument(
                content=content,
                metadata=metadata,
                original_score=0.0,  # 元のスコアがあれば設定
                rerank_score=float(score),
                final_rank=0  # 後で設定
            ))
        
        # スコアでソート
        ranked_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # ランク番号を設定
        for i, doc in enumerate(ranked_docs):
            doc.final_rank = i + 1
        
        # top_kで切り詰め
        if top_k is not None:
            ranked_docs = ranked_docs[:top_k]
        
        return ranked_docs
    
    def _fallback_scoring(self, query: str, texts: List[str]) -> List[float]:
        """フォールバック用の単純スコアリング"""
        import re
        
        # クエリのキーワードを抽出
        query_terms = set(re.findall(r'[一-龥ぁ-んァ-ン]+|[a-zA-Z]+|\d+', query.lower()))
        
        scores = []
        for text in texts:
            text_lower = text.lower()
            # キーワードの出現回数をカウント
            match_count = sum(1 for term in query_terms if term in text_lower)
            # 正規化
            score = match_count / max(len(query_terms), 1)
            scores.append(score)
        
        return scores


class HybridReranker:
    """Hybrid検索結果専用の再ランキング"""
    
    def __init__(
        self,
        reranker: Optional[Reranker] = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.reranker = reranker or Reranker(model_name=model_name)
    
    def rerank_hybrid_results(
        self,
        query: str,
        vector_results: List[Any],
        bm25_results: List[Any],
        top_k: int = 20
    ) -> List[RankedDocument]:
        """
        Hybrid検索の結果を再ランキング
        
        Args:
            query: 検索クエリ
            vector_results: ベクトル検索結果
            bm25_results: BM25検索結果
            top_k: 返す上位件数
            
        Returns:
            再ランキングされたドキュメントリスト
        """
        # 重複除去しながらマージ
        seen_contents = set()
        merged = []
        
        for doc in vector_results + bm25_results:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get('text', doc.get('content', ''))
            else:
                content = str(doc)
            
            # 重複チェック（先頭100文字で判定）
            content_key = content[:100] if content else ''
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(doc)
        
        # 再ランキング
        return self.reranker.rerank(query, merged, top_k=top_k)
