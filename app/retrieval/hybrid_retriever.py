"""
HybridRetriever: BM25とベクトル検索の組み合わせ
"""
import hashlib
import logging
from typing import List, Dict, Any, Literal
from .base import BaseRetriever, Document
from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """BM25とベクトル検索のハイブリッド"""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        fusion_method: Literal["rrf", "weighted_rrf", "weighted"] = "rrf",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 2
    ):
        """
        Args:
            vector_retriever: ベクトル検索Retriever
            bm25_retriever: BM25 Retriever
            fusion_method: スコア統合方法
                - 'rrf': 標準RRF（重み付けなし、論文の標準実装）
                - 'weighted_rrf': 重み付きRRF（検索手法ごとに重みを設定）
                - 'weighted': 正規化後の重み付き加算
            vector_weight: weighted/weighted_rrf方式でのベクトル検索の重み
            bm25_weight: weighted/weighted_rrf方式でのBM25の重み
            rrf_k: RRF (Reciprocal Rank Fusion) のkパラメータ（デフォルト60）
            fetch_k_multiplier: 各Retrieverから取得する候補数の倍率（デフォルト2）
        """
        # パラメータのバリデーション
        if vector_weight < 0:
            raise ValueError(f"vector_weight must be non-negative, got {vector_weight}")
        if bm25_weight < 0:
            raise ValueError(f"bm25_weight must be non-negative, got {bm25_weight}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        if fetch_k_multiplier <= 0:
            raise ValueError(f"fetch_k_multiplier must be positive, got {fetch_k_multiplier}")
        
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_method = fusion_method
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.fetch_k_multiplier = fetch_k_multiplier
        
        logger.info(f"HybridRetriever initialized with fusion_method='{fusion_method}', "
                   f"weights=(vector:{vector_weight}, bm25:{bm25_weight}), "
                   f"rrf_k={rrf_k}, fetch_k_multiplier={fetch_k_multiplier}")
    
    def _get_doc_id(self, doc: Document) -> str:
        """ドキュメントの一意なIDを生成（SHA256ハッシュベース）"""
        # メタデータに基づいてIDを生成
        meta = doc.metadata
        id_parts = [
            meta.get("law_title", ""),
            str(meta.get("article", "")),
            str(meta.get("paragraph", "")),
            str(meta.get("item", "")),
            doc.page_content[:200]  # 最初の200文字を含める（衝突リスク低減）
        ]
        id_string = "|".join(id_parts)
        # SHA256を使用してより衝突しにくいハッシュを生成
        return hashlib.sha256(id_string.encode()).hexdigest()
    
    def add_documents(self, documents: List[Dict[str, Any]], **kwargs):
        """両方のRetrieverにドキュメントを追加"""
        self.vector_retriever.add_documents(documents)
        self.bm25_retriever.add_documents(documents)
    
    def _normalize_scores(self, documents: List[Document]) -> List[Document]:
        """Min-Max正規化でスコアを0-1の範囲に正規化
        
        注: VectorRetrieverとBM25Retrieverは両方とも類似度スコア（大きいほど良い）を返す
        """
        if not documents:
            return documents
        
        scores = [doc.score for doc in documents]
        min_score = min(scores)
        max_score = max(scores)
        
        # 全て同じスコアの場合、中間値を割り当てる
        if max_score == min_score:
            for doc in documents:
                doc.score = 0.5
        else:
            # Min-Max正規化: (score - min) / (max - min)
            for doc in documents:
                doc.score = (doc.score - min_score) / (max_score - min_score)
        
        logger.debug(f"Normalized scores: min={min_score:.4f}, max={max_score:.4f}")
        return documents
    
    def _rrf_fusion(
        self, 
        vector_results: List[Document], 
        bm25_results: List[Document]
    ) -> List[Document]:
        """標準RRF (Reciprocal Rank Fusion) でスコアを統合
        
        重み付けなしの標準実装。各検索手法を平等に扱う。
        """
        score_map = {}
        
        # ベクトル検索結果のランクベーススコア
        for rank, doc in enumerate(vector_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (self.rrf_k + rank)
            score_map[doc_id] = {
                "doc": doc,
                "score": rrf_score
            }
        
        # BM25結果のランクベーススコア（単純加算）
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_id in score_map:
                score_map[doc_id]["score"] += rrf_score
            else:
                score_map[doc_id] = {
                    "doc": doc,
                    "score": rrf_score
                }
        
        # スコアでソート
        results = []
        for doc_id, data in score_map.items():
            doc = data["doc"]
            doc.score = data["score"]
            results.append(doc)
        
        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"RRF fusion: {len(results)} unique documents from {len(vector_results)} vector + {len(bm25_results)} BM25 results")
        return results
    
    def _weighted_rrf_fusion(
        self, 
        vector_results: List[Document], 
        bm25_results: List[Document]
    ) -> List[Document]:
        """重み付きRRF (Weighted RRF) でスコアを統合
        
        各検索手法のランクスコアに重みを適用。
        """
        score_map = {}
        
        # ベクトル検索結果のランクベーススコア（重み付き）
        for rank, doc in enumerate(vector_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (self.rrf_k + rank)
            score_map[doc_id] = {
                "doc": doc,
                "score": rrf_score * self.vector_weight
            }
        
        # BM25結果のランクベーススコア（重み付き）
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_id in score_map:
                score_map[doc_id]["score"] += rrf_score * self.bm25_weight
            else:
                score_map[doc_id] = {
                    "doc": doc,
                    "score": rrf_score * self.bm25_weight
                }
        
        # スコアでソート
        results = []
        for doc_id, data in score_map.items():
            doc = data["doc"]
            doc.score = data["score"]
            results.append(doc)
        
        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Weighted RRF fusion: {len(results)} unique documents")
        return results
    
    def _weighted_fusion(
        self, 
        vector_results: List[Document], 
        bm25_results: List[Document]
    ) -> List[Document]:
        """重み付き加算でスコアを統合（正規化後）
        
        スコアを正規化してから重み付き加算を行う。
        """
        # スコアを正規化
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        score_map = {}
        
        for doc in vector_results:
            doc_id = self._get_doc_id(doc)
            score_map[doc_id] = {
                "doc": doc,
                "vector_score": doc.score,
                "bm25_score": 0.0
            }
        
        for doc in bm25_results:
            doc_id = self._get_doc_id(doc)
            if doc_id in score_map:
                score_map[doc_id]["bm25_score"] = doc.score
            else:
                score_map[doc_id] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "bm25_score": doc.score
                }
        
        # 重み付き加算
        results = []
        for doc_id, data in score_map.items():
            combined_score = (
                self.vector_weight * data["vector_score"] +
                self.bm25_weight * data["bm25_score"]
            )
            doc = data["doc"]
            doc.score = combined_score
            results.append(doc)
        
        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Weighted fusion: {len(results)} unique documents")
        return results
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """ハイブリッド検索: BM25とベクトル検索の結果を統合"""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        
        # より多くの候補を取得（設定可能な倍率を使用）
        fetch_k = top_k * self.fetch_k_multiplier
        
        logger.debug(f"Hybrid retrieval for query: '{query[:50]}...', top_k={top_k}, fetch_k={fetch_k}")
        
        vector_results = self.vector_retriever.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)
        
        # スコア統合方法に応じて処理
        if self.fusion_method == "rrf":
            results = self._rrf_fusion(vector_results, bm25_results)
        elif self.fusion_method == "weighted_rrf":
            results = self._weighted_rrf_fusion(vector_results, bm25_results)
        else:  # weighted
            results = self._weighted_fusion(vector_results, bm25_results)
        
        final_results = results[:top_k]
        logger.info(f"Hybrid retrieval returned {len(final_results)} documents (method: {self.fusion_method})")
        return final_results
    
    def save_index(self):
        """両方のインデックスを保存"""
        self.vector_retriever.save_index()
        self.bm25_retriever.save_index()
    
    def load_index(self):
        """両方のインデックスをロード"""
        self.vector_retriever.load_index()
        self.bm25_retriever.load_index()
