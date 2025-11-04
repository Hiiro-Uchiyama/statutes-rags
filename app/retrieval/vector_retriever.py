"""
FAISSベースのベクトル検索Retriever
"""
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument
from .base import BaseRetriever, Document

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """FAISSを使ったベクトル検索"""
    
    def __init__(self, embedding_model: str, index_path: str = None, use_mmr: bool = False, mmr_lambda: float = 0.5, mmr_fetch_k_max: int = 50):
        # パラメータのバリデーション
        if not embedding_model:
            raise ValueError("embedding_model must not be empty")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in range [0.0, 1.0], got {mmr_lambda}")
        
        self.embedding_model_name = embedding_model
        
        # デバイスの自動検出（CUDA利用可能ならGPU、なければCPU）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for embeddings")
        
        # バッチサイズの自動調整（GPUの場合は大きく、CPUの場合は小さく）
        batch_size = 256 if device == 'cuda' else 32
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True,
                           'batch_size': batch_size
                           }
        )
        self.index_path = index_path
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.mmr_fetch_k_max = mmr_fetch_k_max
        self.vector_store = None
        
        if index_path and Path(index_path).exists():
            self.load_index()
    
    def _get_doc_id(self, doc: LangChainDocument) -> str:
        """ドキュメントの一意なIDを生成（SHA256ハッシュベース）"""
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
        """ドキュメントをベクトルストアに追加"""
        if not documents:
            logger.warning("Empty document list provided, nothing to add")
            return
        
        texts = []
        metadatas = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = {k: v for k, v in doc.items() if k != "text"}
            texts.append(text)
            metadatas.append(metadata)
        
        langchain_docs = [
            LangChainDocument(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        else:
            self.vector_store.add_documents(langchain_docs)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """ベクトル検索を実行"""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        if self.vector_store is None:
            return []
        
        if self.use_mmr:
            # MMR使用時：候補数を最適化（fetch_k = top_k * 2, 上限は設定可能）
            fetch_k = min(top_k * 2, self.mmr_fetch_k_max)
            logger.debug(f"MMR retrieval: top_k={top_k}, fetch_k={fetch_k}, lambda={self.mmr_lambda}, max={self.mmr_fetch_k_max}")
            
            # MMRで多様性を考慮した検索
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=top_k,
                lambda_mult=self.mmr_lambda,
                fetch_k=fetch_k
            )
            
            # MMRの順位ベースでシンプルにスコアリング
            # 1位: 1.0, 2位: 0.5, 3位: 0.333...
            results = []
            for rank, doc in enumerate(docs, start=1):
                rank_score = 1.0 / rank
                results.append((doc, rank_score))
            
            logger.debug(f"MMR returned {len(results)} documents")
        else:
            # 通常の類似度検索
            candidates = self.vector_store.similarity_search_with_score(query, k=top_k)
            # FAISSの距離スコアを類似度スコアに変換（小さいほど良い距離を、大きいほど良いスコアに）
            results = []
            for doc, distance in candidates:
                similarity_score = 1.0 / (1.0 + distance)
                results.append((doc, similarity_score))
        
        documents = []
        for doc, score in results:
            documents.append(Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
                score=float(score)
            ))
        
        return documents
    
    def save_index(self):
        """インデックスを保存"""
        if self.vector_store is None or not self.index_path:
            logger.warning("Cannot save index - vector store or index path not set")
            return
        
        try:
            index_path = Path(self.index_path)
            index_path.mkdir(parents=True, exist_ok=True)
            
            self.vector_store.save_local(str(index_path))
            
            # ドキュメント数を取得
            doc_count = self.vector_store.index.ntotal if self.vector_store.index else 0
            logger.info(f"Vector index saved to {index_path} ({doc_count} documents)")
        except Exception as e:
            logger.error(f"Error saving index to {self.index_path}: {e}", exc_info=True)
            raise
    
    def load_index(self):
        """インデックスをロード
        
        Raises:
            RuntimeError: インデックスファイルが破損している場合
        """
        if not self.index_path:
            logger.warning("Vector index path not set")
            return
        
        index_path = Path(self.index_path)
        
        if not index_path.exists():
            logger.info(f"Vector index not found at {index_path}, will be created on first use")
            return
        
        try:
            # NOTE: allow_dangerous_deserialization=True を使用
            # 信頼できるローカルファイルからのみロードすること
            self.vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # ドキュメント数を取得
            doc_count = self.vector_store.index.ntotal if self.vector_store.index else 0
            logger.info(f"Vector index loaded from {index_path} ({doc_count} documents)")
        except Exception as e:
            logger.error(f"Error loading index from {index_path}: {e}", exc_info=True)
            self.vector_store = None
            logger.warning("Failed to load index. A new index will be created when documents are added.")
