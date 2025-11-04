"""
FAISSベースのベクトル検索Retriever
"""
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument
from .base import BaseRetriever, Document
import faiss

class VectorRetriever(BaseRetriever):
    """FAISSを使ったベクトル検索 (GPU対応修正版)"""
    
    def __init__(self, embedding_model: str, index_path: str = None, use_mmr: bool = False, mmr_lambda: float = 0.5):
        self.embedding_model_name = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True,
                           'batch_size': 256
                           }
        )
        self.index_path = index_path
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.vector_store = None
        self.metadata_list = []
        
        # GPUリソースと状態フラグを追加
        self.gpu_res = None
        self.on_gpu = False
        
        if index_path and Path(index_path).exists():
            self.load_index()
            # ロード後にGPUに移動
            self._move_to_gpu()

    def _move_to_gpu(self):
        """インデックスをGPUに移動するヘルパー関数"""
        if self.vector_store and not self.on_gpu:
            if faiss.get_num_gpus() > 0:
                print("Moving FAISS index to GPU...")
                try:
                    if self.gpu_res is None:
                        # GPUリソースを初期化
                        self.gpu_res = faiss.StandardGpuResources()
                    
                    # CPUインデックスをGPUインデックスに変換
                    self.vector_store.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.vector_store.index)
                    self.on_gpu = True
                    print("FAISS index successfully moved to GPU.")
                except Exception as e:
                    print(f"Failed to move FAISS index to GPU: {e}")
                    print("Continuing with CPU-based FAISS.")
            else:
                print("faiss-gpu not properly installed or no GPU found. Using CPU.")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """ドキュメントをベクトルストアに追加"""
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
            # 初回作成 (CPUで作成される)
            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
            # 作成後にGPUに移動
            self._move_to_gpu()
        else:
            # 既存のインデックスに追加
            # (インデックスがGPUにあれば、add_documentsはGPUで実行される)
            self.vector_store.add_documents(langchain_docs)
        
        self.metadata_list.extend(metadatas)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """ベクトル検索を実行"""
        if self.vector_store is None:
            return []
        
        if self.use_mmr:
            # MMR検索 (GPUでも動作)
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=top_k,
                lambda_mult=self.mmr_lambda,
                fetch_k=top_k * 3
            )
            results = [(doc, 0.0) for doc in docs]
        else:
            # 類似度検索 (GPUでも動作)
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
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
            return
        
        index_path = Path(self.index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # 保存する前にCPUに戻す
        if self.on_gpu:
            print("Moving FAISS index back to CPU for saving...")
            try:
                self.vector_store.index = faiss.index_gpu_to_cpu(self.vector_store.index)
                self.on_gpu = False # CPUに戻した
                print("FAISS index moved to CPU.")
            except Exception as e:
                print(f"Failed to move FAISS index to CPU: {e}")

        # CPUインデックスを保存
        self.vector_store.save_local(str(index_path))
        
        # (オプション) 保存後に再度GPUに戻す
        print("Re-moving index to GPU after saving...")
        self._move_to_gpu()
        
        metadata_path = index_path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata_list, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self):
        """インデックスをロード"""
        index_path = Path(self.index_path)
        
        if not index_path.exists():
            print(f"Index not found at {index_path}")
            return
        
        self.vector_store = FAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        metadata_path = index_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata_list = pickle.load(f)
        
        print(f"Index loaded from {index_path}")
        # ロード直後はCPUにあるため、__init__内のself._move_to_gpu()でGPUに転送されます