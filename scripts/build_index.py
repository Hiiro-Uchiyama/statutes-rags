#!/usr/bin/env python3
"""
JSONLファイルからベクトルインデックスを構築
"""
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag_config import load_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from tqdm import tqdm


def load_jsonl(file_path: Path, limit: int = None):
    """JSONLファイルからドキュメントをロード"""
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                doc = json.loads(line)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading line {i}: {e}")
    return documents


def main():
    parser = argparse.ArgumentParser(description="ベクトルインデックスを構築")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="JSONLデータファイル"
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        help="インデックス保存先"
    )
    parser.add_argument(
        "--retriever-type",
        choices=["vector", "bm25", "hybrid"],
        help="Retrieverタイプ"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理するドキュメント数の上限"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="バッチサイズ"
    )
    
    args = parser.parse_args()
    
    config = load_config()
    
    data_path = args.data_path or Path(config.data_path)
    index_path = args.index_path or Path(config.vector_store_path)
    retriever_type = args.retriever_type or config.retriever.retriever_type
    
    print(f"Loading documents from {data_path}")
    documents = load_jsonl(data_path, limit=args.limit)
    print(f"Loaded {len(documents)} documents")
    
    if retriever_type == "vector":
        print("Building FAISS vector index...")
        retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda
        )
    elif retriever_type == "bm25":
        print("Building BM25 index...")
        retriever = BM25Retriever(
            index_path=str(index_path / "bm25")
        )
    else:
        print("Building hybrid index (FAISS + BM25)...")
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda
        )
        bm25_retriever = BM25Retriever(
            index_path=str(index_path / "bm25")
        )
        retriever = HybridRetriever(vector_retriever, bm25_retriever)
    
    print("Adding documents to index...")
    batch_size = args.batch_size
    for i in tqdm(range(0, len(documents), batch_size), desc="Building index"):
        batch = documents[i:i + batch_size]
        retriever.add_documents(batch)
    
    print("Saving index...")
    retriever.save_index()
    
    print(f"Index built successfully and saved to {index_path}")


if __name__ == "__main__":
    main()
