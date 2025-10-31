#!/usr/bin/env python3
"""
検索機能のみテスト
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag_config import load_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever


def main():
    config = load_config()
    index_path = Path(config.vector_store_path)
    
    print("Loading retrievers...")
    vector_retriever = VectorRetriever(
        embedding_model=config.embedding.model_name,
        index_path=str(index_path / "vector"),
        use_mmr=config.retriever.use_mmr,
        mmr_lambda=config.retriever.mmr_lambda
    )
    bm25_retriever = BM25Retriever(index_path=str(index_path / "bm25"))
    retriever = HybridRetriever(vector_retriever, bm25_retriever)
    
    print("\n検索テスト:")
    query = "窃盗罪の法定刑は何ですか？"
    print(f"Query: {query}\n")
    
    documents = retriever.retrieve(query, top_k=5)
    
    print(f"Found {len(documents)} documents:\n")
    for i, doc in enumerate(documents, 1):
        print(f"[{i}] Score: {doc.score:.4f}")
        print(f"    Law: {doc.metadata.get('law_title', 'Unknown')}")
        if doc.metadata.get('article'):
            print(f"    Article: {doc.metadata.get('article')}")
        print(f"    Content: {doc.page_content[:200]}...")
        print()


if __name__ == "__main__":
    main()
