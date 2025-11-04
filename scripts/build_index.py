#!/usr/bin/env python3
"""
JSONLファイルからベクトルインデックスを構築
"""
import json
import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# プロジェクトルートの.envファイルを読み込み
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from app.core.rag_config import load_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from tqdm import tqdm


def load_jsonl(file_path: Path, limit: int = None):
    """JSONLファイルからドキュメントをロード"""
    documents = []
    error_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:  # 最初の5件のみ表示
                    print(f"Warning: Skipping invalid JSON at line {i+1}: {e}")
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Warning: Error loading line {i+1}: {e}")
    
    if error_count > 5:
        print(f"... and {error_count - 5} more errors")
    
    return documents


def build_arg_parser() -> argparse.ArgumentParser:
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
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=10000,
        help="バッチサイズ"
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    config = load_config()
    
    data_path = args.data_path or Path(config.data_path)
    index_path = args.index_path or Path(config.vector_store_path)
    retriever_type = args.retriever_type or config.retriever.retriever_type
    
    # データファイルの存在確認
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("\nPlease run preprocessing first:")
        print("  make preprocess")
        print("\nOr with file limit:")
        print("  make preprocess PREPROCESS_LIMIT=100")
        sys.exit(1)
    
    print(f"Loading documents from {data_path}")
    documents = load_jsonl(data_path, limit=args.limit)
    print(f"Loaded {len(documents)} documents")
    
    # 空のドキュメントリストチェック
    if not documents:
        print(f"Error: No documents loaded from {data_path}")
        print("Please check if the file contains valid JSONL data.")
        print("Each line should be a valid JSON object.")
        sys.exit(1)
    
    vector_retriever = None
    bm25_retriever = None

    if retriever_type == "vector":
        print("Building FAISS vector index...")
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
        retriever = vector_retriever
    elif retriever_type == "bm25":
        print("Building BM25 index...")
        bm25_retriever = BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
        retriever = bm25_retriever
    else:
        print("Building hybrid index (FAISS + BM25)...")
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
        bm25_retriever = BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
        retriever = HybridRetriever(
            vector_retriever,
            bm25_retriever,
            fusion_method=config.retriever.fusion_method,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            rrf_k=config.retriever.rrf_k,
            fetch_k_multiplier=config.retriever.fetch_k_multiplier
        )

    batch_size = args.batch_size

    if vector_retriever is not None:
        print("Adding documents to vector index...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Building vector index"):
            batch = documents[i:i + batch_size]
            vector_retriever.add_documents(batch)

    if bm25_retriever is not None:
        print("Tokenizing documents for BM25 index...")
        bm25_retriever.add_documents(documents)
    
    print("Saving index...")
    retriever.save_index()
    
    print(f"Index built successfully and saved to {index_path}")


if __name__ == "__main__":
    main()
