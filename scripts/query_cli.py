#!/usr/bin/env python3
"""
RAGシステムCLIツール
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag_config import load_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rag_pipeline import RAGPipeline


def create_retriever(config):
    """設定に基づいてRetrieverを作成"""
    retriever_type = config.retriever.retriever_type
    index_path = Path(config.vector_store_path)
    
    if retriever_type == "vector":
        return VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda
        )
    elif retriever_type == "bm25":
        return BM25Retriever(index_path=str(index_path / "bm25"))
    else:
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda
        )
        bm25_retriever = BM25Retriever(index_path=str(index_path / "bm25"))
        return HybridRetriever(vector_retriever, bm25_retriever)


def main():
    parser = argparse.ArgumentParser(description="RAGシステムCLI")
    parser.add_argument("question", nargs="?", help="質問文")
    parser.add_argument("--interactive", "-i", action="store_true", help="対話モード")
    parser.add_argument("--output", "-o", type=Path, help="結果をJSONファイルに保存")
    
    args = parser.parse_args()
    
    config = load_config()
    
    print("Loading retriever...")
    retriever = create_retriever(config)
    
    reranker = None
    if config.reranker.enabled:
        print("Loading reranker...")
        reranker = CrossEncoderReranker(model_name=config.reranker.model_name)
    
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model_name,
        temperature=config.llm.temperature,
        reranker=reranker,
        top_k=config.retriever.top_k,
        rerank_top_n=config.reranker.top_n if config.reranker.enabled else config.retriever.top_k
    )
    
    print("Ready!\n")
    
    if args.interactive:
        print("Interactive mode. Type 'exit' or 'quit' to exit.\n")
        while True:
            try:
                question = input("Question: ")
                if question.lower() in ["exit", "quit"]:
                    break
                
                if not question.strip():
                    continue
                
                print("\nSearching and generating answer...\n")
                result = pipeline.query(question)
                
                print("=" * 80)
                print("ANSWER:")
                print("=" * 80)
                print(result["answer"])
                print("\n" + "=" * 80)
                print("CITATIONS:")
                print("=" * 80)
                for citation in result["citations"]:
                    cite_str = citation["law_title"]
                    if citation.get("article"):
                        cite_str += f" 第{citation['article']}条"
                    if citation.get("paragraph"):
                        cite_str += f" 第{citation['paragraph']}項"
                    if citation.get("item"):
                        cite_str += f" 第{citation['item']}号"
                    print(f"- {cite_str}")
                print("\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        if not args.question:
            parser.error("Please provide a question or use --interactive mode")
        
        print(f"Question: {args.question}\n")
        print("Searching and generating answer...\n")
        
        result = pipeline.query(args.question)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print("=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(result["answer"])
            print("\n" + "=" * 80)
            print("CITATIONS:")
            print("=" * 80)
            for citation in result["citations"]:
                cite_str = citation["law_title"]
                if citation.get("article"):
                    cite_str += f" 第{citation['article']}条"
                if citation.get("paragraph"):
                    cite_str += f" 第{citation['paragraph']}項"
                if citation.get("item"):
                    cite_str += f" 第{citation['item']}号"
                print(f"- {cite_str}")
            print()


if __name__ == "__main__":
    main()
