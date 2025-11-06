#!/usr/bin/env python3
"""
RAGAS評価スクリプト
lawqa_jp selection.jsonを使用してRAGシステムを評価
"""
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rag_config import load_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rag_pipeline import RAGPipeline

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
except ImportError:
    print("Error: ragas and datasets are required. Install with:")
    print("  pip install ragas datasets")
    sys.exit(1)


def load_lawqa_dataset(file_path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """lawqa_jp selection.jsonをロード"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    return data


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


def evaluate_rag_system(
    pipeline: RAGPipeline,
    test_data: List[Dict[str, Any]],
    output_path: Path = None
) -> Dict[str, float]:
    """RAGシステムをRAGASで評価"""
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print("Running RAG queries...")
    for item in tqdm(test_data):
        question = item.get("question", "")
        ground_truth = item.get("answer", "")
        
        try:
            result = pipeline.query(question)
            
            questions.append(question)
            answers.append(result["answer"])
            
            context_list = [ctx["text"] for ctx in result["contexts"]]
            contexts.append(context_list)
            
            ground_truths.append(ground_truth)
            
        except Exception as e:
            print(f"Error processing question: {question[:50]}... Error: {e}")
            continue
    
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print("\nEvaluating with RAGAS...")
    metrics = [faithfulness, answer_relevancy, context_precision]
    
    result = evaluate(eval_dataset, metrics=metrics)
    
    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "metrics": scores,
            "num_samples": len(questions),
            "detailed_results": [
                {
                    "question": q,
                    "answer": a,
                    "ground_truth": gt,
                    "num_contexts": len(c)
                }
                for q, a, gt, c in zip(questions, answers, ground_truths, contexts)
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed report saved to {output_path}")
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="RAGASでRAGシステムを評価")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "lawqa_jp" / "data" / "selection.json",
        help="lawqa_jp selection.jsonパス"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluations" / "ragas_evaluation.json",
        help="評価レポート出力先"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="評価するサンプル数"
    )
    
    args = parser.parse_args()
    
    config = load_config()
    
    print("Loading test dataset...")
    test_data = load_lawqa_dataset(args.dataset, limit=args.limit)
    print(f"Loaded {len(test_data)} test samples")
    
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
    
    print("\nStarting evaluation...")
    scores = evaluate_rag_system(pipeline, test_data, output_path=args.output)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Faithfulness:       {scores['faithfulness']:.4f}")
    print(f"Answer Relevancy:   {scores['answer_relevancy']:.4f}")
    print(f"Context Precision:  {scores['context_precision']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
