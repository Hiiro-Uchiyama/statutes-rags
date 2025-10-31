#!/usr/bin/env python3
"""
4択法令データ（デジタル庁）を用いたRAG評価スクリプト
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag_config import load_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
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


def create_multiple_choice_prompt(question: str, choices: str, context: str = "") -> str:
    """4択問題用のプロンプトを作成"""
    prompt = """あなたは日本の法律に精通した法律アシスタントです。以下の法令条文に基づいて、4択問題に答えてください。

【法令条文】
{context}

【問題文】
{question}

【選択肢】
{choices}

【指示】
上記の法令条文に基づいて、選択肢a、b、c、dの中から最も適切なものを1つ選んでください。
回答は必ず「a」「b」「c」「d」のいずれか1文字のみを返してください。説明は不要です。

回答: """
    
    return prompt.format(context=context, question=question, choices=choices)


def extract_answer(response: str) -> str:
    """LLM応答から回答(a/b/c/d)を抽出"""
    response = response.strip().lower()
    
    # パターン1: 単独の a, b, c, d
    if response in ['a', 'b', 'c', 'd']:
        return response
    
    # パターン2: 「回答: a」「答え: b」などの形式
    match = re.search(r'(?:回答|答え|選択肢)?\s*[:：]?\s*([abcd])', response)
    if match:
        return match.group(1)
    
    # パターン3: 最初に出現する a, b, c, d
    match = re.search(r'([abcd])', response)
    if match:
        return match.group(1)
    
    return "unknown"


def evaluate_sample(pipeline: RAGPipeline, sample: Dict[str, Any], use_rag: bool = True) -> Dict[str, Any]:
    """1サンプルを評価"""
    question = sample['問題文']
    choices = sample['選択肢']
    correct_answer = sample['output'].lower()
    
    if use_rag:
        # RAGで関連法令を検索
        documents = pipeline.retrieve_documents(question)
        context = pipeline.format_context(documents)
    else:
        # コンテキストなし（LLMのみ）
        documents = []
        context = "法令条文が提供されていません。あなたの知識に基づいて回答してください。"
    
    # 4択プロンプトを作成
    prompt = create_multiple_choice_prompt(question, choices, context)
    
    # LLMに質問
    try:
        response = pipeline.llm.invoke(prompt)
        predicted_answer = extract_answer(response)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        predicted_answer = "error"
        response = str(e)
    
    # 正解判定
    is_correct = predicted_answer == correct_answer
    
    return {
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response": response,
        "retrieved_docs_count": len(documents),
        "file_name": sample.get('ファイル名', ''),
        "references": sample.get('references', [])
    }


def main():
    parser = argparse.ArgumentParser(description="4択法令データを用いたRAG評価")
    parser.add_argument("--data", type=Path, 
                       default=Path("datasets/lawqa_jp/data/selection.json"),
                       help="4択データセットのパス")
    parser.add_argument("--output", type=Path, 
                       default=Path("evaluation_results.json"),
                       help="評価結果の出力パス")
    parser.add_argument("--samples", type=int, default=None,
                       help="評価するサンプル数（指定しない場合は全て）")
    parser.add_argument("--no-rag", action="store_true",
                       help="RAGを使用せずLLMのみで評価")
    parser.add_argument("--top-k", type=int, default=5,
                       help="検索する文書数")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="使用するLLMモデル名（指定しない場合は設定ファイルから）")
    
    args = parser.parse_args()
    
    # データセット読み込み
    print(f"Loading dataset from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    if args.samples:
        samples = samples[:args.samples]
    
    print(f"Total samples to evaluate: {len(samples)}")
    
    # RAGパイプライン初期化
    print("\nInitializing RAG pipeline...")
    config = load_config()
    
    # top_kを引数で上書き
    config.retriever.top_k = args.top_k
    
    # LLMモデル名を引数で上書き
    llm_model = args.llm_model if args.llm_model else config.llm.model_name
    
    retriever = create_retriever(config)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider=config.llm.provider,
        llm_model=llm_model,
        temperature=0.0,  # 4択問題では決定的な回答が望ましい
        reranker=None,  # Rerankerは無効化
        top_k=config.retriever.top_k,
        rerank_top_n=config.retriever.top_k
    )
    
    print(f"RAG Mode: {'Disabled (LLM only)' if args.no_rag else 'Enabled'}")
    print(f"Retriever Type: {config.retriever.retriever_type}")
    print(f"LLM Model: {llm_model}")
    print(f"Top-K: {args.top_k}\n")
    
    # 評価実行
    results = []
    correct_count = 0
    
    for sample in tqdm(samples, desc="Evaluating"):
        result = evaluate_sample(pipeline, sample, use_rag=not args.no_rag)
        results.append(result)
        
        if result['is_correct']:
            correct_count += 1
    
    # 精度計算
    accuracy = correct_count / len(results) if results else 0
    
    # 結果サマリー
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(results) - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*50)
    
    # 詳細結果を保存
    output_data = {
        "config": {
            "rag_enabled": not args.no_rag,
            "retriever_type": config.retriever.retriever_type,
            "llm_model": llm_model,
            "top_k": args.top_k,
            "total_samples": len(results)
        },
        "summary": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(results)
        },
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # エラーケースの表示
    error_cases = [r for r in results if not r['is_correct']]
    if error_cases and len(error_cases) <= 5:
        print("\n" + "="*50)
        print("INCORRECT CASES:")
        print("="*50)
        for i, case in enumerate(error_cases[:5], 1):
            print(f"\n[Case {i}]")
            print(f"Question: {case['question'][:100]}...")
            print(f"Correct: {case['correct_answer']}")
            print(f"Predicted: {case['predicted_answer']}")


if __name__ == "__main__":
    main()
