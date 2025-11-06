#!/usr/bin/env python3
"""
4択法令データ（デジタル庁）を用いたRAG評価スクリプト
"""
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import re
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
from app.retrieval.rag_pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker


def create_retriever(config):
    """設定に基づいてRetrieverを作成"""
    retriever_type = config.retriever.retriever_type
    index_path = Path(config.vector_store_path)
    
    if retriever_type == "vector":
        return VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
    elif retriever_type == "bm25":
        return BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
    else:
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
        return HybridRetriever(
            vector_retriever, 
            bm25_retriever,
            fusion_method=config.retriever.fusion_method,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            rrf_k=config.retriever.rrf_k,
            fetch_k_multiplier=config.retriever.fetch_k_multiplier
        )


def create_cot_prompt(question: str, choices: str, context: str = "") -> str:
    """Chain-of-Thought推論用のプロンプトを作成"""
    prompt = """You are a legal assistant specialized in Japanese law. Based on legal provisions, answer multiple-choice questions using step-by-step reasoning.

【Legal Provisions】
{context}

【Question】
{question}

【Choices】
{choices}

【Instructions】
Let's solve this step by step:

1. First, identify what the question is asking (correct statement or incorrect statement?)
2. Find the relevant provisions in the legal text above
3. Analyze each choice against the provisions
4. Determine which choice is correct (or incorrect, depending on the question type)

Please provide your reasoning process, then give your final answer.

Format:
Reasoning: [Your step-by-step analysis]
Answer: [single letter: a, b, c, or d]
"""
    return prompt.format(context=context, question=question, choices=choices)


def create_multiple_choice_prompt(question: str, choices: str, context: str = "", use_few_shot: bool = True) -> str:
    """4択問題用のプロンプトを作成"""
    
    if use_few_shot:
        # Few-shot例を含むプロンプト（多様な法令分野をカバー）
        prompt = """You are a legal assistant specialized in Japanese law. Based on legal provisions, answer multiple-choice questions.

Example 1 (金融商品取引法 - 正しいものを選ぶ):
【Legal Provisions】
金融商品取引法第24条第1項: 有価証券報告書の提出について、外国会社は事業年度経過後六月以内に提出しなければならない。

【Question】
外国会社が有価証券報告書を提出する場合の期限として、正しいものを教えてください。
【Choices】
a 三月以内
b 六月以内
c 九月以内
d 一年以内
【Answer】b

Example 2 (金融商品取引法 - 誤っているものを選ぶ):
【Legal Provisions】
金融商品取引法第8条第1項: 届出は、内閣総理大臣が届出書を受理した日から15日を経過した日に効力を生ずる。

【Question】
次のうち、誤っているものを教えてください。
【Choices】
a 届出は15日経過で効力発生
b 届出は10日経過で効力発生
c 内閣総理大臣は期間を短縮できる
d 内閣総理大臣は即日効力を認められる
【Answer】b

Example 3 (民法 - 正しいものを選ぶ):
【Legal Provisions】
民法第90条: 公の秩序又は善良の風俗に反する法律行為は、無効とする。
民法第91条: 法律行為の当事者が法令中の公の秩序に関しない規定と異なる意思を表示したときは、その意思に従う。

【Question】
民法における法律行為の効力に関して、正しいものを教えてください。
【Choices】
a 公序良俗に反する法律行為は取り消すことができる
b 公序良俗に反する法律行為は無効である
c 当事者の意思表示は公の秩序に関する規定に優先する
d 法律行為の効力は常に法令の規定に従う
【Answer】b

Example 4 (行政手続法 - 誤っているものを選ぶ):
【Legal Provisions】
行政手続法第2条第3号: 申請とは、法令に基づき、行政庁の許可、認可、免許その他の自己に対し何らかの利益を付与する処分を求める行為であって、当該行為に対して行政庁が諾否の応答をすべきこととされているものをいう。

【Question】
行政手続法における申請に関して、誤っているものを教えてください。
【Choices】
a 申請は法令に基づいて行う
b 申請は自己に利益を付与する処分を求める行為である
c 行政庁は申請に対して必ず許可しなければならない
d 申請に対して行政庁は諾否の応答をする
【Answer】c

Now answer this question:

【Legal Provisions】
{context}

【Question】
{question}

【Choices】
{choices}

【Critical Instructions】
- If question asks for "正しいもの" (correct), choose the TRUE statement
- If question asks for "誤っているもの" (incorrect), choose the FALSE statement
- Base your answer ONLY on the legal provisions above
- Respond with ONLY ONE letter: a, b, c, or d
- No explanation, no reasoning, just the letter
- You MUST choose one answer even if uncertain

Answer (single letter only): """
    else:
        # オリジナルのプロンプト
        prompt = """You are a legal assistant specialized in Japanese law. Based on the following legal provisions, answer the multiple-choice question.

【Legal Provisions】
{context}

【Question】
{question}

【Choices】
{choices}

【Instructions】
Based on the above legal provisions, select the most appropriate answer from choices a, b, c, or d.
You MUST respond with ONLY ONE letter: a, b, c, or d. Do not provide any explanation.
Do not say you cannot answer or that you need more information. You must choose one letter.

Answer (single letter only): """
    
    return prompt.format(context=context, question=question, choices=choices)


def extract_answer(response: str, is_cot: bool = False) -> str:
    """LLM応答から回答(a/b/c/d)を抽出"""
    response_lower = response.strip().lower()
    
    # CoT形式の場合、"Answer:"以降を優先的に探す
    if is_cot:
        # "Answer:" または "Final answer:" の後を探す
        answer_match = re.search(r'(?:final\s+)?answer\s*[:：]\s*([abcd])', response_lower)
        if answer_match:
            return answer_match.group(1)
        
        # 最後の行を確認（CoTでは最後に回答を書く傾向）
        lines = [line.strip() for line in response_lower.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            match = re.search(r'\b([abcd])\b', last_line)
            if match:
                return match.group(1)
    
    # パターン1: 単独の a, b, c, d
    if response_lower in ['a', 'b', 'c', 'd']:
        return response_lower
    
    # パターン2: "answer: a" or "回答: a" などの形式
    match = re.search(r'(?:answer|response|回答|答え|選択肢)\s*[:：]?\s*([abcd])(?![a-z])', response_lower)
    if match:
        return match.group(1)
    
    # パターン3: 最初の行に出現する a, b, c, d
    first_line = response_lower.split('\n')[0].strip()
    match = re.search(r'^[\s"\']?([abcd])[\s"\'\.]?$', first_line)
    if match:
        return match.group(1)
    
    # パターン4: 最初に出現する a, b, c, d
    match = re.search(r'\b([abcd])\b', response_lower)
    if match:
        return match.group(1)
    
    return "unknown"


def evaluate_sample(pipeline: RAGPipeline, sample: Dict[str, Any], use_rag: bool = True, use_few_shot: bool = True, use_cot: bool = False) -> Dict[str, Any]:
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
    
    # プロンプトを作成
    if use_cot:
        prompt = create_cot_prompt(question, choices, context)
    else:
        prompt = create_multiple_choice_prompt(question, choices, context, use_few_shot=use_few_shot)
    
    # LLMに質問
    try:
        response = pipeline.llm.invoke(prompt)
        predicted_answer = extract_answer(response, is_cot=use_cot)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        predicted_answer = "error"
        response = str(e)
    
    # 正解判定
    is_correct = predicted_answer == correct_answer
    
    result = {
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
    
    # CoT使用時は推論プロセスも記録
    if use_cot:
        result["reasoning"] = response
    
    return result


def evaluate_sample_with_ensemble(pipeline: RAGPipeline, sample: Dict[str, Any], 
                                 n_runs: int = 3, use_rag: bool = True, 
                                 use_few_shot: bool = True, use_cot: bool = False) -> Dict[str, Any]:
    """アンサンブル評価（複数回推論して多数決）"""
    from collections import Counter
    
    question = sample['問題文']
    choices = sample['選択肢']
    correct_answer = sample['output'].lower()
    
    # RAG検索は1回のみ
    if use_rag:
        documents = pipeline.retrieve_documents(question)
        context = pipeline.format_context(documents)
    else:
        documents = []
        context = "法令条文が提供されていません。あなたの知識に基づいて回答してください。"
    
    # 複数回推論
    predictions = []
    responses = []
    
    for i in range(n_runs):
        # プロンプト作成
        if use_cot:
            prompt = create_cot_prompt(question, choices, context)
        else:
            prompt = create_multiple_choice_prompt(question, choices, context, use_few_shot=use_few_shot)
        
        try:
            response = pipeline.llm.invoke(prompt)
            predicted = extract_answer(response, is_cot=use_cot)
            predictions.append(predicted)
            responses.append(response)
        except Exception as e:
            predictions.append("error")
            responses.append(str(e))
    
    # 多数決
    vote_counts = Counter(predictions)
    predicted_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[predicted_answer] / n_runs
    
    # 正解判定
    is_correct = predicted_answer == correct_answer
    
    result = {
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response": responses[0],  # 最初の応答を記録
        "ensemble_votes": dict(vote_counts),
        "ensemble_confidence": confidence,
        "ensemble_runs": n_runs,
        "retrieved_docs_count": len(documents),
        "file_name": sample.get('ファイル名', ''),
        "references": sample.get('references', [])
    }
    
    # CoT使用時は複数の推論プロセスも記録
    if use_cot:
        result["reasoning_samples"] = responses[:3]  # 最初の3つを記録
    
    return result


def main():
    parser = argparse.ArgumentParser(description="4択法令データを用いたRAG評価")
    parser.add_argument("--data", type=Path, 
                       default=Path("datasets/lawqa_jp/data/selection.json"),
                       help="4択データセットのパス")
    parser.add_argument("--output", type=Path, 
                       default=Path("results/evaluations/evaluation_results.json"),
                       help="評価結果の出力パス")
    parser.add_argument("--samples", type=int, default=None,
                       help="評価するサンプル数（指定しない場合は全て）")
    parser.add_argument("--no-rag", action="store_true",
                       help="RAGを使用せずLLMのみで評価")
    parser.add_argument("--top-k", type=int, default=3,
                       help="検索する文書数")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="使用するLLMモデル名（指定しない場合は設定ファイルから）")
    parser.add_argument("--no-few-shot", action="store_true",
                       help="Few-shotプロンプトを無効化")
    parser.add_argument("--use-reranker", action="store_true",
                       help="Rerankerを有効化")
    parser.add_argument("--reranker-model", type=str, 
                       default="cross-encoder/ms-marco-MiniLM-L-12-v2",
                       help="Rerankerモデル名")
    parser.add_argument("--rerank-top-n", type=int, default=3,
                       help="Reranker後の文書数")
    parser.add_argument("--ensemble", type=int, default=1,
                       help="アンサンブル推論回数（1=無効、3推奨）")
    parser.add_argument("--use-cot", action="store_true",
                       help="Chain-of-Thought推論を有効化")
    
    args = parser.parse_args()
    
    # データセットファイルの存在確認
    if not args.data.exists():
        print(f"Error: Dataset file not found: {args.data}")
        print("\nPlease download the dataset first. See docs/02-SETUP.md for instructions.")
        print("For Heart01 users, you can copy from shared directory:")
        print("  cp -r /home/jovyan/shared/datasets/statutes2025/* ./datasets/")
        sys.exit(1)
    
    # データセット読み込み
    print(f"Loading dataset from {args.data}...")
    try:
        with open(args.data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {args.data}: {e}")
        sys.exit(1)
    
    # データ構造の検証
    if 'samples' not in data:
        print(f"Error: Invalid dataset format. 'samples' key not found.")
        print(f"Expected format: {{'samples': [...]}}")
        print(f"Please check the dataset file or re-download it.")
        sys.exit(1)
    
    samples = data['samples']
    if not samples:
        print(f"Error: No samples found in dataset.")
        sys.exit(1)
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
    
    # インデックスの存在確認（RAG有効時のみ）
    if not args.no_rag:
        index_path = Path(config.vector_store_path)
        if config.retriever.retriever_type == "vector":
            if not (index_path / "vector").exists():
                print(f"Error: Vector index not found at {index_path / 'vector'}")
                print("\nPlease build the index first:")
                print("  make index")
                sys.exit(1)
        elif config.retriever.retriever_type == "bm25":
            if not (index_path / "bm25").exists():
                print(f"Error: BM25 index not found at {index_path / 'bm25'}")
                print("\nPlease build the index first:")
                print("  make index")
                sys.exit(1)
        elif config.retriever.retriever_type == "hybrid":
            if not (index_path / "vector").exists() or not (index_path / "bm25").exists():
                print(f"Error: Hybrid index not found at {index_path}")
                print("\nPlease build the index first:")
                print("  make index")
                sys.exit(1)
    
    retriever = create_retriever(config) if not args.no_rag else None
    
    # RAG無効時はダミーのRetrieverを使用
    if args.no_rag:
        from unittest.mock import Mock
        retriever = Mock()
        retriever.retrieve = Mock(return_value=[])
    
    # Rerankerの初期化
    reranker = None
    if args.use_reranker:
        print(f"Initializing Reranker: {args.reranker_model}...")
        try:
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
            print("Reranker initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Reranker: {e}")
            print("Continuing without Reranker...")
            reranker = None
    
    # Top-KをReranker使用時に調整
    retriever_top_k = args.top_k if not args.use_reranker else max(args.top_k, args.rerank_top_n * 2)
    
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider=config.llm.provider,
        llm_model=llm_model,
        temperature=0.0,  # 4択問題では決定的な回答が望ましい
        reranker=reranker,
        top_k=retriever_top_k,
        rerank_top_n=args.rerank_top_n if args.use_reranker else 5,
        request_timeout=120  # タイムアウトを120秒に延長
    )
    
    print(f"RAG Mode: {'Disabled (LLM only)' if args.no_rag else 'Enabled'}")
    print(f"Retriever Type: {config.retriever.retriever_type}")
    print(f"LLM Model: {llm_model}")
    print(f"Few-shot Prompt: {'Disabled' if args.no_few_shot else 'Enabled'}")
    print(f"Chain-of-Thought: {'Enabled' if args.use_cot else 'Disabled'}")
    print(f"Reranker: {'Enabled' if args.use_reranker else 'Disabled'}")
    if args.use_reranker:
        print(f"  Model: {args.reranker_model}")
        print(f"  Top-N: {args.rerank_top_n}")
    print(f"Ensemble: {'Enabled ('+str(args.ensemble)+' runs)' if args.ensemble > 1 else 'Disabled'}")
    print(f"Top-K: {retriever_top_k}\n")
    
    # 評価実行
    results = []
    correct_count = 0
    
    for sample in tqdm(samples, desc="Evaluating"):
        if args.ensemble > 1:
            # アンサンブル評価
            result = evaluate_sample_with_ensemble(
                pipeline, sample, 
                n_runs=args.ensemble,
                use_rag=not args.no_rag, 
                use_few_shot=not args.no_few_shot,
                use_cot=args.use_cot
            )
        else:
            # 通常評価
            result = evaluate_sample(
                pipeline, sample, 
                use_rag=not args.no_rag, 
                use_few_shot=not args.no_few_shot,
                use_cot=args.use_cot
            )
        results.append(result)
        
        if result['is_correct']:
            correct_count += 1
    
    # 精度計算
    accuracy = correct_count / len(results) if results else 0
    
    # エラー統計
    error_count = sum(1 for r in results if r['predicted_answer'] == 'error')
    unknown_count = sum(1 for r in results if r['predicted_answer'] == 'unknown')
    timeout_errors = sum(1 for r in results if 'timeout' in r.get('response', '').lower())
    
    # 結果サマリー
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(results) - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nError Analysis:")
    print(f"  Timeout Errors: {timeout_errors}")
    print(f"  Parse Errors (unknown): {unknown_count}")
    print(f"  Other Errors: {error_count}")
    print("="*50)
    
    # 詳細結果を保存
    output_data = {
        "config": {
            "rag_enabled": not args.no_rag,
            "retriever_type": config.retriever.retriever_type,
            "llm_model": llm_model,
            "few_shot_enabled": not args.no_few_shot,
            "cot_enabled": args.use_cot,
            "reranker_enabled": args.use_reranker,
            "reranker_model": args.reranker_model if args.use_reranker else None,
            "rerank_top_n": args.rerank_top_n if args.use_reranker else None,
            "ensemble_runs": args.ensemble if args.ensemble > 1 else None,
            "top_k": retriever_top_k,
            "total_samples": len(results)
        },
        "summary": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(results)
        },
        "results": results
    }
    
    # 出力ディレクトリが存在しない場合は作成
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # 改善提案
    if timeout_errors > 0:
        print("\n⚠️  Timeout errors detected. Consider:")
        print("  - Increasing --top-k parameter to reduce context size")
        print("  - Using a smaller/faster LLM model")
    if unknown_count > 5:
        print(f"\n⚠️  {unknown_count} parse errors detected. The LLM may not be following instructions.")
        print("  - Check if the LLM model supports Japanese and English instructions")
        print("  - Consider adjusting the prompt template")
    
    # エラーケースの表示
    error_cases = [r for r in results if not r['is_correct']]
    if error_cases:
        print("\n" + "="*50)
        print(f"SAMPLE ERROR CASES (showing up to 10):")
        print("="*50)
        for i, case in enumerate(error_cases[:10], 1):
            print(f"\n[Case {i}] {case['file_name']}")
            print(f"Question: {case['question'][:100]}...")
            print(f"Correct: {case['correct_answer']} | Predicted: {case['predicted_answer']}")
            if case['predicted_answer'] in ['error', 'unknown']:
                print(f"Response: {case['response'][:150]}...")


if __name__ == "__main__":
    main()
