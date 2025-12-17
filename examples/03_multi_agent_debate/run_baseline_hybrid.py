#!/usr/bin/env python3
"""
Baseline RAG - Hybrid検索 (Vector + BM25 RRF)

エージェントアーキテクチャなしの単純なRAGベースライン
Hybrid検索を使用
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.llms import Ollama
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

# 設定
TOP_K = 30
TIMEOUT = 180
INDEX_PATH = "data/faiss_index_lawqa"
LLM_MODEL = "qwen3:8b"
NUM_CTX = 30000

def load_questions() -> List[Dict]:
    """テストデータをロード"""
    test_file = project_root / "datasets" / "lawqa_jp" / "data" / "selection.json"
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # データ形式を統一
    samples = []
    for i, item in enumerate(data.get("samples", [])[:140]):
        # 選択肢をパース（改行区切り）
        choices_text = item.get("選択肢", "")
        choices = []
        for line in choices_text.split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())  # "a " を除去
        
        samples.append({
            "question_id": i,
            "question": item.get("問題文", ""),
            "choices": choices,
            "answer": item.get("output", ""),
            "context": item.get("コンテキスト", "")
        })
    return samples


def baseline_hybrid_answer(llm, retriever, question: str, choices: List[str]) -> Dict[str, Any]:
    """Hybrid RAG: 検索 + LLM直接回答"""
    start_time = time.time()
    
    # Step 1: Hybrid検索 (Vector + BM25 RRF)
    docs = retriever.retrieve(question, top_k=TOP_K)
    
    # Step 2: コンテキスト構築
    context_parts = []
    for i, doc in enumerate(docs[:15]):  # 上位15件を使用
        law = doc.metadata.get("law_title", "")
        text = doc.page_content[:500]
        context_parts.append(f"[{i+1}] {law}\n{text}")
    context = "\n\n".join(context_parts)
    
    # Step 3: LLMに直接質問
    choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
    
    prompt = f"""以下の法令条文を参考に、質問に対する最適な回答を選んでください。

【参考法令】
{context}

【質問】
{question}

【選択肢】
{choices_text}

回答は a, b, c, d のいずれか1文字のみで答えてください。
回答:"""

    try:
        response = llm.invoke(prompt)
        # 回答を抽出
        answer = None
        response_lower = response.lower().strip()
        for char in ['a', 'b', 'c', 'd']:
            if char in response_lower[:50]:
                answer = char
                break
        if not answer:
            answer = 'a'  # デフォルト
    except Exception as e:
        print(f"LLM error: {e}")
        answer = 'a'
    
    elapsed = time.time() - start_time
    
    return {
        "answer": answer,
        "elapsed_time": elapsed,
        "docs_retrieved": len(docs)
    }


def save_results(filepath: Path, data: Dict):
    """結果を保存"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("Baseline RAG Evaluation (Hybrid: Vector + BM25 RRF)")
    print("=" * 60)
    print(f"Index: {INDEX_PATH}")
    print(f"Model: {LLM_MODEL}")
    print(f"top_k: {TOP_K}")
    print()
    
    # LLM初期化
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.1,
        base_url=ollama_host,
        timeout=TIMEOUT,
        num_ctx=NUM_CTX
    )
    
    # Hybrid Retriever初期化
    vector_path = str(project_root / INDEX_PATH / "vector")
    bm25_path = str(project_root / INDEX_PATH / "bm25")
    
    vector_retriever = VectorRetriever(
        embedding_model="intfloat/multilingual-e5-large",
        index_path=vector_path
    )
    vector_retriever.load_index()
    
    bm25_retriever = BM25Retriever(index_path=bm25_path)
    bm25_retriever.load_index()
    
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fusion_method="rrf",  # 標準RRF
        rrf_k=60
    )
    
    print(f"Vector index: {vector_retriever.vector_store.index.ntotal} docs")
    print(f"BM25 index: {len(bm25_retriever.documents)} docs")
    print(f"Fusion: RRF (k=60)")
    
    # テストデータロード
    questions = load_questions()
    print(f"Questions: {len(questions)}")
    print()
    
    # 結果ファイル
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "baseline_hybrid_results.json"
    
    # 既存結果の読み込み（再開用）
    results = {
        "method": "Baseline RAG (Hybrid: Vector + BM25 RRF)",
        "config": {
            "index": INDEX_PATH,
            "top_k": TOP_K,
            "model": LLM_MODEL,
            "num_ctx": NUM_CTX,
            "fusion_method": "rrf",
            "rrf_k": 60
        },
        "started_at": datetime.now().isoformat(),
        "completed": 0,
        "correct": 0,
        "accuracy": 0.0,
        "details": []
    }
    
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if existing.get("completed", 0) > 0:
                results = existing
                print(f"Resuming from {results['completed']}/140")
    
    start_idx = results["completed"]
    
    # 評価ループ
    for i, q in enumerate(questions[start_idx:], start=start_idx):
        qid = q.get("question_id", i)
        question = q.get("question", "")
        choices = q.get("choices", [])
        correct = q.get("answer", "")
        
        print(f"[{i+1}/140] Q{qid}...", end=" ", flush=True)
        
        try:
            result = baseline_hybrid_answer(llm, retriever, question, choices)
            predicted = result["answer"]
            is_correct = predicted == correct
            
            results["details"].append({
                "question_id": qid,
                "predicted": predicted,
                "correct": correct,
                "is_correct": is_correct,
                "elapsed_time": result["elapsed_time"]
            })
            
            results["completed"] = i + 1
            if is_correct:
                results["correct"] += 1
            results["accuracy"] = results["correct"] / results["completed"]
            
            status = "OK" if is_correct else "NG"
            print(f"{status} ({predicted}/{correct}) {result['elapsed_time']:.1f}s")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results["details"].append({
                "question_id": qid,
                "error": str(e)
            })
            results["completed"] = i + 1
        
        # 定期保存
        if (i + 1) % 5 == 0:
            save_results(results_file, results)
    
    # 最終保存
    results["finished_at"] = datetime.now().isoformat()
    save_results(results_file, results)
    
    print()
    print("=" * 60)
    print(f"Baseline Hybrid RAG Results")
    print("=" * 60)
    print(f"Completed: {results['completed']}/140")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()

