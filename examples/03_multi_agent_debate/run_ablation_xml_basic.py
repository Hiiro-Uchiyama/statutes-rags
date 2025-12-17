#!/usr/bin/env python3
"""
Baseline RAG - XMLインデックス版

XMLインデックス（5,045条文、9法令）でのベースラインRAG
- Hybrid検索（Vector + BM25）
- 数字正規化対応済み
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.llms import Ollama
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.utils.number_normalizer import normalize_article_numbers

# 設定
TOP_K = 30
TIMEOUT = 180
LLM_MODEL = "qwen3:8b"
NUM_CTX = 30000

# XML v2インデックス
VECTOR_PATH = "data/faiss_index_xml_v2/vector"
BM25_PATH = "data/faiss_index_xml_v2/bm25"

RESULTS_DIR = Path(__file__).parent / "results"


def load_questions() -> List[Dict]:
    """テストデータをロード"""
    test_file = project_root / "datasets" / "lawqa_jp" / "data" / "selection.json"
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for i, item in enumerate(data.get("samples", [])[:140]):
        choices_text = item.get("選択肢", "")
        choices = []
        for line in choices_text.split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())
        
        samples.append({
            "question_id": i,
            "question": item.get("問題文", ""),
            "choices": choices,
            "answer": item.get("output", ""),
            "context": item.get("コンテキスト", "")
        })
    return samples


def baseline_rag_answer(llm, retriever, question: str, choices: List[str]) -> Dict[str, Any]:
    """純粋なRAG: 検索 + LLM直接回答"""
    start_time = time.time()
    
    # Step 1: クエリ構築（問題文 + 選択肢）
    choices_text = " ".join(choices)
    full_query = question + " " + choices_text
    
    # Step 2: クエリ正規化（数字→漢数字）
    normalized_query = normalize_article_numbers(full_query, to_kanji=True)
    
    # Step 3: Hybrid検索
    docs = retriever.retrieve(normalized_query, top_k=TOP_K)
    
    # Step 4: コンテキスト構築
    context_parts = []
    for i, doc in enumerate(docs[:15]):
        law = doc.metadata.get("law_title", "")
        article = doc.metadata.get("article_title", "")
        text = doc.page_content[:500]
        context_parts.append(f"[{i+1}] {law} {article}\n{text}")
    context = "\n\n".join(context_parts)
    
    # Step 5: LLMに直接質問
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
        answer = None
        response_lower = response.lower().strip()
        for char in ['a', 'b', 'c', 'd']:
            if char in response_lower[:50]:
                answer = char
                break
        if not answer:
            answer = 'a'
    except Exception as e:
        print(f"LLM error: {e}")
        answer = 'a'
    
    elapsed = time.time() - start_time
    return {
        "answer": answer,
        "elapsed": elapsed,
        "docs_count": len(docs)
    }


def main():
    print("=" * 60)
    print("Baseline RAG - XML v2インデックス (Hybrid)")
    print("=" * 60)
    print(f"Model: {LLM_MODEL}")
    print(f"Index: faiss_index_xml_v2")
    print(f"Search: Hybrid (Vector + BM25)")
    print(f"top_k: {TOP_K}")
    print()
    
    # LLM初期化
    llm = Ollama(model=LLM_MODEL, timeout=TIMEOUT, num_ctx=NUM_CTX)
    
    # Retriever初期化
    vector_retriever = VectorRetriever(
        index_path=str(project_root / VECTOR_PATH),
        embedding_model="intfloat/multilingual-e5-large"
    )
    
    bm25_retriever = BM25Retriever(index_path=str(project_root / BM25_PATH))
    bm25_retriever.load_index()
    
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    print(f"Vector index: {vector_retriever.vector_store.index.ntotal} docs")
    print(f"BM25 index: {len(bm25_retriever.documents)} docs")
    
    # テストデータ
    questions = load_questions()
    print(f"Questions: {len(questions)}")
    print()
    
    # 結果ファイル
    results_file = RESULTS_DIR / "baseline_xml_results.json"
    log_file = RESULTS_DIR / "baseline_xml_run.log"
    
    # 既存結果の読み込み（再開用）
    existing = {}
    start_idx = 0
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                existing = {r["question_id"]: r for r in data.get("details", [])}
                start_idx = len(existing)
                print(f"Resuming from {start_idx}/140")
        except:
            pass
    
    results = list(existing.values())
    correct = sum(1 for r in results if r.get("correct", False))
    
    with open(log_file, "a") as log:
        log.write(f"\n=== Run started at {datetime.now()} ===\n")
        
        for sample in questions:
            q_id = sample["question_id"]
            
            if q_id < start_idx:
                continue
            
            question = sample["question"]
            choices = sample["choices"]
            correct_answer = sample["answer"].strip().lower()
            
            print(f"[{q_id+1}/140] Q{q_id}...", end=" ", flush=True)
            
            result = baseline_rag_answer(llm, retriever, question, choices)
            
            is_correct = result["answer"] == correct_answer
            if is_correct:
                correct += 1
            
            status = "OK" if is_correct else "NG"
            print(f"{status} ({result['answer']}/{correct_answer}) {result['elapsed']:.1f}s")
            
            log.write(f"Q{q_id}: {status} ({result['answer']}/{correct_answer}) {result['elapsed']:.1f}s\n")
            log.flush()
            
            results.append({
                "question_id": q_id,
                "predicted": result["answer"],
                "correct_answer": correct_answer,
                "correct": is_correct,
                "elapsed": result["elapsed"]
            })
            
            # 途中保存
            if (q_id + 1) % 5 == 0:
                with open(results_file, "w") as f:
                    json.dump({
                        "model": LLM_MODEL,
                        "index": "faiss_index_xml_v2",
                        "search_mode": "hybrid",
                        "top_k": TOP_K,
                        "completed": len(results),
                        "correct": correct,
                        "accuracy": correct / len(results) if results else 0,
                        "details": results
                    }, f, ensure_ascii=False, indent=2)
    
    # 最終保存
    final_accuracy = correct / len(results) if results else 0
    with open(results_file, "w") as f:
        json.dump({
            "model": LLM_MODEL,
            "index": "faiss_index_xml_v2",
            "search_mode": "hybrid",
            "top_k": TOP_K,
            "completed": len(results),
            "correct": correct,
            "accuracy": final_accuracy,
            "timestamp": datetime.now().isoformat(),
            "details": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"完了: {correct}/{len(results)} ({final_accuracy*100:.1f}%)")
    print(f"結果: {results_file}")


if __name__ == "__main__":
    main()

