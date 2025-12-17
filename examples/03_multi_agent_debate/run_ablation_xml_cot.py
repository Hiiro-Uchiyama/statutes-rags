#!/usr/bin/env python3
"""
アブレーション実験: XML v2 + Hybrid + CoTプロンプト

条件:
- インデックス: XML v2 (5,045条文)
- 検索: Hybrid (Vector + BM25)
- プロンプト: CoT (選択肢ごとに条文照合を指示)
- Reranker: なし

比較対象:
- run_ablation_xml_basic.py (基本プロンプト)
- run_ablation_xml_cot_rerank.py (CoT + Reranker)
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
CONTEXT_DOCS = 15
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


def detect_question_type(question: str) -> str:
    """質問タイプを判定"""
    if "誤っている" in question or "誤り" in question:
        return "incorrect"
    elif "正しい" in question or "適切" in question:
        return "correct"
    return "unknown"


def build_cot_prompt(question: str, choices: List[str], context: str) -> str:
    """CoTプロンプトを構築"""
    q_type = detect_question_type(question)
    
    if q_type == "incorrect":
        type_instruction = """【重要】この問題は「誤っているもの」を選ぶ問題です。
条文と一致しない選択肢を探してください。"""
    elif q_type == "correct":
        type_instruction = """【重要】この問題は「正しいもの」を選ぶ問題です。
条文と一致する選択肢を探してください。"""
    else:
        type_instruction = """各選択肢を条文と照合し、最適なものを選んでください。"""
    
    choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
    
    prompt = f"""あなたは法令の専門家です。以下の法令条文を参考に、質問に正確に回答してください。

【参考法令】
{context}

【質問】
{question}

{type_instruction}

【選択肢】
{choices_text}

【回答手順】
1. 各選択肢(a,b,c,d)を1つずつ確認する
2. 各選択肢の内容が参考法令のどの条文に対応するか特定する
3. 選択肢の記述が条文と一致するか（数値、条件、期間など細部まで）確認する
4. 質問の指示（正しいもの/誤っているもの）に従い回答を選ぶ

回答は a, b, c, d のいずれか1文字のみで答えてください。
回答:"""
    return prompt, q_type


def ablation_answer(llm, retriever, question: str, choices: List[str]) -> Dict[str, Any]:
    """アブレーション実験用RAG回答"""
    start_time = time.time()
    
    # Step 1: クエリ構築
    choices_text = " ".join(choices)
    full_query = question + " " + choices_text
    
    # Step 2: クエリ正規化
    normalized_query = normalize_article_numbers(full_query, to_kanji=True)
    
    # Step 3: Hybrid検索
    docs = retriever.retrieve(normalized_query, top_k=TOP_K)
    
    # Step 4: コンテキスト構築
    context_parts = []
    for i, doc in enumerate(docs[:CONTEXT_DOCS]):
        law = doc.metadata.get("law_title", "")
        article = doc.metadata.get("article_title", "")
        text = doc.page_content[:600]
        context_parts.append(f"[{i+1}] {law} {article}\n{text}")
    context = "\n\n".join(context_parts)
    
    # Step 5: CoTプロンプト
    prompt, q_type = build_cot_prompt(question, choices, context)
    
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
        q_type = "error"
    
    elapsed = time.time() - start_time
    return {
        "answer": answer,
        "elapsed": elapsed,
        "docs_count": len(docs),
        "question_type": q_type
    }


def main():
    print("=" * 60)
    print("Ablation: XML v2 + Hybrid + CoT Prompt")
    print("=" * 60)
    print(f"Model: {LLM_MODEL}")
    print(f"Index: faiss_index_xml_v2")
    print(f"Search: Hybrid (Vector + BM25)")
    print(f"Prompt: CoT (選択肢ごと条文照合)")
    print(f"Reranker: None")
    print(f"top_k: {TOP_K}, context_docs: {CONTEXT_DOCS}")
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
    results_file = RESULTS_DIR / "ablation_xml_cot_results.json"
    log_file = RESULTS_DIR / "ablation_xml_cot_run.log"
    
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
            
            result = ablation_answer(llm, retriever, question, choices)
            
            is_correct = result["answer"] == correct_answer
            if is_correct:
                correct += 1
            
            status = "OK" if is_correct else "NG"
            print(f"{status} ({result['answer']}/{correct_answer}) {result['elapsed']:.1f}s [{result['question_type']}]")
            
            log.write(f"Q{q_id}: {status} ({result['answer']}/{correct_answer}) {result['elapsed']:.1f}s [{result['question_type']}]\n")
            log.flush()
            
            results.append({
                "question_id": q_id,
                "predicted": result["answer"],
                "correct_answer": correct_answer,
                "correct": is_correct,
                "elapsed": result["elapsed"],
                "question_type": result["question_type"]
            })
            
            # 途中保存
            if (q_id + 1) % 5 == 0:
                accuracy = correct / len(results) if results else 0
                print(f"  進捗: {len(results)}/140, 正解: {correct}, 精度: {accuracy*100:.1f}%")
                with open(results_file, "w") as f:
                    json.dump({
                        "experiment": "ablation_xml_cot",
                        "model": LLM_MODEL,
                        "index": "faiss_index_xml_v2",
                        "search_mode": "hybrid",
                        "prompt": "cot",
                        "reranker": None,
                        "top_k": TOP_K,
                        "context_docs": CONTEXT_DOCS,
                        "completed": len(results),
                        "correct": correct,
                        "accuracy": accuracy,
                        "details": results
                    }, f, ensure_ascii=False, indent=2)
    
    # 最終保存
    final_accuracy = correct / len(results) if results else 0
    with open(results_file, "w") as f:
        json.dump({
            "experiment": "ablation_xml_cot",
            "model": LLM_MODEL,
            "index": "faiss_index_xml_v2",
            "search_mode": "hybrid",
            "prompt": "cot",
            "reranker": None,
            "top_k": TOP_K,
            "context_docs": CONTEXT_DOCS,
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

