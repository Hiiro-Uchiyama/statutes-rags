#!/usr/bin/env python3
"""
提案手法 v3: 選択肢ごとの検証 + 統合 + 再確認 + 最終判定

実験スクリプト
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
sys.path.insert(0, str(Path(__file__).parent))

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from proposed_workflow_v3 import ProposedWorkflowV3, WorkflowConfig

# 設定
TOP_K = 30
TIMEOUT = 180
LLM_MODEL = "qwen3:8b"
NUM_CTX = 16000

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="開始問題番号")
    parser.add_argument("--end", type=int, default=140, help="終了問題番号")
    parser.add_argument("--test", action="store_true", help="テストモード（Q2のみ）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Proposed v3: Choice Verification + Integration + Recheck")
    print("=" * 60)
    print(f"Model: {LLM_MODEL}")
    print(f"Index: faiss_index_xml_v2")
    print(f"Search: Hybrid (Vector + BM25)")
    print()
    
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
    
    # ワークフロー初期化
    config = WorkflowConfig(
        llm_model=LLM_MODEL,
        timeout=TIMEOUT,
        num_ctx=NUM_CTX,
        top_k=TOP_K,
        enable_recheck=True
    )
    workflow = ProposedWorkflowV3(retriever=retriever, config=config)
    
    # テストデータ
    questions = load_questions()
    print(f"Questions: {len(questions)}")
    print()
    
    if args.test:
        # Q2のみテスト
        print("=== テストモード: Q2 ===")
        sample = questions[1]  # Q2 (index 1)
        
        print(f"\n問題: {sample['question']}")
        print(f"正解: {sample['answer']}")
        print()
        
        start_time = time.time()
        result = workflow.query(sample['question'], sample['choices'])
        elapsed = time.time() - start_time
        
        print(f"\n=== 結果 ===")
        print(f"予測: {result['answer']}")
        print(f"正解: {sample['answer'].strip().lower()}")
        print(f"一致: {'OK' if result['answer'] == sample['answer'].strip().lower() else 'NG'}")
        print(f"確信度: {result['confidence']}")
        print(f"処理時間: {elapsed:.1f}秒")
        print(f"\n=== 法令CoT ===")
        print(result['legal_cot'])
        print(f"\n=== 検証サマリ ===")
        print(result['verification_summary'])
        print(f"\n=== 不確実な点 ===")
        print(result['uncertain_points'])
        print(f"\n=== ステージ情報 ===")
        print(result['stages'])
        return
    
    # 本番実行
    results_file = RESULTS_DIR / "proposed_v3_results.json"
    log_file = RESULTS_DIR / "proposed_v3_run.log"
    
    # 既存結果の読み込み
    existing = {}
    start_idx = args.start
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                existing = {r["question_id"]: r for r in data.get("details", [])}
                if args.start == 0:
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
            
            if q_id < start_idx or q_id >= args.end:
                continue
            
            question = sample["question"]
            choices = sample["choices"]
            correct_answer = sample["answer"].strip().lower()
            
            print(f"[{q_id+1}/140] Q{q_id}...", end=" ", flush=True)
            
            start_time = time.time()
            try:
                result = workflow.query(question, choices)
                answer = result["answer"]
            except Exception as e:
                print(f"Error: {e}")
                answer = "a"
                result = {"confidence": 0.0, "legal_cot": f"Error: {e}"}
            
            elapsed = time.time() - start_time
            
            is_correct = answer == correct_answer
            if is_correct:
                correct += 1
            
            status = "OK" if is_correct else "NG"
            print(f"{status} ({answer}/{correct_answer}) {elapsed:.1f}s")
            
            log.write(f"Q{q_id}: {status} ({answer}/{correct_answer}) {elapsed:.1f}s\n")
            log.flush()
            
            results.append({
                "question_id": q_id,
                "predicted": answer,
                "correct_answer": correct_answer,
                "correct": is_correct,
                "elapsed": elapsed,
                "confidence": result.get("confidence", 0.0)
            })
            
            # 途中保存
            if (q_id + 1) % 5 == 0:
                accuracy = correct / len(results) if results else 0
                print(f"  進捗: {len(results)}/140, 正解: {correct}, 精度: {accuracy*100:.1f}%")
                with open(results_file, "w") as f:
                    json.dump({
                        "experiment": "proposed_v3",
                        "model": LLM_MODEL,
                        "index": "faiss_index_xml_v2",
                        "search_mode": "hybrid",
                        "completed": len(results),
                        "correct": correct,
                        "accuracy": accuracy,
                        "details": results
                    }, f, ensure_ascii=False, indent=2)
    
    # 最終保存
    final_accuracy = correct / len(results) if results else 0
    with open(results_file, "w") as f:
        json.dump({
            "experiment": "proposed_v3",
            "model": LLM_MODEL,
            "index": "faiss_index_xml_v2",
            "search_mode": "hybrid",
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

