#!/usr/bin/env python3
"""
v9テストスクリプト: 構造化法令QAシステム
"""
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from proposed_workflow_v9 import ProposedWorkflowV9, WorkflowConfig

logger = logging.getLogger("proposed_v9_runner")


def setup_logging(log_path: Path) -> None:
    """コンソールとファイルの両方にログを吐き出す"""
    log_path.parent.mkdir(exist_ok=True, parents=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(description="Proposed v9 構造化法令QA 実験")
    parser.add_argument("--model", default="qwen3:8b", help="Ollamaモデル名")
    parser.add_argument("--timeout", type=int, default=180, help="LLMタイムアウト秒")
    parser.add_argument("--num-ctx", type=int, default=20000, help="コンテキスト長")
    parser.add_argument("--top-k-simple", type=int, default=30, help="SIMPLE向けtop_k")
    parser.add_argument("--top-k-moderate", type=int, default=30, help="MODERATE向けtop_k")
    parser.add_argument("--top-k-complex", type=int, default=35, help="COMPLEX向けtop_k")
    parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "weighted_rrf", "weighted"], help="Hybrid融合方式")
    parser.add_argument("--vector-weight", type=float, default=0.4, help="weighted系でのベクトル重み")
    parser.add_argument("--bm25-weight", type=float, default=0.6, help="weighted系でのBM25重み")
    parser.add_argument("--start", type=int, default=0, help="開始インデックス（既存結果を無視して再計算する場合に指定）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Proposed v9: 構造化法令DB + 分析統合（検証Agentなし / v9b-L系）")
    print("=" * 60)
    
    # 設定
    config = WorkflowConfig(
        llm_model=args.model,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
        top_k_simple=args.top_k_simple,
        top_k_moderate=args.top_k_moderate,
        top_k_complex=args.top_k_complex
    )
    
    # Retriever初期化
    vector = VectorRetriever(
        index_path=str(project_root / "data/faiss_index_xml_v2/vector"),
        embedding_model="intfloat/multilingual-e5-large"
    )
    
    bm25 = BM25Retriever(
        index_path=str(project_root / "data/faiss_index_xml_v2/bm25")
    )
    bm25.load_index()
    
    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        fusion_method=args.fusion_method,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight
    )
    
    print(f"Vector index: {len(vector.vector_store.docstore._dict)} docs")
    print(f"BM25 index: {len(bm25.documents)} docs")
    logger.info(f"Vector index docs={len(vector.vector_store.docstore._dict)}, BM25 docs={len(bm25.documents)}, fusion_method={args.fusion_method}, weights=({args.vector_weight},{args.bm25_weight})")
    
    # ワークフロー初期化
    workflow = ProposedWorkflowV9(retriever=retriever, config=config)
    
    # データ読み込み
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples'][:140]
    print(f"Questions: {len(samples)}")
    print()
    
    # 結果ファイルパス
    results_path = Path(__file__).parent / "results" / "proposed_v9_results.json"
    results_path.parent.mkdir(exist_ok=True)
    logger.info(f"results_path={results_path}")
    
    # 既存結果の読み込み（再開機能）
    completed_results = []
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            completed_results = existing.get('details', [])
            print(f"Resuming from {len(completed_results)}/{len(samples)}")
            logger.info(f"resume: {len(completed_results)}/{len(samples)}")
    
    # 明示的な開始位置（既存結果を無視する用途）
    if args.start > 0:
        logger.warning(f"--start={args.start} が指定されたため既存結果を破棄して再計算します")
        completed_results = []
        if results_path.exists():
            backup_path = results_path.with_suffix(".bak")
            results_path.replace(backup_path)
            logger.warning(f"既存結果をバックアップ: {backup_path}")
    
    # テスト実行
    correct = sum(1 for r in completed_results if r['correct'])
    difficulty_stats = {}
    difficulty_correct = {}
    
    for r in completed_results:
        diff = r['difficulty']
        difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1
        if r['correct']:
            difficulty_correct[diff] = difficulty_correct.get(diff, 0) + 1
    
    start_idx = len(completed_results)
    
    for i, sample in enumerate(samples[start_idx:], start=start_idx):
        # 選択肢をパース
        choices = []
        for line in sample['選択肢'].split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())
        
        if len(choices) != 4:
            print(f"[{i+1}/140] Q{i}... SKIP (invalid choices)")
            logger.error(f"skip invalid choices: question_id={i}")
            continue
        
        print(f"[{i+1}/140] Q{i}... ", end="", flush=True)
        start_time = time.time()
        
        try:
            result = workflow.query(sample['問題文'], choices)
            predicted = result['answer']
            difficulty = result['difficulty']
            method = result['method']
        except Exception as e:
            print(f"ERROR: {e}")
            logger.exception(f"error at question_id={i}: {e}")
            predicted = 'a'
            difficulty = 'simple'
            method = 'error'
        
        elapsed = time.time() - start_time
        correct_answer = sample['output'].strip().lower()
        is_correct = predicted == correct_answer
        
        if is_correct:
            correct += 1
            status = "OK"
        else:
            status = "NG"
        
        print(f"{status} ({predicted}/{correct_answer}) {difficulty} {elapsed:.1f}s")
        logger.info(f"q={i} status={status} pred={predicted} gold={correct_answer} diff={difficulty} time={elapsed:.1f}s method={method}")
        
        # 難易度統計
        difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
        if is_correct:
            difficulty_correct[difficulty] = difficulty_correct.get(difficulty, 0) + 1
        
        # 結果保存
        completed_results.append({
            'question_id': i,
            'predicted': predicted,
            'correct_answer': correct_answer,
            'correct': is_correct,
            'difficulty': difficulty,
            'method': method,
            'elapsed': elapsed
        })
        
        # 中間保存
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': config.llm_model,
            'method': 'v9_structured',
            'completed': len(completed_results),
            'correct': correct,
            'accuracy': correct / len(completed_results),
            'difficulty_stats': difficulty_stats,
            'difficulty_correct': difficulty_correct,
            'details': completed_results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 進捗表示（5問ごと）
        if (i + 1) % 5 == 0:
            acc = correct / len(completed_results) * 100
            print(f"  進捗: {len(completed_results)}/140, 正解: {correct}, 精度: {acc:.1f}%")
            for diff in ['simple', 'moderate', 'complex']:
                if diff in difficulty_stats:
                    d_total = difficulty_stats[diff]
                    d_correct = difficulty_correct.get(diff, 0)
                    print(f"    {diff}: {d_correct}/{d_total} ({d_correct/d_total*100:.1f}%)")
            logger.info(f"progress: completed={len(completed_results)} correct={correct} acc={acc:.2f}")
    
    # 最終結果
    print()
    print("=" * 60)
    print("最終結果")
    print("=" * 60)
    
    accuracy = correct / len(completed_results) * 100
    print(f"正解: {correct}/{len(completed_results)} ({accuracy:.1f}%)")
    print(f"目標: 112/140 (80%)")
    print()
    
    print("難易度別:")
    for diff in ['simple', 'moderate', 'complex']:
        if diff in difficulty_stats:
            d_total = difficulty_stats[diff]
            d_correct = difficulty_correct.get(diff, 0)
            print(f"  {diff}: {d_correct}/{d_total} ({d_correct/d_total*100:.1f}%)")
            logger.info(f"final diff={diff} {d_correct}/{d_total} acc={d_correct/d_total*100:.2f}%")
    
    if accuracy >= 80:
        print()
        print("★★★ 目標達成！★★★")
    
    logger.info(f"final: correct={correct}/{len(completed_results)} acc={accuracy:.2f}%")
    logger.info("=== Proposed v9 run end ===")


if __name__ == "__main__":
    main()
