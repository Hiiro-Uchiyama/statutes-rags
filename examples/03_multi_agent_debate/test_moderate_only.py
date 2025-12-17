#!/usr/bin/env python3
"""
moderate問題のみをテストするスクリプト

改善の効果を素早く確認するため、moderate問題（16問）のみを対象にテスト
"""
import json
import time
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from proposed_workflow_v7 import ProposedWorkflowV7, WorkflowConfig, DifficultyLevel


def main():
    print("=" * 60)
    print("Moderate問題のみテスト（16問）")
    print("=" * 60)
    
    # 初期化
    vector = VectorRetriever(
        index_path=str(project_root / "data/faiss_index_xml_v2/vector"),
        embedding_model="intfloat/multilingual-e5-large"
    )
    bm25 = BM25Retriever(index_path=str(project_root / "data/faiss_index_xml_v2/bm25"))
    bm25.load_index()
    retriever = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
    
    workflow = ProposedWorkflowV7(retriever=retriever)
    
    # データ読み込み
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r') as f:
        data = json.load(f)
    
    samples = data['samples'][:140]
    
    # moderate問題を特定
    moderate_indices = []
    for i, sample in enumerate(samples):
        choices = []
        for line in sample['選択肢'].split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())
        
        if len(choices) == 4:
            difficulty = workflow._assess_difficulty(sample['問題文'], choices)
            if difficulty == DifficultyLevel.MODERATE:
                moderate_indices.append(i)
    
    print(f"moderate問題数: {len(moderate_indices)}")
    print(f"問題番号: {moderate_indices}")
    print()
    
    # テスト実行
    correct = 0
    errors = []
    
    for idx in moderate_indices:
        sample = samples[idx]
        choices = []
        for line in sample['選択肢'].split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())
        
        print(f"Q{idx}...", end=" ", flush=True)
        start = time.time()
        
        result = workflow.query(sample['問題文'], choices)
        predicted = result['answer']
        correct_answer = sample['output'].strip().lower()
        is_correct = predicted == correct_answer
        
        elapsed = time.time() - start
        
        if is_correct:
            correct += 1
            print(f"OK ({predicted}/{correct_answer}) {elapsed:.1f}s")
        else:
            print(f"NG ({predicted}/{correct_answer}) {elapsed:.1f}s")
            errors.append({
                'idx': idx,
                'predicted': predicted,
                'correct': correct_answer,
                'question_type': result.get('question_type', 'unknown')
            })
    
    # 結果
    print()
    print("=" * 60)
    accuracy = correct / len(moderate_indices) * 100
    print(f"結果: {correct}/{len(moderate_indices)} ({accuracy:.1f}%)")
    print(f"目標: 80% (13/16)")
    print()
    
    if errors:
        print("誤答詳細:")
        for err in errors:
            print(f"  Q{err['idx']}: {err['predicted']} -> {err['correct']} (type={err['question_type']})")


if __name__ == "__main__":
    main()
