#!/usr/bin/env python3
"""
v7sc テスト実行スクリプト

Self-Consistency（多数決）によるマルチエージェント推論
- 複数エージェント（デフォルト3）による並列推論
- 多数決で最終回答決定

v7オリジナル: 75.7% (106/140)
目標: 80%以上
"""
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

from proposed_workflow_v7sc import ProposedWorkflowV7SC, WorkflowConfig


def main():
    # 設定
    NUM_AGENTS = 3  # エージェント数
    TEMPERATURE = 0.3  # 多様性のための温度（低めに設定）
    
    print("=" * 60)
    print("Proposed v7sc: Self-Consistency（多数決）マルチエージェント")
    print("=" * 60)
    print(f"Model: qwen3:8b")
    print(f"Agents: {NUM_AGENTS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Index: faiss_index_xml_v2")
    print(f"Search: Hybrid (Vector + BM25)")
    print(f"v7オリジナル: 75.7% (106/140)")
    print(f"目標: 80%以上 (112/140)")
    print()
    
    # 初期化
    vector = VectorRetriever(
        index_path=str(project_root / "data/faiss_index_xml_v2/vector"),
        embedding_model="intfloat/multilingual-e5-large"
    )
    bm25 = BM25Retriever(index_path=str(project_root / "data/faiss_index_xml_v2/bm25"))
    bm25.load_index()
    retriever = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
    
    print(f"Vector index: {vector.vector_store.index.ntotal} docs")
    print(f"BM25 index: {len(bm25.documents)} docs")
    
    config = WorkflowConfig(
        num_agents=NUM_AGENTS,
        temperature=TEMPERATURE
    )
    workflow = ProposedWorkflowV7SC(retriever=retriever, config=config)
    
    # データ読み込み
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r') as f:
        data = json.load(f)
    
    samples = data['samples'][:140]
    print(f"Questions: {len(samples)}\n")
    
    # 結果ファイルパス
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "proposed_v7sc_results.json"
    log_file = results_dir / "proposed_v7sc_run.log"
    
    # 既存結果を読み込み（再開用）
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        start_idx = results.get('completed', 0)
        if start_idx > 0:
            print(f"Resuming from {start_idx}/{len(samples)}")
    else:
        results = {
            'experiment': 'proposed_v7sc',
            'model': 'qwen3:8b',
            'num_agents': NUM_AGENTS,
            'temperature': TEMPERATURE,
            'index': 'faiss_index_xml_v2',
            'search_mode': 'hybrid',
            'method': 'Self-Consistency (majority vote)',
            'completed': 0,
            'correct': 0,
            'accuracy': 0.0,
            'difficulty_stats': {'simple': 0, 'moderate': 0, 'complex': 0},
            'difficulty_correct': {'simple': 0, 'moderate': 0, 'complex': 0},
            'confidence_stats': {'unanimous': 0, 'majority': 0, 'tie': 0},
            'details': []
        }
        start_idx = 0
    
    # ログファイル
    with open(log_file, 'a') as lf:
        lf.write(f"\n=== Run started at {datetime.now()} ===\n")
        lf.write(f"Config: agents={NUM_AGENTS}, temperature={TEMPERATURE}\n")
    
    # 実行
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        
        # 選択肢をパース
        choices = []
        for line in sample['選択肢'].split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())
        
        if len(choices) != 4:
            continue
        
        print(f"[{i+1}/{len(samples)}] Q{i}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = workflow.query(sample['問題文'], choices)
            predicted = result['answer']
            difficulty = result['difficulty']
            confidence = result['confidence']
            vote_details = result['vote_details']
            agent_answers = result['agent_answers']
        except Exception as e:
            print(f"Error: {e}")
            predicted = 'a'
            difficulty = 'simple'
            confidence = 0.0
            vote_details = {}
            agent_answers = []
        elapsed = time.time() - start_time
        
        correct_answer = sample['output'].strip().lower()
        is_correct = predicted == correct_answer
        
        # 統計更新
        results['completed'] = i + 1
        if is_correct:
            results['correct'] += 1
            results['difficulty_correct'][difficulty] = results['difficulty_correct'].get(difficulty, 0) + 1
        results['difficulty_stats'][difficulty] = results['difficulty_stats'].get(difficulty, 0) + 1
        results['accuracy'] = results['correct'] / results['completed']
        
        # 信頼度統計
        if confidence == 1.0:
            results['confidence_stats']['unanimous'] += 1
        elif confidence >= 0.5:
            results['confidence_stats']['majority'] += 1
        else:
            results['confidence_stats']['tie'] += 1
        
        # 詳細追加
        results['details'].append({
            'question_id': i,
            'predicted': predicted,
            'correct_answer': correct_answer,
            'correct': is_correct,
            'difficulty': difficulty,
            'confidence': confidence,
            'vote_details': vote_details,
            'agent_answers': [a['answer'] for a in agent_answers],
            'elapsed': elapsed
        })
        
        status = "OK" if is_correct else "NG"
        vote_str = str(vote_details) if vote_details else "{}"
        print(f"{status} ({predicted}/{correct_answer}) conf={confidence:.0%} votes={vote_str} {elapsed:.1f}s")
        
        # ログ記録
        with open(log_file, 'a') as lf:
            lf.write(f"Q{i}: {status} ({predicted}/{correct_answer}) conf={confidence:.0%} {difficulty} {elapsed:.1f}s\n")
        
        # 5問ごとに中間レポートと保存
        if (i + 1) % 5 == 0:
            acc = results['correct'] / results['completed'] * 100
            print(f"  進捗: {results['completed']}/{len(samples)}, "
                  f"正解: {results['correct']}, 精度: {acc:.1f}%")
            
            # 難易度別
            for diff in ['simple', 'moderate', 'complex']:
                total = results['difficulty_stats'].get(diff, 0)
                correct_count = results['difficulty_correct'].get(diff, 0)
                if total > 0:
                    print(f"    {diff}: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
            
            # 信頼度統計
            unanimous = results['confidence_stats']['unanimous']
            majority = results['confidence_stats']['majority']
            tie = results['confidence_stats']['tie']
            print(f"    投票: 全会一致={unanimous}, 多数決={majority}, 同数={tie}")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 最終保存
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 最終レポート
    print("\n" + "=" * 60)
    print(f"完了: {results['correct']}/{results['completed']} ({results['accuracy']*100:.1f}%)")
    target_met = "✓" if results['accuracy'] >= 0.8 else "✗"
    print(f"目標達成: {target_met} (目標: 80%)")
    
    print("\n難易度別精度:")
    for diff in ['simple', 'moderate', 'complex']:
        total = results['difficulty_stats'].get(diff, 0)
        correct_count = results['difficulty_correct'].get(diff, 0)
        if total > 0:
            print(f"  {diff}: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    
    print("\n信頼度統計:")
    unanimous = results['confidence_stats']['unanimous']
    majority = results['confidence_stats']['majority']
    tie = results['confidence_stats']['tie']
    total_q = unanimous + majority + tie
    if total_q > 0:
        print(f"  全会一致: {unanimous} ({unanimous/total_q*100:.1f}%)")
        print(f"  多数決: {majority} ({majority/total_q*100:.1f}%)")
        print(f"  同数: {tie} ({tie/total_q*100:.1f}%)")
    
    print(f"\n結果: {results_file}")


if __name__ == "__main__":
    main()
