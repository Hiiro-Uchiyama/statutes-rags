#!/usr/bin/env python3
"""
提案手法 v2 - 改善版テスト
- CoT簡素化（自然な思考共有）
- JudgeAgent妥当性判断特化
- 議論ラウンド追加
"""
import sys
import json
import warnings
import os
import time
import signal
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from proposed_workflow import ProposedWorkflow

NUM_CTX = 30000
TOP_K = 30
TIMEOUT = 120
QUERY_TIMEOUT = 120
MAX_RETRIES = 2


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Query timed out")


def query_with_timeout(workflow, question, timeout_seconds=QUERY_TIMEOUT):
    for attempt in range(MAX_RETRIES):
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            result = workflow.query(question)
            signal.alarm(0)
            return result
        except TimeoutException:
            signal.alarm(0)
            if attempt < MAX_RETRIES - 1:
                print(f"  Timeout, retry {attempt + 1}...")
                time.sleep(5)
                continue
            raise
        except Exception as e:
            signal.alarm(0)
            if attempt < MAX_RETRIES - 1 and ('timeout' in str(e).lower()):
                time.sleep(5)
                continue
            raise


def save_results(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    results_file = RESULTS_DIR / "proposed_v2_140q.json"
    
    print("=" * 60)
    print("Proposed Method v2 - Improved Version")
    print("=" * 60)
    print("Changes:")
    print("  - Simplified CoT (natural thinking)")
    print("  - JudgeAgent: validity check only")
    print("  - Discussion round when needed")
    print()
    
    os.environ['LLM_NUM_CTX'] = str(NUM_CTX)
    os.environ['DEBATE_RETRIEVAL_TOP_K'] = str(TOP_K)
    os.environ['LLM_TIMEOUT'] = str(TIMEOUT)
    config = load_config()
    
    print("Loading workflow...")
    workflow = ProposedWorkflow(config)
    
    if hasattr(workflow.retriever, 'vector_retriever'):
        doc_count = workflow.retriever.vector_retriever.vector_store.index.ntotal
        print(f"Hybrid: Vector ({doc_count} docs) + BM25")
    else:
        doc_count = workflow.retriever.vector_store.index.ntotal
        print(f"Vector only: {doc_count} docs")
    print()
    
    dataset_path = PROJECT_ROOT / 'datasets/lawqa_jp/data/selection.json'
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples'][:140]
    
    results = []
    completed = set()
    correct_count = 0
    discussion_count = 0
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            existing = json.load(f)
            results = existing.get('results', [])
            completed = {r['q_num'] for r in results}
            correct_count = sum(1 for r in results if r.get('is_correct', False))
            discussion_count = sum(1 for r in results if r.get('metadata', {}).get('discussion_rounds', 0) > 0)
            print(f"Resuming from {len(completed)}/140")
    
    print(f"Questions: {len(samples)}")
    print()
    
    for i, q in enumerate(samples):
        q_num = i + 1
        
        if q_num in completed:
            continue
        
        start = time.time()
        
        try:
            question = q['問題文'] + '\n' + q['指示'] + '\n' + q['選択肢']
            correct = q['output']
            
            result = query_with_timeout(workflow, question)
            
            elapsed = time.time() - start
            predicted = result['answer']
            is_correct = predicted == correct
            
            if is_correct:
                correct_count += 1
            
            disc = result.get('metadata', {}).get('discussion_rounds', 0)
            if disc > 0:
                discussion_count += 1
            
            status = 'OK' if is_correct else 'NG'
            disc_mark = ' [D]' if disc > 0 else ''
            acc = correct_count / (len(completed) + 1) * 100
            print(f"[{q_num}/140] {status} ({predicted}/{correct}) {elapsed:.1f}s{disc_mark} - Acc: {acc:.1f}%")
            
            results.append({
                'q_num': q_num,
                'correct': correct,
                'predicted': predicted,
                'is_correct': is_correct,
                'confidence': result.get('confidence', 0.0),
                'elapsed_time': elapsed,
                'metadata': result.get('metadata', {})
            })
            completed.add(q_num)
            
        except TimeoutException:
            elapsed = time.time() - start
            print(f"[{q_num}/140] TIMEOUT ({elapsed:.1f}s)")
            results.append({
                'q_num': q_num,
                'correct': q['output'],
                'predicted': 'TIMEOUT',
                'is_correct': False,
                'elapsed_time': elapsed
            })
            completed.add(q_num)
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"[{q_num}/140] ERROR: {str(e)[:50]}")
            results.append({
                'q_num': q_num,
                'correct': q['output'],
                'predicted': 'ERROR',
                'is_correct': False,
                'error': str(e),
                'elapsed_time': elapsed
            })
            completed.add(q_num)
        
        save_results(results_file, {
            'method': 'Proposed v2 (Simplified CoT + Discussion)',
            'version': 'v2',
            'config': {
                'search': 'Hybrid (Vector + BM25)',
                'top_k': TOP_K,
                'model': config.llm_model,
                'num_ctx': NUM_CTX,
                'changes': [
                    'Simplified CoT prompt',
                    'JudgeAgent validity check only',
                    'Discussion round when needed'
                ]
            },
            'completed': len(completed),
            'correct': correct_count,
            'accuracy': correct_count / len(completed) if completed else 0,
            'discussion_triggered': discussion_count,
            'results': sorted(results, key=lambda x: x['q_num'])
        })
    
    print()
    print("=" * 60)
    print(f"Proposed v2 Results")
    print("=" * 60)
    print(f"Completed: {len(completed)}/140")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count * 100 / len(completed):.1f}%")
    print(f"Discussion rounds: {discussion_count}")


if __name__ == '__main__':
    main()
