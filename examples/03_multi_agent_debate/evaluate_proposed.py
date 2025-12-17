"""
Proposed Method Evaluation Script

提案手法（4役割エージェント + グループシンキング防止 + 法的根拠共有）の評価を実行。
ベースラインとの比較実験も可能。
"""
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent
sys.path.insert(0, str(debate_dir))

from proposed_workflow import ProposedWorkflow, create_proposed_workflow
from proposed_metrics import (
    EvaluationResult,
    calculate_proposed_metrics,
    format_metrics_report
)
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """評価データセットをロード"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データセット構造の判定
    if isinstance(data, dict):
        if 'samples' in data:
            questions = data['samples']
        elif 'results' in data:
            # MCQ生成結果の形式
            questions = []
            for item in data['results']:
                choices_raw = item.get("choices", {})
                if isinstance(choices_raw, dict):
                    choices = [choices_raw.get(label, "") for label in ['a', 'b', 'c', 'd']]
                elif isinstance(choices_raw, list):
                    choices = choices_raw
                else:
                    choices = []
                
                normalized = {
                    "question": item.get("question") or "",
                    "context": item.get("context") or "",
                    "choices": choices,
                    "answer": item.get("correct_choice") or item.get("correct_choice_label") or "a",
                    "question_id": item.get("question_id"),
                }
                questions.append(normalized)
        else:
            raise ValueError(f"Unsupported dataset format: {list(data.keys())}")
    else:
        questions = data
    
    logger.info(f"Loaded {len(questions)} questions from {dataset_path}")
    return questions


def parse_choices(choices_raw) -> List[str]:
    """選択肢をリストに変換"""
    if isinstance(choices_raw, list):
        return choices_raw[:4]
    elif isinstance(choices_raw, dict):
        return [choices_raw.get(label, "") for label in ['a', 'b', 'c', 'd']]
    elif isinstance(choices_raw, str):
        # 改行区切り
        lines = choices_raw.strip().split('\n')
        choices = []
        for line in lines:
            line = line.strip()
            for prefix in ['a ', 'b ', 'c ', 'd ', 'a. ', 'b. ', 'c. ', 'd. ']:
                if line.lower().startswith(prefix):
                    line = line[len(prefix):]
                    break
            if line:
                choices.append(line)
        return choices[:4]
    return ["", "", "", ""]


def create_prompt(question: str, choices: List[str], context: str = "") -> str:
    """プロンプトを作成"""
    prompt_parts = []
    if context:
        prompt_parts.append(str(context).strip())
    
    prompt_parts.append(str(question).strip())
    
    prompt = "\n\n".join(part for part in prompt_parts if part) + "\n\n"
    for i, choice in enumerate(choices):
        label = chr(ord('a') + i)
        prompt += f"{label}. {choice}\n"
    
    prompt += "\n上記の選択肢から最も適切なものを1つ選び、記号（a/b/c/d）で答えてください。"
    
    return prompt


def extract_answer(response: str) -> str:
    """応答から回答を抽出"""
    if not response:
        return "a"
    
    response_lower = response.lower()
    
    # パターンマッチ
    patterns = [
        r"最終回答[:：]?\s*([abcd])",
        r"final[_\s-]*answer[:：]?\s*([abcd])",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            return match.group(1)
    
    # 単独の選択肢ラベル
    for line in response_lower.splitlines():
        stripped = line.strip()
        if stripped in {"a", "b", "c", "d"}:
            return stripped
    
    # フォールバック
    for char in ['a', 'b', 'c', 'd']:
        if char in response_lower:
            return char
    
    return "a"


def evaluate_single_question(
    workflow: ProposedWorkflow,
    question_data: Dict[str, Any],
    question_index: int
) -> EvaluationResult:
    """1問を評価"""
    
    # データ抽出
    question = question_data.get("question") or question_data.get("問題文", "")
    context = question_data.get("context") or question_data.get("文脈", "")
    choices_raw = question_data.get("choices") or question_data.get("選択肢", [])
    correct_answer = (
        question_data.get("answer") or 
        question_data.get("output") or 
        question_data.get("correct_choice") or 
        "a"
    ).lower()
    question_id = question_data.get("question_id", f"Q{question_index + 1}")
    
    # 選択肢を処理
    choices = parse_choices(choices_raw)
    
    # プロンプト作成
    prompt = create_prompt(question, choices, context)
    
    logger.info(f"Question {question_index + 1} [{question_id}]: {question[:50]}...")
    
    # 評価実行
    start_time = time.time()
    
    try:
        result = workflow.query(prompt, choices)
        elapsed_time = time.time() - start_time
        
        # 回答抽出
        answer = result.get("answer", "a")
        if len(answer) > 1:
            answer = extract_answer(answer)
        
        predicted_answer = answer.lower()
        is_correct = predicted_answer == correct_answer
        
        # メタデータ抽出
        metadata = result.get("metadata", {})
        
        eval_result = EvaluationResult(
            question_id=question_id,
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            confidence=result.get("confidence", 0.0),
            is_correct=is_correct,
            interpreter_answer=metadata.get("interpreter_answer", ""),
            critic_alternative=metadata.get("critic_alternative", ""),
            answer_changed=metadata.get("answer_changed", False),
            criticism_severity=metadata.get("criticism_severity", ""),
            citation_count=metadata.get("citation_count", 0),
            citation_rate=metadata.get("citation_rate", 0.0),
            elapsed_time=elapsed_time,
            audit_trail=metadata.get("audit_trail", {})
        )
        
        logger.info(
            f"  -> Correct: {correct_answer}, Predicted: {predicted_answer}, "
            f"Match: {is_correct}, Confidence: {result.get('confidence', 0):.2f}, "
            f"Time: {elapsed_time:.1f}s"
        )
        
        return eval_result
        
    except Exception as e:
        logger.error(f"Question {question_index + 1} failed: {e}", exc_info=True)
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            correct_answer=correct_answer,
            predicted_answer="a",
            confidence=0.0,
            is_correct=False,
            elapsed_time=time.time() - start_time,
            error=str(e)
        )


def save_results(
    results: List[EvaluationResult],
    metrics: Dict[str, Any],
    output_path: Path,
    config_info: Dict[str, Any] = None
):
    """結果を保存"""
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config_info or {},
        "metrics": metrics,
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question[:200],
                "correct_answer": r.correct_answer,
                "predicted_answer": r.predicted_answer,
                "is_correct": r.is_correct,
                "confidence": r.confidence,
                "interpreter_answer": r.interpreter_answer,
                "critic_alternative": r.critic_alternative,
                "answer_changed": r.answer_changed,
                "criticism_severity": r.criticism_severity,
                "citation_count": r.citation_count,
                "citation_rate": r.citation_rate,
                "elapsed_time": r.elapsed_time,
                "error": r.error
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Proposed Method Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/lawqa_jp/data/selection.json",
        help="評価データセットのパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果の出力先"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="評価する問題数の上限"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="FAISSインデックスのパス"
    )
    
    args = parser.parse_args()
    
    # データセットのロード
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / args.dataset
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    questions = load_dataset(dataset_path)
    
    # 問題数の制限
    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"Limited to {args.limit} questions")
    
    # 設定のロード
    config = load_config()
    
    # ワークフローの初期化
    logger.info("Initializing ProposedWorkflow...")
    workflow = create_proposed_workflow(config, args.index_path)
    
    # 評価実行
    logger.info(f"Starting evaluation on {len(questions)} questions...")
    results: List[EvaluationResult] = []
    
    for i, question_data in enumerate(questions):
        result = evaluate_single_question(workflow, question_data, i)
        results.append(result)
    
    # メトリクス計算
    metrics = calculate_proposed_metrics(results)
    
    # レポート表示
    report = format_metrics_report(metrics)
    logger.info("\n" + report)
    
    # 結果保存
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = debate_dir / "results" / f"proposed_evaluation_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_info = {
        "method": "proposed_4role",
        "llm_model": config.llm_model,
        "max_debate_rounds": config.max_debate_rounds,
        "retrieval_top_k": config.retrieval_top_k
    }
    
    save_results(results, metrics, output_path, config_info)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

