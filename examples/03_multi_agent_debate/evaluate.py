"""
Multi-Agent Debate評価スクリプト

デジタル庁4択法令データで評価を実行します。
"""
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent
sys.path.insert(0, str(debate_dir))

from workflow import DebateWorkflow
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    評価データセットをロード
    
    Args:
        dataset_path: データセットファイルのパス
    
    Returns:
        問題のリスト
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データセットが辞書の場合、'samples'キーから取得
    if isinstance(data, dict):
        if 'samples' in data:
            questions = data['samples']
        else:
            raise ValueError(f"Expected 'samples' key in dataset, got keys: {list(data.keys())}")
    else:
        questions = data
    
    logger.info(f"Loaded {len(questions)} questions from {dataset_path}")
    
    return questions


def create_multiple_choice_prompt(question: str, choices: List[str]) -> str:
    """
    4択問題のプロンプトを作成
    
    Args:
        question: 質問文
        choices: 選択肢のリスト
    
    Returns:
        プロンプト文字列
    """
    prompt = f"{question}\n\n"
    for i, choice in enumerate(choices):
        label = chr(ord('a') + i)  # a, b, c, d
        prompt += f"{label}. {choice}\n"
    
    prompt += "\n上記の選択肢から最も適切なものを1つ選び、記号（a/b/c/d）のみを答えてください。"
    
    return prompt


def extract_answer(response: str) -> str:
    """
    LLMの応答から選択肢を抽出
    
    Args:
        response: LLMの応答
    
    Returns:
        選択肢（a/b/c/d）
    """
    response_lower = response.lower().strip()
    
    # 最初に出現する a, b, c, d を抽出
    for char in ['a', 'b', 'c', 'd']:
        if char in response_lower:
            return char
    
    # 見つからない場合は最初の文字
    if response_lower:
        first_char = response_lower[0]
        if first_char in ['a', 'b', 'c', 'd']:
            return first_char
    
    return "a"  # デフォルト


def parse_choices(choices_str: str) -> List[str]:
    """
    選択肢文字列をリストに変換
    
    Args:
        choices_str: 選択肢文字列（例: "a 選択肢1\nb 選択肢2\nc 選択肢3\nd 選択肢4"）
    
    Returns:
        選択肢のリスト
    """
    choices = []
    # 改行で分割
    lines = choices_str.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # "a ", "b ", "c ", "d " を削除
        for prefix in ['a ', 'b ', 'c ', 'd ']:
            if line.startswith(prefix):
                line = line[2:]
                break
        choices.append(line)
    return choices


def evaluate_single_question(
    workflow: DebateWorkflow,
    question_data: Dict[str, Any],
    question_index: int
) -> Dict[str, Any]:
    """
    1問を評価
    
    Args:
        workflow: DebateWorkflow インスタンス
        question_data: 問題データ
        question_index: 問題番号
    
    Returns:
        評価結果
    """
    # データセットの形式に対応（'問題文'/'question', '選択肢'/'choices', 'output'/'answer'）
    question = question_data.get("問題文") or question_data.get("question", "")
    choices_raw = question_data.get("選択肢") or question_data.get("choices")
    correct_answer = question_data.get("output") or question_data.get("answer", "a")
    
    # 選択肢を処理
    if isinstance(choices_raw, str):
        choices = parse_choices(choices_raw)
    elif isinstance(choices_raw, list):
        choices = choices_raw
    else:
        choices = []
    
    # プロンプトを作成
    prompt = create_multiple_choice_prompt(question, choices)
    
    logger.info(f"Question {question_index + 1}: {question[:50]}...")
    
    # 議論を実行
    start_time = time.time()
    
    try:
        result = workflow.query(prompt)
        elapsed_time = time.time() - start_time
        
        answer = result["answer"]
        metadata = result["metadata"]
        
        # 回答を抽出
        predicted_answer = extract_answer(answer)
        
        # 正解判定
        is_correct = predicted_answer == correct_answer
        
        logger.info(
            f"Question {question_index + 1} - "
            f"Correct: {correct_answer}, Predicted: {predicted_answer}, "
            f"Match: {is_correct}, "
            f"Rounds: {metadata.get('rounds', 0)}, "
            f"Agreement: {metadata.get('agreement_score', 0.0):.2f}, "
            f"Time: {elapsed_time:.1f}s"
        )
        
        return {
            "question_index": question_index,
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_response": answer,
            "rounds": metadata.get("rounds", 0),
            "agreement_score": metadata.get("agreement_score", 0.0),
            "debate_history": metadata.get("debate_history", []),
            "elapsed_time": elapsed_time,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Question {question_index + 1} failed: {e}", exc_info=True)
        
        return {
            "question_index": question_index,
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": "a",
            "is_correct": False,
            "full_response": "",
            "rounds": 0,
            "agreement_score": 0.0,
            "debate_history": [],
            "elapsed_time": 0.0,
            "error": str(e)
        }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    評価メトリクスを計算
    
    Args:
        results: 評価結果のリスト
    
    Returns:
        メトリクス
    """
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    
    accuracy = correct / total if total > 0 else 0.0
    
    # ラウンド数の統計
    rounds = [r["rounds"] for r in results if r["rounds"] > 0]
    avg_rounds = sum(rounds) / len(rounds) if rounds else 0.0
    
    # 合意スコアの統計
    agreements = [r["agreement_score"] for r in results if r["agreement_score"] > 0]
    avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0
    
    # 処理時間の統計
    times = [r["elapsed_time"] for r in results if r["elapsed_time"] > 0]
    avg_time = sum(times) / len(times) if times else 0.0
    
    # エラー数
    errors = sum(1 for r in results if r["error"] is not None)
    
    # 合意形成率（最終的に高い合意スコアに達した割合）
    high_agreement_threshold = 0.7
    high_agreement_count = sum(
        1 for r in results if r["agreement_score"] >= high_agreement_threshold
    )
    agreement_formation_rate = high_agreement_count / total if total > 0 else 0.0
    
    return {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "avg_rounds": avg_rounds,
        "avg_agreement_score": avg_agreement,
        "avg_time_per_question": avg_time,
        "agreement_formation_rate": agreement_formation_rate,
        "errors": errors
    }


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path
):
    """
    結果を保存
    
    Args:
        results: 評価結果
        metrics: メトリクス
        output_path: 出力ファイルパス
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Multi-Agent Debate評価")
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
        help="結果の出力先（デフォルト: examples/03_multi_agent_debate/evaluation_results_YYYYMMDD_HHMMSS.json）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="評価する問題数の上限"
    )
    
    args = parser.parse_args()
    
    # データセットのロード
    dataset_path = Path(args.dataset)
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
    logger.info("Initializing DebateWorkflow...")
    workflow = DebateWorkflow(config)
    
    # 評価実行
    logger.info(f"Starting evaluation on {len(questions)} questions...")
    results = []
    
    for i, question_data in enumerate(questions):
        result = evaluate_single_question(workflow, question_data, i)
        results.append(result)
    
    # メトリクス計算
    metrics = calculate_metrics(results)
    
    # 結果表示
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Questions: {metrics['total_questions']}")
    logger.info(f"Correct Answers: {metrics['correct_answers']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Average Rounds: {metrics['avg_rounds']:.2f}")
    logger.info(f"Average Agreement Score: {metrics['avg_agreement_score']:.2f}")
    logger.info(f"Agreement Formation Rate: {metrics['agreement_formation_rate']:.2%}")
    logger.info(f"Average Time per Question: {metrics['avg_time_per_question']:.1f}s")
    logger.info(f"Errors: {metrics['errors']}")
    logger.info("=" * 60)
    
    # 結果保存
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"evaluation_results_{timestamp}.json"
    
    save_results(results, metrics, output_path)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

