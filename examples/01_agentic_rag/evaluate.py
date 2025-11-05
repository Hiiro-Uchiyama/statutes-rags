"""
Agentic RAG評価スクリプト

デジタル庁4択法令データで評価を実行します。
"""
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline import AgenticRAGPipeline
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
    
    # データセットの構造に応じて処理
    if isinstance(data, dict):
        # {"samples": [...]} 形式
        if "samples" in data:
            data = data["samples"]
        # {"key1": {...}, "key2": {...}} 形式
        else:
            data = list(data.values())
    
    logger.info(f"Loaded {len(data)} questions from {dataset_path}")
    
    return data


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


def evaluate_single_question(
    pipeline: AgenticRAGPipeline,
    question_data: Dict[str, Any],
    question_index: int
) -> Dict[str, Any]:
    """
    1問を評価
    
    Args:
        pipeline: AgenticRAGPipeline
        question_data: 問題データ
        question_index: 問題番号
    
    Returns:
        評価結果
    """
    # データセットのフィールド名に対応
    question = question_data.get("問題文", question_data.get("question", ""))
    
    # 選択肢を処理（改行区切りの文字列またはリスト）
    choices_raw = question_data.get("選択肢", question_data.get("choices", []))
    if isinstance(choices_raw, str):
        # "a ...\nb ...\nc ...\nd ..." 形式から選択肢を抽出
        choices = [line.strip() for line in choices_raw.split('\n') if line.strip()]
    else:
        choices = choices_raw
    
    correct_answer = question_data.get("output", question_data.get("answer", "a"))
    
    # プロンプト作成
    prompt = create_multiple_choice_prompt(question, choices)
    
    logger.info(f"Question {question_index + 1}: {question[:50]}...")
    
    try:
        # RAG実行
        result = pipeline.query(prompt)
        
        # 回答抽出
        predicted_answer = extract_answer(result["answer"])
        
        # 正誤判定
        is_correct = predicted_answer == correct_answer.lower()
        
        logger.info(f"  Predicted: {predicted_answer}, Correct: {correct_answer}, Match: {is_correct}")
        logger.info(f"  Metadata: {result['metadata']}")
        
        return {
            "question_index": question_index,
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "raw_answer": result["answer"],
            "citations": result["citations"],
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error evaluating question {question_index + 1}: {e}", exc_info=True)
        return {
            "question_index": question_index,
            "question": question,
            "error": str(e),
            "is_correct": False
        }


def evaluate(
    pipeline: AgenticRAGPipeline,
    dataset: List[Dict[str, Any]],
    max_questions: int = None
) -> Dict[str, Any]:
    """
    データセット全体を評価
    
    Args:
        pipeline: AgenticRAGPipeline
        dataset: 問題のリスト
        max_questions: 評価する最大問題数（Noneの場合は全問題）
    
    Returns:
        評価結果
    """
    if max_questions:
        dataset = dataset[:max_questions]
    
    results = []
    correct_count = 0
    total_iterations = 0
    total_llm_calls = 0
    
    for i, question_data in enumerate(dataset):
        result = evaluate_single_question(pipeline, question_data, i)
        
        if result.get("is_correct", False):
            correct_count += 1
        
        # メタデータの集計
        metadata = result.get("metadata", {})
        total_iterations += metadata.get("iterations", 0)
        # LLM呼び出し回数は概算（エージェント数 × 反復回数）
        agents_count = len(metadata.get("agents_used", []))
        total_llm_calls += agents_count
        
        results.append(result)
    
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    avg_iterations = total_iterations / total if total > 0 else 0.0
    avg_llm_calls = total_llm_calls / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total": total,
        "avg_iterations": avg_iterations,
        "avg_llm_calls": avg_llm_calls,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def save_results(results: Dict[str, Any], output_path: Path):
    """
    評価結果を保存
    
    Args:
        results: 評価結果
        output_path: 出力ファイルパス
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def print_summary(results: Dict[str, Any]):
    """
    評価結果のサマリーを表示
    
    Args:
        results: 評価結果
    """
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Accuracy:        {results['accuracy'] * 100:.2f}%")
    print(f"Correct:         {results['correct_count']} / {results['total']}")
    print(f"Avg Iterations:  {results['avg_iterations']:.2f}")
    print(f"Avg LLM Calls:   {results['avg_llm_calls']:.2f}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG評価スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全問題を評価
  python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json

  # 最初の10問のみ評価
  python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10

  # 出力先を指定
  python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --output my_results.json
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="評価データセットのパス（例: ../../datasets/lawqa_jp/data/selection.json）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果の出力先（デフォルト: results/evaluation_{timestamp}.json）"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="評価する最大問題数（テスト用）"
    )
    
    args = parser.parse_args()
    
    # データセットのロード
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 1
    
    dataset = load_dataset(dataset_path)
    
    # 設定のロード（環境変数から）
    config = load_config()
    
    # パイプラインの初期化
    logger.info("Initializing Agentic RAG pipeline...")
    pipeline = AgenticRAGPipeline(config)
    
    # 評価実行
    logger.info(f"Starting evaluation on {len(dataset)} questions...")
    results = evaluate(pipeline, dataset, args.max_questions)
    
    # サマリー表示
    print_summary(results)
    
    # 結果保存
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"evaluation_{timestamp}.json"
    
    save_results(results, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

