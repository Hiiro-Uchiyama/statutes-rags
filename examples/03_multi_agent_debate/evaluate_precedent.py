"""
判例評価スクリプト

判例データを正解として、マルチエージェント議論システムが同じ結論を出せるかを評価する。
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
from precedent_loader import PrecedentLoader
from binary_classification_metrics import (
    calculate_binary_classification_metrics,
    analyze_threshold_optimization
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_precedent_question_prompt(question: str, case_name: str = "") -> str:
    """
    判例評価用のプロンプトを作成
    
    Args:
        question: 事件の要旨（質問）
        case_name: 事件名（オプション）
    
    Returns:
        プロンプト文字列
    """
    prompt = f"""以下の法律事件について、法的解釈と結論を示してください。

【事件名】
{case_name}

【事件の要旨】
{question}

上記の事件について、関連する法令を参照しながら、以下の点について回答してください：

1. 法的問題点の整理
2. 適用される法令の特定
3. 法的解釈と根拠
4. 結論

判例の要旨に相当する形で、簡潔かつ明確に回答してください。
"""
    return prompt


def calculate_answer_similarity(
    predicted_answer: str,
    correct_answer: str
) -> Dict[str, float]:
    """
    予測回答と正解（判例要旨）の類似度を計算
    
    Args:
        predicted_answer: 予測された回答
        correct_answer: 正解（判例要旨）
    
    Returns:
        類似度メトリクスの辞書
    """
    import os
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import numpy as np
    
    metrics = {}
    
    # 埋め込みモデルを使用した類似度計算
    try:
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        embed_pred = embeddings.embed_query(predicted_answer)
        embed_correct = embeddings.embed_query(correct_answer)
        
        # コサイン類似度
        similarity = np.dot(embed_pred, embed_correct)
        metrics["cosine_similarity"] = float((similarity + 1.0) / 2.0)
        
    except Exception as e:
        logger.warning(f"Failed to calculate embedding similarity: {e}")
        metrics["cosine_similarity"] = 0.0
    
    # キーワード一致率
    words_pred = set(predicted_answer.split())
    words_correct = set(correct_answer.split())
    
    if words_pred and words_correct:
        intersection = words_pred & words_correct
        union = words_pred | words_correct
        metrics["keyword_overlap"] = len(intersection) / len(union) if union else 0.0
    else:
        metrics["keyword_overlap"] = 0.0
    
    # 文字列の長さの比率
    len_pred = len(predicted_answer)
    len_correct = len(correct_answer)
    if len_correct > 0:
        metrics["length_ratio"] = min(len_pred / len_correct, len_correct / len_pred)
    else:
        metrics["length_ratio"] = 0.0
    
    return metrics


def evaluate_single_precedent(
    workflow: DebateWorkflow,
    precedent_data: Dict[str, Any],
    precedent_index: int
) -> Dict[str, Any]:
    """
    1つの判例を評価
    
    Args:
        workflow: DebateWorkflow インスタンス
        precedent_data: 判例データ
        precedent_index: 判例のインデックス
    
    Returns:
        評価結果
    """
    question = precedent_data.get("question", "")
    correct_answer = precedent_data.get("correct_answer", "")
    case_name = precedent_data.get("case_name", "")
    
    # プロンプトを作成
    prompt = create_precedent_question_prompt(question, case_name)
    
    logger.info(f"Precedent {precedent_index + 1}: {case_name}")
    
    # 議論を実行
    start_time = time.time()
    
    try:
        result = workflow.query(prompt)
        elapsed_time = time.time() - start_time
        
        predicted_answer = result["answer"]
        metadata = result["metadata"]
        
        # 類似度を計算
        similarity_metrics = calculate_answer_similarity(
            predicted_answer,
            correct_answer
        )
        
        # 類似度スコア（コサイン類似度を主要指標として使用）
        similarity_score = similarity_metrics.get("cosine_similarity", 0.0)
        
        # 閾値による一致判定（0.7以上で一致とみなす）
        is_similar = similarity_score >= 0.7
        
        logger.info(
            f"Precedent {precedent_index + 1} - "
            f"Similarity: {similarity_score:.3f}, "
            f"Match: {is_similar}, "
            f"Rounds: {metadata.get('rounds', 0)}, "
            f"Agreement: {metadata.get('agreement_score', 0.0):.2f}, "
            f"Time: {elapsed_time:.1f}s"
        )
        
        return {
            "precedent_index": precedent_index,
            "precedent_id": precedent_data.get("precedent_id", ""),
            "case_name": case_name,
            "case_number": precedent_data.get("case_number", ""),
            "court_name": precedent_data.get("court_name", ""),
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "similarity_score": similarity_score,
            "similarity_metrics": similarity_metrics,
            "is_similar": is_similar,
            "rounds": metadata.get("rounds", 0),
            "agreement_score": metadata.get("agreement_score", 0.0),
            "debate_history": metadata.get("debate_history", []),
            "elapsed_time": elapsed_time,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Precedent {precedent_index + 1} failed: {e}", exc_info=True)
        
        return {
            "precedent_index": precedent_index,
            "precedent_id": precedent_data.get("precedent_id", ""),
            "case_name": case_name,
            "case_number": precedent_data.get("case_number", ""),
            "court_name": precedent_data.get("court_name", ""),
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": "",
            "similarity_score": 0.0,
            "similarity_metrics": {},
            "is_similar": False,
            "rounds": 0,
            "agreement_score": 0.0,
            "debate_history": [],
            "elapsed_time": 0.0,
            "error": str(e)
        }


def calculate_precedent_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    判例評価メトリクスを計算
    
    Args:
        results: 評価結果のリスト
    
    Returns:
        メトリクス
    """
    total = len(results)
    similar = sum(1 for r in results if r["is_similar"])
    
    similarity_rate = similar / total if total > 0 else 0.0
    
    # 類似度スコアの統計
    similarity_scores = [r["similarity_score"] for r in results if r["similarity_score"] > 0]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
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
    
    return {
        "total_precedents": total,
        "similar_precedents": similar,
        "similarity_rate": similarity_rate,
        "avg_similarity_score": avg_similarity,
        "avg_rounds": avg_rounds,
        "avg_agreement_score": avg_agreement,
        "avg_time_per_precedent": avg_time,
        "errors": errors
    }


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
    config: Dict[str, Any]
):
    """
    結果を保存
    
    Args:
        results: 評価結果
        metrics: メトリクス
        output_path: 出力ファイルパス
        config: 設定情報
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "results": results
    }
    
    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """メイン関数"""
    import os
    
    parser = argparse.ArgumentParser(description="判例評価スクリプト")
    parser.add_argument(
        "--precedent-dir",
        type=str,
        default="data_set/precedent",
        help="判例データディレクトリのパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果の出力先（デフォルト: results/precedent_evaluation_YYYYMMDD_HHMMSS.json）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="評価する判例数の上限"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="ランダムシード（指定するとランダムサンプリング）"
    )
    
    args = parser.parse_args()
    
    # 判例データのロード
    precedent_dir = Path(args.precedent_dir)
    if not precedent_dir.is_absolute():
        precedent_dir = Path(__file__).parent / precedent_dir
    
    if not precedent_dir.exists():
        logger.error(f"Precedent directory not found: {precedent_dir}")
        return
    
    logger.info(f"Loading precedents from {precedent_dir}")
    loader = PrecedentLoader(precedent_dir)
    evaluation_data = loader.load_evaluation_dataset(
        limit=args.limit,
        random_seed=args.random_seed
    )
    
    if not evaluation_data:
        logger.error("No evaluation data loaded")
        return
    
    # 設定のロード
    config = load_config()
    
    # ワークフローの初期化
    logger.info("Initializing DebateWorkflow...")
    workflow = DebateWorkflow(config)
    
    # 評価実行
    logger.info(f"Starting evaluation on {len(evaluation_data)} precedents...")
    results = []
    
    for i, precedent_data in enumerate(evaluation_data):
        result = evaluate_single_precedent(workflow, precedent_data, i)
        results.append(result)
    
    # メトリクス計算
    metrics = calculate_precedent_metrics(results)
    
    # 2値分類メトリクスの計算
    binary_metrics = calculate_binary_classification_metrics(results, threshold=0.7)
    metrics.update(binary_metrics)
    
    # 閾値最適化の分析（複数サンプルがある場合のみ）
    if len(results) > 1:
        threshold_optimization = analyze_threshold_optimization(results)
        metrics["threshold_optimization"] = threshold_optimization
    
    # 結果表示
    logger.info("\n" + "=" * 60)
    logger.info("PRECEDENT EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Precedents: {metrics['total_precedents']}")
    logger.info(f"Similar Precedents: {metrics['similar_precedents']}")
    logger.info(f"Similarity Rate: {metrics['similarity_rate']:.2%}")
    logger.info(f"Average Similarity Score: {metrics['avg_similarity_score']:.3f}")
    logger.info(f"Average Rounds: {metrics['avg_rounds']:.2f}")
    logger.info(f"Average Agreement Score: {metrics['avg_agreement_score']:.2f}")
    logger.info(f"Average Time per Precedent: {metrics['avg_time_per_precedent']:.1f}s")
    logger.info(f"Errors: {metrics['errors']}")
    
    # 2値分類メトリクスの表示
    if "binary_classification" in metrics:
        bc = metrics["binary_classification"]
        logger.info("\n" + "-" * 60)
        logger.info("BINARY CLASSIFICATION METRICS")
        logger.info("-" * 60)
        logger.info(f"Threshold: {bc['threshold']}")
        logger.info(f"Accuracy: {bc['accuracy']:.3f}")
        logger.info(f"Precision: {bc['precision']:.3f}")
        logger.info(f"Recall: {bc['recall']:.3f}")
        logger.info(f"F1 Score: {bc['f1_score']:.3f}")
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TP: {bc['confusion_matrix']['true_positive']}, "
                   f"TN: {bc['confusion_matrix']['true_negative']}")
        logger.info(f"  FP: {bc['confusion_matrix']['false_positive']}, "
                   f"FN: {bc['confusion_matrix']['false_negative']}")
    
    logger.info("=" * 60)
    
    # 結果保存
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"precedent_evaluation_{timestamp}.json"
    
    config_dict = {
        "max_debate_rounds": config.max_debate_rounds,
        "agreement_threshold": config.agreement_threshold,
        "llm_model": config.llm_model,
        "retrieval_top_k": config.retrieval_top_k
    }
    
    save_results(results, metrics, output_path, config_dict)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

