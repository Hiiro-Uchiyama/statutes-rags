"""
2値分類問題としての判例評価メトリクス計算

判例評価を2値分類問題（「同じ結論を出せた（1）」vs「出せなかった（0）」）として扱う。
"""
from typing import Dict, Any, List
import numpy as np


def calculate_binary_classification_metrics(
    results: List[Dict[str, Any]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    2値分類問題としての評価メトリクスを計算
    
    Args:
        results: 評価結果のリスト
        threshold: 類似度スコアの閾値（デフォルト: 0.7）
    
    Returns:
        2値分類メトリクス
    """
    # 予測ラベル（システムの判定）
    y_pred = [1 if r["similarity_score"] >= threshold else 0 for r in results]
    
    # 実際のラベル（現在は類似度スコアに基づくが、将来的には人間評価に置き換え可能）
    # 注意: 現在は類似度スコアを「正解」として扱っているが、
    # より厳密には人間による評価が必要
    y_true = [1 if r["similarity_score"] >= threshold else 0 for r in results]
    
    # より厳密な評価のため、複数の閾値で評価
    # 高閾値（0.8以上）: 明確に一致
    # 中閾値（0.7以上）: 概ね一致
    # 低閾値（0.6以上）: 部分的に一致
    
    # 混同行列の計算
    tp = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 0)
    
    # 基本メトリクス
    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 複数閾値での評価
    thresholds = [0.6, 0.7, 0.8, 0.9]
    threshold_metrics = {}
    
    for thresh in thresholds:
        y_pred_thresh = [1 if r["similarity_score"] >= thresh else 0 for r in results]
        tp_thresh = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred_thresh[i] == 1)
        tn_thresh = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred_thresh[i] == 0)
        fp_thresh = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred_thresh[i] == 1)
        fn_thresh = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred_thresh[i] == 0)
        
        total_thresh = len(results)
        acc_thresh = (tp_thresh + tn_thresh) / total_thresh if total_thresh > 0 else 0.0
        prec_thresh = tp_thresh / (tp_thresh + fp_thresh) if (tp_thresh + fp_thresh) > 0 else 0.0
        rec_thresh = tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0.0
        f1_thresh = 2 * (prec_thresh * rec_thresh) / (prec_thresh + rec_thresh) if (prec_thresh + rec_thresh) > 0 else 0.0
        
        threshold_metrics[f"threshold_{thresh}"] = {
            "accuracy": acc_thresh,
            "precision": prec_thresh,
            "recall": rec_thresh,
            "f1_score": f1_thresh,
            "true_positive": tp_thresh,
            "true_negative": tn_thresh,
            "false_positive": fp_thresh,
            "false_negative": fn_thresh
        }
    
    return {
        "binary_classification": {
            "threshold": threshold,
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "threshold_analysis": threshold_metrics
        }
    }


def analyze_threshold_optimization(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    最適な閾値を探索
    
    Args:
        results: 評価結果のリスト
    
    Returns:
        閾値最適化の結果
    """
    thresholds = np.arange(0.5, 1.0, 0.05)
    best_threshold = 0.7
    best_f1 = 0.0
    
    threshold_results = []
    
    for thresh in thresholds:
        y_pred = [1 if r["similarity_score"] >= thresh else 0 for r in results]
        y_true = [1 if r["similarity_score"] >= 0.7 else 0 for r in results]  # 基準として0.7を使用
        
        tp = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        threshold_results.append({
            "threshold": float(thresh),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return {
        "best_threshold": float(best_threshold),
        "best_f1_score": float(best_f1),
        "threshold_analysis": threshold_results
    }

2値分類問題としての判例評価メトリクス計算

判例評価を2値分類問題（「同じ結論を出せた（1）」vs「出せなかった（0）」）として扱う。
"""
from typing import Dict, Any, List
import numpy as np


def calculate_binary_classification_metrics(
    results: List[Dict[str, Any]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    2値分類問題としての評価メトリクスを計算
    
    Args:
        results: 評価結果のリスト
        threshold: 類似度スコアの閾値（デフォルト: 0.7）
    
    Returns:
        2値分類メトリクス
    """
    # 予測ラベル（システムの判定）
    y_pred = [1 if r["similarity_score"] >= threshold else 0 for r in results]
    
    # 実際のラベル（現在は類似度スコアに基づくが、将来的には人間評価に置き換え可能）
    # 注意: 現在は類似度スコアを「正解」として扱っているが、
    # より厳密には人間による評価が必要
    y_true = [1 if r["similarity_score"] >= threshold else 0 for r in results]
    
    # より厳密な評価のため、複数の閾値で評価
    # 高閾値（0.8以上）: 明確に一致
    # 中閾値（0.7以上）: 概ね一致
    # 低閾値（0.6以上）: 部分的に一致
    
    # 混同行列の計算
    tp = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 0)
    
    # 基本メトリクス
    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 複数閾値での評価
    thresholds = [0.6, 0.7, 0.8, 0.9]
    threshold_metrics = {}
    
    for thresh in thresholds:
        y_pred_thresh = [1 if r["similarity_score"] >= thresh else 0 for r in results]
        tp_thresh = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred_thresh[i] == 1)
        tn_thresh = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred_thresh[i] == 0)
        fp_thresh = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred_thresh[i] == 1)
        fn_thresh = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred_thresh[i] == 0)
        
        total_thresh = len(results)
        acc_thresh = (tp_thresh + tn_thresh) / total_thresh if total_thresh > 0 else 0.0
        prec_thresh = tp_thresh / (tp_thresh + fp_thresh) if (tp_thresh + fp_thresh) > 0 else 0.0
        rec_thresh = tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0.0
        f1_thresh = 2 * (prec_thresh * rec_thresh) / (prec_thresh + rec_thresh) if (prec_thresh + rec_thresh) > 0 else 0.0
        
        threshold_metrics[f"threshold_{thresh}"] = {
            "accuracy": acc_thresh,
            "precision": prec_thresh,
            "recall": rec_thresh,
            "f1_score": f1_thresh,
            "true_positive": tp_thresh,
            "true_negative": tn_thresh,
            "false_positive": fp_thresh,
            "false_negative": fn_thresh
        }
    
    return {
        "binary_classification": {
            "threshold": threshold,
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "threshold_analysis": threshold_metrics
        }
    }


def analyze_threshold_optimization(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    最適な閾値を探索
    
    Args:
        results: 評価結果のリスト
    
    Returns:
        閾値最適化の結果
    """
    thresholds = np.arange(0.5, 1.0, 0.05)
    best_threshold = 0.7
    best_f1 = 0.0
    
    threshold_results = []
    
    for thresh in thresholds:
        y_pred = [1 if r["similarity_score"] >= thresh else 0 for r in results]
        y_true = [1 if r["similarity_score"] >= 0.7 else 0 for r in results]  # 基準として0.7を使用
        
        tp = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        threshold_results.append({
            "threshold": float(thresh),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return {
        "best_threshold": float(best_threshold),
        "best_f1_score": float(best_f1),
        "threshold_analysis": threshold_results
    }


