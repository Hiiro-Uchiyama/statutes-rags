"""
2値分類問題としての判例評価の改善案

人間による評価ラベルを追加し、より厳密な2値分類問題として扱う。
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def add_human_evaluation_labels(
    results_file: Path,
    evaluation_labels: Dict[int, int]
) -> None:
    """
    人間による評価ラベルを追加
    
    Args:
        results_file: 評価結果ファイルのパス
        evaluation_labels: {precedent_index: label} の辞書
                          label: 1（同じ結論を出せた）or 0（出せなかった）
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 各結果に人間評価ラベルを追加
    for result in data['results']:
        idx = result['precedent_index']
        if idx in evaluation_labels:
            result['human_label'] = evaluation_labels[idx]
        else:
            result['human_label'] = None
    
    # 保存
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_binary_classification_with_human_labels(
    results: List[Dict[str, Any]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    人間による評価ラベルを使用した2値分類メトリクス
    
    Args:
        results: 評価結果のリスト（human_labelフィールドを含む）
        threshold: 類似度スコアの閾値
    
    Returns:
        2値分類メトリクス
    """
    # 人間による評価ラベル（正解）
    y_true = []
    # システムの予測ラベル
    y_pred = []
    
    for r in results:
        # 人間評価ラベルがある場合はそれを使用
        if 'human_label' in r and r['human_label'] is not None:
            y_true.append(r['human_label'])
        else:
            # 人間評価ラベルがない場合は類似度スコアから推定
            y_true.append(1 if r['similarity_score'] >= threshold else 0)
        
        # システムの予測（類似度スコアに基づく）
        y_pred.append(1 if r['similarity_score'] >= threshold else 0)
    
    # 混同行列の計算
    tp = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 0)
    
    # メトリクス計算
    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "binary_classification_with_human_labels": {
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
            "human_labels_count": sum(1 for r in results if 'human_label' in r and r['human_label'] is not None),
            "total_count": total
        }
    }


def create_evaluation_template(results_file: Path) -> Path:
    """
    人間評価用のテンプレートファイルを作成
    
    Args:
        results_file: 評価結果ファイルのパス
    
    Returns:
        テンプレートファイルのパス
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    template_file = results_file.parent / f"{results_file.stem}_human_evaluation.json"
    
    template_data = {
        "instructions": "各判例について、システムの回答が判例要旨と同じ結論を出せているか評価してください。",
        "evaluation_criteria": {
            "1": "同じ結論を出せた - システムの回答が判例要旨の法的論点を正しく理解し、類似した結論を導き出している",
            "0": "同じ結論を出せなかった - システムの回答が判例要旨の法的論点を理解していない、または異なる結論を導き出している"
        },
        "precedents": []
    }
    
    for result in data['results']:
        template_data['precedents'].append({
            "precedent_index": result['precedent_index'],
            "case_name": result['case_name'],
            "case_number": result['case_number'],
            "question": result['question'][:200] + "...",
            "correct_answer": result['correct_answer'],
            "predicted_answer": result['predicted_answer'][:200] + "...",
            "similarity_score": result['similarity_score'],
            "human_label": None,  # ここに1または0を記入
            "notes": ""  # 評価の理由やコメント
        })
    
    with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(template_data, f, ensure_ascii=False, indent=2)
    
    return template_file


