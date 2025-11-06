#!/usr/bin/env python3
"""
評価結果の詳細分析スクリプト
誤答パターンを分析して改善点を特定
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

def load_results(file_path: Path) -> Dict[str, Any]:
    """評価結果を読み込み"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_error_patterns(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """誤答パターンを分析"""
    analysis = {
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0}),
        "by_error_type": defaultdict(int),
        "confusion_matrix": defaultdict(int),
        "incorrect_cases": [],
        "question_types": defaultdict(lambda: {"total": 0, "correct": 0})
    }
    
    for result in results:
        # カテゴリ別分析（ファイル名から抽出）
        file_name = result.get("file_name", "")
        category = file_name.split("_")[0] if "_" in file_name else "unknown"
        
        analysis["by_category"][category]["total"] += 1
        
        if result["is_correct"]:
            analysis["by_category"][category]["correct"] += 1
        else:
            analysis["by_category"][category]["incorrect"] += 1
            
            # エラータイプ分類
            predicted = result["predicted_answer"]
            if predicted == "error":
                analysis["by_error_type"]["timeout_or_error"] += 1
            elif predicted == "unknown":
                analysis["by_error_type"]["parse_failure"] += 1
            else:
                analysis["by_error_type"]["wrong_answer"] += 1
            
            # 混同行列
            correct = result["correct_answer"]
            confusion_key = f"{correct} -> {predicted}"
            analysis["confusion_matrix"][confusion_key] += 1
            
            # 誤答ケース記録
            analysis["incorrect_cases"].append({
                "file_name": file_name,
                "question": result["question"][:100],
                "correct": correct,
                "predicted": predicted,
                "response": result["response"][:200] if isinstance(result["response"], str) else str(result["response"])[:200],
                "retrieved_docs_count": result.get("retrieved_docs_count", 0)
            })
        
        # 質問タイプ分析
        question = result["question"]
        if "正しいもの" in question or "正しい" in question:
            q_type = "正答選択"
        elif "誤っているもの" in question or "誤っている" in question:
            q_type = "誤答選択"
        elif "条文" in question or "根拠" in question:
            q_type = "条文参照"
        else:
            q_type = "その他"
        
        analysis["question_types"][q_type]["total"] += 1
        if result["is_correct"]:
            analysis["question_types"][q_type]["correct"] += 1
    
    return analysis

def print_analysis(analysis: Dict[str, Any], total_samples: int, correct_count: int):
    """分析結果を表示"""
    print("\n" + "="*80)
    print("詳細分析レポート")
    print("="*80)
    
    # 全体サマリー
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    print(f"\n【全体サマリー】")
    print(f"総サンプル数: {total_samples}")
    print(f"正解数: {correct_count}")
    print(f"誤答数: {total_samples - correct_count}")
    print(f"精度: {accuracy:.2%}")
    
    # カテゴリ別分析
    print(f"\n【カテゴリ別精度】")
    categories = sorted(analysis["by_category"].items(), 
                       key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0)
    
    for category, stats in categories:
        cat_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {category:20s}: {cat_accuracy:>6.1%} ({stats['correct']:>3d}/{stats['total']:>3d}) "
              f"[誤答: {stats['incorrect']:>2d}]")
    
    # エラータイプ分析
    print(f"\n【エラータイプ分析】")
    for error_type, count in sorted(analysis["by_error_type"].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"  {error_type:20s}: {count:>3d} 件")
    
    # 質問タイプ別分析
    print(f"\n【質問タイプ別精度】")
    for q_type, stats in sorted(analysis["question_types"].items()):
        q_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {q_type:20s}: {q_accuracy:>6.1%} ({stats['correct']:>3d}/{stats['total']:>3d})")
    
    # 混同行列（上位10件）
    print(f"\n【混同パターン（上位10件）】")
    for confusion, count in sorted(analysis["confusion_matrix"].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {confusion:10s}: {count:>3d} 件")
    
    # 誤答ケースサンプル
    print(f"\n【誤答ケース例（上位5件）】")
    for i, case in enumerate(analysis["incorrect_cases"][:5], 1):
        print(f"\n  [{i}] {case['file_name']}")
        print(f"      質問: {case['question']}...")
        print(f"      正解: {case['correct']} | 予測: {case['predicted']}")
        print(f"      応答: {case['response']}...")
        print(f"      検索文書数: {case['retrieved_docs_count']}")
    
    print("\n" + "="*80)

def suggest_improvements(analysis: Dict[str, Any], accuracy: float):
    """改善提案を生成"""
    print("\n" + "="*80)
    print("改善提案")
    print("="*80)
    
    suggestions = []
    
    # エラータイプに基づく提案
    error_types = analysis["by_error_type"]
    
    if error_types.get("timeout_or_error", 0) > 0:
        suggestions.append({
            "priority": "高",
            "issue": f"タイムアウト/エラー: {error_types['timeout_or_error']}件",
            "actions": [
                "Top-Kを3以下に削減してコンテキストを短縮",
                "request_timeoutを180秒に延長",
                "より高速なLLMモデル（gemma2:9b等）を試用"
            ]
        })
    
    if error_types.get("parse_failure", 0) > 0:
        suggestions.append({
            "priority": "高",
            "issue": f"パース失敗: {error_types['parse_failure']}件",
            "actions": [
                "プロンプトをさらに厳格化（Few-shot例を追加）",
                "LLMモデルを変更（より指示に従順なモデルを選択）",
                "Temperature=0.0で完全に決定的な応答を強制"
            ]
        })
    
    wrong_answer_count = error_types.get("wrong_answer", 0)
    if wrong_answer_count > 0:
        wrong_answer_ratio = wrong_answer_count / sum(error_types.values()) if error_types else 0
        
        if wrong_answer_ratio > 0.7:
            suggestions.append({
                "priority": "最高",
                "issue": f"誤答が多数: {wrong_answer_count}件（{wrong_answer_ratio:.1%}）",
                "actions": [
                    "Retrieverの改善: hybrid → vector or BM25のみで比較",
                    "Top-Kを増やして関連文書を増加（5-10）",
                    "Rerankerを有効化して文書の質を向上",
                    "より大きなLLMモデル（qwen2.5:14b, gemma2:27b等）を使用",
                    "プロンプトに「誤答を選ぶ」等の指示を明確化"
                ]
            })
    
    # カテゴリ別の弱点分析
    weak_categories = []
    for category, stats in analysis["by_category"].items():
        if stats["total"] >= 5:  # 十分なサンプルがある場合のみ
            cat_accuracy = stats["correct"] / stats["total"]
            if cat_accuracy < 0.5:
                weak_categories.append((category, cat_accuracy, stats["total"]))
    
    if weak_categories:
        suggestions.append({
            "priority": "中",
            "issue": f"特定カテゴリで低精度: {', '.join([c[0] for c in weak_categories])}",
            "actions": [
                "該当カテゴリの法令データを確認（インデックスに含まれているか）",
                "カテゴリ固有のプロンプトテンプレートを作成",
                "該当カテゴリの問題でRetrieverの検索結果を目視確認"
            ]
        })
    
    # 質問タイプ別の弱点
    for q_type, stats in analysis["question_types"].items():
        if stats["total"] >= 5:
            q_accuracy = stats["correct"] / stats["total"]
            if q_accuracy < 0.5:
                suggestions.append({
                    "priority": "中",
                    "issue": f"「{q_type}」タイプの問題で低精度（{q_accuracy:.1%}）",
                    "actions": [
                        f"「{q_type}」に特化したプロンプト設計",
                        "質問タイプに応じた検索クエリの調整",
                        "Few-shotプロンプトで同種の例を提示"
                    ]
                })
    
    # 精度目標に基づく提案
    if accuracy < 0.65:
        suggestions.insert(0, {
            "priority": "最高",
            "issue": f"精度が低すぎる（{accuracy:.1%}）→ 80-90%を目指す",
            "actions": [
                "LLMモデルを大型化（qwen2.5:14b, qwen2.5:32b, gemma2:27b等）",
                "Few-shotプロンプト（正解例を2-3個提示）を導入",
                "Chain-of-Thoughtで推論ステップを明示",
                "複数回推論して多数決（アンサンブル）"
            ]
        })
    
    # 提案を表示
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n【提案 {i}】優先度: {suggestion['priority']}")
        print(f"問題: {suggestion['issue']}")
        print("対策:")
        for action in suggestion['actions']:
            print(f"  • {action}")
    
    print("\n" + "="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_evaluation_results.py <evaluation_results.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    # 結果を読み込み
    data = load_results(results_file)
    results = data["results"]
    summary = data["summary"]
    
    # 分析実行
    analysis = analyze_error_patterns(results)
    
    # 結果表示
    print_analysis(analysis, summary["total_count"], summary["correct_count"])
    
    # 改善提案
    suggest_improvements(analysis, summary["accuracy"])

if __name__ == "__main__":
    main()
