#!/usr/bin/env python3
"""
ベンチマーク結果比較ツール

Vectorベース評価とMCPエージェント評価の結果を比較します。
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_result(filepath: Path) -> Dict[str, Any]:
    """結果ファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """結果から主要指標を抽出（フォーマット統一）"""
    # evaluation_typeで判定
    eval_type = result.get("evaluation_type", "vector")
    
    if eval_type == "mcp_agent":
        # MCPエージェント結果
        metrics = result.get("metrics", {})
        return {
            "type": "MCP Agent",
            "mode": result.get("mode", "unknown"),
            "accuracy": metrics.get("accuracy", 0.0),
            "correct": metrics.get("correct", 0),
            "total": metrics.get("total", 0),
            "avg_time": metrics.get("avg_time", 0.0),
            "error_rate": metrics.get("error_rate", 0.0),
            "api_usage": metrics.get("api_metrics", {}).get("api_usage_rate", 0.0),
            "by_source": metrics.get("by_source", {}),
            "timestamp": result.get("timestamp", "")
        }
    else:
        # Vectorベース結果
        metrics = result.get("metrics", {})
        return {
            "type": "Vector-based",
            "mode": result.get("metadata", {}).get("retriever_type", "vector"),
            "accuracy": metrics.get("accuracy", 0.0),
            "correct": metrics.get("correct", 0),
            "total": metrics.get("total", 0),
            "avg_time": metrics.get("avg_time", 0.0),
            "error_rate": 0.0,  # Vectorベースにはエラー率がない
            "api_usage": 0.0,   # API使用なし
            "by_source": {},
            "timestamp": result.get("metadata", {}).get("timestamp", "")
        }


def print_comparison(vector_metrics: Dict[str, Any], mcp_metrics: Dict[str, Any]):
    """比較結果を表示"""
    print("=" * 100)
    print("ベンチマーク結果比較")
    print("=" * 100)
    print()
    
    # 基本情報
    print("【評価システム】")
    print(f"  Vector-based: {vector_metrics['mode']} モード")
    print(f"  MCP Agent:    {mcp_metrics['mode']} モード")
    print()
    
    # タイムスタンプ
    print("【評価日時】")
    print(f"  Vector-based: {vector_metrics['timestamp']}")
    print(f"  MCP Agent:    {mcp_metrics['timestamp']}")
    print()
    
    # 主要指標の比較
    print("=" * 100)
    print("【主要指標の比較】")
    print("=" * 100)
    print()
    
    # テーブルヘッダー
    print(f"{'指標':<20} {'Vector-based':<20} {'MCP Agent':<20} {'差分':<20}")
    print("-" * 100)
    
    # 正答率
    acc_diff = mcp_metrics['accuracy'] - vector_metrics['accuracy']
    acc_diff_str = f"{acc_diff:+.2%}"
    print(f"{'正答率':<20} {vector_metrics['accuracy']:>18.2%} {mcp_metrics['accuracy']:>18.2%} {acc_diff_str:>20}")
    
    # 正答数
    correct_diff = mcp_metrics['correct'] - vector_metrics['correct']
    print(f"{'正答数':<20} {vector_metrics['correct']:>18} {mcp_metrics['correct']:>18} {correct_diff:>+20}")
    
    # 総問題数
    print(f"{'総問題数':<20} {vector_metrics['total']:>18} {mcp_metrics['total']:>18} {'-':>20}")
    
    # 平均応答時間
    time_diff = mcp_metrics['avg_time'] - vector_metrics['avg_time']
    time_diff_str = f"{time_diff:+.2f}秒"
    print(f"{'平均応答時間':<20} {vector_metrics['avg_time']:>16.2f}秒 {mcp_metrics['avg_time']:>16.2f}秒 {time_diff_str:>20}")
    
    # エラー率
    print(f"{'エラー率':<20} {vector_metrics['error_rate']:>18.2%} {mcp_metrics['error_rate']:>18.2%} {'-':>20}")
    
    print()
    
    # MCP特有の情報
    if mcp_metrics['api_usage'] > 0:
        print("=" * 100)
        print("【MCP Agent 特有情報】")
        print("=" * 100)
        print(f"API使用率: {mcp_metrics['api_usage']:.2%}")
        print()
        
        if mcp_metrics['by_source']:
            print("データソース別正答率:")
            for source, stats in mcp_metrics['by_source'].items():
                accuracy = stats.get('accuracy', 0.0)
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                print(f"  {source:<15}: {accuracy:.2%} ({correct}/{total})")
            print()
    
    # 結論
    print("=" * 100)
    print("【結論】")
    print("=" * 100)
    
    if acc_diff > 0.01:
        improvement = "向上"
        symbol = "📈"
    elif acc_diff < -0.01:
        improvement = "低下"
        symbol = "📉"
    else:
        improvement = "ほぼ同等"
        symbol = "➡️"
    
    print(f"{symbol} MCPエージェントは、Vectorベースと比較して正答率が {abs(acc_diff):.2%} {improvement}しました。")
    print()
    
    if time_diff > 0:
        print(f"⏱️  応答時間は {time_diff:.2f}秒 遅くなりました（API呼び出しのオーバーヘッド）。")
    else:
        print(f"⚡ 応答時間は {abs(time_diff):.2f}秒 速くなりました。")
    print()


def save_comparison_report(
    vector_metrics: Dict[str, Any],
    mcp_metrics: Dict[str, Any],
    output_path: Path
):
    """比較レポートをJSON形式で保存"""
    report = {
        "comparison_timestamp": datetime.now().isoformat(),
        "vector_based": vector_metrics,
        "mcp_agent": mcp_metrics,
        "comparison": {
            "accuracy_diff": mcp_metrics['accuracy'] - vector_metrics['accuracy'],
            "time_diff": mcp_metrics['avg_time'] - vector_metrics['avg_time'],
            "mcp_advantage": mcp_metrics['accuracy'] > vector_metrics['accuracy']
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 比較レポートを {output_path} に保存しました。")
    print()


def main():
    parser = argparse.ArgumentParser(description="ベンチマーク結果比較ツール")
    parser.add_argument(
        "--vector",
        type=Path,
        default=Path("evaluation_results_final.json"),
        help="Vectorベース評価結果のパス"
    )
    parser.add_argument(
        "--mcp",
        type=Path,
        default=Path("mcp_benchmark_results.json"),
        help="MCPエージェント評価結果のパス"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_comparison.json"),
        help="比較レポートの出力先"
    )
    
    args = parser.parse_args()
    
    # 結果ファイルの読み込み
    print("📥 結果ファイルを読み込み中...")
    
    if not args.vector.exists():
        print(f"❌ エラー: Vectorベース評価結果が見つかりません: {args.vector}")
        print("   先に Vector-based 評価を実行してください:")
        print("   ./scripts/evaluate.sh 50")
        return 1
    
    if not args.mcp.exists():
        print(f"❌ エラー: MCPエージェント評価結果が見つかりません: {args.mcp}")
        print("   先に MCP Agent 評価を実行してください:")
        print("   python scripts/evaluate_mcp_benchmark.py --samples 50")
        return 1
    
    vector_result = load_result(args.vector)
    mcp_result = load_result(args.mcp)
    
    print("   ✓ 読み込み完了")
    print()
    
    # 指標の抽出
    vector_metrics = extract_metrics(vector_result)
    mcp_metrics = extract_metrics(mcp_result)
    
    # 比較結果の表示
    print_comparison(vector_metrics, mcp_metrics)
    
    # レポートの保存
    save_comparison_report(vector_metrics, mcp_metrics, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
