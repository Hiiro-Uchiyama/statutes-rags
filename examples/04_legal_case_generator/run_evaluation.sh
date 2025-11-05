#!/bin/bash
# Legal Case Generator 評価実行スクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "======================================"
echo "Legal Case Generator - 評価実行"
echo "======================================"
echo ""

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# Ollamaの起動確認
echo "[1/4] Ollamaの起動確認..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "エラー: Ollamaが起動していません"
    echo "   次のコマンドでOllamaを起動してください:"
    echo "   ollama serve"
    exit 1
fi
echo "✓ Ollama起動確認完了"
echo ""

# qwen3:8bモデルの確認
echo "[2/4] LLMモデルの確認..."
if ! curl -s http://localhost:11434/api/tags | grep -q "qwen3:8b"; then
    echo "   警告: qwen3:8bモデルが見つかりません"
    echo "   次のコマンドでモデルをダウンロードしてください:"
    echo "   ollama pull qwen3:8b"
fi
echo "✓ モデル確認完了"
echo ""

# テストの実行
echo "[3/4] 基本テストの実行..."
if python3 examples/04_legal_case_generator/tests/test_legal_case_generator.py; then
    echo "✓ 基本テスト成功"
else
    echo "エラー: 基本テストに失敗しました"
    exit 1
fi
echo ""

# 評価の実行
echo "[4/4] 評価の実行..."
echo "   （3つの法令で9事例を生成します。約5-10分かかります）"
echo ""

cd examples/04_legal_case_generator

if python3 evaluate.py --output evaluation_results.json; then
    echo ""
    echo "✓ 評価完了"
    echo ""
    echo "======================================"
    echo "評価結果サマリー"
    echo "======================================"
    cat evaluation_results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
s = r['summary']
print(f'テストケース数: {s[\"total_test_cases\"]}')
print(f'生成事例数: {s[\"total_cases_generated\"]}')
print(f'成功数: {s[\"success_count\"]}')
print(f'成功率: {s[\"success_rate\"] * 100:.1f}%')
print(f'総実行時間: {s[\"total_time\"]:.1f}秒')
print(f'平均生成時間: {s[\"average_time_per_case\"]:.1f}秒/事例')
print(f'平均反復回数: {s[\"average_iterations\"]:.2f}回')
"
    echo ""
    echo "詳細結果: examples/04_legal_case_generator/evaluation_results.json"
else
    echo "エラー: 評価に失敗しました"
    exit 1
fi

echo ""
echo "======================================"
echo "評価完了！"
echo "======================================"
