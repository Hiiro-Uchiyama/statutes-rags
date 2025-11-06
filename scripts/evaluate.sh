#!/bin/bash
# 最終評価スクリプト（Vector-Onlyモード）

set -e

echo "=========================================="
echo "Final RAG Evaluation (Vector-Only Mode)"
echo "=========================================="
echo ""

# 環境変数を設定
export RETRIEVER_TYPE=vector

# スクリプトのディレクトリに移動
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# 仮想環境を有効化
source .venv/bin/activate

# デフォルト設定
SAMPLES=${1:-50}
TOP_K=${2:-10}
OUTPUT=${3:-results/evaluations/evaluation_results_final.json}

echo "Configuration:"
echo "  RETRIEVER_TYPE: $RETRIEVER_TYPE (BM25無効化、メモリ問題回避)"
echo "  Samples: $SAMPLES"
echo "  Top-K: $TOP_K"
echo "  Output: $OUTPUT"
echo ""

# GPU使用状況を表示
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# 評価実行
echo "Starting evaluation..."
python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output "$OUTPUT" \
    --samples "$SAMPLES" \
    --top-k "$TOP_K"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT"
