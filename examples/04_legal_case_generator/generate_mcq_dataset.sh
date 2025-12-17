#!/bin/bash
# 4択問題の事例シナリオをバックグラウンドで生成するスクリプト
#
# 使用例:
#   ./generate_mcq_dataset.sh \
#       --dataset /home/toronto02/statutes-rags/datasets/lawqa_jp/data/selection.json \
#       --start 0 \
#       --count 50 \
#       --output /home/toronto02/statutes-rags/examples/04_legal_case_generator/results/mcq_cases_001.json
#
# 実行すると nohup でパイプラインをバックグラウンド起動し、ログとPIDを表示します。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_DATASET="$PROJECT_ROOT/datasets/lawqa_jp/data/selection.json"
DEFAULT_START_INDEX=0
DEFAULT_COUNT=50
DEFAULT_OUTPUT="$SCRIPT_DIR/results/mcq_cases_$(date +%Y%m%d_%H%M%S).json"
DEFAULT_LOG_DIR="$SCRIPT_DIR/logs"

DATASET="$DEFAULT_DATASET"
START_INDEX="$DEFAULT_START_INDEX"
COUNT="$DEFAULT_COUNT"
OUTPUT="$DEFAULT_OUTPUT"
LOG_DIR="$DEFAULT_LOG_DIR"

print_usage() {
    cat <<'EOF'
使い方:
  ./generate_mcq_dataset.sh [オプション]

オプション:
  --dataset PATH   入力データセット（JSON）のパス
  --start N        開始インデックス（0始まり、デフォルト: 0）
  --count N        生成件数（デフォルト: 50）
  --output PATH    出力JSONファイルのパス
  --log-dir PATH   ログファイルを保存するディレクトリ
  -h, --help       このヘルプを表示

実行後、nohupでバックグラウンド処理が開始され、PIDとログファイルの場所を表示します。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --start)
            START_INDEX="$2"
            shift 2
            ;;
        --count)
            COUNT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "不明なオプション: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

if ! command -v nohup >/dev/null 2>&1; then
    echo "エラー: nohup コマンドが見つかりません。インストールしてください。" >&2
    exit 1
fi

if ! command -v realpath >/dev/null 2>&1; then
    echo "エラー: realpath コマンドが見つかりません。" >&2
    echo "coreutils をインストールするか、realpath を利用可能にしてください。" >&2
    exit 1
fi

DATASET_ABS="$(realpath "$DATASET")"
OUTPUT_ABS="$(realpath -m "$OUTPUT")"
LOG_DIR_ABS="$(realpath -m "$LOG_DIR")"
LOG_FILE="$LOG_DIR_ABS/mcq_generation_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "$DATASET_ABS" ]]; then
    echo "エラー: データセットが見つかりません: $DATASET_ABS" >&2
    exit 1
fi

if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
    echo "エラー: --start には0以上の整数を指定してください: $START_INDEX" >&2
    exit 1
fi

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -eq 0 ]]; then
    echo "エラー: --count には1以上の整数を指定してください: $COUNT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_ABS")"
mkdir -p "$LOG_DIR_ABS"

cd "$SCRIPT_DIR"

COMMAND=(python3 pipeline.py mcq
    --dataset "$DATASET_ABS"
    --index "$START_INDEX"
    --count "$COUNT"
    --output "$OUTPUT_ABS"
)

echo "======================================"
echo "MCQ事例データセット生成 (バックグラウンド)"
echo "======================================"
echo "開始時刻       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "データセット   : $DATASET_ABS"
echo "開始インデックス: $START_INDEX"
echo "生成件数       : $COUNT"
echo "出力ファイル   : $OUTPUT_ABS"
echo "ログディレクトリ: $LOG_DIR_ABS"
echo "ログファイル   : $LOG_FILE"
echo ""
echo "nohup コマンド:"
printf '  %q ' "${COMMAND[@]}"
echo ""
echo ""

nohup "${COMMAND[@]}" >"$LOG_FILE" 2>&1 &
PID=$!

echo "バックグラウンド実行を開始しました。"
echo "  PID : $PID"
echo "  ログ: $LOG_FILE"
echo ""
echo "進捗は tail -f \"$LOG_FILE\" で確認できます。"
echo "完了後、結果は $OUTPUT_ABS に保存されます。"
echo "======================================"

