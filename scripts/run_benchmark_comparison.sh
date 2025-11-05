#!/bin/bash
#
# ベンチマーク比較実行スクリプト
# 
# Vector-based評価とMCPエージェント評価を両方実行し、結果を比較します。
#

set -e

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 引数の取得
SAMPLES=${1:-50}
VECTOR_OUTPUT="evaluation_results_final.json"
MCP_OUTPUT="mcp_benchmark_results.json"
COMPARISON_OUTPUT="benchmark_comparison.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ベンチマーク比較実行${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "評価サンプル数: ${GREEN}${SAMPLES}${NC}"
echo ""

# 環境復元の確認
if [ -z "$LLM_BASE_URL" ]; then
    echo -e "${YELLOW}⚠️  環境変数が設定されていません。環境を復元します...${NC}"
    source setup/restore_env.sh
fi

echo -e "${BLUE}[ステップ 1/3] Vector-based 評価を実行中...${NC}"
echo ""

# Vector-based評価
python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output "$VECTOR_OUTPUT" \
    --samples "$SAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Vector-based 評価完了${NC}"
else
    echo ""
    echo -e "${RED}❌ Vector-based 評価に失敗しました${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}[ステップ 2/3] MCP Agent 評価を実行中...${NC}"
echo ""

# MCPエージェント評価
python scripts/evaluate_mcp_benchmark.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output "$MCP_OUTPUT" \
    --mode api_preferred \
    --samples "$SAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ MCP Agent 評価完了${NC}"
else
    echo ""
    echo -e "${RED}❌ MCP Agent 評価に失敗しました${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}[ステップ 3/3] 結果を比較中...${NC}"
echo ""

# 結果比較
python scripts/compare_benchmark_results.py \
    --vector "$VECTOR_OUTPUT" \
    --mcp "$MCP_OUTPUT" \
    --output "$COMPARISON_OUTPUT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 比較完了${NC}"
else
    echo -e "${RED}❌ 比較に失敗しました${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}全ての評価が完了しました${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "📁 生成されたファイル:"
echo -e "  - ${GREEN}${VECTOR_OUTPUT}${NC} (Vector-based 評価結果)"
echo -e "  - ${GREEN}${MCP_OUTPUT}${NC} (MCP Agent 評価結果)"
echo -e "  - ${GREEN}${COMPARISON_OUTPUT}${NC} (比較レポート)"
echo ""
echo -e "📊 結果の確認:"
echo -e "  cat ${COMPARISON_OUTPUT} | python3 -m json.tool | less"
echo ""
