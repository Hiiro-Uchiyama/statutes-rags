#!/bin/bash
set -e

echo "==================================="
echo "Legal RAG System - Test Runner"
echo "==================================="

cd "$(dirname "$0")/.."

# 仮想環境の確認
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run 'make setup-uv' or './setup/setup_uv_env.sh' first."
    exit 1
fi

source .venv/bin/activate

echo ""
echo "Running tests..."
echo ""

# デフォルトはユニットテストのみ
TEST_TYPE=${1:-"unit"}

case $TEST_TYPE in
    "unit")
        echo "Running unit tests only (fast)..."
        pytest tests/ -v -m unit
        ;;
    "integration")
        echo "Running integration tests..."
        pytest tests/ -v -m integration
        ;;
    "all")
        echo "Running all tests..."
        pytest tests/ -v
        ;;
    "coverage")
        echo "Running tests with coverage..."
        pytest tests/ -v --cov=app --cov=scripts --cov-report=html --cov-report=term
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    "quick")
        echo "Running quick tests (unit tests, no slow tests)..."
        pytest tests/ -v -m "unit and not slow"
        ;;
    "slow")
        echo "Running slow tests..."
        pytest tests/ -v -m slow
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo ""
        echo "Usage: $0 [unit|integration|all|coverage|quick|slow]"
        echo ""
        echo "  unit        - Run unit tests only (default, fast)"
        echo "  integration - Run integration tests"
        echo "  all         - Run all tests"
        echo "  coverage    - Run tests with coverage report"
        echo "  quick       - Run quick tests (unit, no slow)"
        echo "  slow        - Run slow tests only"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo "Tests completed!"
echo "==================================="
