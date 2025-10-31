#!/bin/bash
set -e

echo "==================================="
echo "Legal RAG System - UV Setup Script"
echo "==================================="

# プロジェクトルートに移動
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# uvがインストールされているか確認
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # uvをパスに追加
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # 再度確認
    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed"
        exit 1
    fi
    
    echo "✓ uv installed successfully"
else
    echo "✓ uv is already installed"
    uv --version
fi

echo ""
echo "Creating virtual environment with uv..."

# 既存の.venvがあれば削除（非インタラクティブ）
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
    echo "✓ Removed existing .venv"
fi

# uv venvで仮想環境を作成
if [ ! -d ".venv" ]; then
    uv venv .venv
    echo "✓ Virtual environment created"
fi

# 仮想環境を有効化
source .venv/bin/activate

echo ""
echo "Installing dependencies with uv..."

# pyproject.tomlから依存関係をインストール
uv pip install -e .

# 追加の開発用依存関係
echo ""
echo "Installing development dependencies..."
uv pip install \
    "pytest>=7.4.0" \
    "pytest-cov>=4.1.0" \
    "pytest-asyncio>=0.21.0" \
    "pytest-mock>=3.11.0" \
    "black>=23.11.0" \
    "ruff>=0.1.6" \
    "mypy>=1.7.0"

echo ""
echo "Installing RAG-specific dependencies..."
uv pip install \
    "langchain>=0.1.0" \
    "langchain-community>=0.0.10" \
    "langchain-core>=1.0.0" \
    "faiss-cpu>=1.7.4" \
    "sentence-transformers>=2.2.0" \
    "rank-bm25>=0.2.2" \
    "mecab-python3>=1.0.6" \
    "ragas>=0.1.0" \
    "pandas>=2.0.0"

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo "  pytest tests/ -v --cov=app"
echo ""
echo "To format code:"
echo "  black app/ scripts/ tests/"
echo "  ruff check app/ scripts/ tests/"
echo ""
