#!/bin/bash
set -e

echo "==================================="
echo "Legal RAG System - UV Setup Script"
echo "==================================="

# プロジェクトルートに移動
# スクリプトがどこから呼ばれてもいいように、スクリプト自身の場所を基準にする
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR/.."
PROJECT_ROOT=$(pwd)

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# uvがインストールされているか確認
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    # インストールスクリプトを実行
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # uvのデフォルトインストール先
    UV_BIN_DIR="$HOME/.local/bin"
    
    # 現在のセッションのPATHを更新
    export PATH="$UV_BIN_DIR:$PATH"
    
    echo "Updating shell configuration file..."

    # .bashrc や .zshrc にPATHを追加
    # (zshを使っている場合も考慮)
    SHELL_CONFIG_FILE=""
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG_FILE="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_CONFIG_FILE="$HOME/.bashrc"
    else
        # デフォルトまたはフォールバック
        SHELL_CONFIG_FILE="$HOME/.bashrc"
        if [ ! -f "$SHELL_CONFIG_FILE" ]; then
             SHELL_CONFIG_FILE="$HOME/.profile" # .profile を使う環境もある
        fi
    fi
    
    # (touchでファイルが存在しない場合に作成)
    touch "$SHELL_CONFIG_FILE"

    UV_PATH_STRING="export PATH=\"\$HOME/.local/bin:\$PATH\""
    
    # 既にPATH設定が書き込まれていないか確認
    if ! grep -qF "$UV_PATH_STRING" "$SHELL_CONFIG_FILE"; then
        echo "" >> "$SHELL_CONFIG_FILE"
        echo "# Add uv (installed by legal-rag setup)" >> "$SHELL_CONFIG_FILE"
        echo "$UV_PATH_STRING" >> "$SHELL_CONFIG_FILE"
        echo "✓ Added uv to PATH in $SHELL_CONFIG_FILE."
        echo "   Please restart your shell or run 'source $SHELL_CONFIG_FILE' to apply changes."
    else
        echo "✓ uv PATH already configured in $SHELL_CONFIG_FILE."
    fi

    # 再度確認
    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed or PATH not set correctly."
        echo "Please ensure '$UV_BIN_DIR' is in your PATH."
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
    # Python 3.10以降を指定（推奨）
    uv venv .venv --python 3.10
    echo "✓ Virtual environment created"
fi

# 仮想環境を有効化
source .venv/bin/activate

echo ""
echo "Installing dependencies with uv..."

# pyproject.tomlから依存関係をインストール
# -e . は編集可能モードでのインストール
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
