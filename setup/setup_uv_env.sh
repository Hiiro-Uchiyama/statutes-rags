#!/bin/bash
set -e

echo "==================================="
echo "Legal RAG System - UV Setup Script (Persistent)"
echo "==================================="

# プロジェクトルートに移動
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR/.."
PROJECT_ROOT=$(pwd)

echo ""
echo "Project root (persistent volume): $PROJECT_ROOT"
echo ""

# 永続的なツールのインストール先を定義
PERSISTENT_TOOLS_DIR="$HOME/work/tools"
UV_DIR="$PERSISTENT_TOOLS_DIR/uv"
UV_BIN_DIR="$UV_DIR/bin"
UV_BIN="$UV_BIN_DIR/uv"

# CARGO_HOMEを設定（インストーラがこれを見ることを期待）
export CARGO_HOME="$UV_DIR"
mkdir -p "$UV_BIN_DIR"

# uvがインストールされているか確認
if [ ! -f "$UV_BIN" ]; then
    echo "Installing uv..."
    # インストールスクリプトを実行 (これは ~/.local/bin にインストールされる)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    DEFAULT_UV_INSTALL_DIR="$HOME/.local/bin"


    # インストールされたバイナリを永続的な場所に「コピー」する
    if [ -f "$DEFAULT_UV_INSTALL_DIR/uv" ]; then
        echo "Copying uv binary to persistent location: $UV_BIN_DIR"
        cp "$DEFAULT_UV_INSTALL_DIR/uv" "$UV_BIN_DIR/uv"
        cp "$DEFAULT_UV_INSTALL_DIR/uvx" "$UV_BIN_DIR/uvx"
        
        # envスクリプトは元の場所 (~/.local/bin) に残しておく
        echo "[OK] uv/uvx binaries copied."
    else
        echo "Error: uv was not found in the default location ($DEFAULT_UV_INSTALL_DIR) after installation."
        exit 1
    fi


    # 現在のセッションのPATHを更新 (永続パスを優先的に追加)
    export PATH="$UV_BIN_DIR:$PATH"
    
    # 再度確認 (今度は $UV_BIN のパスで確認)
    if [ ! -f "$UV_BIN" ]; then
        echo "Error: uv installation failed. Binary not found at $UV_BIN."
        exit 1
    fi
    
    echo "[OK] uv installed successfully to $UV_BIN"
else
    echo "[OK] uv is already installed at $UV_BIN"
    export PATH="$UV_BIN_DIR:$PATH"
    uv --version
fi

SHELL_CONFIG_FILE="$HOME/.bashrc"
UV_PATH_STRING="export PATH=\"$UV_BIN_DIR:\$PATH\""

# .bashrc が存在するか確認し、なければ作成
touch "$SHELL_CONFIG_FILE"

# 既にPATH設定が書き込まれていないか確認
if ! grep -qF "$UV_PATH_STRING" "$SHELL_CONFIG_FILE"; then
    echo ""
    echo "Adding uv persistent path to $SHELL_CONFIG_FILE..."
    # .bashrcの末尾に追記
    echo "" >> "$SHELL_CONFIG_FILE"
    echo "# Added by statutes-rags setup (uv persistent path)" >> "$SHELL_CONFIG_FILE"
    echo "$UV_PATH_STRING" >> "$SHELL_CONFIG_FILE"
    echo "[OK] uv path added to $SHELL_CONFIG_FILE."
    echo "  Run 'source $SHELL_CONFIG_FILE' or restart your shell to apply changes permanently."
else
    echo ""
    echo "[OK] uv persistent path already configured in $SHELL_CONFIG_FILE."
fi

echo ""
echo "Creating virtual environment in $PROJECT_ROOT/.venv ..."

# 既存の.venvがあれば削除（非インタラクティブ）
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
    echo "[OK] Removed existing .venv"
fi

# uv venvで仮想環境を作成
if [ ! -d ".venv" ]; then
    # Python 3.10以降を指定（推奨）
    uv venv .venv --python 3.10
    echo "[OK] Virtual environment created"
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
    "torch" \
    "torchvision" \
    "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu121 \
    "langchain>=0.1.0" \
    "langchain-community>=0.0.10" \
    "langchain-core>=1.0.0" \
    "faiss-cpu>=1.7.4" \
    "sentence-transformers>=2.2.0" \
    "rank-bm25>=0.2.2" \
    "ragas>=0.1.0" \
    "pandas>=2.0.0"

echo ""
echo "Installing Japanese tokenizers (no admin rights required)..."
echo "  - SudachiPy (recommended, MeCab alternative)"
echo "  - Janome (lightweight fallback)"
uv pip install \
    "sudachipy>=0.6.0" \
    "sudachidict-core>=20230927" \
    "janome>=0.5.0"

echo ""
echo "Installing examples dependencies (for advanced features)..."
echo "  - LangGraph (for Multi-Agent Debate and Agentic RAG)"
echo "  - LangSmith (optional, for tracing)"
echo "  - xmltodict (for e-Gov API XML parsing)"
uv pip install \
    "langgraph>=0.2.0" \
    "langsmith>=0.1.0" \
    "xmltodict>=0.13.0"

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "IMPORTANT: The persistent uv path has been added to $SHELL_CONFIG_FILE."
echo "Please run 'source $SHELL_CONFIG_FILE' or restart your terminal to ensure it's active everywhere."