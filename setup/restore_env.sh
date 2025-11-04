#!/bin/bash
#
# Environment Restoration Script
# 
# 重要: このスクリプトは必ず source コマンドで実行してください
# 
# 正しい実行方法:
#   source setup/restore_env.sh
#   または
#   . setup/restore_env.sh
# 
# 誤った実行方法（環境変数が反映されません）:
#   ./setup/restore_env.sh
#   bash setup/restore_env.sh
# 
# このスクリプトは以下を行います:
# - uv, ollama へのPATHを設定
# - OLLAMA_MODELS 環境変数を設定
# - Python仮想環境（.venv）を有効化
# - Ollamaサーバーの起動確認と必要に応じた自動起動
#

echo "Restoring persistent environment..."

# 1. プロジェクトルートを基準にパスを設定 (スクリプトの場所から一つ上)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)
fi

# 2. uv へのパスを通す (setup_uv_env.sh が永続化したパス)
UV_BIN_DIR="$HOME/work/tools/uv/bin"
if [ -d "$UV_BIN_DIR" ]; then
    export PATH="$UV_BIN_DIR:$PATH"
    echo "[OK] Added persistent uv to PATH."
else
    echo "[WARNING] uv not found. Please run 'setup/setup_uv_env.sh' first."
fi

# 3. Ollama へのパスを通す (setup_ollama.sh が配置したパス)
OLLAMA_BIN_DIR="$PROJECT_ROOT/setup/bin"
if [ -d "$OLLAMA_BIN_DIR" ]; then
    export PATH="$OLLAMA_BIN_DIR:$PATH"
    echo "[OK] Added persistent ollama to PATH."
else
    echo "[WARNING] ollama not found. Please run 'setup/setup_ollama.sh' first."
fi

# 4. Ollama モデルの場所を設定 (setup_ollama.sh が設定したパス)
# 環境変数が既に設定されている場合はそれを優先、なければデフォルトパスを使用
if [ -z "$OLLAMA_MODELS" ]; then
    export OLLAMA_MODELS="$HOME/work/.ollama-models"
fi
echo "[OK] Set OLLAMA_MODELS to $OLLAMA_MODELS"

# 5. トークナイザーの確認（SudachiPyがデフォルトでインストール済み）
# MeCabはレガシーサポートのため、インストールされていれば使用可能
MECAB_SETUP_DIR="$PROJECT_ROOT/setup"
MECAB_BIN_DIR="$MECAB_SETUP_DIR/bin"
MECAB_LIB_DIR="$MECAB_SETUP_DIR/lib/mecab"

if [ -f "$MECAB_BIN_DIR/mecab" ]; then
    export PATH="$MECAB_BIN_DIR:$PATH"
    export LD_LIBRARY_PATH="$MECAB_LIB_DIR/lib:${LD_LIBRARY_PATH}"
    export MECABRC="$MECAB_LIB_DIR/etc/mecabrc"
    echo "[OK] MeCab found (legacy support)."
else
    echo "[INFO] Using SudachiPy tokenizer (default)."
fi

# 6. Python仮想環境を有効化 (setup_uv_env.sh が作成したパス)
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[OK] Activated Python virtual environment."
else
    echo "[WARNING] .venv not found. Please run 'setup/setup_uv_env.sh' first."
fi

echo ""
echo "Environment restored."


# 7. Ollamaが既に起動しているか確認
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "[OK] Ollama server is already running."
else
    echo "Ollama server not running. Starting server..."
    
    # GPU環境変数を設定
    export CUDA_VISIBLE_DEVICES=0
    export OLLAMA_NUM_GPU=1
    export OLLAMA_GPU_OVERHEAD=0
    export OLLAMA_FLASH_ATTENTION=1

    # Ollamaサーバーをバックグラウンドで起動
    OLLAMA_LOG_FILE="$PROJECT_ROOT/setup/ollama.log"
    ollama serve > "$OLLAMA_LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    
    echo "Ollama server started (PID: $OLLAMA_PID). Log: $OLLAMA_LOG_FILE"
    echo "Waiting for server to be ready..."
    sleep 5

    # 8. curlで起動確認
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "[OK] Ollama server started successfully and API is accessible."
    else
        echo "[ERROR] Ollama server failed to start. Check logs:"
        echo "  tail -f $OLLAMA_LOG_FILE"
    fi
fi

echo ""
echo "All setup complete. You can now run your scripts."