#!/bin/bash

set -e

# スクリプト自身のディレクトリに移動する
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

echo "=== Ollama Setup Script (Persistent) ==="

# 永続的なモデル保存先を定義
# 環境変数が既に設定されている場合はそれを優先、なければデフォルトパスを使用
if [ -z "$OLLAMA_MODELS" ]; then
    export OLLAMA_MODELS="$HOME/work/.ollama-models"
fi
mkdir -p "$OLLAMA_MODELS"
echo "Setting OLLAMA_MODELS to $OLLAMA_MODELS"

# Download Ollama
echo "Downloading Ollama..."
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz

# Extract
echo "Extracting Ollama..."
tar -xzf ollama.tgz

# Make executable
chmod +x ./bin/ollama

# Set GPU device and Ollama environment variables
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_FLASH_ATTENTION=1

# Start Ollama server in background
echo "Starting Ollama server on GPU 0..."
./bin/ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Ollama server started (PID: $OLLAMA_PID)"
echo "Waiting for server to be ready..."
sleep 5

# Pull LLM model
# Note: Embedding model (intfloat/multilingual-e5-large) is downloaded via HuggingFace, not Ollama
echo "Pulling LLM model (qwen3:8b) to $OLLAMA_MODELS..."
./bin/ollama pull qwen3:8b

echo ""
echo "=== Setup Complete ==="
echo "Ollama server is running (PID: $OLLAMA_PID)"
echo "Models are stored in $OLLAMA_MODELS"
echo "Log file: ollama.log"
echo ""

# Verify API is accessible
echo "Verifying API endpoint..."
sleep 2
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "[OK] API is accessible at http://localhost:11434"
else
    echo "[WARNING] API endpoint not responding"
fi

echo ""
echo "Available models:"
./bin/ollama list

echo ""
echo "=== API Usage Examples ==="
echo ""
echo "# Chat with qwen3:8b:"
echo "curl http://localhost:11434/api/generate -d '{"
echo '  "model": "qwen3:8b",'
echo '  "prompt": "Why is the sky blue?",'
echo '  "stream": false'
echo "}'"
echo ""
echo "# List models via API:"
echo "curl http://localhost:11434/api/tags"
echo ""
echo "=== Server Control ==="
echo "Stop server: kill $OLLAMA_PID"
echo "View logs: tail -f ollama.log"
echo "API base URL: http://localhost:11434"
echo ""
echo "IMPORTANT: After container reset, run 'source restore_env.sh' from your project root."
