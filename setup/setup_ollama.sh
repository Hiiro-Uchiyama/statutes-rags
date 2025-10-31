#!/bin/bash

set -e

echo "=== Ollama Setup Script ==="

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

# Pull embedding model
echo "Pulling embedding model (nomic-embed-text)..."
./bin/ollama pull nomic-embed-text

# Pull LLM model
echo "Pulling LLM model (gpt-oss:20b)..."
./bin/ollama pull gpt-oss:20b

echo ""
echo "=== Setup Complete ==="
echo "Ollama server is running (PID: $OLLAMA_PID)"
echo "Log file: ollama.log"
echo ""

# Verify API is accessible
echo "Verifying API endpoint..."
sleep 2
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✓ API is accessible at http://localhost:11434"
else
    echo "⚠ Warning: API endpoint not responding"
fi

echo ""
echo "Available models:"
./bin/ollama list

echo ""
echo "=== API Usage Examples ==="
echo ""
echo "# Chat with gpt-oss:20b:"
echo "curl http://localhost:11434/api/generate -d '{"
echo '  "model": "gpt-oss:20b",'
echo '  "prompt": "Why is the sky blue?",'
echo '  "stream": false'
echo "}'"
echo ""
echo "# Generate embeddings with nomic-embed-text:"
echo "curl http://localhost:11434/api/embeddings -d '{"
echo '  "model": "nomic-embed-text",'
echo '  "prompt": "Hello world"'
echo "}'"
echo ""
echo "# List models via API:"
echo "curl http://localhost:11434/api/tags"
echo ""
echo "=== Server Control ==="
echo "Stop server: kill $OLLAMA_PID"
echo "View logs: tail -f ollama.log"
echo "API base URL: http://localhost:11434"
