# MCP e-Gov Agent - セットアップガイド

このガイドでは、MCP e-Gov Agentを実行するための環境構築手順を説明します。

## 目次

- [前提条件](#前提条件)
- [セットアップ手順](#セットアップ手順)
- [動作確認](#動作確認)
- [トラブルシューティング](#トラブルシューティング)

## 前提条件

### システム要件

- **OS**: Linux / macOS / Windows (WSL)
- **Python**: 3.10以上
- **メモリ**: 最低8GB（16GB以上推奨）
- **ディスク容量**: 10GB以上の空き容量
- **ネットワーク**: インターネット接続（e-Gov API使用時）

### 必要なソフトウェア

1. **Python 3.10+**
   ```bash
   python3 --version  # 3.10以上であることを確認
   ```

2. **uv** (高速Pythonパッケージマネージャー)
   ```bash
   # uvのインストール（未インストールの場合）
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # インストール確認
   uv --version
   ```

3. **Ollama** (ローカルLLM実行環境)
   ```bash
   # Ollamaのインストール
   curl -fsSL https://ollama.com/install.sh | sh
   
   # インストール確認
   ollama --version
   ```

## セットアップ手順

### ステップ1: プロジェクトのクローン

```bash
# 既にクローン済みの場合はスキップ
cd /path/to/workspace
git clone <repository-url> statutes-rags
cd statutes-rags
```

### ステップ2: 環境設定ファイルの準備

```bash
# .env.exampleをコピーして.envを作成
cp .env.example .env

# .envファイルを編集（必要に応じて）
# デフォルト設定のままでも動作します
```

### ステップ3: Ollamaモデルのセットアップ

```bash
# Ollamaサービスの起動
ollama serve &

# 推奨モデルのダウンロード
ollama pull qwen3:8b

# 動作確認
ollama run qwen3:8b "こんにちは"
```

**別のモデルを使用する場合**:
```bash
# 軽量モデル（高速、精度は中程度）
ollama pull gpt-oss:7b

# 大規模モデル（高精度、低速）
ollama pull qwen3:14b
```

### ステップ4: Python仮想環境のセットアップ

#### 方法A: 全自動セットアップ（推奨）

```bash
# プロジェクトルートで実行
./setup/setup_uv_env.sh
```

このスクリプトは以下を自動的に実行します：
- Python仮想環境の作成
- 基本RAGシステムの依存関係インストール
- Examples用の依存関係インストール（langgraph, httpx, tenacity等）

#### 方法B: 手動セットアップ

```bash
# 仮想環境の作成
uv venv

# 仮想環境の有効化
source .venv/bin/activate

# 依存関係のインストール
uv pip install -e .
uv pip install httpx tenacity tqdm python-dotenv
```

### ステップ5: 仮想環境の有効化

```bash
source .venv/bin/activate

# 仮想環境が有効化されたことを確認
which python  # .venv/bin/python を指していることを確認
```

### ステップ6: データインデックスの構築

```bash
# プロジェクトルートで実行
python scripts/build_index.py
```

このコマンドは以下を実行します：
- 法令データのダウンロード（未取得の場合）
- ベクトルインデックスの構築
- BM25インデックスの構築

**注意**: 初回実行時は時間がかかります（10-30分程度）。

### ステップ7: API接続テスト（簡易）

```bash
cd examples/02_mcp_egov_agent
python3 demo.py --simple
```

**期待される出力**:
```
e-Gov API v2 接続テスト
テスト1: 法令一覧取得
結果: XXX件の法律を取得しました
テスト1: OK

テスト2: キーワード検索
キーワード: '個人情報保護'
結果: XX件の法令が見つかりました
テスト2: OK
```

### ステップ8: 完全デモの実行

```bash
# 仮想環境内で実行
python3 demo.py
```

**期待される出力**:
```
デモ1: e-Gov APIへの接続テスト
1. API疎通確認...
   結果: OK - APIに正常に接続できました

2. キーワード検索テスト...
   結果: XX件の法令が見つかりました

デモ2: 設定の確認
設定を読み込みました:
  API URL: https://laws.e-gov.go.jp/api/2
  ...

デモ3: データパスの確認
データパス:
  ベクトルストア: /path/to/data/faiss_index
    存在: Yes
```

### ステップ9: ユニットテストの実行

```bash
# API接続テスト
python3 tests/test_api_simple.py

# 詳細APIテスト
python3 tests/test_api_connection.py

# パイプラインテスト
python3 tests/test_pipeline.py

# pytestで全テスト実行
pytest tests/ -v
```

## 動作確認

すべてのセットアップが完了したら、以下のコマンドで動作確認を行います：

### 最小限の評価テスト（10問）

```bash
cd examples/02_mcp_egov_agent

python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --output results/quick_test.json \
  --mode api_preferred \
  --limit 10
```

**期待される出力**:
```
評価結果
全体正答率: XX.XX%
総問題数: 10
正答数: X
平均応答時間: X.XX秒

データソース別正答率:
  api: XX.XX% (X/X)
  local: XX.XX% (X/X)
  hybrid: XX.XX% (X/X)
```

### エンドツーエンドテスト

Python APIとして使用:
```python
import importlib
import sys
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path.cwd().parent.parent))

# モジュールのインポート
mcp_module = importlib.import_module('examples.02_mcp_egov_agent')
MCPEgovPipeline = mcp_module.MCPEgovPipeline
load_config = mcp_module.load_config

from app.core.rag_config import load_config as load_base_config
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever

# 設定のロード
base_config = load_base_config()
mcp_config = load_config(validate=False)

# Retrieverの初期化
vector_retriever = VectorRetriever(
    embedding_model=base_config.embedding.model_name,
    index_path="../../data/faiss_index/vector"
)
bm25_retriever = BM25Retriever(
    index_path="../../data/faiss_index/bm25"
)
retriever = HybridRetriever(vector_retriever, bm25_retriever)

# パイプラインの作成
pipeline = MCPEgovPipeline(config=mcp_config, retriever=retriever)

# 質問
result = pipeline.query("個人情報保護法の目的について教えてください")

print(f"回答: {result['answer']}")
print(f"データソース: {result['source']}")
print(f"引用数: {len(result['citations'])}")
```

## トラブルシューティング

### 問題1: uvが見つからない

```bash
# uvを再インストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# パスを追加（bashの場合）
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 問題2: Ollamaが起動しない

```bash
# Ollamaのステータス確認
systemctl status ollama  # systemdを使用している場合

# 手動起動
ollama serve

# ポート確認
curl http://localhost:11434/api/version
```

### 問題3: 依存関係のエラー

```bash
# 仮想環境の再作成
deactivate
rm -rf .venv
./setup/setup_uv_env.sh
source .venv/bin/activate

# 個別に依存関係を再インストール
uv pip install httpx tenacity tqdm python-dotenv
```

### 問題4: API接続エラー

```bash
# ネットワーク接続確認
ping laws.e-gov.go.jp

# curlでAPI確認
curl "https://laws.e-gov.go.jp/api/2/laws?law_type=Act"

# プロキシ設定（必要な場合）
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### 問題5: データが見つからない

```bash
# データパスの確認
ls -la data/egov_laws.jsonl
ls -la data/faiss_index/

# インデックスの再構築
cd /path/to/statutes-rags
python scripts/build_index.py
```

### 問題6: ModuleNotFoundError

```bash
# 仮想環境が有効化されているか確認
which python  # .venv/bin/python を指すべき

# 仮想環境の有効化
source .venv/bin/activate

# プロジェクトルートから実行しているか確認
pwd  # /path/to/statutes-rags であるべき
```

### 問題7: メモリ不足エラー

```bash
# より軽量なモデルを使用
export LLM_MODEL="gpt-oss:7b"

# バッチサイズを小さく
# evaluate.pyに --limit 5 を追加
```

### 問題8: LLMタイムアウト

```bash
# タイムアウト時間を延長
export LLM_TIMEOUT=120

# 別のモデルを試す
ollama pull qwen3:8b
export LLM_MODEL="qwen3:8b"
```

## 環境変数リファレンス

主要な環境変数の一覧:

```bash
# e-Gov API設定
export EGOV_API_BASE_URL="https://laws.e-gov.go.jp/api/2"
export EGOV_API_TIMEOUT=30
export EGOV_API_MAX_RETRIES=3

# ハイブリッド戦略
export MCP_PREFER_API=true
export MCP_FALLBACK_TO_LOCAL=true
export MCP_USE_API_FOR_RECENT=true
export MCP_RECENT_LAW_DAYS=90

# LLM設定
export LLM_MODEL="qwen3:8b"
export LLM_TEMPERATURE=0.1
export LLM_TIMEOUT=60
export OLLAMA_HOST="http://localhost:11434"

# 検索設定
export MCP_RETRIEVAL_TOP_K=10
export MCP_RERANK_TOP_N=5

# データパス
export VECTOR_STORE_PATH="data/faiss_index"
export DATA_PATH="data/egov_laws.jsonl"
```

## 次のステップ

セットアップが完了したら、[USAGE.md](USAGE.md)を参照して以下を実行してください：

1. **評価実験の実行**
   ```bash
   python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 10
   ```

2. **ベンチマーク比較**
   ```bash
   cd /path/to/statutes-rags
   python scripts/evaluate_mcp_benchmark.py --samples 50
   ```

3. **詳細な使用方法**
   - [USAGE.md](USAGE.md) - コマンドオプション、Python API、評価方法
   - [README.md](README.md) - アーキテクチャ、コンポーネント詳細

## サポート

問題が解決しない場合:

1. ログレベルを上げて詳細情報を取得:
   ```bash
   export LOG_LEVEL=DEBUG
   python demo.py 2>&1 | tee debug.log
   ```

2. システム情報を確認:
   ```bash
   python3 --version
   uv --version
   ollama --version
   which python
   ```

3. Issue報告時に以下を含める:
   - Python/uv/Ollamaのバージョン
   - エラーメッセージ全文
   - 実行したコマンド
   - debug.logの内容
