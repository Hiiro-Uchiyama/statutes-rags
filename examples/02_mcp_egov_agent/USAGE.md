# MCP e-Gov Agent - 使用方法ガイド

## クイックスタート

### 1. セットアップ

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 全ての依存関係を一度にインストール（推奨）
./setup/setup_uv_env.sh

# 仮想環境を有効化
source .venv/bin/activate
```

または、既存環境にExamples依存関係を追加:

```bash
cd /path/to/statutes-rags
source .venv/bin/activate

# Examples用の依存関係のみ追加
uv pip install httpx tenacity
```

### 2. API接続テスト

最小限の依存関係でAPI接続を確認:

```bash
# システムPythonで実行可能
python3 -m pip install httpx tenacity

cd examples/02_mcp_egov_agent
python3 demo.py --simple
```

### 3. 完全デモの実行

仮想環境内で全機能をテスト:

```bash
cd examples/02_mcp_egov_agent
python3 demo.py
```

## コマンドオプション

### デモスクリプト

```bash
# 完全デモ（全機能のチェック）
python3 demo.py

# 簡易デモ（API接続テストのみ）
python3 demo.py --simple
```

### 評価スクリプト

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `--dataset` | 評価データセットのパス（必須） | - |
| `--output` | 結果の出力先 | `results/evaluation_{timestamp}.json` |
| `--mode` | 評価モード | `api_preferred` |
| `--limit` | 評価する最大問題数 | なし（全問題） |

## 使用例

### 基本的な評価

```bash
cd examples/02_mcp_egov_agent

# テスト実行（最初の10問のみ）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/test_eval.json \
    --mode api_preferred \
    --limit 10
```

### 全問題の評価

```bash
# 全問題を評価（API優先モード）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/full_eval.json \
    --mode api_preferred
```

### 評価モード

| モード | 説明 | 用途 |
|--------|------|------|
| `api_preferred` | API優先、ローカルフォールバック | 推奨：本番想定 |
| `local_preferred` | ローカル優先 | オフライン環境 |
| `api_only` | API強制（フォールバックなし） | API性能評価 |
| `local_only` | ローカル強制（APIなし） | ローカル性能評価 |

## 環境変数による設定

### e-Gov API設定

```bash
export EGOV_API_BASE_URL="https://laws.e-gov.go.jp/api/2"
export EGOV_API_TIMEOUT=30
export EGOV_API_MAX_RETRIES=3
```

### ハイブリッド戦略

```bash
export MCP_PREFER_API=true              # API優先モード
export MCP_FALLBACK_TO_LOCAL=true       # ローカルフォールバック
export MCP_USE_API_FOR_RECENT=true      # 最近の法令はAPI優先
export MCP_RECENT_LAW_DAYS=90           # 最近と判定する日数
```

### LLM設定

```bash
# より軽量なモデルを使用（高速）
export LLM_MODEL=gpt-oss:7b
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 10

# より大規模なモデルを使用（高精度）
export LLM_MODEL=qwen3:8b
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 10

# タイムアウト設定
export LLM_TIMEOUT=60
```

### 検索設定

```bash
export MCP_RETRIEVAL_TOP_K=10
export MCP_RERANK_TOP_N=5
```

## Python APIとして使用

```python
import importlib

# モジュールのインポート（数字で始まるため動的インポート）
mcp_module = importlib.import_module('examples.02_mcp_egov_agent')
MCPEgovPipeline = mcp_module.MCPEgovPipeline
load_config = mcp_module.load_config

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever

# 設定のロード
config = load_config(validate=False)

# ローカルRetrieverの初期化
vector_retriever = VectorRetriever(
    embedding_model="intfloat/multilingual-e5-large",
    index_path="data/faiss_index/vector"
)
bm25_retriever = BM25Retriever(
    index_path="data/faiss_index/bm25"
)
retriever = HybridRetriever(vector_retriever, bm25_retriever)

# パイプラインの作成
pipeline = MCPEgovPipeline(config=config, retriever=retriever)

# 質問
result = pipeline.query("個人情報保護法の最新規定について教えてください")

print(f"回答: {result['answer']}")
print(f"データソース: {result['source']}")  # "api", "local", "hybrid"
```

### データソースの強制指定

```python
# API優先で検索
result = pipeline.query(
    "令和5年の改正について",
    force_api=True
)

# ローカルのみで検索
result = pipeline.query(
    "民法第1条について",
    force_local=True
)
```

## テストの実行

### API接続テスト

```bash
# 簡易テスト
python3 tests/test_api_simple.py

# 詳細テスト
python3 tests/test_api_connection.py
```

### パイプラインテスト

```bash
python3 tests/test_pipeline.py
```

### pytestによる実行

```bash
# すべてのテストを実行
pytest tests/ -v

# カバレッジ付きで実行
pytest tests/ --cov=agents --cov-report=html
```

## ベンチマーク評価

### 標準的な評価方法（推奨）

プロジェクトルートから統一されたベンチマーク評価を実行:

```bash
cd /path/to/statutes-rags

# MCPエージェント評価（50サンプル、API優先モード）
python scripts/evaluate_mcp_benchmark.py --samples 50 --mode api_preferred

# Vector-basedとの比較評価
./scripts/run_benchmark_comparison.sh 50
```

### 比較結果の確認

```bash
# 比較レポートの確認
cat benchmark_comparison.json | python3 -m json.tool | less

# 主要指標のみ抽出
cat benchmark_comparison.json | python3 -m json.tool | grep -A 5 '"comparison"'
```

## トラブルシューティング

### 依存関係のエラー

```bash
# 必要なパッケージを再インストール
uv pip install httpx tenacity pydantic
```

### API接続エラー

```bash
# API疎通確認
python3 -c "
import httpx
response = httpx.get('https://laws.e-gov.go.jp/api/2/laws')
print(f'Status: {response.status_code}')
"
```

### データが見つからない

```bash
# データパスの確認
ls -la data/egov_laws.jsonl
ls -la data/faiss_index/

# データがない場合は再ビルド
cd /path/to/statutes-rags
python scripts/build_index.py
```

### ModuleNotFoundError

```bash
# 仮想環境を使用してください
source .venv/bin/activate
```

### Python環境の問題

```bash
# 仮想環境を再作成
deactivate
rm -rf .venv
./setup/setup_uv_env.sh
source .venv/bin/activate
cd examples/02_mcp_egov_agent
```

## ワンライナーコマンド集

### テスト評価（10問、API優先）

```bash
cd examples/02_mcp_egov_agent && \
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 10 --mode api_preferred --output results/quick_test.json
```

### 本番評価（全問題、標準設定）

```bash
cd examples/02_mcp_egov_agent && \
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --mode api_preferred --output results/full_evaluation_$(date +%Y%m%d_%H%M%S).json
```

### デバッグモード（詳細ログ付き）

```bash
cd examples/02_mcp_egov_agent && \
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 1 2>&1 | tee debug.log
```

## 評価結果の形式

評価結果は以下の形式で出力されます:

```json
{
  "accuracy": 0.75,
  "correct_count": 3,
  "total": 4,
  "api_call_success_rate": 0.9,
  "api_usage_rate": 0.8,
  "results": [
    {
      "question_index": 0,
      "question": "個人情報保護法第24条に...",
      "choices": ["a ...", "b ...", "c ...", "d ..."],
      "correct_answer": "b",
      "predicted_answer": "b",
      "is_correct": true,
      "raw_answer": "回答の全文...",
      "data_source": "api",
      "metadata": {
        "api_called": true,
        "api_success": true,
        "fallback_used": false
      }
    }
  ],
  "timestamp": "2025-11-06T01:30:00.123456"
}
```

## 次のステップ

- 詳細な実装情報: [README.md](README.md)
- 動作確認結果: README.md の最後のセクション
- プロジェクト全体のドキュメント: [../../docs/](../../docs/)
