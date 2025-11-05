# MCP e-Gov Agent - e-Gov API連携エージェント

e-Gov API v2を活用した動的法令検索エージェントです。最新の法令データを動的に取得し、ローカルデータと組み合わせたハイブリッド検索を実現します。

## 目次

- [概要](#概要)
- [アーキテクチャ](#アーキテクチャ)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [評価](#評価)
- [主要コンポーネント](#主要コンポーネント)
- [ディレクトリ構成](#ディレクトリ構成)
- [トラブルシューティング](#トラブルシューティング)

**使用方法の詳細**: [USAGE.md](USAGE.md) を参照してください。

## 概要

### 主な特徴

- e-Gov API v2との統合による最新法令データの取得
- API優先のハイブリッド検索戦略
- ローカルデータへの自動フォールバック
- 質問の種類に応じた動的なデータソース選択
- タイムアウトとリトライによる堅牢なエラーハンドリング

### アーキテクチャ

```
┌─────────────┐
│   質問入力  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  MCPEgovAgent   │ ← 質問を分析し、データソースを選択
└────┬────────┬───┘
     │        │
     ▼        ▼
┌────────┐ ┌──────────┐
│ e-Gov  │ │ ローカル │
│  API   │ │ データ   │
└────┬───┘ └────┬─────┘
     │          │
     └────┬─────┘
          ▼
    ┌──────────┐
    │   LLM    │
    └────┬─────┘
         ▼
    ┌──────────┐
    │   回答   │
    └──────────┘
```

## セットアップ

### 前提条件

- Python >= 3.10
- 既存のRAGシステムがセットアップ済み
- インターネット接続（e-Gov API使用時）

### 方法1: 完全セットアップ（推奨）

```bash
cd /path/to/statutes-rags

# 全ての依存関係を一度にインストール
./setup/setup_uv_env.sh

# 仮想環境を有効化
source .venv/bin/activate
```

これで、基本のRAGシステム + Examples用の依存関係（langgraph, tenacity等）が全てインストールされます。

### 方法2: 既存環境にExamples依存関係を追加

既に基本セットアップが完了している場合:

```bash
cd /path/to/statutes-rags
source .venv/bin/activate

# Examples用の依存関係のみ追加
uv pip install httpx tenacity
```

### 方法3: 最小限（API接続テストのみ）

```bash
# システムPythonで実行可能
python3 -m pip install httpx tenacity

cd examples/02_mcp_egov_agent
python3 demo.py --simple
```

### デモを実行

```bash
cd examples/02_mcp_egov_agent

# 完全デモ（仮想環境内で全機能をテスト）
python3 demo.py

# 簡易デモ（最小依存でAPI接続テストのみ）
python3 demo.py --simple
```

**デモモード:**
- `demo.py`: 完全デモ - 依存関係の確認、API接続テスト、設定確認、データパス確認
- `demo.py --simple`: 簡易デモ - 最小限の依存関係でAPI接続テストのみ実行

## 使用方法

### e-Gov API接続テスト

```bash
python3 tests/test_api_simple.py
```

このテストは以下を確認します:
- 法令一覧取得
- キーワード検索
- 法令本文取得
- エラーハンドリング

### Python APIとして使用

```python
# 数字で始まるモジュール名のため、動的インポートを使用
import importlib
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

## 設定

環境変数で動作をカスタマイズできます:

```bash
# e-Gov API設定
export EGOV_API_BASE_URL="https://laws.e-gov.go.jp/api/2"
export EGOV_API_TIMEOUT=30
export EGOV_API_MAX_RETRIES=3

# ハイブリッド戦略
export MCP_PREFER_API=true              # API優先モード
export MCP_FALLBACK_TO_LOCAL=true       # ローカルフォールバック
export MCP_USE_API_FOR_RECENT=true      # 最近の法令はAPI優先
export MCP_RECENT_LAW_DAYS=90           # 最近と判定する日数

# LLM設定
export LLM_MODEL="qwen3:8b"
export LLM_TEMPERATURE=0.1
export LLM_TIMEOUT=60

# 検索設定
export MCP_RETRIEVAL_TOP_K=10
export MCP_RERANK_TOP_N=5
```

## 評価

### 標準的な評価方法（推奨）

プロジェクトルートから統一されたベンチマーク評価を実行：

```bash
cd /path/to/statutes-rags

# MCPエージェント評価（50サンプル、API優先モード）
python scripts/evaluate_mcp_benchmark.py --samples 50 --mode api_preferred

# Vector-basedとの比較評価
./scripts/run_benchmark_comparison.sh 50
```

**利点**:
- 既存のVector-based評価と同じフォーマットで結果を出力
- 自動的に比較レポートを生成
- プロジェクト全体のベンチマークとして管理可能

### 個別評価（開発・デバッグ用）

MCPエージェント固有の詳細評価：

```bash
cd examples/02_mcp_egov_agent

# 全問題を評価
python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --output results.json \
  --mode api_preferred

# 最初の10問のみ評価（テスト用）
python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --output results.json \
  --mode api_preferred \
  --limit 10
```

### 評価モードの説明

| モード | 説明 | 用途 |
|--------|------|------|
| `api_preferred` | API優先、ローカルフォールバック | 推奨：本番想定 |
| `local_preferred` | ローカル優先 | オフライン環境 |
| `api_only` | API強制（フォールバックなし） | API性能評価 |
| `local_only` | ローカル強制（APIなし） | ローカル性能評価 |

### ベンチマーク比較結果の確認

```bash
# 比較レポートの確認
cat benchmark_comparison.json | python3 -m json.tool | less

# 主要指標のみ抽出
cat benchmark_comparison.json | python3 -m json.tool | grep -A 5 '"comparison"'
```

### 評価指標

**共通指標**（Vector-basedと比較可能）:
- 正答率（4択問題）
- 正答数
- 平均応答時間
- エラー率

**MCP固有指標**:
- API呼び出し成功率
- API使用率
- データソース別の精度比較（API/ローカル/ハイブリッド）
- API経由での正答率

## 主要コンポーネント

### EGovAPIClient

`agents/egov_client.py` - e-Gov API v2との通信を管理するクライアント。

```python
# 数字で始まるモジュール名のため、動的インポートを使用
import importlib
mcp_module = importlib.import_module('examples.02_mcp_egov_agent')
EGovAPIClient = mcp_module.EGovAPIClient

client = EGovAPIClient()

# キーワード検索
result = client.search_by_keyword("個人情報保護")

# 法令本文取得
law_data = client.get_law_data("平成二十八年個人情報保護委員会規則第六号")

# ヘルスチェック
if client.health_check():
    print("API接続OK")
```

### MCPEgovAgent

`agents/mcp_agent.py` - 質問を分析し、適切なデータソースを選択するエージェント。

- 最近の法令に関する質問はAPI優先
- 法令番号が含まれる場合は直接取得を試行
- APIエラー時はローカルデータにフォールバック

### MCPEgovPipeline

`pipeline.py` - ハイブリッド検索とLLM応答生成を統合したパイプライン。

### Tools

`agents/tools.py` - e-Gov APIとローカル検索のLangChain Tools定義。

## API仕様

使用しているe-Gov API v2のエンドポイント:

- `GET /keyword` - キーワード検索
- `GET /law_data/{law_id_or_num}` - 法令本文取得
- `GET /laws` - 法令一覧取得

詳細: https://laws.e-gov.go.jp/api/2/swagger-ui

## エラーハンドリング

- タイムアウト: デフォルト30秒、3回までリトライ
- HTTPエラー: 適切なエラーメッセージを返却
- API障害: 自動的にローカルデータにフォールバック

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

### APIに接続できない

```python
import importlib
mcp_module = importlib.import_module('examples.02_mcp_egov_agent')
EGovAPIClient = mcp_module.EGovAPIClient

client = EGovAPIClient()
if client.health_check():
    print("API接続OK")
else:
    print("API接続失敗 - ローカルモードを使用してください")
```

### ローカルフォールバックが動作しない

- `MCP_FALLBACK_TO_LOCAL=true` が設定されているか確認
- Retrieverが正しく初期化されているか確認

### Python環境の問題

```bash
# 仮想環境を再作成
deactivate
rm -rf .venv
./setup/setup_uv_env.sh
source .venv/bin/activate
cd examples/02_mcp_egov_agent
```

## ディレクトリ構成

```
02_mcp_egov_agent/
├── __init__.py                   # パッケージ定義
├── README.md                     # このファイル
├── USAGE.md                      # 使用方法ガイド
├── config.py                     # 設定管理
├── pipeline.py                   # RAGパイプライン
├── evaluate.py                   # 評価スクリプト
├── demo.py                       # デモスクリプト（--simpleで簡易モード）
├── agents/                       # エージェントコンポーネント
│   ├── __init__.py
│   ├── egov_client.py            # e-Gov APIクライアント
│   ├── mcp_agent.py              # ハイブリッド検索エージェント
│   └── tools.py                  # LangChain Tools
└── tests/                        # テストディレクトリ
    ├── __init__.py
    ├── conftest.py
    ├── test_api_simple.py        # API接続テスト
    ├── test_api_connection.py    # 詳細APIテスト
    └── test_pipeline.py          # パイプライン統合テスト
```

## 制約事項

- e-Gov APIの試行版機能（JSON形式）を使用しているため、仕様変更の可能性あり
- レート制限は明示されていないが、常識的な範囲で使用すること
- ローカルデータとAPIデータの完全な整合性は保証されない

## 次のステップ

1. **ドキュメントを読む**
   - このREADME.mdで全体概要を把握
   
2. **実験する**
   - 異なる質問で試す
   - データソース戦略を変更する
   - 設定をカスタマイズする

3. **評価を実施**
   - 全評価データセットで評価
   - ベースライン（既存RAG）と比較
   - 結果を分析

4. **改善する**
   - キャッシュ機構を追加
   - 並列検索を実装
   - スコアリングを改善

## テスト

```bash
cd examples/02_mcp_egov_agent

# API接続テスト
python3 tests/test_api_simple.py

# 詳細API接続テスト
python3 tests/test_api_connection.py

# pytest実行（全テスト）
pytest tests/ -v
```

## 動作確認済み

このサンプルは以下の環境で動作確認済みです：

### 動作確認日
2025年1月（e-Gov API v2使用）

### テスト結果
- API接続テスト（demo.py --simple）: 成功
- 詳細APIテスト（tests/test_api_simple.py）: 全テスト成功
- 完全接続テスト（tests/test_api_connection.py）: 全テスト成功
- フルデモ（demo.py）: 成功
- パイプラインテスト（tests/test_pipeline.py）: 成功

### 主な修正点
- **API応答キーの修正**: e-Gov API v2の応答キーが`laws`から`items`に変更されていたため、全ファイルで修正
- **LangChainインポートの修正**: `langchain.tools`から`langchain_core.tools`に変更し、バージョン互換性を改善

### 確認されたAPI機能
- キーワード検索（/keyword）: 正常動作、検索結果取得確認
- 法令一覧取得（/laws）: 正常動作
- 法令本文取得（/law_data）: 正常動作
- エラーハンドリング: 404エラーなど適切に処理

### 依存関係
```bash
# 必須
httpx>=0.25.0
tenacity>=8.2.0
pydantic>=2.5.0

# RAG統合時に必要
langchain>=0.1.0
langchain-core>=1.0.0
langgraph>=0.2.0
```

## サポート

問題が発生した場合:

1. **仮想環境を使用**: `source .venv/bin/activate`
2. デモスクリプトを実行して診断: `python3 demo.py`
3. 簡易デモでAPI接続確認: `python3 demo.py --simple`
4. API接続テストを実行: `python3 tests/test_api_simple.py`
5. パイプラインテスト: `python3 tests/test_pipeline.py`
6. ログを確認（ログレベルを上げる）

### よくある問題

**Q: `ModuleNotFoundError: No module named 'langchain_core'`**
```bash
# 仮想環境を使用してください
source .venv/bin/activate
```

**Q: キーワード検索で0件の結果**
- この問題は修正済みです。最新のコードを使用してください。
- API応答キーが`items`に正しく設定されているか確認してください。

**Q: "law_title" が "不明" と表示される**
- これはAPI応答のキー構造の違いによるものです。
- `revision_info`内の`law_title`を参照する必要があります（今後の改善予定）。
