# statutes RAG System ドキュメント

日本の法令文書を対象としたRetrieval-Augmented Generation（RAG）システムの包括的ドキュメントです。

## ドキュメント構成

このディレクトリには、statutes RAGシステムの設計、実装、使用方法に関する詳細ドキュメントが含まれています。

### 1. [05-ARCHITECTURE.md](./05-ARCHITECTURE.md)
**システムアーキテクチャと設計ドキュメント**

- システム概要と主要機能
- アーキテクチャ設計（ディレクトリ構造、コンポーネント図）
- コアモジュール詳細
  - 設定管理（RAGConfig）
  - 抽象インターフェース（BaseRetriever, BaseReranker）
  - ベクトル検索（FAISS）
  - BM25検索
  - ハイブリッド検索
  - リランキング（Cross-encoder）
  - RAGパイプライン
- データフロー（前処理、インデックス構築、検索、評価）
- 技術スタック（依存パッケージ、環境変数）

**対象読者**: 開発者、システム設計者、アーキテクト

### 2. [03-USAGE.md](./03-USAGE.md)
**使用方法ガイド**

- セットアップ手順
  - uv環境セットアップ
  - 日本語トークナイザー（SudachiPy - 自動インストール済み）
  - Ollamaセットアップ
  - 環境変数設定
- 基本的な使い方
  - ステップ1: 前処理（XML → JSONL）
  - ステップ2: インデックス構築
  - ステップ3: 質問応答（対話型・単発クエリ）
  - ステップ4: 評価（RAGAS）
- Makefileコマンド一覧
- スクリプト詳細
- 設定カスタマイズ
- トラブルシューティング
- パフォーマンスチューニング

**対象読者**: ユーザー、オペレーター、初学者

### 3. [06-DEVELOPMENT.md](./06-DEVELOPMENT.md)
**開発ガイド**

- 開発環境セットアップ
- コーディング規約（PEP 8、型ヒント、Docstring）
- コードフォーマット（black、ruff）
- テスト戦略
  - テスト階層（ユニット、統合、評価）
  - pytestマーカー
  - テストフィクスチャ
  - テスト作成ガイドライン
- 今後の開発方針
  - 短期目標（FastAPI Web API、Qdrant統合、ストリーミング応答）
  - 中期目標（ファインチューニング、マルチターン対話、評価データセット拡充）
  - 長期目標（マルチモーダル対応、判例検索、自動更新パイプライン）
- 拡張ポイント
  - 新しいRetrieverの追加方法
  - 新しいRerankerの追加方法
  - 新しい評価メトリクスの追加方法
- デバッグガイド（ログ出力、診断スクリプト、プロファイリング）

**対象読者**: 開発者、コントリビューター

### 4. [04-TESTING.md](./04-TESTING.md)
**テストドキュメント**

- テスト概要（3層のテスト階層）
- テスト環境（pytest設定、前提条件）
- テストファイル詳細
  - tests/conftest.py（フィクスチャ定義）
  - tests/test_config.py（設定管理テスト）
  - tests/test_preprocessing.py（XML前処理テスト）
  - tests/test_retrieval.py（Retrieverテスト）
  - tests/test_rag_pipeline.py（RAGパイプラインテスト）
  - tests/test_rag_components.py（統合テスト）
- テスト実行方法
  - 基本実行
  - Makefile経由
  - マーカー指定
  - 並列実行
- カバレッジレポート（計測、HTML生成、目標値）
- テストデータ（サンプルデータ、一時ファイル）
- トラブルシューティング

**対象読者**: テスター、QAエンジニア、開発者

### 5. [07-ALGORITHM.md](./07-ALGORITHM.md)
**アルゴリズム詳細ガイド**

- システムアーキテクチャ概要
- 文書前処理とチャンキング
- ベクトル検索アルゴリズム（FAISS、MMR）
- BM25検索アルゴリズム（日本語トークナイザー）
- ハイブリッド検索とスコア統合（RRF、重み付き統合）
- Rerankerアルゴリズム（Cross-Encoder）
- RAGパイプライン全体フロー
- パラメータチューニングガイド

**対象読者**: アルゴリズム研究者、実装者、上級開発者

### 6. [08-CODE-REFERENCE.md](./08-CODE-REFERENCE.md)
**コードリファレンス**

- ディレクトリ構造詳細
- コアモジュール（app/）の実装詳細
- スクリプト（scripts/）の使用方法
- テスト（tests/）の構成
- 設定ファイルの解説
- 主要クラスのAPIリファレンス

**対象読者**: 開発者、コントリビューター、実装詳細を知りたい方

## クイックスタート

### 1. 環境構築

```bash
# 仮想環境セットアップ
./setup/setup_uv_env.sh
source .venv/bin/activate

# 日本語トークナイザーの確認（自動インストール済み）
python -c "from app.retrieval.bm25_retriever import BM25Retriever; r = BM25Retriever(); print(f'使用中: {r.tokenizer_type}')"

# Ollamaセットアップ
cd setup && ./setup_ollama.sh && cd ..
```

### 2. データ処理

```bash
# 前処理（XML → JSONL）
make preprocess

# インデックス構築
make index
```

### 3. 質問応答

```bash
# 対話型モード
make qa

# または単発クエリ
make query Q="博物館法の目的は何ですか？"
```

### 4. テスト実行

```bash
# ユニットテスト
make test

# カバレッジ付き
make test-coverage
```

## システム構成図

```
statutes RAG System
├── データ準備
│   ├── e-Gov法令XML (datasets/egov_laws/)
│   └── 評価データセット (datasets/lawqa_jp/)
│
├── 前処理パイプライン
│   ├── XML解析 (scripts/preprocess_egov_xml.py)
│   └── JSONL生成 (data/egov_laws.jsonl)
│
├── インデックス構築
│   ├── ベクトルインデックス (FAISS) → data/faiss_index/vector/
│   ├── BM25インデックス → data/faiss_index/bm25/
│   └── ハイブリッドインデックス
│
├── RAGパイプライン
│   ├── Retriever (Vector/BM25/Hybrid)
│   ├── Reranker (Cross-encoder, オプション)
│   └── LLM (Ollama qwen3:8b)
│
├── インターフェース
│   ├── CLI (scripts/query_cli.py)
│   └── Web API (将来実装予定)
│
└── 評価
    ├── RAGAS (scripts/evaluate_ragas.py)
    └── カスタムメトリクス
```

## 主要コンポーネント

| コンポーネント | ファイルパス | 説明 |
|--------------|------------|------|
| 設定管理 | `app/core/rag_config.py` | Pydantic BaseModelによる型安全な設定 |
| ベクトル検索 | `app/retrieval/vector_retriever.py` | FAISSベースの密ベクトル検索 |
| BM25検索 | `app/retrieval/bm25_retriever.py` | BM25Okapiによるキーワード検索 |
| ハイブリッド検索 | `app/retrieval/hybrid_retriever.py` | ベクトル+BM25の統合検索 |
| リランキング | `app/retrieval/reranker.py` | Cross-encoderによる再スコアリング |
| RAGパイプライン | `app/retrieval/rag_pipeline.py` | エンドツーエンドRAG処理 |
| XML前処理 | `scripts/preprocess_egov_xml.py` | e-Gov法令XMLの構造化パース |
| インデックス構築 | `scripts/build_index.py` | ベクトル/BM25インデックス生成 |
| 質問応答CLI | `scripts/query_cli.py` | 対話型・単発クエリインターフェース |
| RAGAS評価 | `scripts/evaluate_ragas.py` | RAGシステム評価フレームワーク |

## データセット

| データセット | サイズ | 用途 | 場所 |
|------------|--------|------|------|
| e-Gov法令XML | 264MB | メインコーパス | `datasets/egov_laws/` |
| lawqa_jp | 4.9MB | RAG評価 | `datasets/lawqa_jp/` |
| 民法Instruction（任意） | 128KB | Few-shot/Fine-tuning | `datasets/civil_law_instructions/` |
| 刑法試験問題（任意） | 472KB | 高難度評価 | `datasets/criminal_law_exams/` |

## 技術スタック

- **言語**: Python 3.10+
- **フレームワーク**: LangChain
- **ベクトル検索**: FAISS
- **埋め込みモデル**: HuggingFace Transformers（intfloat/multilingual-e5-large）
- **キーワード検索**: rank-bm25
- **トークナイザ**: SudachiPy（デフォルト、管理者権限不要）
- **LLM**: Ollama（qwen3:8b）
- **リランキング**: sentence-transformers（Cross-encoder）
- **評価**: RAGAS
- **テスト**: pytest
- **設定管理**: Pydantic

## 設定項目

主要な環境変数（`.env`ファイルで設定）:

```bash
# 埋め込みモデル
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024

# LLM
LLM_MODEL=qwen3:8b
LLM_TEMPERATURE=0.1

# Retriever
RETRIEVER_TYPE=hybrid  # vector/bm25/hybrid
RETRIEVER_TOP_K=10
USE_MMR=true

# Reranker
RERANKER_ENABLED=false
RERANKER_TOP_N=5
```

詳細は [03-USAGE.md](./03-USAGE.md) の「設定カスタマイズ」セクションを参照してください。

## トラブルシューティング

問題が発生した場合は以下を確認:

1. [03-USAGE.md](./03-USAGE.md) の「トラブルシューティング」セクション
2. [04-TESTING.md](./04-TESTING.md) の「トラブルシューティング」セクション

---

最終更新: 2024-11-04
