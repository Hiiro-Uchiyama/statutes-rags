# statutes RAG System ドキュメント

日本の法令文書を対象としたRetrieval-Augmented Generation（RAG）システムの包括的ドキュメントです。

## ドキュメント構成

このディレクトリには、statutes RAGシステムの設計、実装、使用方法に関する詳細ドキュメントが含まれています。

### 1. [ARCHITECTURE.md](./ARCHITECTURE.md)
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

### 2. [USAGE.md](./USAGE.md)
**使用方法ガイド**

- セットアップ手順
  - uv環境セットアップ
  - MeCabセットアップ
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

### 3. [DEVELOPMENT.md](./DEVELOPMENT.md)
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

### 4. [TESTING.md](./TESTING.md)
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

## クイックスタート

### 1. 環境構築

```bash
# 仮想環境セットアップ
./setup/setup_uv_env.sh
source .venv/bin/activate

# MeCabセットアップ
./setup/setup_mecab.sh
source setup/mecab_env.sh

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
│   └── LLM (Ollama qwen2.5:7b)
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
| 民法Instruction | 128KB | Few-shot/Fine-tuning | `datasets/civil_law_instructions/` |
| 刑法試験問題 | 472KB | 高難度評価 | `datasets/criminal_law_exams/` |

## 技術スタック

- **言語**: Python 3.10+
- **フレームワーク**: LangChain
- **ベクトル検索**: FAISS
- **埋め込みモデル**: HuggingFace Transformers（intfloat/multilingual-e5-large）
- **キーワード検索**: rank-bm25
- **トークナイザ**: MeCab
- **LLM**: Ollama（qwen2.5:7b）
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
LLM_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.1

# Retriever
RETRIEVER_TYPE=hybrid  # vector/bm25/hybrid
RETRIEVER_TOP_K=10
USE_MMR=true

# Reranker
RERANKER_ENABLED=false
RERANKER_TOP_N=5
```

詳細は [USAGE.md](./USAGE.md) の「設定カスタマイズ」セクションを参照してください。

## パフォーマンス指標

### 処理時間

- **前処理**: 10,000ファイル → 約10分
- **インデックス構築**: 100,000文書 → 約50分（CPU、multilingual-e5-large使用）
- **検索**: クエリあたり1-3秒（Top-K=10、ハイブリッド検索）
- **LLM生成**: クエリあたり5-15秒（qwen2.5:7b、CPU）

### 評価メトリクス（lawqa_jp、50サンプル）

- **Faithfulness**: 0.82（回答の忠実性）
- **Answer Relevancy**: 0.79（回答の関連性）
- **Context Precision**: 0.85（コンテキストの精度）

注: 実際の値は設定やデータセットによって変動します。

## 今後のロードマップ

### 短期（1-3ヶ月）
- FastAPI Web API実装
- Qdrant統合
- ストリーミング応答

### 中期（3-6ヶ月）
- ファインチューニング（民法Instructionデータセット使用）
- マルチターン対話対応
- 評価データセット拡充（刑法試験問題）

### 長期（6ヶ月以上）
- マルチモーダル対応（図表解析）
- 判例検索統合
- 自動更新パイプライン

詳細は [DEVELOPMENT.md](./DEVELOPMENT.md) の「今後の開発方針」を参照してください。

## コントリビューション

開発に参加する場合は以下を確認してください:

1. [DEVELOPMENT.md](./DEVELOPMENT.md) のコーディング規約
2. [TESTING.md](./TESTING.md) のテスト作成ガイドライン
3. テストカバレッジ80%以上を維持

## トラブルシューティング

問題が発生した場合は以下を確認:

1. [USAGE.md](./USAGE.md) の「トラブルシューティング」セクション
2. [TESTING.md](./TESTING.md) の「トラブルシューティング」セクション

よくある問題:
- MeCabが見つからない → `source setup/mecab_env.sh`
- Ollamaに接続できない → Ollamaサーバーが起動しているか確認
- メモリ不足 → より小さいモデルを使用、またはドキュメント数を制限

## ライセンス

（プロジェクトのライセンスを記載）

## 連絡先

（開発者・メンテナーの連絡先を記載）

---

最終更新: 2025-10-30
