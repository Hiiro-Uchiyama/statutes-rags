# statutes RAG アーキテクチャドキュメント

このドキュメントでは、statutes-ragプロジェクトのコードベース構造、各モジュールの役割、データフローを詳しく説明します。

## 目次

1. [プロジェクト構造](#プロジェクト構造)
2. [コアモジュール](#コアモジュール)
3. [データフロー](#データフロー)
4. [設定管理](#設定管理)
5. [拡張方法](#拡張方法)

## プロジェクト構造

```
statutes-rags/
├── app/                        # メインアプリケーションコード
│   ├── core/                   # コア設定
│   │   └── rag_config.py       # 全体設定管理
│   └── retrieval/              # 検索・RAG関連
│       ├── base.py             # 基底クラス定義
│       ├── vector_retriever.py # ベクトル検索
│       ├── bm25_retriever.py   # BM25検索
│       ├── hybrid_retriever.py # ハイブリッド検索
│       ├── reranker.py         # リランカー
│       └── rag_pipeline.py     # RAGパイプライン
│
├── scripts/                    # 実行スクリプト
│   ├── preprocess_egov_xml.py  # XML→JSONL前処理
│   ├── build_index.py          # インデックス構築
│   ├── query_cli.py            # 対話型CLI
│   ├── evaluate_multiple_choice.py  # 4択評価
│   ├── evaluate_ragas.py       # RAGAS評価
│   ├── run_tests.sh            # テスト実行
│   └── test_retrieval.py       # 検索テスト
│
├── setup/                      # セットアップスクリプト
│   ├── setup_uv_env.sh         # Python環境セットアップ
│   ├── setup_ollama.sh         # Ollamaセットアップ
│   └── restore_env.sh          # 環境復元スクリプト（再起動後）
│
├── tests/                      # テストコード
│   ├── conftest.py             # pytest設定とフィクスチャ
│   ├── test_config.py          # 設定テスト
│   ├── test_preprocessing.py   # 前処理テスト
│   ├── test_rag_components.py  # RAGコンポーネントテスト
│   ├── test_rag_pipeline.py    # パイプラインテスト
│   └── test_retrieval.py       # 検索テスト
│
├── data/                       # データディレクトリ
│   ├── egov_laws.jsonl         # 前処理済み法令データ
│   └── faiss_index/            # 検索インデックス
│       ├── vector/             # FAISSベクトルインデックス
│       └── bm25/               # BM25インデックス
│
├── datasets/                   # 元データセット
│   ├── egov_laws/              # e-Gov法令XML（要ダウンロード）
│   └── lawqa_jp/               # デジタル庁4択データ（要ダウンロード）
│
├── docs/                       # ドキュメント
│   ├── 02-SETUP.md             # セットアップガイド
│   ├── 05-ARCHITECTURE.md      # このファイル
│   ├── 03-USAGE.md             # 使用方法
│   └── 04-TESTING.md           # テストガイド
│
├── pyproject.toml              # Python依存関係定義
├── pytest.ini                  # pytest設定
├── Makefile                    # ビルド・実行タスク
├── .env                        # 環境変数設定
└── README.md                   # プロジェクト概要
```

## コアモジュール

### 1. 設定管理 (`app/core/`)

#### `rag_config.py`

全体の設定を管理するモジュール。環境変数または`.env`ファイルから設定を読み込みます。

**主要クラス:**

- `EmbeddingConfig` - 埋め込みモデル設定
  - `provider`: "huggingface"
  - `model_name`: 埋め込みモデル名（デフォルト: `intfloat/multilingual-e5-large`）
  - `dimension`: 埋め込み次元数（デフォルト: 1024）

- `LLMConfig` - LLM設定
  - `provider`: "ollama"
  - `model_name`: モデル名（デフォルト: `qwen3:8b`）
  - `temperature`: 生成温度（デフォルト: 0.1）
  - `max_tokens`: 最大トークン数（デフォルト: 2048）

- `RetrieverConfig` - 検索設定
  - `retriever_type`: "vector", "bm25", "hybrid"
  - `top_k`: 取得する文書数（デフォルト: 10）
  - `use_mmr`: MMR（Maximal Marginal Relevance）使用（デフォルト: true）
  - `mmr_lambda`: MMRの多様性パラメータ（デフォルト: 0.5）

- `RerankerConfig` - リランカー設定
  - `enabled`: リランカー有効化（デフォルト: false）
  - `model_name`: Cross-Encoderモデル名
  - `top_n`: 再ランク後の文書数（デフォルト: 5）

- `RAGConfig` - 全体設定（上記を統合）

**使用例:**

```python
from app.core.rag_config import load_config

config = load_config()
print(config.llm.model_name)  # "qwen3:8b"
print(config.retriever.top_k)  # 10
```

### 2. 検索・RAGモジュール (`app/retrieval/`)

#### `base.py`

基底クラスと共通データ型の定義。

**主要クラス:**

- `Document` - 検索結果の文書を表現
  - `page_content`: 文書本文
  - `metadata`: メタデータ（法令名、条文番号等。デフォルトは空辞書）
  - `score`: 検索スコア（オプション、デフォルト0.0）

- `BaseRetriever` - Retrieverの抽象基底クラス
  - `retrieve(query, top_k)`: 検索を実行

- `BaseReranker` - Rerankerの抽象基底クラス
  - `rerank(query, documents, top_n)`: 再ランクを実行

#### `vector_retriever.py`

FAISSを使用したベクトル検索の実装。

**主要クラス:**

- `VectorRetriever(BaseRetriever)`

**機能:**
- Sentence Transformersで文書を埋め込み
- FAISSインデックスで高速近似最近傍探索
- MMR（Maximal Marginal Relevance）による多様性確保

**主要メソッド:**
- `add_documents(documents)`: ドキュメントをベクトルストアへ追加
- `retrieve(query, top_k)`: クエリに対して関連文書を取得
- `save_index(path)`: インデックスを永続化
- `load_index(path)`: インデックスをロード

**使用例:**

```python
from app.retrieval.vector_retriever import VectorRetriever

retriever = VectorRetriever(
    embedding_model="intfloat/multilingual-e5-large",
    index_path="data/faiss_index/vector",
    use_mmr=True,
    mmr_lambda=0.5
)

documents = retriever.retrieve("会社法第26条について", top_k=5)
for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
```

**技術詳細:**
- 埋め込みモデル: `intfloat/multilingual-e5-large` (1024次元)
- インデックス: FAISS IndexFlatIP (内積ベース)
- MMR: 関連性と多様性のバランスを取る（λ=0.5）

#### `bm25_retriever.py`

BM25アルゴリズムを使用したキーワードベース検索。

**主要クラス:**

- `BM25Retriever(BaseRetriever)`

**機能:**
- SudachiPyによる日本語トークナイゼーション（デフォルト）
- BM25スコアリング（TF-IDFの改良版）
- インデックスの永続化

**主要メソッド:**
- `add_documents(documents)`: ドキュメントをBM25インデックスに追加
- `retrieve(query, top_k)`: クエリに対して関連文書を取得
- `save_index(path)`: インデックスを保存
- `load_index(path)`: インデックスをロード

**使用例:**

```python
from app.retrieval.bm25_retriever import BM25Retriever

retriever = BM25Retriever(index_path="data/faiss_index/bm25")
documents = retriever.retrieve("労働時間 制限", top_k=5)
```

**技術詳細:**
- トークナイザー: auto選択時の優先順位 - SudachiPy > Janome > MeCab > n-gram > simple
- BM25パラメータ: k1=1.5, b=0.75（デフォルト）
- 日本語特化: SudachiPyによる形態素解析で高精度（管理者権限不要）
- トークナイザー選択: `BM25_TOKENIZER`環境変数で指定可能（auto/sudachi/janome/mecab/ngram/simple）

#### `hybrid_retriever.py`

ベクトル検索とBM25検索を組み合わせたハイブリッド検索。

**主要クラス:**

- `HybridRetriever(BaseRetriever)`

**機能:**
- ベクトル検索とBM25検索を並列実行
- スコアの正規化と統合（デフォルト: 均等加重）
- Reciprocal Rank Fusion（RRF）オプション

**主要メソッド:**
- `add_documents(documents)`: 両方のRetrieverにドキュメントを追加
- `retrieve(query, top_k)`: ハイブリッド検索
- `save_index() / load_index()`: 両方のインデックスを保存・読み込み

**使用例:**

```python
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever

vector_ret = VectorRetriever(...)
bm25_ret = BM25Retriever(...)
hybrid_ret = HybridRetriever(vector_ret, bm25_ret)

documents = hybrid_ret.retrieve("会社設立の手続き", top_k=10)
```

**技術詳細:**
- スコア統合: `final_score = α * vector_score + (1-α) * bm25_score`
- デフォルト重み: α=0.5（均等）
- 重複除去: 同じ文書が両検索で取得された場合は統合

#### `reranker.py`

Cross-Encoderを使用したリランキング。

**主要クラス:**

- `CrossEncoderReranker(BaseReranker)`

**機能:**
- クエリと文書のペアを直接スコアリング
- Bi-Encoderより高精度だが低速
- 初期検索結果の精度向上

**主要メソッド:**
- `rerank(query, documents, top_n)`: 文書を再スコアリング

**使用例:**

```python
from app.retrieval.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# 初期検索結果を再ランク
reranked_docs = reranker.rerank("質問文", initial_documents, top_n=5)
```

**技術詳細:**
- モデル: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- 処理時間: 10文書で約0.5-1秒
- 推奨: top_k=50 → rerank top_n=5（高精度）

#### `rag_pipeline.py`

Retriever、Reranker、LLMを統合したRAGパイプライン。

**主要クラス:**

- `RAGPipeline`

**機能:**
- エンドツーエンドのRAG実行
- 検索 → （再ランク） → プロンプト生成 → LLM推論
- LangChain Expression Language（LCEL）使用

**主要メソッド:**
- `retrieve_documents(query)`: 文書検索のみ実行
- `format_context(documents)`: 文書をプロンプト用に整形
- `query(question)`: 質問に対する完全な回答を生成

**使用例:**

```python
from app.retrieval.rag_pipeline import RAGPipeline
from app.retrieval.hybrid_retriever import HybridRetriever

pipeline = RAGPipeline(
    retriever=hybrid_retriever,
    llm_provider="ollama",
    llm_model="qwen3:8b",
    temperature=0.1,
    top_k=10,
    rerank_top_n=5
)

result = pipeline.query("会社法第26条について教えてください")
print(result["answer"])      # LLMの回答
print(result["citations"])   # 引用元法令
```

**技術詳細:**
- LangChain統合: LCEL（`|`演算子）でパイプライン構築
- プロンプトテンプレート: 法令条文を明示的に提示
- 出力パーサー: 文字列として回答を抽出

## データフロー

### 1. インデックス構築フロー

```
[XMLファイル] 
    ↓ preprocess_egov_xml.py
[JSONL] 
    ↓ build_index.py
[Document List]
    ↓ (並列)
    ├─→ VectorRetriever.build_index()
    │       ↓ Sentence Transformers
    │   [埋め込みベクトル]
    │       ↓ FAISS
    │   [vector/index.faiss]
    │   [vector/index.pkl]
    │
    └─→ BM25Retriever.build_index()
            ↓ SudachiPy
        [トークン列]
            ↓ rank-bm25
        [bm25/index.pkl]
```

### 2. クエリ実行フロー

```
[ユーザー質問]
    ↓
RAGPipeline.query()
    ↓
retrieve_documents()
    ↓ (並列)
    ├─→ VectorRetriever.retrieve()
    │       ↓ 埋め込み + FAISS検索
    │   [Vector Results]
    │
    └─→ BM25Retriever.retrieve()
            ↓ トークナイズ + BM25
        [BM25 Results]
    ↓
HybridRetriever (スコア統合)
    ↓
[Top-K Documents]
    ↓ (オプション)
Reranker.rerank()
    ↓ Cross-Encoder
[Top-N Documents]
    ↓
format_context()
    ↓
[プロンプト文字列]
    ↓
LLM (Ollama)
    ↓
[回答テキスト]
    ↓
[結果 + 引用]
```

### 3. 評価フロー

```
[4択データセット (selection.json)]
    ↓
evaluate_multiple_choice.py
    ↓ (各サンプルに対して)
    ├─→ RAGPipeline.retrieve_documents()
    │       ↓
    │   [関連法令条文]
    │       ↓
    │   create_multiple_choice_prompt()
    │       ↓
    │   [4択プロンプト]
    │
    └─→ LLM.generate()
            ↓
        [回答 (a/b/c/d)]
            ↓
        extract_answer()
            ↓
        [正解/不正解判定]
    ↓
[評価結果JSON]
- accuracy
- correct_count
- 詳細結果
```

## 設定管理

### 環境変数 (`.env`)

全ての設定は`.env`ファイルで管理されます：

```bash
# 埋め込みモデル
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024

# LLM
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Retriever
RETRIEVER_TYPE=hybrid
RETRIEVER_TOP_K=10
USE_MMR=true
MMR_LAMBDA=0.5

# Hybrid Retriever設定
FUSION_METHOD=rrf                    # スコア統合方法: rrf/weighted_rrf/weighted
VECTOR_WEIGHT=0.5                    # ベクトル検索の重み（weighted/weighted_rrf使用時）
BM25_WEIGHT=0.5                      # BM25検索の重み（weighted/weighted_rrf使用時）
RRF_K=60                             # RRF のkパラメータ
FETCH_K_MULTIPLIER=2                 # 各Retrieverから取得する候補数の倍率

# BM25設定
BM25_TOKENIZER=auto                  # トークナイザー: auto/sudachi/janome/mecab/ngram/simple

# Reranker
RERANKER_ENABLED=false
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER_TOP_N=5

# パス（相対パスまたはプロジェクトルートからの絶対パス）
VECTOR_STORE_PATH=data/faiss_index
DATA_PATH=data/egov_laws.jsonl

# Ollama
OLLAMA_HOST=http://localhost:11434
```

### 設定の変更方法

#### 1. LLMモデルの変更

```bash
# .envを編集
nano .env

# LLM_MODELを変更
LLM_MODEL=qwen3:8b
```

#### 2. Retrieverタイプの変更

```bash
# Vectorのみ
RETRIEVER_TYPE=vector

# BM25のみ
RETRIEVER_TYPE=bm25

# ハイブリッド（推奨）
RETRIEVER_TYPE=hybrid
```

#### 3. Rerankerの有効化

```bash
RERANKER_ENABLED=true
RERANKER_TOP_N=5
```

## 主要スクリプト

### `scripts/preprocess_egov_xml.py`

**目的:** e-Gov法令XMLをJSONL形式に変換

**入力:** `datasets/egov_laws/*.xml` (10,435ファイル)

**出力:** `data/egov_laws.jsonl`

**処理内容:**
1. XMLファイルを再帰的にスキャン
2. 各法令から以下を抽出:
   - 法令名、法令番号
   - 条文番号、項番号、号番号
   - 条文本文
3. JSONL形式で出力

**データ構造:**

```json
{
  "law_title": "会社法",
  "law_num": "平成十七年法律第八十六号",
  "article": "26",
  "article_caption": "株式会社の設立",
  "article_title": "第26条",
  "paragraph": "1",
  "item": null,
  "text": "株式会社を設立するには..."
}
```

### `scripts/build_index.py`

**目的:** JSONLからFAISSインデックスとBM25インデックスを構築

**入力:** `data/egov_laws.jsonl`

**出力:** 
- `data/faiss_index/vector/index.faiss`
- `data/faiss_index/vector/index.pkl`
- `data/faiss_index/bm25/index.pkl`

**処理内容:**
1. JSONLファイルをロード
2. VectorRetrieverでベクトルインデックスを構築
3. BM25RetrieverでBM25インデックスを構築
4. 両方を永続化

### `scripts/query_cli.py`

**目的:** 対話型RAG CLI

**機能:**
- 対話モードまたは単発クエリ
- RAGパイプラインの完全実行
- 回答と引用元の表示

### `scripts/evaluate_multiple_choice.py`

**目的:** 4択法令データでRAG評価

**入力:** `datasets/lawqa_jp/data/selection.json`

**出力:** `evaluation_results.json`

**評価指標:**
- Accuracy（正解率）
- 各問題の詳細結果

## 拡張方法

### 1. 新しいRetrieverの追加

```python
# app/retrieval/my_retriever.py
from .base import BaseRetriever, Document

class MyRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # カスタム検索ロジック
        pass
```

### 2. 新しいLLMプロバイダーの追加

```python
# app/retrieval/rag_pipeline.py内
if llm_provider == "my_provider":
    self.llm = MyLLM(model=llm_model, temperature=temperature)
```

### 3. 新しい評価スクリプトの追加

```python
# scripts/evaluate_custom.py
from app.retrieval.rag_pipeline import RAGPipeline

# カスタム評価ロジック
```

### 4. プロンプトのカスタマイズ

```python
# app/retrieval/rag_pipeline.py内
self.prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""カスタムプロンプト...
    {context}
    {question}
    """
)
```

詳細な使用方法は [03-USAGE.md](./03-USAGE.md) を参照してください。

---

最終更新: 2024-11-04
