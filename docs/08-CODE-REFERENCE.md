# コードリファレンス

本ドキュメントでは、法令RAGシステムの各コンポーネントの役割と実装詳細を説明します。

## 目次

1. [ディレクトリ構造](#ディレクトリ構造)
2. [コアモジュール (`app/`)](#コアモジュール-app)
3. [スクリプト (`scripts/`)](#スクリプト-scripts)
4. [テスト (`tests/`)](#テスト-tests)
5. [設定ファイル](#設定ファイル)
6. [主要クラスの API リファレンス](#主要クラスの-api-リファレンス)

---

## ディレクトリ構造

```
statutes-rags/
├── app/                      # コアアプリケーション
│   ├── core/                # コア設定・ユーティリティ
│   │   └── rag_config.py   # 設定管理
│   └── retrieval/           # 検索関連モジュール
│       ├── __init__.py
│       ├── base.py          # 抽象クラス
│       ├── vector_retriever.py    # ベクトル検索
│       ├── bm25_retriever.py      # BM25検索
│       ├── hybrid_retriever.py    # ハイブリッド検索
│       ├── reranker.py            # Reranker
│       └── rag_pipeline.py        # RAGパイプライン
│
├── scripts/                 # 実行スクリプト
│   ├── preprocess_egov_xml.py     # XML前処理
│   ├── build_index.py             # インデックス構築
│   ├── query_cli.py               # CLI検索ツール
│   ├── evaluate_multiple_choice.py # 4択評価
│   └── evaluate_ragas.py          # RAGAS評価
│
├── tests/                   # テストコード
│   ├── conftest.py         # pytest設定
│   ├── test_rag_pipeline.py
│   ├── test_retrieval.py
│   └── ...
│
├── data/                    # データ保存ディレクトリ
│   ├── egov_laws.jsonl     # 前処理済み法令データ
│   └── faiss_index/        # ベクトルインデックス
│       ├── vector/         # FAISSインデックス
│       └── bm25/           # BM25インデックス
│
├── datasets/                # 元データセット
│   ├── egov_laws/          # e-Gov法令XML
│   └── lawqa_jp/           # 4択評価データ
│
├── docs/                    # ドキュメント
└── setup/                   # セットアップスクリプト
```

---

## コアモジュール (`app/`)

### `app/core/rag_config.py`

**役割:** システム全体の設定を管理

**主要クラス:**

#### `RAGConfig`

システム全体の設定を保持するPydanticモデル。

```python
class RAGConfig(BaseModel):
    embedding: EmbeddingConfig      # 埋め込みモデル設定
    llm: LLMConfig                  # LLM設定
    retriever: RetrieverConfig      # Retriever設定
    reranker: RerankerConfig        # Reranker設定
    vector_store_path: str          # インデックス保存パス
    data_path: str                  # データパス
```

**設定項目:**

```python
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
RETRIEVER_TYPE=hybrid              # vector, bm25, hybrid
RETRIEVER_TOP_K=10
USE_MMR=true
MMR_LAMBDA=0.5

# Hybrid検索
FUSION_METHOD=rrf                  # rrf, weighted_rrf, weighted
VECTOR_WEIGHT=0.5
BM25_WEIGHT=0.5
RRF_K=60
FETCH_K_MULTIPLIER=2

# BM25
BM25_TOKENIZER=auto                # auto, sudachi, janome, mecab, ngram

# Reranker
RERANKER_ENABLED=false
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER_TOP_N=5
```

**使用例:**

```python
from app.core.rag_config import load_config

config = load_config()
print(f"使用モデル: {config.llm.model_name}")
print(f"Retrieverタイプ: {config.retriever.retriever_type}")
```

---

### `app/retrieval/base.py`

**役割:** 抽象クラスとデータモデルの定義

**主要クラス:**

#### `Document`

検索結果のドキュメントを表すデータモデル。

```python
from pydantic import BaseModel, Field


class Document(BaseModel):
    page_content: str               # テキスト本文
    metadata: Dict[str, Any] = Field(default_factory=dict)   # メタデータ（法令名、条文番号など）
    score: float = 0.0              # 関連性スコア
```

**メタデータ例:**

```python
{
    "law_title": "博物館法",
    "law_num": "昭和二十六年法律第二百八十五号",
    "article": "2",
    "article_caption": "定義",
    "article_title": "第2条",
    "paragraph": "1",
    "item": None
}
```

#### `BaseRetriever`

Retrieverの抽象基底クラス。

```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """クエリに対して関連ドキュメントを検索"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """ドキュメントを追加"""
        pass
```

#### `BaseReranker`

Rerankerの抽象基底クラス。

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], 
               top_n: int = 5) -> List[Document]:
        """ドキュメントを再ランキング"""
        pass
```

---

### `app/retrieval/vector_retriever.py`

**役割:** FAISSベースのベクトル検索

**主要クラス:**

#### `VectorRetriever`

```python
class VectorRetriever(BaseRetriever):
    def __init__(
        self, 
        embedding_model: str,              # 埋め込みモデル名
        index_path: str = None,            # インデックス保存パス
        use_mmr: bool = False,             # MMR使用
        mmr_lambda: float = 0.5            # MMRパラメータ
    ):
```

**主要メソッド:**

##### `add_documents(documents: List[Dict[str, Any]])`

ドキュメントをベクトルストアに追加。

```python
# 使用例
retriever = VectorRetriever("intfloat/multilingual-e5-large")
retriever.add_documents([
    {
        "text": "博物館は資料を収集する機関です。",
        "law_title": "博物館法",
        "article": "2"
    }
])
```

**内部処理:**
1. テキストを埋め込みベクトルに変換
2. FAISSインデックスに追加
3. メタデータを保存

##### `retrieve(query: str, top_k: int = 10) -> List[Document]`

ベクトル検索を実行。

```python
# 使用例
results = retriever.retrieve("博物館とは何ですか？", top_k=5)
for doc in results:
    print(f"スコア: {doc.score:.3f}")
    print(f"法令: {doc.metadata['law_title']}")
    print(f"本文: {doc.page_content[:100]}...")
```

**検索モード:**

1. **標準検索** (`use_mmr=False`)
   - FAISS k-NN検索
   - L2距離を類似度に変換: `score = 1.0 / (1.0 + distance)`

2. **MMR検索** (`use_mmr=True`)
   - 多様性を考慮した検索
   - 順位ベーススコア: `score = 1.0 / rank`（1位: 1.0, 2位: 0.5, 3位: 0.333...）

##### `save_index()` / `load_index()`

インデックスの保存・読み込み。

```python
# 保存
retriever.save_index()

# 読み込み
retriever = VectorRetriever(
    embedding_model="intfloat/multilingual-e5-large",
    index_path="data/faiss_index/vector"
)
# 自動的に読み込まれる
```

**保存内容:**
- FAISSインデックス（ベクトルデータとメタデータを含む）

---

### `app/retrieval/bm25_retriever.py`

**役割:** BM25ベースのキーワード検索

**主要クラス:**

#### `BM25Retriever`

```python
class BM25Retriever(BaseRetriever):
    def __init__(
        self, 
        index_path: str = None,
        tokenizer: Literal["auto", "sudachi", "janome", 
                          "mecab", "ngram", "simple"] = "auto"
    ):
```

**トークナイザーの選択:**

| トークナイザー | 特徴 | 推奨用途 |
|--------------|------|---------|
| `auto` | 自動選択（sudachi → janome → mecab → ngram） | 通常使用 |
| `sudachi` | 高精度形態素解析 | 精度重視 |
| `janome` | 軽量形態素解析 | バランス型 |
| `mecab` | システム辞書利用 | レガシー環境 |
| `ngram` | 辞書不要 | ロバスト性重視 |
| `simple` | 簡易分割 | デバッグ用 |

**主要メソッド:**

##### `tokenize(text: str) -> List[str]`

テキストをトークン化。

```python
retriever = BM25Retriever(tokenizer="sudachi")

# Sudachi例
tokens = retriever.tokenize("博物館は資料を収集する機関です。")
# → ["博物館", "は", "資料", "を", "収集", "する", "機関", "です", "。"]

# N-gram例
retriever_ngram = BM25Retriever(tokenizer="ngram")
tokens = retriever_ngram.tokenize("博物館")
# → ["博物", "物館", "博物館", "博", "物", "館"]
```

##### `retrieve(query: str, top_k: int = 10) -> List[Document]`

BM25検索を実行。

```python
results = retriever.retrieve("博物館の定義", top_k=5)
```

**内部処理:**
1. クエリをトークン化
2. 各ドキュメントに対してBM25スコアを計算
3. スコア降順でソート
4. 上位top_k件を返す

##### `save_index()` / `load_index()`

BM25インデックスの保存・読み込み。

**保存内容:**
- BM25オブジェクト（pickle形式）
- ドキュメントリスト（pickle形式）
- トークナイザー情報（互換性チェック用）

**注意:** インデックスをロードする際、保存時と異なるトークナイザーを使用すると警告が表示されます。トークナイザーの一貫性を保つことが重要です。

---

### `app/retrieval/hybrid_retriever.py`

**役割:** ベクトル検索とBM25検索の統合

**主要クラス:**

#### `HybridRetriever`

```python
class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        fusion_method: Literal["rrf", "weighted_rrf", "weighted"] = "rrf",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 2
    ):
```

**スコア統合方式:**

1. **`rrf`**: 標準RRF（推奨）
   - 重み付けなし
   - 各検索手法を平等に扱う
   - 論文標準実装

2. **`weighted_rrf`**: 重み付きRRF
   - 検索手法ごとに重みを設定
   - ベクトル/BM25の比率調整可能

3. **`weighted`**: 正規化後の重み付き加算
   - 生スコアをMin-Max正規化
   - 重み付き加算

**主要メソッド:**

##### `retrieve(query: str, top_k: int = 10) -> List[Document]`

ハイブリッド検索を実行。

```python
# 初期化
vector_retriever = VectorRetriever("intfloat/multilingual-e5-large")
bm25_retriever = BM25Retriever(tokenizer="sudachi")
hybrid_retriever = HybridRetriever(
    vector_retriever, 
    bm25_retriever,
    fusion_method="rrf"
)

# 検索
results = hybrid_retriever.retrieve("博物館の定義", top_k=10)
```

**内部処理:**
1. Vector検索: `fetch_k = top_k * fetch_k_multiplier` 件取得
2. BM25検索: `fetch_k` 件取得
3. スコア統合（RRFなど）
4. 上位 `top_k` 件を返す

##### `_get_doc_id(doc: Document) -> str`

ドキュメントの一意なIDを生成（重複判定用）。

```python
def _get_doc_id(self, doc: Document) -> str:
    """
    SHA256ハッシュベースのID生成
    
    使用するメタデータ:
    - law_title: 法令名
    - article: 条番号
    - paragraph: 項番号
    - item: 号番号
    - page_content[:200]: 本文の最初の200文字
    
    利点:
    - 衝突リスクが極めて低い（SHA256）
    - 同一文書の確実な判定
    """
```

##### `_rrf_fusion(vector_results, bm25_results) -> List[Document]`

RRFによるスコア統合。

```python
# 内部で使用されるが、理解のため例を示す
def _rrf_fusion(self, vector_results, bm25_results):
    score_map = {}
    
    # ベクトル検索の順位スコア
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = self._get_doc_id(doc)
        score_map[doc_id] = 1.0 / (self.rrf_k + rank)
    
    # BM25検索の順位スコア（加算）
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = self._get_doc_id(doc)
        if doc_id in score_map:
            score_map[doc_id] += 1.0 / (self.rrf_k + rank)
        else:
            score_map[doc_id] = 1.0 / (self.rrf_k + rank)
    
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)
```

---

### `app/retrieval/reranker.py`

**役割:** Cross-Encoderによる再ランキング

**主要クラス:**

#### `CrossEncoderReranker`

```python
class CrossEncoderReranker(BaseReranker):
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
```

**主要メソッド:**

##### `rerank(query: str, documents: List[Document], top_n: int = 5) -> List[Document]`

Cross-Encoderで再ランキング。

```python
# 初期化
reranker = CrossEncoderReranker()

# Retrieverから取得した文書を再ランキング
documents = retriever.retrieve(query, top_k=20)
reranked = reranker.rerank(query, documents, top_n=5)
```

**パラメータバリデーション:**
- `top_n` は正の整数である必要があります
- 空のクエリは警告を出し、元のドキュメントを返します
- `top_n` がドキュメント数より多い場合、自動的に調整されます

**内部処理:**
1. パラメータバリデーション
2. クエリと各文書をペアにする
3. Cross-Encoderで関連性スコアを計算
4. スコア降順でソート
5. 上位 `top_n` 件を返す

---

### `app/retrieval/rag_pipeline.py`

**役割:** RAGの全体パイプラインを統合

**主要クラス:**

#### `RAGPipeline`

```python
class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: str = "ollama",
        llm_model: str = "qwen3:8b",
        temperature: float = 0.1,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 10,
        rerank_top_n: int = 5,
        max_context_length: int = 4000,
        request_timeout: int = 60  # LLMリクエストのタイムアウト（秒）
    ):
```

**パラメータ:**
- `request_timeout`: LLMリクエストのタイムアウト時間（秒）。タイムアウトエラーは自動的に検出され、適切なエラーメッセージが返されます。
- `max_context_length`: コンテキストの最大文字数（デフォルト: 4000）

**主要メソッド:**

##### `query(question: str) -> Dict[str, Any]`

質問に対して回答を生成。

```python
# 初期化
from app.core.rag_config import load_config
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rag_pipeline import RAGPipeline

config = load_config()
retriever = HybridRetriever(...)
pipeline = RAGPipeline(retriever)

# 質問
result = pipeline.query("博物館とは何ですか？")

print(result["answer"])      # LLMの回答
print(result["citations"])   # 引用情報
print(result["contexts"])    # 検索された文書
```

**返却値:**

```python
{
    "answer": "博物館とは、博物館法第2条によれば...",
    "citations": [
        {
            "law_title": "博物館法",
            "article": "2",
            "paragraph": "1",
            "item": None
        }
    ],
    "contexts": [
        {
            "law_title": "博物館法",
            "article": "2",
            "paragraph": "1",
            "text": "博物館は、歴史、芸術...",
            "score": 0.85
        }
    ],
    "error": "timeout" (optional)  # エラー情報（タイムアウト等）
}
```

**エラーハンドリング:**
- タイムアウトエラー: エラーメッセージ文字列に "timeout" が含まれる場合を検出
- その他のエラー: 例外の詳細をログに記録し、ユーザーにエラーメッセージを返す

##### `retrieve_documents(query: str) -> List[Document]`

文書検索のみ実行（LLM呼び出しなし）。

```python
documents = pipeline.retrieve_documents("博物館")
```

**内部処理:**
1. Retrieverで検索
2. Reranker（設定されている場合）で再ランキング
3. 結果を返す

##### `format_context(documents: List[Document]) -> str`

文書をコンテキスト文字列に整形。

```python
context = pipeline.format_context(documents)
```

**出力例:**

```
[1] 博物館法 第2条 第1項
博物館は、歴史、芸術、民俗、産業、自然科学等に関する資料を
収集し、保管し、展示して教育的配慮の下に一般公衆の利用に
供し、その教養、調査研究、レクリエーション等に資するために
必要な事業を行う機関をいう。

[2] 博物館法 第3条
...
```

##### `extract_citations(documents: List[Document]) -> List[Dict[str, Any]]`

引用情報を抽出（重複排除）。

```python
citations = pipeline.extract_citations(documents)
```

---

## スクリプト (`scripts/`)

### `scripts/preprocess_egov_xml.py`

**役割:** e-Gov法令XMLをJSONL形式に変換

**使用法:**

```bash
python scripts/preprocess_egov_xml.py \
    --input-dir datasets/egov_laws \
    --output-file data/egov_laws.jsonl \
    --limit 100  # 最初の100ファイルのみ（テスト用）
```

**主要関数:**

##### `parse_law_xml(xml_path: Path) -> List[Dict[str, Any]]`

単一のXMLファイルをパースしてチャンクのリストを返す。

```python
chunks = parse_law_xml(Path("datasets/egov_laws/博物館法.xml"))
# → [
#     {"law_title": "博物館法", "article": "1", ...},
#     {"law_title": "博物館法", "article": "2", ...},
#     ...
# ]
```

##### `process_directory(input_dir: Path, output_file: Path, limit: int = None)`

ディレクトリ内の全XMLファイルを処理。

**出力形式（JSONL）:**

各行が1つのチャンク（JSON）:

```json
{"law_title":"博物館法","law_num":"昭和二十六年法律第二百八十五号","article":"1","article_caption":"","article_title":"第1条","paragraph":null,"item":null,"text":"この法律は、..."}
{"law_title":"博物館法","law_num":"昭和二十六年法律第二百八十五号","article":"2","article_caption":"定義","article_title":"第2条","paragraph":"1","item":null,"text":"博物館は、..."}
```

---

### `scripts/build_index.py`

**役割:** JSONLファイルからベクトルインデックスを構築

**使用法:**

```bash
# ハイブリッドインデックス構築
python scripts/build_index.py \
    --data-path data/egov_laws.jsonl \
    --index-path data/faiss_index \
    --retriever-type hybrid \
    --batch-size 10000

# ベクトルインデックスのみ
python scripts/build_index.py --retriever-type vector

# BM25インデックスのみ
python scripts/build_index.py --retriever-type bm25
```

**主要処理:**

1. JSONLファイルを読み込み
2. Retrieverを初期化
3. バッチでドキュメントを追加
4. インデックスを保存

**出力:**

```
data/faiss_index/
├── vector/
│   ├── index.faiss
│   └── index.pkl
└── bm25/
    ├── bm25.pkl
    ├── documents.pkl
    └── tokenizer_info.pkl
```

---

### `scripts/query_cli.py`

**役割:** コマンドラインから検索・質問

**使用法:**

```bash
# 基本的な使用（単発クエリ）
python scripts/query_cli.py "博物館とは何ですか？"

# 対話モード
python scripts/query_cli.py --interactive

# 結果をJSONに保存
python scripts/query_cli.py "博物館とは何ですか？" --output result.json
```

**引数:**

- 位置引数で質問文を指定
- `--interactive` / `-i`: 対話モード
- `--output` / `-o`: 結果をJSONファイルへ保存

Retriever や LLM の設定を変更したい場合は `.env` または環境変数を利用します。

---

### `scripts/evaluate_multiple_choice.py`

**役割:** 4択法令データで評価

**使用法:**

```bash
# 基本的な評価
python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output evaluation_results.json \
    --samples 20

# LLMのみ（RAGなし）
python scripts/evaluate_multiple_choice.py \
    --no-rag \
    --output evaluation_results_llm_only.json

# 異なるモデルで評価
python scripts/evaluate_multiple_choice.py \
    --llm-model qwen3:8b \
    --top-k 10
```

**引数:**

- `--data`: 4択データセットのパス
- `--output`: 評価結果の出力パス
- `--samples`: 評価するサンプル数
- `--no-rag`: RAGを使用せずLLMのみ
- `--top-k`: 検索件数
- `--llm-model`: 使用するLLMモデル

**出力例:**

```json
{
  "config": {
    "rag_enabled": true,
    "retriever_type": "hybrid",
    "llm_model": "qwen3:8b",
    "top_k": 10,
    "total_samples": 20
  },
  "summary": {
    "accuracy": 0.75,
    "correct_count": 15,
    "total_count": 20
  },
  "results": [...]
}
```

---

### `scripts/evaluate_ragas.py`

**役割:** RAGASフレームワークで評価

**使用法:**

```bash
python scripts/evaluate_ragas.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output ragas_results.json \
    --samples 10
```

**評価指標:**

- **Faithfulness**: 回答が検索された文書に基づいているか
- **Answer Relevancy**: 回答が質問に関連しているか
- **Context Precision**: 検索された文書の精度
- **Context Recall**: 必要な情報を検索できているか

---

## テスト (`tests/`)

### `tests/conftest.py`

**役割:** pytest設定とフィクスチャ定義

**主要フィクスチャ:**

```python
@pytest.fixture
def sample_jsonl_data():
    """テスト用のサンプルデータ"""
    return [...]

@pytest.fixture
def temp_index_dir(tmp_path):
    """一時インデックスディレクトリ"""
    return tmp_path / "test_index"
```

### テストの実行

```bash
# 全テスト実行
pytest tests/

# 特定のテストのみ
pytest tests/test_rag_pipeline.py

# マーカーでフィルタ
pytest -m unit          # ユニットテストのみ
pytest -m integration   # 統合テストのみ
pytest -m "not slow"    # 遅いテストを除外

# カバレッジ測定
pytest --cov=app --cov-report=html
```

---

## 設定ファイル

### `pyproject.toml`

**役割:** プロジェクトメタデータと依存関係

**主要依存関係:**

```toml
dependencies = [
    "langchain>=0.1.0",              # LLM統合
    "langchain-community>=0.0.10",   # コミュニティ拡張
    "faiss-cpu>=1.7.4",             # ベクトル検索
    "sentence-transformers>=2.2.0",  # 埋め込みモデル
    "rank-bm25>=0.2.2",             # BM25検索
    "python-dotenv>=1.0.0",         # 環境変数管理
    "sudachipy>=0.6.0",             # 日本語トークナイザー
    "janome>=0.5.0",                # 日本語トークナイザー
    # ...
]
```

### `pytest.ini`

**役割:** pytestの設定

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: ユニットテスト
    integration: 統合テスト
    slow: 時間がかかるテスト
    rag: RAGパイプライン関連
```

### `Makefile`

**役割:** よく使うコマンドのショートカット

```bash
make preprocess   # XMLを前処理
make index        # インデックス構築
make test         # ユニットテストのみ（高速）
make test-all     # 全テスト実行
make clean        # クリーンアップ
```

---

## 主要クラスの API リファレンス

### クイックリファレンス

#### ベクトル検索のみ

```python
from app.retrieval.vector_retriever import VectorRetriever

retriever = VectorRetriever(
    embedding_model="intfloat/multilingual-e5-large",
    index_path="data/faiss_index/vector",
    use_mmr=True,
    mmr_lambda=0.5
)

results = retriever.retrieve("博物館とは", top_k=5)
```

#### BM25検索のみ

```python
from app.retrieval.bm25_retriever import BM25Retriever

retriever = BM25Retriever(
    index_path="data/faiss_index/bm25",
    tokenizer="sudachi"
)

results = retriever.retrieve("博物館の定義", top_k=5)
```

#### ハイブリッド検索

```python
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

vector_ret = VectorRetriever("intfloat/multilingual-e5-large")
bm25_ret = BM25Retriever(tokenizer="sudachi")

hybrid_ret = HybridRetriever(
    vector_ret, 
    bm25_ret,
    fusion_method="rrf",
    rrf_k=60,
    fetch_k_multiplier=2
)

results = hybrid_ret.retrieve("博物館", top_k=10)
```

#### 完全なRAGパイプライン

```python
from app.core.rag_config import load_config
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rag_pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker

config = load_config()

# Retriever初期化
vector_ret = VectorRetriever(
    config.embedding.model_name,
    index_path=str(config.vector_store_path / "vector")
)
bm25_ret = BM25Retriever(
    index_path=str(config.vector_store_path / "bm25")
)
retriever = HybridRetriever(vector_ret, bm25_ret)

# Reranker初期化（オプション）
reranker = CrossEncoderReranker() if config.reranker.enabled else None

# RAGパイプライン初期化
pipeline = RAGPipeline(
    retriever=retriever,
    llm_provider=config.llm.provider,
    llm_model=config.llm.model_name,
    temperature=config.llm.temperature,
    reranker=reranker,
    top_k=config.retriever.top_k,
    rerank_top_n=config.reranker.top_n
)

# 質問
result = pipeline.query("博物館とは何ですか？")
print(result["answer"])
```

---

## デバッグとログ

### ログレベルの設定

```python
import logging

# アプリケーション全体のログレベル
logging.basicConfig(level=logging.DEBUG)

# 特定のモジュールのみ
logging.getLogger("app.retrieval.vector_retriever").setLevel(logging.DEBUG)
```

### 環境変数でログレベル設定

```bash
export LOG_LEVEL=DEBUG
python scripts/query_cli.py "博物館"
```

### デバッグ情報の出力

```python
# Retriever内部の動作を確認
import logging
logging.getLogger("app.retrieval.hybrid_retriever").setLevel(logging.DEBUG)

results = hybrid_retriever.retrieve("博物館", top_k=5)

# 出力例:
# DEBUG - Hybrid retrieval for query: '博物館', top_k=5, fetch_k=10
# DEBUG - RRF fusion: 12 unique documents from 10 vector + 10 BM25 results
# INFO - Hybrid retrieval returned 5 documents (method: rrf)
```

---

## よくあるパターン

### パターン1: カスタムRetrieverの作成

```python
from app.retrieval.base import BaseRetriever, Document
from typing import List, Dict, Any

class CustomRetriever(BaseRetriever):
    def __init__(self):
        # 初期化処理
        pass
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        # ドキュメント追加処理
        pass
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        # 検索処理
        results = []
        # ...
        return results
```

### パターン2: 複数のクエリをバッチ処理

```python
queries = [
    "博物館とは何ですか？",
    "個人情報の定義は？",
    "著作権法について"
]

results = []
for query in queries:
    result = pipeline.query(query)
    results.append(result)
```

### パターン3: 検索結果のフィルタリング

```python
# 特定の法令のみに絞る
all_results = retriever.retrieve("定義", top_k=20)
filtered = [
    doc for doc in all_results 
    if doc.metadata.get("law_title") == "博物館法"
]
```

### パターン4: スコア閾値での フィルタリング

```python
results = retriever.retrieve("博物館", top_k=20)
high_quality = [doc for doc in results if doc.score > 0.7]
```

---

## 関連ドキュメント

- [05-ARCHITECTURE.md](05-ARCHITECTURE.md) - システムアーキテクチャ
- [07-ALGORITHM.md](07-ALGORITHM.md) - アルゴリズム詳細
- [03-USAGE.md](03-USAGE.md) - 使用方法
- [04-TESTING.md](04-TESTING.md) - テストガイド
- [06-DEVELOPMENT.md](06-DEVELOPMENT.md) - 開発ガイド

---

最終更新: 2024-11-04
