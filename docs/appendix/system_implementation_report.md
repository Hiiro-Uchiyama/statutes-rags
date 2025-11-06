# 法令RAGシステム 実装報告書

本ドキュメントは、statutes-ragsプロジェクトの技術実装を詳細に記述した報告書です。

作成日: 2025年11月6日

## 目次

1. [システム概要](#1-システム概要)
2. [技術スタック](#2-技術スタック)
3. [コンポーネント実装](#3-コンポーネント実装)
4. [データ処理パイプライン](#4-データ処理パイプライン)
5. [設定管理](#5-設定管理)

## 1. システム概要

### 1.1 プロジェクト目的

本プロジェクトは、日本の法令に関する質問に対して、適切な法令根拠を基に回答を生成できるRAG（Retrieval-Augmented Generation）システムを実装し、デジタル庁が公開した4択法令問題データセット（140問）を使用して定量的に評価するものです。

### 1.2 システムアーキテクチャ

```
質問
  ↓
Retriever（ベクトル検索）
  ↓
Reranker（再ランキング）
  ↓
LLM（生成）
  ↓
回答
```

主要コンポーネント:
- データソース: e-Gov法令XML（10,435ファイル）
- 埋め込みモデル: intfloat/multilingual-e5-large（1024次元）
- 検索エンジン: FAISSベクトルインデックス
- リランカー: cross-encoder/ms-marco-MiniLM-L-12-v2
- LLM: Ollama qwen3:14b（ローカル実行）

### 1.3 データセット

法令データ:
- ソース: e-Gov法令API
- ファイル数: 10,435法令
- 総チャンク数: 約280万件
- チャンクサイズ: 500文字（オーバーラップ50文字）

評価データ:
- ソース: デジタル庁公開データ
- 問題数: 140問（4択）
- カバー範囲: 会社法、金融商品取引法等
- 形式: JSON形式

## 2. 技術スタック

### 2.1 開発環境

```yaml
言語: Python 3.11+
パッケージ管理: uv
仮想環境: .venv
依存関係管理: pyproject.toml
```

### 2.2 主要ライブラリ

検索・RAG:
- langchain: RAGパイプライン構築
- langchain-community: Ollama統合
- faiss-cpu: ベクトル検索インデックス
- sentence-transformers: 埋め込み生成
- rank-bm25: BM25検索（オプション）

自然言語処理:
- sudachipy: 日本語形態素解析
- sudachidict-core: 日本語辞書

LLM:
- ollama: ローカルLLM実行環境

データ処理:
- lxml: XML解析
- pydantic: データ検証と設定管理

評価:
- ragas: RAG評価フレームワーク（オプション）

## 3. コンポーネント実装

### 3.1 検索システム

#### ベクトル検索（VectorRetriever）

実装ファイル: `app/retrieval/vector_retriever.py`

```python
class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model: str,
        index_path: str,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ):
        # 埋め込みモデルの初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # FAISSインデックスのロード
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings
        )
```

技術仕様:
- 埋め込みモデル: intfloat/multilingual-e5-large
- ベクトル次元: 1024次元
- インデックスタイプ: FAISS IndexFlatIP（内積ベース）
- MMR（Maximal Marginal Relevance）: Lambda=0.5
- 検索速度: 約0.1-0.2秒/クエリ
- メモリ使用量: 約4-6GB

#### リランキング（CrossEncoderReranker）

実装ファイル: `app/retrieval/reranker.py`

```python
class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5
    ) -> List[Document]:
        # クエリと文書のペアを作成
        pairs = [[query, doc.page_content] for doc in documents]
        
        # スコアリング
        scores = self.model.predict(pairs)
        
        # スコアでソートして上位top_nを返す
        sorted_docs = [
            doc for _, doc in sorted(
                zip(scores, documents),
                reverse=True
            )
        ]
        return sorted_docs[:top_n]
```

技術仕様:
- モデル: cross-encoder/ms-marco-MiniLM-L-12-v2
- 入力: クエリと文書のペア
- 出力: 関連度スコア（-10〜10）
- 処理時間: 約0.5-1秒/10文書
- 効果: 精度向上約5-10ポイント

### 3.2 LLMパイプライン

#### RAGパイプライン

実装ファイル: `app/retrieval/rag_pipeline.py`

```python
class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: str = "ollama",
        llm_model: str = "qwen3:14b",
        temperature: float = 0.1,
        top_k: int = 10,
        reranker: Optional[BaseReranker] = None,
        rerank_top_n: int = 5
    ):
        # LLMの初期化
        self.llm = Ollama(
            model=llm_model,
            temperature=temperature
        )
        
        # Retrieverとrerankerの設定
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
```

処理フロー:

```python
def query(self, question: str) -> Dict[str, Any]:
    # 1. 文書検索
    documents = self.retriever.retrieve(question, self.top_k)
    
    # 2. リランキング（オプション）
    if self.reranker:
        documents = self.reranker.rerank(
            question,
            documents,
            self.rerank_top_n
        )
    
    # 3. コンテキスト生成
    context = self.format_context(documents)
    
    # 4. プロンプト生成とLLM実行
    prompt = self.create_prompt(context, question)
    response = self.llm.invoke(prompt)
    
    # 5. 結果返却
    return {
        "answer": response,
        "context": context,
        "sources": [doc.metadata for doc in documents]
    }
```

#### プロンプト設計

Few-shotプロンプト構造:

```python
PROMPT_TEMPLATE = """You are a legal assistant specialized in Japanese law.

Example 1:
Question: [問題例1]
Answer: a

Example 2:
Question: [問題例2]
Answer: b

Legal Provisions:
{context}

Question: {question}

Choices:
{choices}

Answer (a, b, c, or d):"""
```

設計原則:
- 明確な役割定義
- Few-shot学習（2例）
- 構造化された入力
- 短く明確な出力指示

### 3.3 LLMモデル設定

```yaml
プロバイダー: Ollama
モデル: qwen3:14b
パラメータ数: 14.8B
量子化: Q4_K_M
コンテキスト長: 40,960トークン
Temperature: 0.1（決定論的生成）
最大トークン数: 2,048
```

Qwen3-14Bの特徴:
- 多言語対応（日本語を含む100+言語）
- 高い推論能力
- ローカル実行可能
- GPUメモリ: 約8-10GB必要

## 4. データ処理パイプライン

### 4.1 前処理（XML → JSONL）

実装ファイル: `scripts/preprocess_egov_xml.py`

処理フロー:

```
[e-Gov法令XML] 
    ↓
XML解析（lxml）
    ↓
条文抽出
    ↓
メタデータ付与
    ↓
[JSONL出力]
```

出力形式:

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

処理統計:
- 入力: 10,435 XMLファイル
- 出力: 約280万行のJSONL
- 処理時間: 約5-10分

### 4.2 インデックス構築

実装ファイル: `scripts/build_index.py`

処理フロー:

```
[JSONL]
    ↓
Document作成
    ↓
    ├─→ VectorRetriever
    │     ↓
    │   埋め込み生成
    │     ↓
    │   FAISSインデックス
    │     ↓
    │   [vector/index.faiss]
    │   [vector/index.pkl]
    │
    └─→ BM25Retriever（オプション）
          ↓
        トークナイズ
          ↓
        BM25インデックス
          ↓
        [bm25/index.pkl]
```

ベクトルインデックス構築:
- 処理時間: 約30-60分（GPUあり）
- メモリピーク: 約20-30GB
- 出力サイズ: 約3-4GB

### 4.3 評価パイプライン

実装ファイル: `scripts/evaluate_multiple_choice.py`

評価フロー:

```python
def evaluate():
    # 1. データセット読み込み
    dataset = load_dataset("selection.json")
    
    # 2. 各問題を評価
    results = []
    for sample in dataset:
        # a. 文書検索
        docs = pipeline.retrieve_documents(sample["question"])
        
        # b. プロンプト生成
        prompt = create_prompt(docs, sample)
        
        # c. LLM推論
        response = pipeline.llm.invoke(prompt)
        
        # d. 回答抽出
        predicted = extract_answer(response)
        
        # e. 正解判定
        is_correct = (predicted == sample["correct_answer"])
        
        results.append({
            "question": sample["question"],
            "predicted_answer": predicted,
            "correct_answer": sample["correct_answer"],
            "is_correct": is_correct
        })
    
    # 3. 精度計算
    accuracy = sum(r["is_correct"] for r in results) / len(results)
    
    return {
        "accuracy": accuracy,
        "results": results
    }
```

評価指標:
- Accuracy（正解率）
- 正解数/総問題数
- タイムアウト数
- パースエラー数

## 5. 設定管理

### 5.1 環境変数設定

ファイル: `.env`

```bash
# 埋め込みモデル
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024

# LLM
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Retriever
RETRIEVER_TYPE=vector
RETRIEVER_TOP_K=10
USE_MMR=true
MMR_LAMBDA=0.5

# Reranker
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER_TOP_N=3

# パス
VECTOR_STORE_PATH=data/faiss_index
DATA_PATH=data/egov_laws.jsonl

# Ollama
OLLAMA_HOST=http://localhost:11434
```

### 5.2 設定クラス

実装ファイル: `app/core/rag_config.py`

```python
class RAGConfig(BaseModel):
    embedding: EmbeddingConfig
    llm: LLMConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    vector_store_path: str
    data_path: str

def load_config() -> RAGConfig:
    """環境変数から設定をロード"""
    return RAGConfig()
```

設定の優先順位:
1. コマンドライン引数
2. 環境変数
3. .envファイル
4. デフォルト値

## 6. プロジェクト構造

```
statutes-rags/
├── app/
│   ├── core/
│   │   └── rag_config.py          # 設定管理
│   └── retrieval/
│       ├── base.py                # 基底クラス
│       ├── vector_retriever.py    # ベクトル検索
│       ├── bm25_retriever.py      # BM25検索
│       ├── hybrid_retriever.py    # ハイブリッド
│       ├── reranker.py            # リランカー
│       └── rag_pipeline.py        # RAGパイプライン
│
├── scripts/
│   ├── preprocess_egov_xml.py     # 前処理
│   ├── build_index.py             # インデックス構築
│   ├── query_cli.py               # CLI
│   └── evaluate_multiple_choice.py # 評価
│
├── data/
│   ├── egov_laws.jsonl            # 前処理済みデータ
│   └── faiss_index/               # インデックス
│
├── datasets/
│   ├── egov_laws/                 # 元XMLファイル
│   └── lawqa_jp/                  # 評価データ
│
└── docs/
    ├── 02-SETUP.md                # セットアップガイド
    ├── 03-USAGE.md                # 使用方法
    ├── 04-TESTING.md              # テストガイド
    └── 05-ARCHITECTURE.md         # アーキテクチャ
```

最終更新: 2025年11月6日
