# statutes RAG System 開発ガイド

## 目次

1. [開発環境セットアップ](#開発環境セットアップ)
2. [コーディング規約](#コーディング規約)
3. [テスト戦略](#テスト戦略)
4. [今後の開発方針](#今後の開発方針)
5. [拡張ポイント](#拡張ポイント)
6. [デバッグガイド](#デバッグガイド)

## 開発環境セットアップ

### 必要なツール

```bash
# 開発環境のセットアップ
make dev-setup
```

これにより以下がインストールされます:
- pytest: テストフレームワーク
- pytest-cov: カバレッジ計測
- black: コードフォーマッタ
- ruff: 高速リンター
- mypy: 型チェッカー（オプション）

### エディタ設定

#### VSCode推奨設定 (`.vscode/settings.json`)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests", "-v"],
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### 開発サーバー起動

将来的なWeb API開発用:

```bash
# FastAPIサーバー起動（実装後）
uvicorn app.main:app --reload --port 8000
```

## コーディング規約

### Pythonスタイルガイド

- **PEP 8準拠**: blackとruffで自動フォーマット
- **型ヒント**: 関数シグネチャに型ヒントを記述
- **Docstring**: Google Style Docstring推奨

#### 型ヒントの例

```python
from typing import List, Dict, Any, Optional

def retrieve_documents(
    query: str,
    top_k: int = 10,
    use_reranker: bool = False
) -> List[Document]:
    """ドキュメントを検索
    
    Args:
        query: 検索クエリ
        top_k: 取得件数
        use_reranker: リランキング使用フラグ
    
    Returns:
        検索結果のDocumentリスト
    
    Raises:
        ValueError: top_kが0以下の場合
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    # 実装...
    return documents
```

### コードフォーマット

```bash
# 全体をフォーマット
make format

# または
black app/ scripts/ tests/

# チェックのみ（CI用）
black --check app/ scripts/ tests/
```

### リンティング

```bash
# リンターチェック
make lint

# または
ruff check app/ scripts/ tests/

# 自動修正
ruff check --fix app/ scripts/ tests/
```

### インポート順序

ruffが自動的にソートしますが、手動の場合は以下の順序:

1. 標準ライブラリ
2. サードパーティライブラリ
3. ローカルモジュール

```python
# 標準ライブラリ
import json
from pathlib import Path
from typing import List, Dict

# サードパーティ
import numpy as np
from langchain_core.documents import Document

# ローカル
from app.core.rag_config import load_config
from app.retrieval.base import BaseRetriever
```

## テスト戦略

### テスト階層

statutes RAGシステムは3層のテスト構造を採用しています:

1. **ユニットテスト** (`@pytest.mark.unit`)
   - 個別関数・メソッドのテスト
   - 外部依存なし、高速
   - モック使用推奨

2. **統合テスト** (`@pytest.mark.integration`)
   - 複数コンポーネントの統合動作
   - 実際のモデル・インデックス使用
   - 中速

3. **評価テスト** (`@pytest.mark.eval`)
   - エンドツーエンド評価
   - RAGAS使用
   - 低速

### pytestマーカー

`pytest.ini`で定義されているマーカー:

```ini
[pytest]
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (may require external services)
    slow: Slow tests (may take several minutes)
    xmlparse: XML parsing tests
    retrieval: Retrieval system tests
    rag: RAG pipeline tests
    eval: Evaluation tests
```

### テスト実行

```bash
# ユニットテストのみ（高速）
make test
# または
pytest tests/ -v -m unit

# 統合テストのみ
make test-integration
# または
pytest tests/ -v -m integration

# 全テスト
make test-all
# または
pytest tests/ -v

# カバレッジ付き
make test-coverage
# または
pytest tests/ -v --cov=app --cov=scripts --cov-report=html

# 特定のテストファイル
pytest tests/test_retrieval.py -v

# 特定のテスト関数
pytest tests/test_retrieval.py::test_vector_retriever_basic -v

# slowマーカー除外（高速テスト）
pytest tests/ -v -m "unit and not slow"
```

### テストフィクスチャ

`tests/conftest.py`で共通フィクスチャを定義しています:

**主要フィクスチャ**:

1. **project_root_path**: プロジェクトルートパス
2. **test_data_dir**: テストデータディレクトリ
3. **sample_xml_content**: サンプルXML文字列
4. **sample_xml_file**: 一時XMLファイル
5. **sample_jsonl_data**: サンプルJSONLデータ（Pythonリスト）
6. **sample_jsonl_file**: 一時JSONLファイル
7. **temp_index_dir**: 一時インデックスディレクトリ（自動クリーンアップ）
8. **mock_config**: モック設定オブジェクト
9. **sample_lawqa_data**: サンプル評価データ

**使用例**:

```python
import pytest
from pathlib import Path

@pytest.mark.unit
def test_preprocessing(sample_xml_file):
    """sample_xml_fileフィクスチャを使用"""
    from scripts.preprocess_egov_xml import parse_law_xml
    
    chunks = parse_law_xml(sample_xml_file)
    
    assert len(chunks) > 0
    assert chunks[0]["law_title"] == "博物館法"

@pytest.mark.integration
def test_retrieval(sample_jsonl_data, temp_index_dir):
    """複数フィクスチャを使用"""
    from app.retrieval.vector_retriever import VectorRetriever
    
    retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector")
    )
    retriever.add_documents(sample_jsonl_data)
    
    results = retriever.retrieve("博物館", top_k=3)
    assert len(results) > 0
```

### テスト作成ガイドライン

#### ユニットテストの例

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
def test_format_context():
    """コンテキスト整形のテスト（外部依存なし）"""
    from app.retrieval.rag_pipeline import RAGPipeline
    from app.retrieval.base import Document
    
    # モックRetrieverを作成
    mock_retriever = Mock()
    
    # LLMをパッチしてモック化
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    # テストデータ
    documents = [
        Document(
            page_content="博物館は資料を収集する機関です。",
            metadata={"law_title": "博物館法", "article": "2"}
        )
    ]
    
    # テスト実行
    context = pipeline.format_context(documents)
    
    # アサーション
    assert "博物館法" in context
    assert "第2条" in context
    assert "博物館は資料を収集する機関です。" in context
```

#### 統合テストの例

```python
import pytest

@pytest.mark.integration
@pytest.mark.retrieval
def test_hybrid_retrieval(sample_jsonl_data, temp_index_dir):
    """ハイブリッド検索の統合テスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    
    # Retriever構築
    vector_retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector")
    )
    bm25_retriever = BM25Retriever(
        index_path=str(temp_index_dir / "bm25")
    )
    
    hybrid_retriever = HybridRetriever(
        vector_retriever, bm25_retriever,
        vector_weight=0.5, bm25_weight=0.5
    )
    
    # ドキュメント追加
    hybrid_retriever.add_documents(sample_jsonl_data)
    
    # 検索実行
    results = hybrid_retriever.retrieve("博物館", top_k=5)
    
    # アサーション
    assert len(results) > 0
    assert all(hasattr(doc, 'score') for doc in results)
    assert results[0].score >= results[-1].score  # スコア順
```

### カバレッジ目標

- **全体**: 80%以上
- **コアモジュール** (`app/core/`, `app/retrieval/`): 90%以上
- **スクリプト** (`scripts/`): 70%以上

カバレッジレポート確認:
```bash
make test-coverage
# htmlcov/index.htmlをブラウザで開く
```

## 今後の開発方針

### 短期目標（1-3ヶ月）

#### 1. FastAPI Web API実装

**目的**: CLIだけでなくHTTP APIでRAG機能を提供

**実装内容**:
- `app/api/`: FastAPIルーター
  - `POST /api/query`: 質問応答エンドポイント
  - `GET /api/health`: ヘルスチェック
  - `GET /api/citations/{law_title}`: 法令条文取得
- `app/main.py`: FastAPIアプリケーションエントリポイント
- 非同期処理対応（async/await）
- Pydanticによるリクエスト/レスポンス検証

**設計案**:
```python
# app/api/query.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    use_reranker: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    contexts: List[Dict[str, Any]]

@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # RAGパイプライン実行
    result = pipeline.query(request.question)
    return QueryResponse(**result)
```

#### 2. Qdrant統合

**目的**: FAISSからスケーラブルなベクトルDBへ移行

**実装内容**:
- `app/retrieval/qdrant_retriever.py`: QdrantベースのRetriever
- Qdrantコレクション管理
- フィルタリング機能（法令名、年度など）
- 分散検索対応

**設計案**:
```python
# app/retrieval/qdrant_retriever.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class QdrantRetriever(BaseRetriever):
    def __init__(self, url: str, collection_name: str, embedding_model: str):
        self.client = QdrantClient(url)
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # コレクション作成
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [self._convert_to_document(hit) for hit in results]
```

#### 3. ストリーミング応答

**目的**: LLM生成をリアルタイムでストリーミング

**実装内容**:
- `app/retrieval/rag_pipeline.py`にstream_queryメソッド追加
- FastAPIのServerSentEvents（SSE）対応
- フロントエンドでの逐次表示

**設計案**:
```python
# app/retrieval/rag_pipeline.py
from typing import Iterator

class RAGPipeline:
    def stream_query(self, question: str) -> Iterator[str]:
        """ストリーミング生成"""
        documents = self.retrieve_documents(question)
        context = self.format_context(documents)
        
        # Ollamaのストリーミング機能を使用
        for chunk in self.llm.stream({"context": context, "question": question}):
            yield chunk

# app/api/query.py
from fastapi.responses import StreamingResponse

@router.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    def generate():
        for chunk in pipeline.stream_query(request.question):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 中期目標（3-6ヶ月）

#### 4. ファインチューニング

**目的**: 法律ドメイン特化LLMの構築

**実装内容**:
- `scripts/finetune.py`: ファインチューニングスクリプト
- 法律QAデータセットを使用（将来的には専用の学習データセットを準備予定）
- LoRA/QLoRAによる効率的なチューニング
- モデル評価フレームワーク

**学習データ準備の例**:
```python
# scripts/prepare_finetune_data.py（将来の実装例）
import json

def prepare_instruction_data():
    # 法律QAデータセットから学習データを生成
    with open("datasets/lawqa_jp/data/selection.json") as f:
        data = json.load(f)
    
    # Instruction形式に変換
    instructions = []
    for sample in data["samples"]:
        instructions.append({
            "instruction": sample["問題文"],
            "input": "",
            "output": f"正解: {sample['output']}"
        })
    
    return instructions
```

#### 5. マルチターン対話対応

**目的**: 対話履歴を考慮した質問応答

**実装内容**:
- `app/retrieval/conversation_rag.py`: 会話履歴管理
- LangChainのConversationBufferMemory使用
- 代名詞解決（「それ」「その法律」など）

**設計案**:
```python
# app/retrieval/conversation_rag.py
from langchain.memory import ConversationBufferMemory

class ConversationalRAGPipeline(RAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    
    def query_with_history(self, question: str) -> Dict[str, Any]:
        # 会話履歴を考慮したクエリ書き換え
        reformulated_query = self._reformulate_query(question)
        
        # 通常のRAG実行
        result = self.query(reformulated_query)
        
        # 履歴を保存
        self.memory.save_context({"input": question}, {"output": result["answer"]})
        
        return result
```

#### 6. 評価データセット拡充

**目的**: より包括的なRAG評価

**実装内容**:
- 追加の法律問題データセットの作成または取得
- カスタム評価メトリクス
  - 法令引用精度
  - 条文特定精度
- 人間評価フレームワーク

**実装案**:
```python
# scripts/evaluate_custom.py（将来の実装例）
def evaluate_custom_dataset():
    # カスタムデータセット読み込み
    # （データセットは別途準備が必要）
    with open("datasets/custom_law_qa/test_data.json") as f:
        test_data = json.load(f)
    
    # 評価実行
    results = []
    for item in test_data:
        question = item["question"]
        ground_truth = item["answer"]
        
        result = pipeline.query(question)
        
        # カスタムメトリクス計算
        citation_accuracy = calculate_citation_accuracy(
            result["citations"], ground_truth
        )
        
        results.append({
            "question": question,
            "citation_accuracy": citation_accuracy,
            "ragas_score": ragas_evaluate(result, ground_truth)
        })
    
    return results
```

### 長期目標（6ヶ月以上）

#### 7. マルチモーダル対応

**目的**: 図表を含む法令文書の処理

**実装内容**:
- PDF/画像からの表・図抽出
- Vision LLMによる図表理解
- 画像埋め込みとテキスト埋め込みの統合検索

#### 8. 判例検索統合

**目的**: 法令だけでなく判例も検索

**実装内容**:
- 判例データベースの構築
- 法令-判例リンク機能
- 判例ベース推論

#### 9. 自動更新パイプライン

**目的**: 法改正への自動追従

**実装内容**:
- e-Gov APIからの自動データ取得
- 差分更新機能
- インデックスの増分更新

## 拡張ポイント

### 新しいRetrieverの追加

**手順**:

1. `app/retrieval/`に新しいRetrieverクラスを作成
2. `BaseRetriever`を継承
3. `retrieve()`と`add_documents()`を実装

**例: Elasticsearchベースのretriever**:

```python
# app/retrieval/elasticsearch_retriever.py
from elasticsearch import Elasticsearch
from typing import List, Dict, Any
from .base import BaseRetriever, Document

class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, hosts: List[str], index_name: str):
        self.client = Elasticsearch(hosts)
        self.index_name = index_name
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """ドキュメントをElasticsearchにインデックス"""
        for i, doc in enumerate(documents):
            self.client.index(
                index=self.index_name,
                id=i,
                document=doc
            )
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Elasticsearch検索を実行"""
        response = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "law_title"]
                    }
                },
                "size": top_k
            }
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append(Document(
                page_content=hit['_source']['text'],
                metadata=hit['_source'],
                score=hit['_score']
            ))
        
        return results
```

3. `app/core/rag_config.py`に設定を追加
4. `scripts/build_index.py`でRetrieverを選択可能にする
5. テストを追加（`tests/test_elasticsearch_retriever.py`）

### 新しいRerankerの追加

**手順**:

1. `app/retrieval/reranker.py`に新しいRerankerクラスを追加
2. `BaseReranker`を継承
3. `rerank()`を実装

**例: ColBERTベースのreranker**:

```python
# app/retrieval/reranker.py
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

class ColBERTReranker(BaseReranker):
    def __init__(self, checkpoint_path: str):
        self.config = ColBERTConfig()
        self.checkpoint = Checkpoint(checkpoint_path, colbert_config=self.config)
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """ColBERTでリランキング"""
        passages = [doc.page_content for doc in documents]
        
        # ColBERTスコアリング
        scores = self.checkpoint.score(query, passages)
        
        # スコア更新とソート
        for doc, score in zip(documents, scores):
            doc.score = float(score)
        
        documents.sort(key=lambda x: x.score, reverse=True)
        
        return documents[:top_n]
```

### 新しい評価メトリクスの追加

**手順**:

1. `scripts/evaluate_ragas.py`にカスタムメトリクスを追加
2. RAGASの`Metric`クラスを継承

**例: 法令引用精度メトリクス**:

```python
# scripts/custom_metrics.py
from ragas.metrics.base import Metric
from typing import List, Dict

class CitationAccuracy(Metric):
    """法令引用の精度を評価"""
    
    def __init__(self):
        self.name = "citation_accuracy"
    
    def score(
        self,
        contexts: List[str],
        ground_truth: str,
        **kwargs
    ) -> float:
        """
        ground_truthに含まれる法令名・条文番号が
        contextsに含まれているかを評価
        """
        # ground_truthから法令参照を抽出
        gt_citations = self._extract_citations(ground_truth)
        
        # contextsから法令参照を抽出
        context_citations = []
        for ctx in contexts:
            context_citations.extend(self._extract_citations(ctx))
        
        # 一致率を計算
        if not gt_citations:
            return 1.0
        
        matched = sum(1 for cite in gt_citations if cite in context_citations)
        
        return matched / len(gt_citations)
    
    def _extract_citations(self, text: str) -> List[str]:
        """テキストから法令参照を抽出"""
        import re
        pattern = r'([^「」]+?法)(?:第(\d+)条)?(?:第(\d+)項)?'
        matches = re.findall(pattern, text)
        
        citations = []
        for law, article, paragraph in matches:
            cite = law
            if article:
                cite += f"第{article}条"
            if paragraph:
                cite += f"第{paragraph}項"
            citations.append(cite)
        
        return citations

# scripts/evaluate_ragas.pyで使用
from custom_metrics import CitationAccuracy

metrics = [faithfulness, answer_relevancy, context_precision, CitationAccuracy()]
result = evaluate(eval_dataset, metrics=metrics)
```

## デバッグガイド

### ログ出力

**基本ログ設定**:

```python
# scripts/query_cli.py に追加
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 使用例
logger.info(f"Loading retriever from {config.vector_store_path}")
logger.debug(f"Query: {query}, Top-K: {top_k}")
```

**デバッグレベルログ**:

```bash
# 環境変数で制御
export LOG_LEVEL=DEBUG
python3 scripts/query_cli.py --interactive
```

### Retriever検索結果の確認

```python
# デバッグスクリプト例
from app.retrieval.vector_retriever import VectorRetriever

retriever = VectorRetriever(
    embedding_model="intfloat/multilingual-e5-large",
    index_path="data/faiss_index/vector"
)

query = "博物館の目的"
results = retriever.retrieve(query, top_k=10)

for i, doc in enumerate(results, 1):
    print(f"\n[{i}] Score: {doc.score:.4f}")
    print(f"Law: {doc.metadata.get('law_title', 'N/A')}")
    print(f"Article: {doc.metadata.get('article', 'N/A')}")
    print(f"Text: {doc.page_content[:100]}...")
```

### LLMプロンプトの確認

```python
# RAGパイプラインでプロンプトを出力
from app.retrieval.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(retriever, llm_provider="ollama", llm_model="qwen3:8b")

question = "博物館法の目的は？"
documents = pipeline.retrieve_documents(question)
context = pipeline.format_context(documents)

# プロンプト確認
prompt = pipeline.prompt_template.format(context=context, question=question)
print("=" * 80)
print("PROMPT:")
print("=" * 80)
print(prompt)
```

### インデックスの診断

```python
# インデックス診断スクリプト
from pathlib import Path

index_path = Path("data/faiss_index/vector")

# FAISSインデックス情報
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# NOTE: allow_dangerous_deserialization=True を使用
# 信頼できるローカルファイルからのみロードすること（セキュリティリスクに注意）
vector_store = FAISS.load_local(
    str(index_path),
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Total documents: {vector_store.index.ntotal}")

# Docstore内のメタデータ確認
docstore = vector_store.docstore
print(f"Metadata count: {len(docstore._dict)}")

first_doc = next(iter(docstore._dict.values()), None)
if first_doc:
    print(f"Sample metadata: {first_doc.metadata}")
else:
    print("Docstore is empty")
```

### パフォーマンスプロファイリング

```python
# タイミング計測
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end - start:.2f}s")
        return result
    return wrapper

# 使用例
@timing_decorator
def retrieve_documents(query, top_k):
    return retriever.retrieve(query, top_k)

@timing_decorator
def generate_answer(context, question):
    return llm.generate(context, question)
```

### メモリ使用量監視

```python
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    print(f"Memory: {mem.rss / 1024 / 1024:.2f} MB")

# 使用例
print_memory_usage()
retriever = VectorRetriever(...)
print_memory_usage()
retriever.add_documents(documents)
print_memory_usage()
```

### CI/CD統合

将来的なCI/CD設定例（GitHub Actions）:

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv venv
        source .venv/bin/activate
        uv pip install -e .
        uv pip install pytest pytest-cov black ruff
    
    - name: Run linters
      run: |
        source .venv/bin/activate
        black --check app/ scripts/ tests/
        ruff check app/ scripts/ tests/
    
    - name: Run tests
      run: |
        source .venv/bin/activate
        pytest tests/ -v -m unit --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

今後の開発では、これらの設計原則を維持しながら、FastAPI Web API、Qdrant統合、ファインチューニングなどの機能拡張を進めていきます。

---

最終更新: 2024-11-04
