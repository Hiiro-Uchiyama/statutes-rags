# statutes RAG System テストドキュメント

## 目次

1. [テスト概要](#テスト概要)
2. [テスト環境](#テスト環境)
3. [テストファイル詳細](#テストファイル詳細)
4. [テスト実行方法](#テスト実行方法)
5. [カバレッジレポート](#カバレッジレポート)
6. [テストデータ](#テストデータ)

## テスト概要

statutes RAGシステムのテストは、pytest フレームワークを使用して実装されています。テストは以下の3つのカテゴリに分類されます:

### テスト階層

| テストタイプ | マーカー | 目的 | 実行時間 | 外部依存 |
|------------|---------|------|---------|---------|
| ユニットテスト | `@pytest.mark.unit` | 個別関数・メソッドの動作確認 | 高速（<1秒/テスト） | なし |
| 統合テスト | `@pytest.mark.integration` | 複数コンポーネントの連携確認 | 中速（1-10秒/テスト） | あり（モデル・インデックス） |
| 評価テスト | `@pytest.mark.eval` | エンドツーエンド評価 | 低速（>10秒/テスト） | あり（LLM・データセット） |

### 追加マーカー

特定の機能領域を示すマーカー:

- `@pytest.mark.xmlparse`: XML解析テスト
- `@pytest.mark.retrieval`: 検索システムテスト
- `@pytest.mark.rag`: RAGパイプラインテスト
- `@pytest.mark.slow`: 時間のかかるテスト

## テスト環境

### 前提条件

```bash
# 仮想環境の有効化
source .venv/bin/activate

# テスト依存のインストール
pip install pytest pytest-cov pytest-asyncio pytest-mock
```

### pytest設定 (pytest.ini)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (may require external services)
    slow: Slow tests (may take several minutes)
    xmlparse: XML parsing tests
    retrieval: Retrieval system tests
    rag: RAG pipeline tests
    eval: Evaluation tests
```

## テストファイル詳細

### 1. tests/conftest.py

pytest設定とフィクスチャの定義。全テストで共有される基盤。

**主要フィクスチャ**:

#### project_root_path
```python
@pytest.fixture
def project_root_path():
    """プロジェクトルートのパス"""
    return Path(__file__).parent.parent
```

#### sample_xml_content
```python
@pytest.fixture
def sample_xml_content():
    """サンプルXMLコンテンツ（博物館法）"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Law Era="Showa" Lang="ja" LawType="Act" Num="285" Year="26">
    <LawNum>昭和二十六年法律第二百八十五号</LawNum>
    <LawBody>
        <LawTitle Kana="はくぶつかんほう">博物館法</LawTitle>
        <MainProvision>
            <Chapter Num="1">
                <Article Num="1">
                    <ArticleCaption>（目的）</ArticleCaption>
                    <Paragraph Num="1">
                        <ParagraphSentence>
                            <Sentence>この法律は、博物館の設置及び運営に関して必要な事項を定める。</Sentence>
                        </ParagraphSentence>
                    </Paragraph>
                </Article>
            </Chapter>
        </MainProvision>
    </LawBody>
</Law>"""
```

#### sample_jsonl_data
```python
@pytest.fixture
def sample_jsonl_data():
    """サンプルJSONLデータ（Pythonリスト）"""
    return [
        {
            "law_title": "博物館法",
            "law_num": "昭和二十六年法律第二百八十五号",
            "article": "1",
            "article_caption": "（目的）",
            "article_title": "第一条",
            "paragraph": "1",
            "item": None,
            "text": "この法律は、博物館の設置及び運営に関して必要な事項を定める。"
        },
        {
            "law_title": "個人情報保護法",
            "law_num": "平成十五年法律第五十七号",
            "article": "27",
            "article_caption": "（第三者提供の制限）",
            "article_title": "第二十七条",
            "paragraph": "1",
            "item": None,
            "text": "個人情報取扱事業者は、次に掲げる場合を除くほか、あらかじめ本人の同意を得ないで、個人データを第三者に提供してはならない。"
        }
    ]
```

#### temp_index_dir
```python
@pytest.fixture
def temp_index_dir(tmp_path):
    """一時インデックスディレクトリ（自動クリーンアップ）"""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    yield index_dir
    # テスト後に自動削除
    if index_dir.exists():
        shutil.rmtree(index_dir)
```

#### mock_config
```python
@pytest.fixture
def mock_config():
    """モック設定（軽量モデルで高速テスト）"""
    from app.core.rag_config import RAGConfig, EmbeddingConfig, LLMConfig
    
    return RAGConfig(
        embedding=EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 軽量モデル
            dimension=384
        ),
        llm=LLMConfig(
            provider="ollama",
            model_name="qwen3:8b",
            temperature=0.1,
            max_tokens=512
        ),
        retriever=RetrieverConfig(
            retriever_type="vector",
            top_k=3,
            use_mmr=False
        ),
        reranker=RerankerConfig(
            enabled=False
        ),
        vector_store_path="/tmp/test_index",
        data_path="/tmp/test_data.jsonl"
    )
```

### 2. tests/test_config.py

設定管理のテスト（179行）。

**テストケース例**:

```python
@pytest.mark.unit
def test_load_default_config():
    """デフォルト設定のロード"""
    from app.core.rag_config import load_config
    
    config = load_config()
    
    assert config.embedding.provider == "huggingface"
    assert config.llm.provider == "ollama"
    assert config.retriever.retriever_type in ["vector", "bm25", "hybrid"]

@pytest.mark.unit
def test_embedding_config_from_env(monkeypatch):
    """環境変数からの設定読み込み"""
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_DIM", "512")
    
    from app.core.rag_config import EmbeddingConfig
    
    config = EmbeddingConfig()
    
    assert config.model_name == "test-model"
    assert config.dimension == 512

@pytest.mark.unit
def test_config_validation():
    """設定のバリデーション"""
    from app.core.rag_config import RetrieverConfig
    from pydantic import ValidationError
    
    # 不正なretriever_type
    with pytest.raises(ValidationError):
        RetrieverConfig(retriever_type="invalid")
```

### 3. tests/test_preprocessing.py

XML前処理のテスト（152行）。

**テストケース例**:

```python
@pytest.mark.unit
@pytest.mark.xmlparse
def test_extract_text_recursive():
    """再帰的テキスト抽出"""
    from scripts.preprocess_egov_xml import extract_text_recursive
    import xml.etree.ElementTree as ET
    
    xml = """<Article>
        <ArticleTitle>第一条</ArticleTitle>
        <Paragraph>
            <Sentence>テスト文</Sentence>
        </Paragraph>
    </Article>"""
    
    element = ET.fromstring(xml)
    text = extract_text_recursive(element)
    
    assert "第一条" in text
    assert "テスト文" in text

@pytest.mark.unit
@pytest.mark.xmlparse
def test_parse_article(sample_xml_content):
    """条文パース"""
    from scripts.preprocess_egov_xml import parse_law_xml
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(sample_xml_content)
        xml_path = f.name
    
    chunks = parse_law_xml(Path(xml_path))
    
    assert len(chunks) > 0
    assert chunks[0]["law_title"] == "博物館法"
    assert chunks[0]["article"] == "1"
    assert "博物館の設置" in chunks[0]["text"]

@pytest.mark.unit
@pytest.mark.xmlparse
def test_parse_article_with_items():
    """号を含む条文のパース"""
    from scripts.preprocess_egov_xml import parse_article
    import xml.etree.ElementTree as ET
    
    xml = """<Article Num="5">
        <ArticleTitle>第五条</ArticleTitle>
        <Paragraph Num="1">
            <Item Num="1">
                <ItemTitle>一</ItemTitle>
                <ItemSentence>第一号の内容</ItemSentence>
            </Item>
            <Item Num="2">
                <ItemTitle>二</ItemTitle>
                <ItemSentence>第二号の内容</ItemSentence>
            </Item>
        </Paragraph>
    </Article>"""
    
    element = ET.fromstring(xml)
    chunks = parse_article(element, "テスト法", "テスト法令番号")
    
    # 各号が個別のチャンクになること
    assert len(chunks) == 2
    assert chunks[0]["item"] == "1"
    assert chunks[1]["item"] == "2"
```

### 4. tests/test_retrieval.py

Retrieverテスト（244行）。

**テストケース例**:

```python
@pytest.mark.unit
def test_document_model():
    """Documentモデルのテスト"""
    from app.retrieval.base import Document
    
    doc = Document(
        page_content="テスト本文",
        metadata={"law_title": "テスト法", "article": "1"},
        score=0.95
    )
    
    assert doc.page_content == "テスト本文"
    assert doc.metadata["law_title"] == "テスト法"
    assert doc.score == 0.95

@pytest.mark.integration
@pytest.mark.retrieval
def test_vector_retriever_basic(sample_jsonl_data, temp_index_dir):
    """ベクトル検索の基本動作"""
    from app.retrieval.vector_retriever import VectorRetriever
    
    retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector_test")
    )
    
    # ドキュメント追加
    retriever.add_documents(sample_jsonl_data)
    
    # 検索実行
    results = retriever.retrieve("博物館", top_k=2)
    
    assert len(results) > 0
    assert all(isinstance(doc.score, float) for doc in results)
    assert "博物館" in results[0].page_content or "博物館" in results[0].metadata.get("law_title", "")

@pytest.mark.integration
@pytest.mark.retrieval
def test_bm25_retriever_basic(sample_jsonl_data, temp_index_dir):
    """BM25検索の基本動作"""
    from app.retrieval.bm25_retriever import BM25Retriever
    
    retriever = BM25Retriever(index_path=str(temp_index_dir / "bm25_test"))
    
    retriever.add_documents(sample_jsonl_data)
    
    results = retriever.retrieve("博物館", top_k=2)
    
    assert len(results) > 0
    assert all(hasattr(doc, 'score') for doc in results)

@pytest.mark.integration
@pytest.mark.retrieval
@pytest.mark.slow
def test_hybrid_retriever(sample_jsonl_data, temp_index_dir):
    """ハイブリッド検索"""
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    
    vector_retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector_hybrid")
    )
    bm25_retriever = BM25Retriever(
        index_path=str(temp_index_dir / "bm25_hybrid")
    )
    
    hybrid_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    hybrid_retriever.add_documents(sample_jsonl_data)
    
    results = hybrid_retriever.retrieve("個人情報", top_k=3)
    
    assert len(results) > 0
    # スコアが降順であること
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score

@pytest.mark.unit
def test_bm25_tokenizer_sudachi():
    """SudachiPyトークナイザのテスト"""
    from app.retrieval.bm25_retriever import BM25Retriever
    
    retriever = BM25Retriever(tokenizer="sudachi")
    
    text = "これはテストです。"
    tokens = retriever.tokenize(text)
    
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)

@pytest.mark.unit
def test_bm25_tokenizer_fallback():
    """フォールバックトークナイザのテスト"""
    from app.retrieval.bm25_retriever import BM25Retriever
    
    retriever = BM25Retriever(tokenizer="simple")
    
    text = "これはテストです。"
    tokens = retriever.tokenize(text)
    
    assert len(tokens) > 0
```

### 5. tests/test_rag_pipeline.py

RAGパイプラインテスト（244行）。

**テストケース例**:

```python
@pytest.mark.unit
def test_rag_pipeline_format_context():
    """コンテキスト整形"""
    from app.retrieval.rag_pipeline import RAGPipeline
    from app.retrieval.base import Document
    from unittest.mock import Mock, patch
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    documents = [
        Document(
            page_content="博物館は資料を収集する機関です。",
            metadata={
                "law_title": "博物館法",
                "article": "2",
                "paragraph": "1"
            },
            score=0.95
        )
    ]
    
    context = pipeline.format_context(documents)
    
    assert "博物館法" in context
    assert "第2条" in context
    assert "第1項" in context
    assert "博物館は資料を収集する機関です。" in context

@pytest.mark.unit
def test_rag_pipeline_extract_citations():
    """引用情報抽出"""
    from app.retrieval.rag_pipeline import RAGPipeline
    from app.retrieval.base import Document
    from unittest.mock import Mock, patch
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    documents = [
        Document(
            page_content="本文1",
            metadata={"law_title": "博物館法", "article": "1", "paragraph": "1"}
        ),
        Document(
            page_content="本文2",
            metadata={"law_title": "博物館法", "article": "2", "paragraph": "1"}
        ),
        Document(
            page_content="本文3",
            metadata={"law_title": "個人情報保護法", "article": "27", "paragraph": "1"}
        )
    ]
    
    citations = pipeline.extract_citations(documents)
    
    assert len(citations) >= 2  # 最低2つの異なる引用
    assert all("law_title" in c for c in citations)
    assert all("article" in c for c in citations)

@pytest.mark.unit
def test_rag_pipeline_no_documents():
    """ドキュメントが見つからない場合"""
    from app.retrieval.rag_pipeline import RAGPipeline
    from unittest.mock import Mock, patch
    
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = []
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    result = pipeline.query("テスト質問")
    
    assert result["answer"] == "関連する法令条文が見つかりませんでした。"
    assert result["citations"] == []
    assert result["contexts"] == []

@pytest.mark.unit
def test_citation_deduplication():
    """引用の重複排除"""
    from app.retrieval.rag_pipeline import RAGPipeline
    from app.retrieval.base import Document
    from unittest.mock import Mock, patch
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    # 同じ法令・条文の重複ドキュメント
    documents = [
        Document(
            page_content="本文1",
            metadata={"law_title": "博物館法", "article": "1", "paragraph": "1"}
        ),
        Document(
            page_content="本文2",
            metadata={"law_title": "博物館法", "article": "1", "paragraph": "1"}
        ),
        Document(
            page_content="本文3",
            metadata={"law_title": "博物館法", "article": "1", "paragraph": "2"}
        )
    ]
    
    citations = pipeline.extract_citations(documents)
    
    # 重複が排除されること
    unique_keys = set((c["law_title"], c["article"], c["paragraph"]) for c in citations)
    assert len(citations) == len(unique_keys)
```

### 6. tests/test_rag_components.py

コンポーネント統合テスト（65行）。

**テストケース例**:

```python
@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.slow
def test_end_to_end_rag(sample_jsonl_data, temp_index_dir):
    """エンドツーエンドRAGテスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.rag_pipeline import RAGPipeline
    from unittest.mock import patch, Mock
    
    # Retriever構築
    retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "e2e_vector")
    )
    retriever.add_documents(sample_jsonl_data)
    
    # モックLLMを使用（実際のOllamaは不要）
    mock_llm = Mock()
    mock_llm.return_value = "モック回答"
    
    with patch('app.retrieval.rag_pipeline.Ollama', return_value=mock_llm):
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
        
        # 質問実行
        result = pipeline.query("博物館法の目的は？")
    
    # 結果検証
    assert "answer" in result
    assert "citations" in result
    assert "contexts" in result
    assert len(result["contexts"]) > 0
```

## テスト実行方法

### 基本実行

```bash
# 全テスト実行
pytest tests/ -v

# ユニットテストのみ（高速）
pytest tests/ -v -m unit

# 統合テストのみ
pytest tests/ -v -m integration

# 特定のテストファイル
pytest tests/test_retrieval.py -v

# 特定のテスト関数
pytest tests/test_retrieval.py::test_vector_retriever_basic -v
```

### Makefile経由

```bash
# ユニットテスト（デフォルト）
make test

# 全テスト
make test-all

# 統合テスト
make test-integration

# カバレッジ付き
make test-coverage

# 高速テスト（slowマーカー除外）
make test-quick
```

### マーカー指定

```bash
# 複数マーカーのAND条件
pytest tests/ -v -m "unit and xmlparse"

# OR条件
pytest tests/ -v -m "unit or integration"

# NOT条件（slowを除外）
pytest tests/ -v -m "unit and not slow"

# 特定の機能領域
pytest tests/ -v -m retrieval
pytest tests/ -v -m rag
```

### 並列実行（高速化）

```bash
# pytest-xdist使用
pip install pytest-xdist

# 4並列で実行
pytest tests/ -v -n 4

# 自動並列数（CPU数）
pytest tests/ -v -n auto
```

### 詳細出力

```bash
# より詳細な出力
pytest tests/ -vv

# 標準出力を表示
pytest tests/ -v -s

# 失敗したテストのみ再実行
pytest tests/ -v --lf

# 最初の失敗で停止
pytest tests/ -v -x
```

## カバレッジレポート

### カバレッジ計測

```bash
# テスト実行+カバレッジ計測
pytest tests/ -v --cov=app --cov=scripts

# HTML報告書生成
pytest tests/ -v --cov=app --cov=scripts --cov-report=html

# ターミナルに詳細表示
pytest tests/ -v --cov=app --cov=scripts --cov-report=term-missing
```

### Makefile経由

```bash
make test-coverage
# htmlcov/index.htmlが生成される
```

### カバレッジレポートの見方

HTMLレポート（`htmlcov/index.html`）を開くと:

1. **全体サマリ**: モジュールごとのカバレッジ率
2. **ファイル詳細**: 各ファイルの実行行・未実行行
3. **カラーコード**:
   - 緑: 実行された行
   - 赤: 実行されなかった行
   - 黄: 部分的に実行された行

### 現在のカバレッジ状況

```bash
# カバレッジ確認
pytest tests/ -v --cov=app --cov=scripts --cov-report=term

# 期待される出力例:
# app/core/rag_config.py           95%
# app/retrieval/base.py            100%
# app/retrieval/vector_retriever.py 87%
# app/retrieval/bm25_retriever.py  82%
# app/retrieval/hybrid_retriever.py 90%
# app/retrieval/reranker.py        75%
# app/retrieval/rag_pipeline.py    88%
# scripts/preprocess_egov_xml.py   78%
# scripts/build_index.py           65%
# ------------------------------------------
# TOTAL                            85%
```

## テストデータ

### サンプルデータの構成

テストでは以下のデータを使用:

1. **博物館法XML**: 2条3項のサンプル条文
2. **JSONL形式データ**: 博物館法と個人情報保護法の2件
3. **lawqa_jpサンプル**: 2件の質問・回答ペア

### データ生成

新しいテストデータを追加する場合:

```python
# tests/conftest.pyに追加
@pytest.fixture
def sample_new_data():
    """新しいテストデータ"""
    return [
        {
            "law_title": "新規法",
            "law_num": "令和五年法律第一号",
            "article": "1",
            "text": "テスト条文"
        }
    ]
```

### 一時ファイル・ディレクトリ

pytestの`tmp_path`フィクスチャを使用して一時領域を作成:

```python
@pytest.mark.unit
def test_with_temp_file(tmp_path):
    """一時ファイルを使用するテスト"""
    test_file = tmp_path / "test.jsonl"
    
    with open(test_file, "w") as f:
        f.write('{"test": "data"}\n')
    
    # テスト実行...
    
    # tmp_pathはテスト終了後に自動削除される
```

## トラブルシューティング

### よくあるエラー

#### 1. インポートエラー

```
ModuleNotFoundError: No module named 'app'
```

**解決策**:
```bash
# パスが正しく設定されているか確認
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# または conftest.pyで設定（既に実装済み）
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### 2. トークナイザーエラー

```
SudachiPy not available. Using simple tokenizer.
```

**解決策**:
```bash
# トークナイザーをインストール
pip install sudachipy sudachidict-core janome

# テスト実行
pytest tests/ -v -m unit
```

#### 3. Ollamaエラー（統合テスト）

```
httpx.ConnectError: Connection refused
```

**解決策**:
```bash
# ユニットテストのみ実行（Ollama不要）
pytest tests/ -v -m unit

# または統合テストをスキップ
pytest tests/ -v -m "not integration"
```

#### 4. メモリ不足

```
MemoryError: Cannot allocate memory
```

**解決策**:
```bash
# 並列実行を減らす
pytest tests/ -v -n 2

# または統合テストを除外
pytest tests/ -v -m "unit and not slow"
```

### デバッグ方法

```bash
# pdbデバッガ起動
pytest tests/ -v --pdb

# 失敗時にpdb起動
pytest tests/ -v --pdb --pdbcls=IPython.terminal.debugger:Pdb

# 特定のテストにブレークポイント
# テストコード内に追加:
import pdb; pdb.set_trace()
```

テストの追加・修正時は、この設計方針に従ってください。

---

最終更新: 2024-11-04
