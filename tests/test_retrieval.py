"""
Retrievalシステムのテスト
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.base import BaseRetriever, BaseReranker, Document


@pytest.mark.unit
def test_document_creation():
    """Documentオブジェクトの作成テスト"""
    doc = Document(
        page_content="テスト本文",
        metadata={"law_title": "テスト法", "article": "1"},
        score=0.95
    )
    
    assert doc.page_content == "テスト本文"
    assert doc.metadata["law_title"] == "テスト法"
    assert doc.metadata["article"] == "1"
    assert doc.score == 0.95


@pytest.mark.unit
def test_document_defaults():
    """Documentのデフォルト値テスト"""
    doc = Document(page_content="本文のみ")
    
    assert doc.page_content == "本文のみ"
    assert doc.metadata == {}
    assert doc.score == 0.0


@pytest.mark.unit
def test_base_retriever_interface():
    """BaseRetrieverインターフェースのテスト"""
    # 抽象クラスなので直接インスタンス化できないことを確認
    with pytest.raises(TypeError):
        BaseRetriever()


@pytest.mark.unit
def test_base_reranker_interface():
    """BaseRerankerインターフェースのテスト"""
    # 抽象クラスなので直接インスタンス化できないことを確認
    with pytest.raises(TypeError):
        BaseReranker()


@pytest.mark.integration
@pytest.mark.retrieval
@pytest.mark.slow
def test_vector_retriever_basic(sample_jsonl_data, temp_index_dir):
    """VectorRetrieverの基本機能テスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    
    # 小さいモデルでテスト
    retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector"),
        use_mmr=False
    )
    
    # ドキュメント追加
    retriever.add_documents(sample_jsonl_data)
    
    # 検索実行
    results = retriever.retrieve("博物館とは何ですか", top_k=2)
    
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)
    assert all(hasattr(doc, 'score') for doc in results)


@pytest.mark.integration
@pytest.mark.retrieval
def test_bm25_retriever_basic(sample_jsonl_data, temp_index_dir):
    """BM25Retrieverの基本機能テスト"""
    from app.retrieval.bm25_retriever import BM25Retriever
    
    # Note: テスト環境では依存関係を最小化するため、simpleトークナイザーを使用
    # 本番環境ではauto（SudachiPy優先）が推奨
    retriever = BM25Retriever(
        index_path=str(temp_index_dir / "bm25"),
        tokenizer="simple"
    )
    
    # ドキュメント追加
    retriever.add_documents(sample_jsonl_data)
    
    # 検索実行
    results = retriever.retrieve("博物館", top_k=2)
    
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.integration
@pytest.mark.retrieval
@pytest.mark.slow
def test_hybrid_retriever_basic(sample_jsonl_data, temp_index_dir):
    """HybridRetrieverの基本機能テスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    
    vector_retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector")
    )
    
    # Note: テスト環境では依存関係を最小化するため、simpleトークナイザーを使用
    bm25_retriever = BM25Retriever(
        index_path=str(temp_index_dir / "bm25"),
        tokenizer="simple"
    )
    
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    # ドキュメント追加
    hybrid.add_documents(sample_jsonl_data)
    
    # 検索実行
    results = hybrid.retrieve("博物館", top_k=2)
    
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.integration
@pytest.mark.retrieval
@pytest.mark.slow
def test_vector_retriever_save_load(sample_jsonl_data, temp_index_dir):
    """VectorRetrieverの保存/ロードテスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    
    index_path = str(temp_index_dir / "vector_save_test")
    
    # 作成と保存
    retriever1 = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=index_path
    )
    retriever1.add_documents(sample_jsonl_data)
    retriever1.save_index()
    
    # ロード
    retriever2 = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=index_path
    )
    
    # 検索できることを確認
    results = retriever2.retrieve("博物館", top_k=2)
    assert len(results) > 0


@pytest.mark.integration
@pytest.mark.retrieval
def test_bm25_retriever_save_load(sample_jsonl_data, temp_index_dir):
    """BM25Retrieverの保存/ロードテスト"""
    from app.retrieval.bm25_retriever import BM25Retriever
    
    index_path = str(temp_index_dir / "bm25_save_test")
    
    # 作成と保存
    retriever1 = BM25Retriever(index_path=index_path, tokenizer="simple")
    retriever1.add_documents(sample_jsonl_data)
    retriever1.save_index()
    
    # ロード
    retriever2 = BM25Retriever(index_path=index_path, tokenizer="simple")
    
    # 検索できることを確認
    results = retriever2.retrieve("博物館", top_k=2)
    assert len(results) > 0


@pytest.mark.unit
@pytest.mark.retrieval
def test_document_metadata_extraction():
    """ドキュメントメタデータの抽出テスト"""
    doc = Document(
        page_content="個人情報保護法第27条の規定",
        metadata={
            "law_title": "個人情報保護法",
            "article": "27",
            "paragraph": "1",
            "item": None
        },
        score=0.88
    )
    
    assert doc.metadata["law_title"] == "個人情報保護法"
    assert doc.metadata["article"] == "27"
    assert doc.metadata["paragraph"] == "1"
    assert doc.metadata["item"] is None


@pytest.mark.integration
@pytest.mark.retrieval
@pytest.mark.slow
def test_retriever_top_k_parameter(sample_jsonl_data, temp_index_dir):
    """top_kパラメータのテスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    
    retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector_topk")
    )
    retriever.add_documents(sample_jsonl_data)
    
    # top_k=1
    results_1 = retriever.retrieve("博物館", top_k=1)
    assert len(results_1) == 1
    
    # top_k=2
    results_2 = retriever.retrieve("博物館", top_k=2)
    assert len(results_2) <= 2  # データ数が少ない場合は2未満もあり得る


@pytest.mark.integration
@pytest.mark.retrieval
def test_bm25_tokenization():
    """BM25のトークン化テスト"""
    from app.retrieval.bm25_retriever import BM25Retriever
    
    # テスト環境では確実に動作するsimpleトークナイザーを使用
    retriever = BM25Retriever(tokenizer="simple")
    
    # 日本語のトークン化
    tokens = retriever.tokenize("これは博物館に関するテストです。")
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    
    # トークン化された結果に主要な文字が含まれることを確認
    text_content = "".join(tokens)
    assert "博物館" in text_content or "博" in text_content
