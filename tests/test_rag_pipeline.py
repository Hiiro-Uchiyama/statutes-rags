"""
RAGパイプラインのテスト
"""
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.base import Document


@pytest.mark.unit
def test_rag_pipeline_format_context():
    """コンテキスト整形のテスト"""
    from app.retrieval.rag_pipeline import RAGPipeline
    
    # モックRetrieverを作成
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    # テスト用ドキュメント
    documents = [
        Document(
            page_content="博物館は資料を収集する機関です。",
            metadata={
                "law_title": "博物館法",
                "article": "2",
                "paragraph": "1",
                "item": None
            },
            score=0.95
        ),
        Document(
            page_content="個人データの第三者提供には同意が必要です。",
            metadata={
                "law_title": "個人情報保護法",
                "article": "27",
                "paragraph": "1",
                "item": None
            },
            score=0.88
        )
    ]
    
    context = pipeline.format_context(documents)
    
    assert "博物館法" in context
    assert "第2条" in context
    assert "個人情報保護法" in context
    assert "第27条" in context


@pytest.mark.unit
def test_rag_pipeline_extract_citations():
    """引用情報抽出のテスト"""
    from app.retrieval.rag_pipeline import RAGPipeline
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    documents = [
        Document(
            page_content="テスト本文1",
            metadata={
                "law_title": "博物館法",
                "article": "1",
                "paragraph": "1",
                "item": None
            }
        ),
        Document(
            page_content="テスト本文2",
            metadata={
                "law_title": "博物館法",
                "article": "2",
                "paragraph": "1",
                "item": "1"
            }
        )
    ]
    
    citations = pipeline.extract_citations(documents)
    
    assert len(citations) > 0
    assert all("law_title" in c for c in citations)
    assert all("article" in c for c in citations)
    
    # 重複が排除されること
    law_titles = [c["law_title"] for c in citations]
    assert "博物館法" in law_titles


@pytest.mark.unit
def test_rag_pipeline_no_documents():
    """ドキュメントが見つからない場合のテスト"""
    from app.retrieval.rag_pipeline import RAGPipeline
    
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


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.slow
def test_rag_pipeline_retrieve_documents(sample_jsonl_data, temp_index_dir):
    """ドキュメント検索の統合テスト"""
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.rag_pipeline import RAGPipeline
    
    retriever = VectorRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path=str(temp_index_dir / "vector_rag")
    )
    retriever.add_documents(sample_jsonl_data)
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider="ollama",
            llm_model="test-model",
            top_k=2
        )
    
    documents = pipeline.retrieve_documents("博物館")
    
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)


@pytest.mark.unit
def test_rag_pipeline_prompt_template():
    """プロンプトテンプレートのテスト"""
    from app.retrieval.rag_pipeline import RAGPipeline
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    assert pipeline.prompt_template is not None
    assert "context" in pipeline.prompt_template.input_variables
    assert "question" in pipeline.prompt_template.input_variables


@pytest.mark.unit
def test_citation_deduplication():
    """引用の重複排除テスト"""
    from app.retrieval.rag_pipeline import RAGPipeline
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    # 同じ法令・条文のドキュメント
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


@pytest.mark.unit
def test_context_with_item_numbers():
    """号番号を含むコンテキストの整形テスト"""
    from app.retrieval.rag_pipeline import RAGPipeline
    
    mock_retriever = Mock()
    
    with patch('app.retrieval.rag_pipeline.Ollama'):
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider="ollama",
            llm_model="test-model"
        )
    
    documents = [
        Document(
            page_content="号の内容",
            metadata={
                "law_title": "テスト法",
                "article": "5",
                "paragraph": "2",
                "item": "3"
            }
        )
    ]
    
    context = pipeline.format_context(documents)
    
    assert "テスト法" in context
    assert "第5条" in context
    assert "第2項" in context
    assert "第3号" in context
