"""
pytest設定とフィクスチャ（01_agentic_rag用）
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_llm():
    """モックLLMを提供"""
    llm = Mock()
    llm.invoke = Mock(return_value="モック応答")
    return llm


@pytest.fixture
def mock_document():
    """モックDocumentを提供"""
    from app.retrieval.base import Document
    
    return Document(
        page_content="これは第1条の内容です。",
        metadata={
            "law_title": "テスト法",
            "law_num": "令和元年法律第1号",
            "article": "1",
            "paragraph": "1",
            "item": None
        },
        score=0.95
    )


@pytest.fixture
def mock_documents(mock_document):
    """複数のモックDocumentを提供"""
    docs = []
    for i in range(5):
        from app.retrieval.base import Document
        doc = Document(
            page_content=f"これは第{i+1}条の内容です。",
            metadata={
                "law_title": "テスト法",
                "law_num": "令和元年法律第1号",
                "article": str(i+1),
                "paragraph": "1",
                "item": None
            },
            score=0.95 - i * 0.05
        )
        docs.append(doc)
    
    return docs


@pytest.fixture
def agentic_rag_config():
    """Agentic RAG設定を提供"""
    # examplesディレクトリをパスに追加
    examples_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(examples_dir))
    
    from config import AgenticRAGConfig
    
    return AgenticRAGConfig(
        max_iterations=2,
        confidence_threshold=0.7,
        enable_reasoning=True,
        enable_validation=True,
        llm_model="qwen3:8b",
        llm_temperature=0.1,
        retrieval_top_k=5
    )

