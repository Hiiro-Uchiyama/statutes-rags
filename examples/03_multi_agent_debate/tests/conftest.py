"""
pytest設定とフィクスチャ（03_multi_agent_debate用）
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent.parent
sys.path.insert(0, str(debate_dir))


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
def debate_config():
    """Multi-Agent Debate設定を提供"""
    from config import MultiAgentDebateConfig
    
    return MultiAgentDebateConfig(
        max_debate_rounds=3,
        agreement_threshold=0.8,
        llm_model="qwen3:8b",
        llm_temperature=0.1,
        retrieval_top_k=10
    )





