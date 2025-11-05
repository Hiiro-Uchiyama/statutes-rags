"""
pytest設定とフィクスチャ（02_mcp_egov_agent用）
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
def mcp_egov_config():
    """MCP e-Gov Agent設定を提供"""
    # 数字で始まるモジュール名のため、動的インポートを使用
    import importlib
    config_module = importlib.import_module('examples.02_mcp_egov_agent.config')
    MCPEgovConfig = config_module.MCPEgovConfig
    
    return MCPEgovConfig(
        api_base_url="https://laws.e-gov.go.jp/api/2",
        api_timeout=30,
        api_max_retries=3,
        prefer_api=True,
        fallback_to_local=True,
        retrieval_top_k=10
    )


@pytest.fixture
def mock_api_response():
    """モックAPI応答を提供"""
    return {
        "laws": [
            {
                "law_id": "12345",
                "law_num": "令和元年法律第1号",
                "law_title": "テスト法"
            }
        ]
    }

