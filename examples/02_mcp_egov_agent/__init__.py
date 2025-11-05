"""
e-Gov API MCP Agent

e-Gov API v2を使用して最新の法令データを動的に取得するエージェントシステム。
LangChain Toolsを基盤とし、API優先のハイブリッド検索戦略を採用。
"""

__version__ = "0.1.0"

from .config import MCPEgovConfig, load_config
from .agents import EGovAPIClient, MCPEgovAgent
from .pipeline import MCPEgovPipeline

__all__ = [
    "MCPEgovConfig",
    "load_config",
    "EGovAPIClient",
    "MCPEgovAgent",
    "MCPEgovPipeline",
]

