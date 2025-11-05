"""
Multi-Agent Debate

複数のエージェントが法的解釈について議論し、合意形成を行うシステム。

主要コンポーネント:
- DebaterAgent: 肯定的/批判的解釈を行うエージェント
- ModeratorAgent: 議論を調整し、最終判断を行うエージェント
- DebateWorkflow: LangGraphベースの議論ワークフロー

注意: このモジュールは数字で始まるディレクトリ名のため、
通常のPythonインポートではなく、sys.pathを使用してインポートする必要があります。
"""

__version__ = "0.1.0"

# このモジュールは数字で始まるため、直接インポートは構文エラーになります
# 代わりに以下のように使用してください：
#
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path("examples/03_multi_agent_debate")))
# from workflow import DebateWorkflow
# from config import MultiAgentDebateConfig, load_config

__all__ = [
    "DebateWorkflow",
    "MultiAgentDebateConfig",
    "load_config",
]

