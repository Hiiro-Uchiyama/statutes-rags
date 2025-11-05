"""
Multi-Agent Debate Agents

議論に参加するエージェントを定義。

注意: 親モジュールが数字で始まるため、sys.pathを使用してインポートしてください。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.03_multi_agent_debate.agents.debater import DebaterAgent
from examples.03_multi_agent_debate.agents.moderator import ModeratorAgent

__all__ = [
    "DebaterAgent",
    "ModeratorAgent",
]

