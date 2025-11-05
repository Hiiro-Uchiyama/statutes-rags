"""
Multi-Agent Debate Agents

議論に参加するエージェントを定義。

注意: 親モジュールが数字で始まるため、sys.pathを使用してインポートしてください。
"""

from .debater import DebaterAgent
from .moderator import ModeratorAgent

__all__ = [
    "DebaterAgent",
    "ModeratorAgent",
]

