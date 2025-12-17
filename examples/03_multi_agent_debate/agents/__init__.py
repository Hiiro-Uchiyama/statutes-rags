"""
Multi-Agent Debate Agents

議論に参加するエージェントを定義。

【従来のエージェント（2役＋モデレーター）】
- DebaterAgent: 肯定的/批判的議論者
- ModeratorAgent: 調停者

【提案エージェント（3役割分離）】
- RetrieverAgent: 法令検索役
- InterpreterAgent: 法解釈役
- JudgeAgent: 最終判断者役

【法的根拠共有】
- CitationRegistry: 条文引用の一元管理
- Citation: 法的根拠の構造化表現
- ReasoningStep: 推論ステップの記録

注意: 親モジュールが数字で始まるため、sys.pathを使用してインポートしてください。
"""

# 従来のエージェント（後方互換性のため維持）
from .debater import DebaterAgent
from .moderator import ModeratorAgent

# 提案エージェント（3役割分離）
from .retriever_agent import RetrieverAgent
from .interpreter_agent import InterpreterAgent
from .judge_agent import JudgeAgent

# 法的根拠共有
from .citation import (
    Citation,
    CitationStatus,
    CitationRegistry,
    ReasoningStep,
)

__all__ = [
    # 従来
    "DebaterAgent",
    "ModeratorAgent",
    # 提案（3役割）
    "RetrieverAgent",
    "InterpreterAgent",
    "JudgeAgent",
    # 法的根拠
    "Citation",
    "CitationStatus",
    "CitationRegistry",
    "ReasoningStep",
]
