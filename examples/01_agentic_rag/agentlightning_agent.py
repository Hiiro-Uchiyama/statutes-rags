"""
Agent-lightning対応エージェント

AgenticRAGPipelineをAgent-lightningのLitAgentとしてラップします。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import agentlightning as agl
from agentlightning.emitter import emit_reward

from config import load_config
from pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class AgenticRAGLightningAgent(agl.LitAgent[Dict[str, Any]]):
    """Agent-lightning用のAgentic RAGエージェント。"""

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        self.pipeline = AgenticRAGPipeline(self.config)

    def rollout(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> float:
        """
        Agent-lightningのrollout処理。

        Args:
            task: {"prompt": str, "answer": str} を想定
            resources: Agent-lightningのリソース（未使用）
            rollout: Rolloutメタデータ

        Returns:
            float: 報酬（正解なら1.0, それ以外は0.0）
        """
        prompt = task.get("prompt") or task.get("question")
        correct_answer = task.get("answer")

        if not prompt or correct_answer is None:
            raise ValueError("taskには'prompt'（または'question'）と'answer'が必要です。")

        try:
            result = self.pipeline.query(prompt)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Pipeline execution failed: %s", exc, exc_info=True)
            emit_reward(0.0)
            return 0.0

        predicted_answer = self._extract_answer_label(result.get("answer", ""))
        correct_label = str(correct_answer).lower()
        reward = 1.0 if predicted_answer == correct_label else 0.0

        emit_reward(reward)
        logger.info(
            "Rollout %s: predicted=%s, correct=%s, reward=%.1f",
            rollout.rollout_id,
            predicted_answer,
            correct_label,
            reward,
        )

        return reward

    @staticmethod
    def _extract_answer_label(response: str) -> str:
        """
        LLM応答から最初に出現する選択肢ラベルを抽出。
        evaluate.extract_answer と同等の処理。
        """
        normalized = response.lower().strip()
        for label in ("a", "b", "c", "d"):
            if label in normalized:
                return label
        return normalized[:1] if normalized else ""


