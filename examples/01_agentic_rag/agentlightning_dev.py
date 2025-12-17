"""
Agent-lightning開発用デモスクリプト

小規模データセットを用いてAgentic RAGエージェントをTrainer.devで動作確認します。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import agentlightning as agl
from agentlightning.algorithm.fast import Baseline
from agentlightning.instrumentation import instrument_all
from agentlightning.adapter import TraceAdapter

from agentlightning_agent import AgenticRAGLightningAgent
from evaluate import create_multiple_choice_prompt

logger = logging.getLogger(__name__)


class NoOpAdapter(TraceAdapter[Any]):
    """開発用途: Spanがなくてもエラーにならないアダプタ。"""

    def adapt(self, spans: List[Any]) -> List[Any]:  # type: ignore[override]
        return []


def load_demo_dataset(dataset_path: Path, limit: int) -> List[Dict[str, Any]]:
    """評価データセットから少数件を抽出し、Trainer用フォーマットへ変換。"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "samples" in data:
            data = data["samples"]
        else:
            data = list(data.values())

    tasks: List[Dict[str, Any]] = []
    for sample in data[:limit]:
        question = sample.get("問題文", sample.get("question", ""))
        choices_raw = sample.get("選択肢", sample.get("choices", []))
        if isinstance(choices_raw, str):
            choices = [line.strip() for line in choices_raw.split("\n") if line.strip()]
        else:
            choices = choices_raw

        prompt = create_multiple_choice_prompt(question, choices)
        answer = str(sample.get("output", sample.get("answer", "a"))).lower()
        tasks.append({"prompt": prompt, "answer": answer})

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-lightning dev run for Agentic RAG.")
    project_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "datasets/lawqa_jp/data/selection.json",
        help="評価用データセットのJSONパス（デフォルト: <project_root>/datasets/lawqa_jp/data/selection.json）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="使用する問題数（小規模動作確認用）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    instrument_all()
    logger.info("Agent-lightning instrumentation enabled for dev run.")

    dataset = load_demo_dataset(args.dataset, args.limit)
    if not dataset:
        raise ValueError("デモ用データセットが空です。データファイルを確認してください。")

    agent = AgenticRAGLightningAgent()
    trainer = agl.Trainer(
        dev=True,
        n_runners=1,
        max_rollouts=args.limit,
        initial_resources={},
        adapter=NoOpAdapter(),
        algorithm=Baseline(n_epochs=1, train_split=0.5, span_verbosity="keys"),
    )

    logger.info("Starting Trainer.dev with %d tasks.", len(dataset))
    trainer.dev(agent, train_dataset=dataset)
    logger.info("Trainer.dev completed.")


if __name__ == "__main__":
    main()


