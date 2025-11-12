"""
MCQ Refiner Agent

検証で指摘された点を踏まえて事例を改善するエージェント。
"""
from typing import Dict, Any, List

from examples.shared.base_agent import BaseAgent


class MCQRefinerAgent(BaseAgent):
    """
    4択問題用シナリオをフィードバックに基づき修正するエージェント
    """

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        シナリオを改善

        Args:
            input_data: {
                "scenario": str,
                "feedback": List[str],
                "question": str,
                "choices": Dict[str, str],
                "correct_choice": str,
                "parser_result": Dict[str, Any],
                "target_length": int,
                "min_length": int,
                "max_length": int,
                "current_length": int
            }

        Returns:
            {
                "scenario": str,
                "character_count": int
            }
        """
        scenario = input_data.get("scenario", "")
        feedback: List[str] = input_data.get("feedback", [])
        question = input_data.get("question", "")
        choices: Dict[str, str] = input_data.get("choices", {})
        correct_choice = input_data.get("correct_choice", "")
        parser_result = input_data.get("parser_result", {})
        target_length = input_data.get("target_length", 500)
        min_length = input_data.get("min_length", 460)
        max_length = input_data.get("max_length", 540)
        current_length = input_data.get("current_length", len(scenario))
        length_deficit = max(0, min_length - current_length)

        correct_choice_text = parser_result.get("correct_choice_text") or choices.get(correct_choice, "")
        legal_requirements = parser_result.get("legal_requirements", [])

        feedback_lines = "\n".join(f"- {item}" for item in feedback) if feedback else "- 特に指摘事項なし"
        requirement_lines = "\n".join(f"- {req}" for req in legal_requirements) if legal_requirements else "- 条文要件に即した記述を補強"

        prompt = f"""以下の事例に対するフィードバックを踏まえて、内容を改善してください。

【元の事例】
{scenario}

【フィードバック】
{feedback_lines}

【問題文（再掲）】
{question}

【参照用の結論】
{correct_choice_text}

【強調したい要件】
{requirement_lines}

        【改善指示】
        - 指摘事項をすべて解消する
        - 現在の文字数は約{current_length}文字で、最低文字数まで残り{length_deficit}文字。不足がある場合は追加記述で補う
        - 最終的には{min_length}〜{max_length}文字（目標{target_length}文字）に収める
        - {min_length}文字未満になっている場合は段落を拡張・追加し、必ず下限を上回る
        - 12〜14文程度に分け、各文の長さを40〜60文字程度とし、段落を適切に構成する
        - 3段落構成（背景→協議・検討→最終判断）とし、段落ごとに2〜3文で構成する
        - 具体的な年・場所・登場人物の肩書を示し、会話や意思表明を「」付きで最低1箇所盛り込む（空の引用は不可）
        - 時系列で状況→問題→対応→結論を描写し、法的評価は事実の流れの中で示す
        - 上記の結論文面をそのまま引用せず、事実で根拠を説明する
        - 最終段落では条文要件と結論を整理しつつ、問題形式を示唆しない
        - 「選択肢」「正解」「回答」など問題形式を示唆する語やラベルを用いない
        - 「第○条」「第○項」「第○号」など条文番号・項号の明示的引用を避け、制度趣旨を言い換えて説明する
        - 他の選択肢に言及しない
        - 文章の途中や末尾に文字数などのメタ情報を付記しない

改善後の事例のみを出力してください。
"""

        response = self._safe_llm_invoke(prompt)

        if not response:
            self.logger.error("MCQRefinerAgent: refinement failed")
            return {
                "scenario": scenario,
                "character_count": len(scenario)
            }

        refined = response.strip()

        return {
            "scenario": refined,
            "character_count": len(refined)
        }

