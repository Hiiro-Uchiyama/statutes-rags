"""
MCQ Case Generator Agent

4択問題の正解選択肢を裏付ける具体的な事例を生成するエージェント。
"""
from typing import Dict, Any, List

from examples.shared.base_agent import BaseAgent


class MCQCaseGeneratorAgent(BaseAgent):
    """
    正解選択肢に沿った具体的な事例（約500文字）を生成するエージェント
    """

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        具体的なシナリオを生成

        Args:
            input_data: {
                "question": str,
                "context": str,
                "choices": Dict[str, str],
                "correct_choice": str,
                "parser_result": Dict[str, Any],
                "target_length": int,
                "min_length": int,
                "max_length": int
            }

        Returns:
            {
                "scenario": str,
                "character_count": int
            }
        """
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        choices: Dict[str, str] = input_data.get("choices", {})
        correct_choice = input_data.get("correct_choice", "")
        parser_result = input_data.get("parser_result", {})
        target_length = input_data.get("target_length", 500)
        min_length = input_data.get("min_length", 460)
        max_length = input_data.get("max_length", 540)

        correct_choice_text = parser_result.get("correct_choice_text") or choices.get(correct_choice, "")
        legal_requirements: List[str] = parser_result.get("legal_requirements", [])
        key_facts: List[str] = parser_result.get("key_facts", [])
        scenario_outline = parser_result.get("scenario_outline", "")
        tone_guideline = parser_result.get("tone_guideline", "")

        requirement_lines = "\n".join(f"- {req}" for req in legal_requirements) if legal_requirements else "- 条文の要件を端的に満たす"
        fact_lines = "\n".join(f"- {fact}" for fact in key_facts) if key_facts else "- 登場人物・時系列・意思決定の流れを明確にする"

        prompt = f"""あなたは法律教育向けのケースライターです。以下の4択問題の正解選択肢を裏付ける具体的な事例を日本語で作成してください。

【問題文】
{question}

【参照用の結論】
{correct_choice_text}

【重要な要件・ポイント】
{requirement_lines}

【事例に盛り込みたい事実】
{fact_lines}

【コンテキスト（参照用任意）】
{context}

【事例作成の指示】
- 文字数は全角換算で約{target_length}文字（許容範囲: {min_length}〜{max_length}文字、{min_length}文字未満は禁止）
- 必要に応じて事実や背景を追加し、460文字以上を確保する
- {min_length}文字未満となる場合は段落を追記・拡張して必ず下限を上回る
- 最低でも2名の固有名・肩書を示し、誰が何を決めたのかを描写する
- 具体的な年・場所・経緯を設定し、時系列に沿って状況→問題→対応→結論を記述する
- 少なくとも一度、登場人物の具体的な発言や意思表明を「」で示し、空の引用は厳禁
- 3段落構成（背景→協議・検討→最終判断）とし、段落ごとに2〜3文で構成する
- 12〜14文程度に分け、各文は40〜60文字を目安に散文で書く（改行は段落単位で可）
- 法的評価は事実の描写を通じて示し、上記の結論文面をそのまま転載しない
- 文章内では「選択肢」「正解」「回答」といった語や選択肢ラベル（a, b, c, d 等）を用いない
- 最終段落で免除の判断根拠と条文要件がそろったことを簡潔に整理するが、問題形式を示唆しない
- 条文番号や項・号などの表記（例:「第5条第6項」）を直接引用せず、制度の趣旨と要件を言い換えて説明する
- 対象は実務家志望の受験生。専門用語を用いつつも読みやすくする
- 他の選択肢が誤りとなる理由には踏み込まない
- 文章の途中や末尾に文字数などのメタ情報を付記しない
{f"- 筆致・トーン: {tone_guideline}" if tone_guideline else ""}

【出力形式】
事例のみを記述し、余計なヘッダーや箇条書きは使用しない。
"""

        response = self._safe_llm_invoke(prompt)

        if not response:
            self.logger.error("MCQCaseGeneratorAgent: generation failed")
            return {
                "scenario": "",
                "character_count": 0
            }

        scenario = response.strip()
        character_count = len(scenario)

        return {
            "scenario": scenario,
            "character_count": character_count
        }

