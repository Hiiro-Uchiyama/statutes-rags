"""
MCQ Parser Agent

4択問題の正解肢と法的要点を抽出するエージェント。
"""
from typing import Dict, Any, List

from examples.shared.base_agent import BaseAgent


class MCQParserAgent(BaseAgent):
    """
    4択問題から正解肢の要点を抽出するエージェント

    責務:
    - 問題文とコンテキストの要旨整理
    - 正解選択肢の論点把握
    - シナリオ生成用の骨子作成
    """

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        正解肢の要点を抽出

        Args:
            input_data: {
                "question": str,
                "context": str,
                "choices": Dict[str, str],
                "correct_choice": str
            }

        Returns:
            {
                "correct_choice_label": str,
                "correct_choice_text": str,
                "legal_requirements": List[str],
                "key_facts": List[str],
                "scenario_outline": str,
                "tone_guideline": str
            }
        """
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        choices: Dict[str, str] = input_data.get("choices", {})
        correct_choice = input_data.get("correct_choice", "").strip().lower()

        choice_texts = "\n".join(
            f"{label}) {text}"
            for label, text in choices.items()
        )

        prompt = f"""あなたは法律教育向けのケースライターです。以下の4択問題について、
正解となる選択肢の根拠・要件・論点を整理し、具体例を作りやすい骨子を抽出してください。

【問題文】
{question}

【選択肢】
{choice_texts}

【正解】選択肢{correct_choice}

【コンテキスト（参考）】
{context}

次のJSON形式で出力してください。文章は日本語で簡潔に記述してください。

```json
{{
  "correct_choice_label": "正解の選択肢ラベル（例: \"b\"）",
  "correct_choice_text": "正解選択肢の内容を要約または引用",
  "legal_requirements": ["適用要件1", "適用要件2"],
  "key_facts": ["事例で押さえるべき事実1", "事例で押さえるべき事実2"],
  "scenario_outline": "シナリオの流れ（起点→問題→結論）を2〜3文で要約",
  "tone_guideline": "筆致や読み手への配慮（例: 実務家向けにややフォーマルに）"
}}
```"""

        response = self._safe_llm_invoke(prompt)

        default_result = {
            "correct_choice_label": correct_choice,
            "correct_choice_text": choices.get(correct_choice, ""),
            "legal_requirements": [],
            "key_facts": [],
            "scenario_outline": "",
            "tone_guideline": ""
        }

        if not response:
            self.logger.warning("MCQParserAgent: no response, using defaults")
            return default_result

        parsed = self._parse_json_response(response, default=default_result)

        # フィールドの整形
        parsed["correct_choice_label"] = str(parsed.get("correct_choice_label", correct_choice)).strip().lower()
        if not parsed["correct_choice_label"]:
            parsed["correct_choice_label"] = correct_choice

        parsed["correct_choice_text"] = parsed.get("correct_choice_text") or choices.get(parsed["correct_choice_label"], "")
        if not isinstance(parsed.get("legal_requirements"), list):
            parsed["legal_requirements"] = [str(parsed["legal_requirements"])]
        if not isinstance(parsed.get("key_facts"), list):
            parsed["key_facts"] = [str(parsed["key_facts"])]

        parsed["legal_requirements"] = self._sanitize_list(parsed["legal_requirements"])
        parsed["key_facts"] = self._sanitize_list(parsed["key_facts"])
        parsed["scenario_outline"] = parsed.get("scenario_outline", "").strip()
        parsed["tone_guideline"] = parsed.get("tone_guideline", "").strip()

        return parsed

    @staticmethod
    def _sanitize_list(values: List[Any]) -> List[str]:
        """文字列リストに正規化"""
        cleaned = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                cleaned.append(text)
        return cleaned

