"""
MCQ Consistency Checker Agent

生成された事例が正解選択肢と整合しているかを検証するエージェント。
"""
from typing import Dict, Any, List

from examples.shared.base_agent import BaseAgent


class MCQConsistencyCheckerAgent(BaseAgent):
    """
    4択問題の正解選択肢に沿った事例かどうかを検証するエージェント
    """

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        事例の整合性を検証

        Args:
            input_data: {
                "question": str,
                "context": str,
                "choices": Dict[str, str],
                "correct_choice": str,
                "scenario": str,
                "parser_result": Dict[str, Any]
            }

        Returns:
            {
                "is_valid": bool,
                "score": float,
                "feedback": List[str]
            }
        """
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        choices: Dict[str, str] = input_data.get("choices", {})
        correct_choice = input_data.get("correct_choice", "")
        scenario = input_data.get("scenario", "")
        parser_result = input_data.get("parser_result", {})

        correct_choice_text = parser_result.get("correct_choice_text") or choices.get(correct_choice, "")
        legal_requirements = parser_result.get("legal_requirements", [])

        choice_lines = "\n".join(f"{label}) {text}" for label, text in choices.items())

        requirements_text = "\n".join(f"- {req}" for req in legal_requirements) if legal_requirements else "（要件情報なし）"

        prompt = f"""以下の4択問題と生成された事例を読み、事例が正解選択肢を正しく裏付けているか検証してください。
評価は日本語で行い、指定フォーマットのJSONのみを出力してください。

【問題文】
{question}

【選択肢】
{choice_lines}

【正解】選択肢{correct_choice}: {correct_choice_text}

【想定される主要要件】
{requirements_text}

【生成された事例】
{scenario}

【評価基準】
- 事例が正解選択肢の内容と論点を明確に裏付けているか
- 条文の要件を満たす具体的事実が示されているか
- 他の選択肢を誤って是認していないか
- 文章構成が論理的で読みやすいか
- 「選択肢」「正解」「回答」など問題形式を示す語が含まれていないか
- 参照用の結論文面をそのまま転載せず、事実描写で結論を示しているか
- 「第〇条」「第〇項」「第〇号」など条文番号・項号を直接記載せず、制度趣旨を言い換えられているか

【出力形式】
```json
{{
  "is_valid": true/false,
  "score": 0.0〜1.0の範囲の数値,
  "feedback": ["改善が必要な点を50文字以内で列挙", "..."]
}}
```
"""

        response = self._safe_llm_invoke(prompt)

        default_result = {
            "is_valid": False,
            "score": 0.0,
            "feedback": ["評価に失敗しました。プロンプトや事例を確認してください。"]
        }

        if not response:
            self.logger.error("MCQConsistencyCheckerAgent: no response")
            return default_result

        parsed = self._parse_json_response(response, default=default_result)

        parsed["is_valid"] = bool(parsed.get("is_valid", False))
        try:
            parsed["score"] = float(parsed.get("score", 0.0))
        except (TypeError, ValueError):
            parsed["score"] = 0.0

        feedback = parsed.get("feedback", [])
        if not isinstance(feedback, list):
            feedback = [str(feedback)]
        parsed["feedback"] = [str(item).strip() for item in feedback if str(item).strip()]

        if not parsed["feedback"]:
            parsed["feedback"] = ["評価コメントが得られませんでした。"]

        return parsed

