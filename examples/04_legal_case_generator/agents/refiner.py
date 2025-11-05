"""
Refiner Agent

フィードバックに基づき事例を洗練するエージェント。
"""
import sys
from pathlib import Path
from typing import Dict, Any, List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class RefinerAgent(BaseAgent):
    """
    フィードバックに基づき事例を洗練するエージェント
    
    責務:
    - Legal Checkerのフィードバックに基づく事例修正
    - 具体性と教育的価値の向上
    - 最終的な事例の整形
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        事例を洗練
        
        Args:
            input_data: {
                "scenario": str,              # 現在のシナリオ
                "legal_analysis": str,        # 現在の法的分析
                "educational_point": str,     # 現在の教育的ポイント
                "feedback": List[str],        # 改善点のリスト
                "article_content": str,       # 条文内容
                "case_type": str,             # 事例タイプ
                "law_title": str,             # 法令名
                "article": str                # 条文番号
            }
        
        Returns:
            {
                "scenario": str,              # 洗練されたシナリオ
                "legal_analysis": str,        # 洗練された法的分析
                "educational_point": str      # 洗練された教育的ポイント
            }
        """
        scenario = input_data.get("scenario", "")
        legal_analysis = input_data.get("legal_analysis", "")
        educational_point = input_data.get("educational_point", "")
        feedback = input_data.get("feedback", [])
        article_content = input_data.get("article_content", "")
        case_type = input_data.get("case_type", "applicable")
        law_title = input_data.get("law_title", "")
        article = input_data.get("article", "")
        
        self.logger.info(f"Refining {case_type} case for {law_title} Article {article}")
        self.logger.info(f"Feedback items: {len(feedback)}")
        
        # フィードバックが空の場合は修正不要
        if not feedback:
            self.logger.info("No feedback to apply, returning original")
            return {
                "scenario": scenario,
                "legal_analysis": legal_analysis,
                "educational_point": educational_point
            }
        
        # 洗練プロンプトを生成
        prompt = self._create_refinement_prompt(
            law_title=law_title,
            article=article,
            article_content=article_content,
            scenario=scenario,
            legal_analysis=legal_analysis,
            educational_point=educational_point,
            feedback=feedback,
            case_type=case_type
        )
        
        # LLMで洗練
        response = self._safe_llm_invoke(prompt)
        
        if not response:
            self.logger.error("Failed to refine scenario, returning original")
            return {
                "scenario": scenario,
                "legal_analysis": legal_analysis,
                "educational_point": educational_point
            }
        
        # 応答をパース
        result = self._parse_refinement_response(response)
        
        # パースに失敗した場合は元の内容を返す
        if not any(result.values()):
            self.logger.warning("Failed to parse refinement, returning original")
            return {
                "scenario": scenario,
                "legal_analysis": legal_analysis,
                "educational_point": educational_point
            }
        
        self.logger.info("Refinement completed successfully")
        
        return result
    
    def _create_refinement_prompt(
        self,
        law_title: str,
        article: str,
        article_content: str,
        scenario: str,
        legal_analysis: str,
        educational_point: str,
        feedback: List[str],
        case_type: str
    ) -> str:
        """
        洗練用プロンプトを生成
        
        Args:
            law_title: 法令名
            article: 条文番号
            article_content: 条文内容
            scenario: 現在のシナリオ
            legal_analysis: 現在の法的分析
            educational_point: 現在の教育的ポイント
            feedback: 改善点のリスト
            case_type: 事例タイプ
        
        Returns:
            プロンプト文字列
        """
        feedback_text = "\n".join([f"- {item}" for item in feedback])
        
        prompt = f"""以下の法令条文について生成された事例を、フィードバックに基づいて改善してください。

【法令】{law_title} 第{article}条
{article_content}

【現在の事例（{case_type}）】

シナリオ:
{scenario}

法的分析:
{legal_analysis}

教育的ポイント:
{educational_point}

【改善フィードバック】
{feedback_text}

【改善要件】
- フィードバックの各項目に対応する
- 具体性を高める
- 法的正確性を確保する
- 教育的価値を向上させる
- シナリオの文字数は200-400文字程度を維持

【出力形式】
以下の形式で改善された事例を出力してください：

シナリオ:
（改善されたシナリオ）

法的分析:
（改善された法的分析）

教育的ポイント:
（改善された教育的ポイント）

【注意事項】
- フィードバックに対応しつつ、元の事例の良い部分は維持する
- 不必要な変更は避ける
- より具体的で教育的な内容にする
"""
        
        return prompt
    
    def _parse_refinement_response(self, response: str) -> Dict[str, Any]:
        """
        洗練応答をパース
        
        Args:
            response: LLMの応答文字列
        
        Returns:
            {
                "scenario": str,
                "legal_analysis": str,
                "educational_point": str
            }
        """
        import re
        
        # デフォルト値
        result = {
            "scenario": "",
            "legal_analysis": "",
            "educational_point": ""
        }
        
        # シナリオの抽出
        scenario_match = re.search(
            r'シナリオ[:：]\s*(.*?)(?=法的分析|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if scenario_match:
            result["scenario"] = scenario_match.group(1).strip()
        
        # 法的分析の抽出
        analysis_match = re.search(
            r'法的分析[:：]\s*(.*?)(?=教育的ポイント|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if analysis_match:
            result["legal_analysis"] = analysis_match.group(1).strip()
        
        # 教育的ポイントの抽出
        point_match = re.search(
            r'教育的ポイント[:：]\s*(.*?)$',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if point_match:
            result["educational_point"] = point_match.group(1).strip()
        
        return result

