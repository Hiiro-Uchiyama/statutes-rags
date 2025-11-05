"""
Scenario Generator Agent

法令条文から具体的な適用事例を生成するエージェント。
"""
import sys
from pathlib import Path
from typing import Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class ScenarioGeneratorAgent(BaseAgent):
    """
    法令条文から具体的な適用事例を生成するエージェント
    
    責務:
    - 法令条文の分析
    - 3種類の事例（適用・非適用・境界）の生成
    - 具体的で現実的なシナリオの作成
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        事例シナリオを生成
        
        Args:
            input_data: {
                "article_content": str,  # 条文内容
                "case_type": str,        # "applicable" | "non_applicable" | "boundary"
                "law_title": str,        # 法令名
                "article": str           # 条文番号
            }
        
        Returns:
            {
                "scenario": str,          # シナリオ（状況説明）
                "legal_analysis": str,    # 法的分析
                "educational_point": str  # 教育的ポイント
            }
        """
        article_content = input_data.get("article_content", "")
        case_type = input_data.get("case_type", "applicable")
        law_title = input_data.get("law_title", "")
        article = input_data.get("article", "")
        
        self.logger.info(f"Generating {case_type} case for {law_title} Article {article}")
        
        # 事例タイプに応じたプロンプトを生成
        prompt = self._create_prompt(
            law_title=law_title,
            article=article,
            article_content=article_content,
            case_type=case_type
        )
        
        # LLMで事例を生成
        response = self._safe_llm_invoke(prompt)
        
        if not response:
            self.logger.error("Failed to generate scenario")
            return {
                "scenario": "",
                "legal_analysis": "",
                "educational_point": ""
            }
        
        # 応答をパース
        result = self._parse_response(response)
        
        self.logger.info(f"Generated scenario: {len(result.get('scenario', ''))} chars")
        
        return result
    
    def _create_prompt(
        self,
        law_title: str,
        article: str,
        article_content: str,
        case_type: str
    ) -> str:
        """
        事例タイプに応じたプロンプトを生成
        
        Args:
            law_title: 法令名
            article: 条文番号
            article_content: 条文内容
            case_type: 事例タイプ
        
        Returns:
            プロンプト文字列
        """
        # 事例タイプの説明
        type_descriptions = {
            "applicable": "この法令条文が明確に適用される",
            "non_applicable": "この法令条文が適用されない",
            "boundary": "この法令条文の適用について判断が分かれる可能性がある"
        }
        
        type_desc = type_descriptions.get(case_type, "")
        
        # 事例タイプ別の追加指示
        type_instructions = {
            "applicable": """
- 条文の要件を全て満たす状況を設定
- 適用が明確であることを強調
- 典型的なケースを例示""",
            "non_applicable": """
- 条文の要件のうち、重要な要件が欠けている状況を設定
- なぜ適用されないかを明確に説明
- よくある誤解を解消する観点を含める""",
            "boundary": """
- 条文の要件の解釈に幅がある状況を設定
- 判断が分かれるポイントを明確に指摘
- 両方の解釈の可能性を示す"""
        }
        
        type_instruction = type_instructions.get(case_type, "")
        
        prompt = f"""以下の法令条文について、{type_desc}具体的な事例を生成してください。

【法令】{law_title} 第{article}条
{article_content}

【要件】
- 現実的で具体的なシナリオ（200-400文字程度）
- 法的分析が明確で論理的
- 教育的価値が高い
{type_instruction}

【出力形式】
以下の形式で出力してください：

シナリオ:
（具体的な状況説明を記述）

法的分析:
（なぜこの条文が適用される/されないか、または判断が分かれるかを説明）

教育的ポイント:
（この事例から学べることを簡潔に記述）

【注意事項】
- 実在の人物・企業名は使用しない
- 具体的な数値や期日を含める
- 法律用語を適切に使用する
"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        LLMの応答をパースして構造化
        
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
        
        # パースに失敗した場合は全体を返す
        if not any(result.values()):
            self.logger.warning("Failed to parse response structure, using raw response")
            result["scenario"] = response
        
        return result

