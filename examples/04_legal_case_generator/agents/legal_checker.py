"""
Legal Checker Agent

生成された事例の法的整合性を検証するエージェント。
"""
import sys
from pathlib import Path
from typing import Dict, Any, List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class LegalCheckerAgent(BaseAgent):
    """
    生成された事例の法的整合性を検証するエージェント
    
    責務:
    - 生成事例の法的整合性検証
    - 条文の適用要件との照合
    - 矛盾や不正確な点の指摘
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        法的整合性を検証
        
        Args:
            input_data: {
                "article_content": str,    # 条文内容
                "scenario": str,           # シナリオ
                "legal_analysis": str,     # 法的分析
                "case_type": str,          # 事例タイプ
                "law_title": str,          # 法令名
                "article": str             # 条文番号
            }
        
        Returns:
            {
                "is_valid": bool,              # 法的整合性があるか
                "validation_score": float,     # 検証スコア（0-1）
                "feedback": List[str]          # 改善点のリスト
            }
        """
        article_content = input_data.get("article_content", "")
        scenario = input_data.get("scenario", "")
        legal_analysis = input_data.get("legal_analysis", "")
        case_type = input_data.get("case_type", "applicable")
        law_title = input_data.get("law_title", "")
        article = input_data.get("article", "")
        
        self.logger.info(f"Validating {case_type} case for {law_title} Article {article}")
        
        # 検証プロンプトを生成
        prompt = self._create_validation_prompt(
            law_title=law_title,
            article=article,
            article_content=article_content,
            scenario=scenario,
            legal_analysis=legal_analysis,
            case_type=case_type
        )
        
        # LLMで検証
        response = self._safe_llm_invoke(prompt)
        
        if not response:
            self.logger.error("Failed to validate scenario")
            return {
                "is_valid": False,
                "validation_score": 0.0,
                "feedback": ["検証プロセスでエラーが発生しました"]
            }
        
        # 応答をパース
        result = self._parse_validation_response(response)
        
        self.logger.info(
            f"Validation result: valid={result['is_valid']}, "
            f"score={result['validation_score']:.3f}"
        )
        
        return result
    
    def _create_validation_prompt(
        self,
        law_title: str,
        article: str,
        article_content: str,
        scenario: str,
        legal_analysis: str,
        case_type: str
    ) -> str:
        """
        検証用プロンプトを生成
        
        Args:
            law_title: 法令名
            article: 条文番号
            article_content: 条文内容
            scenario: シナリオ
            legal_analysis: 法的分析
            case_type: 事例タイプ
        
        Returns:
            プロンプト文字列
        """
        # 事例タイプ別の検証ポイント
        validation_points = {
            "applicable": """
1. シナリオは条文の要件を全て満たしているか
2. 法的分析は条文の適用を正確に説明しているか
3. 適用が明確であることが示されているか""",
            "non_applicable": """
1. シナリオは条文の要件の少なくとも1つを欠いているか
2. 法的分析は非適用の理由を正確に説明しているか
3. なぜ適用されないかが明確に示されているか""",
            "boundary": """
1. シナリオは判断が分かれる要素を含んでいるか
2. 法的分析は両方の解釈の可能性を示しているか
3. 判断が分かれるポイントが明確に指摘されているか"""
        }
        
        points = validation_points.get(case_type, "")
        
        prompt = f"""以下の法令条文と、それについて生成された事例を検証してください。

【法令】{law_title} 第{article}条
{article_content}

【生成された事例】
事例タイプ: {case_type}

シナリオ:
{scenario}

法的分析:
{legal_analysis}

【検証項目】
{points}

4. シナリオは具体的で現実的か
5. 法的分析は論理的に正しいか
6. 条文の引用や解釈に誤りはないか

【出力形式】
以下の形式で検証結果を出力してください：

検証結果: 合格 または 不合格

スコア: 0.0-1.0の数値（1.0が最高評価）

フィードバック:
- （改善点1）
- （改善点2）
- ...

【注意事項】
- フィードバックは具体的に記述する
- 合格の場合でも改善の余地があれば指摘する
- スコアは厳格に評価する
"""
        
        return prompt
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        検証応答をパース
        
        Args:
            response: LLMの応答文字列
        
        Returns:
            {
                "is_valid": bool,
                "validation_score": float,
                "feedback": List[str]
            }
        """
        import re
        
        # デフォルト値
        result = {
            "is_valid": False,
            "validation_score": 0.5,
            "feedback": []
        }
        
        # 検証結果の抽出
        result_match = re.search(
            r'検証結果[:：]\s*(合格|不合格)',
            response,
            re.IGNORECASE
        )
        if result_match:
            result["is_valid"] = result_match.group(1) == "合格"
        
        # スコアの抽出
        score_match = re.search(
            r'スコア[:：]\s*([0-9]*\.?[0-9]+)',
            response,
            re.IGNORECASE
        )
        if score_match:
            try:
                score = float(score_match.group(1))
                result["validation_score"] = max(0.0, min(1.0, score))
            except ValueError:
                self.logger.warning("Failed to parse validation score")
        
        # フィードバックの抽出
        feedback_match = re.search(
            r'フィードバック[:：]\s*(.*?)(?=\n\n|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if feedback_match:
            feedback_text = feedback_match.group(1).strip()
            # 箇条書きを解析
            feedback_items = re.findall(r'[-•]\s*(.+)', feedback_text)
            if feedback_items:
                result["feedback"] = [item.strip() for item in feedback_items]
            else:
                # 箇条書きでない場合は行で分割
                lines = [line.strip() for line in feedback_text.split('\n') if line.strip()]
                result["feedback"] = lines
        
        # スコアベースで is_valid を再判定（設定の閾値を使用）
        threshold = getattr(self.config, 'validation_threshold', 0.7)
        if result["validation_score"] >= threshold:
            result["is_valid"] = True
        
        return result

