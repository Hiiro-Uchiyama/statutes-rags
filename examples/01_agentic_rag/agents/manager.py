"""
Manager Agent

クエリを分析し、適切なワークフローを決定するエージェント。
"""
import re
from typing import Dict, Any
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class ManagerAgent(BaseAgent):
    """
    マネージャーエージェント
    
    責務:
    - クエリの複雑度判定
    - クエリタイプの分類
    - ワークフローの決定
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        クエリを分析
        
        Args:
            input_data: {"query": str}
        
        Returns:
            {
                "complexity": str,  # "simple" | "medium" | "complex"
                "query_type": str,  # "lookup" | "interpretation" | "application"
            }
        """
        query = input_data.get("query", "")
        
        complexity = self.classify_complexity(query)
        query_type = self.classify_query_type(query)
        
        return {
            "complexity": complexity,
            "query_type": query_type,
        }
    
    def classify_complexity(self, query: str) -> str:
        """
        クエリの複雑度を判定
        
        Args:
            query: ユーザーの質問
        
        Returns:
            "simple" | "medium" | "complex"
        """
        # まずルールベースで簡易判定
        
        # ルール1: 具体的な条文番号のみを聞いている場合 → simple
        if re.search(r'^.*第\d+条.*(?:とは|について|は何|内容).*$', query):
            if '複数' not in query and 'また' not in query and 'および' not in query:
                return "simple"
        
        # ルール2: 複数の法令が言及されている → complex
        law_count = len(re.findall(r'[^。、]+法', query))
        if law_count >= 2:
            return "complex"
        
        # ルール3: 「場合」「適用」「関係」などの複雑なキーワード → complex
        complex_keywords = ['場合', '適用', '関係', '違い', '比較', 'どちらが', 'いつ']
        if any(kw in query for kw in complex_keywords):
            return "complex"
        
        # LLMによる判定
        prompt = f"""以下の法律質問の複雑度を判定してください。

質問: {query}

判定基準:
- simple: 単一条文の直接的な照会（「第○条の内容は？」など）
- medium: 複数条文の参照が必要、または簡単な解釈が必要
- complex: 複数法令にまたがる、または高度な推論が必要

複雑度のみを1語で返してください（simple/medium/complexのいずれか）:"""
        
        response = self._safe_llm_invoke(prompt)
        
        if response:
            response_lower = response.lower()
            if "simple" in response_lower:
                return "simple"
            elif "complex" in response_lower:
                return "complex"
            else:
                return "medium"
        
        # LLM失敗時はmediumをデフォルト
        return "medium"
    
    def classify_query_type(self, query: str) -> str:
        """
        クエリのタイプを判定
        
        Args:
            query: ユーザーの質問
        
        Returns:
            "lookup" | "interpretation" | "application"
        """
        # ルールベースで簡易判定
        
        # ルール1: 「内容」「とは」など → lookup
        if any(kw in query for kw in ['内容', 'とは', 'どういう', '何を']):
            return "lookup"
        
        # ルール2: 「場合」「適用」など → application
        if any(kw in query for kw in ['場合', '適用', 'どうなる', 'できる']):
            return "application"
        
        # LLMによる判定
        prompt = f"""以下の法律質問のタイプを判定してください。

質問: {query}

タイプ:
- lookup: 条文の内容を照会
- interpretation: 法令の解釈を求める
- application: 具体的な事例への適用を求める

タイプのみを1語で返してください:"""
        
        response = self._safe_llm_invoke(prompt)
        
        if response:
            response_lower = response.lower()
            if "lookup" in response_lower:
                return "lookup"
            elif "application" in response_lower:
                return "application"
            else:
                return "interpretation"
        
        # デフォルト
        return "interpretation"
    
    def decide_workflow(self, complexity: str, query_type: str) -> str:
        """
        ワークフローを決定
        
        Args:
            complexity: 複雑度
            query_type: クエリタイプ
        
        Returns:
            ワークフロー名
        """
        if complexity == "simple":
            return "simple_workflow"
        elif complexity == "medium":
            return "medium_workflow"
        else:
            return "complex_workflow"

