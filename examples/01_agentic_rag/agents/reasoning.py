"""
Reasoning Agent

法的推論を行うエージェント。
"""
from typing import Dict, Any, List
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class ReasoningAgent(BaseAgent):
    """
    推論エージェント
    
    責務:
    - 複数条文の関連性分析
    - 法的推論の構築
    - 適用順序・優先順位の判定
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        法的推論を実行
        
        Args:
            input_data: {
                "query": str,
                "documents": List[Document],
                "complexity": str (optional)
            }
        
        Returns:
            {
                "reasoning": str,
                "legal_structure": Dict[str, Any]
            }
        """
        query = input_data.get("query", "")
        documents = input_data.get("documents", [])
        complexity = input_data.get("complexity", "medium")
        
        if not documents:
            return {
                "reasoning": "検索結果が不十分なため、推論を実行できませんでした。",
                "legal_structure": {}
            }
        
        # 複雑度に応じた推論
        if complexity == "simple":
            # 簡易推論
            reasoning = self.simple_reasoning(query, documents)
            legal_structure = {}
        else:
            # 詳細推論
            legal_structure = self.analyze_legal_structure(query, documents)
            reasoning = self.construct_reasoning(query, legal_structure, documents)
        
        return {
            "reasoning": reasoning,
            "legal_structure": legal_structure
        }
    
    def simple_reasoning(self, query: str, documents: List[Any]) -> str:
        """
        簡易推論（単純な質問向け）
        
        Args:
            query: 質問文
            documents: 検索結果
        
        Returns:
            推論結果
        """
        context = self._format_documents(documents[:3])  # 上位3件のみ使用
        
        prompt = f"""以下の法令条文に基づいて、質問に簡潔に回答してください。

【法令条文】
{context}

【質問】
{query}

【回答】
該当する法令と条文を明示して回答してください:"""
        
        response = self._safe_llm_invoke(prompt)
        
        return response if response else "回答を生成できませんでした。"
    
    def analyze_legal_structure(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        """
        法的構造を分析
        
        Args:
            query: 質問文
            documents: 検索結果
        
        Returns:
            法的構造の分析結果
        """
        context = self._format_documents(documents[:5])  # 上位5件を使用
        
        prompt = f"""以下の法令条文を分析し、質問に対する法的構造を明確にしてください。

【質問】
{query}

【法令条文】
{context}

以下の観点で分析し、JSON形式で返してください:
{{
    "main_provisions": ["主要な適用条文"],
    "related_provisions": ["関連する条文"],
    "exceptions": ["例外規定"],
    "application_order": ["適用順序"]
}}

JSON:"""
        
        response = self._safe_llm_invoke(prompt)
        
        result = self._parse_json_response(response, {
            "main_provisions": [],
            "related_provisions": [],
            "exceptions": [],
            "application_order": []
        })
        
        self.logger.info(f"Legal structure: {len(result.get('main_provisions', []))} main provisions")
        
        return result
    
    def construct_reasoning(
        self, 
        query: str, 
        legal_structure: Dict[str, Any],
        documents: List[Any]
    ) -> str:
        """
        法的推論を構築
        
        Args:
            query: 質問文
            legal_structure: 法的構造分析結果
            documents: 検索結果
        
        Returns:
            構築された推論
        """
        context = self._format_documents(documents[:5])
        
        main_provisions = legal_structure.get("main_provisions", [])
        related = legal_structure.get("related_provisions", [])
        exceptions = legal_structure.get("exceptions", [])
        
        prompt = f"""以下の法的構造分析に基づいて、質問に対する推論を構築してください。

【質問】
{query}

【法的構造】
- 主要条文: {', '.join(main_provisions) if main_provisions else 'なし'}
- 関連条文: {', '.join(related) if related else 'なし'}
- 例外規定: {', '.join(exceptions) if exceptions else 'なし'}

【参照条文】
{context}

以下の形式で推論を構築してください:
1. 適用される法規範（大前提）
2. 事実関係の当てはめ（小前提）
3. 結論

【推論】"""
        
        response = self._safe_llm_invoke(prompt)
        
        return response if response else "推論を構築できませんでした。"

