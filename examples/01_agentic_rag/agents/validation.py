"""
Validation Agent

回答を検証するエージェント。
"""
from typing import Dict, Any, List
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class ValidationAgent(BaseAgent):
    """
    検証エージェント
    
    責務:
    - 回答の妥当性検証
    - 引用の正確性チェック
    - ハルシネーション検出
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        回答を検証
        
        Args:
            input_data: {
                "query": str,
                "answer": str,
                "documents": List[Document]
            }
        
        Returns:
            {
                "is_valid": bool,
                "confidence": float,
                "issues": List[str],
                "suggestions": List[str]
            }
        """
        query = input_data.get("query", "")
        answer = input_data.get("answer", "")
        documents = input_data.get("documents", [])
        
        if not answer:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues": ["No answer provided"],
                "suggestions": []
            }
        
        # 引用の正確性チェック
        citation_check = self.verify_citations(answer, documents)
        
        # ハルシネーション検出
        hallucination_check = self.detect_hallucination(answer, documents)
        
        # 総合評価
        is_valid = (
            citation_check.get("is_accurate", False) and 
            not hallucination_check.get("has_hallucination", False)
        )
        
        issues = []
        if not citation_check.get("is_accurate", False):
            issues.append("Citation accuracy issue")
        if hallucination_check.get("has_hallucination", False):
            issues.append("Potential hallucination detected")
        
        confidence = citation_check.get("confidence", 0.5)
        
        self.logger.info(f"Validation: valid={is_valid}, confidence={confidence:.3f}, issues={len(issues)}")
        
        return {
            "is_valid": is_valid,
            "confidence": float(confidence),
            "issues": issues,
            "suggestions": hallucination_check.get("suggestions", [])
        }
    
    def verify_citations(self, answer: str, documents: List[Any]) -> Dict[str, Any]:
        """
        引用の正確性を検証
        
        Args:
            answer: 生成された回答
            documents: 検索結果
        
        Returns:
            検証結果
        """
        if not documents:
            return {
                "is_accurate": False,
                "confidence": 0.0,
                "issues": ["No documents to verify against"]
            }
        
        context = self._format_documents(documents[:5])
        
        prompt = f"""以下の回答について、引用されている法令条文が正確か検証してください。

【回答】
{answer}

【参照可能な条文】
{context}

検証結果をJSON形式で返してください:
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "issues": ["問題点があれば列挙"]
}}

JSON:"""
        
        response = self._safe_llm_invoke(prompt)
        
        result = self._parse_json_response(response, {
            "is_accurate": True,
            "confidence": 0.7,
            "issues": []
        })
        
        return result
    
    def detect_hallucination(self, answer: str, documents: List[Any]) -> Dict[str, Any]:
        """
        ハルシネーションを検出
        
        Args:
            answer: 生成された回答
            documents: 検索結果
        
        Returns:
            検出結果
        """
        if not documents:
            return {
                "has_hallucination": True,
                "hallucinated_parts": ["No context provided"],
                "suggestions": ["Provide relevant context"]
            }
        
        context = self._format_documents(documents[:5])
        
        prompt = f"""以下の回答が、提供されたコンテキストに基づいているか確認してください。
コンテキストにない情報を含んでいる場合は指摘してください。

【回答】
{answer}

【コンテキスト】
{context}

検証結果をJSON形式で返してください:
{{
    "has_hallucination": true/false,
    "hallucinated_parts": ["該当箇所"],
    "suggestions": ["改善提案"]
}}

JSON:"""
        
        response = self._safe_llm_invoke(prompt)
        
        result = self._parse_json_response(response, {
            "has_hallucination": False,
            "hallucinated_parts": [],
            "suggestions": []
        })
        
        return result

