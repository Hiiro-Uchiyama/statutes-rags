"""
Debater Agent

肯定的または批判的な立場から法的解釈を行うエージェント。
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Literal

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class DebaterAgent(BaseAgent):
    """
    議論エージェント
    
    責務:
    - 法令文書を基に、指定された立場から解釈を提示
    - 相手の主張を受けて反論または補強
    - 法的根拠を明確にした主張の構築
    """
    
    def __init__(
        self,
        llm,
        config,
        stance: Literal["affirmative", "critical"] = "affirmative"
    ):
        """
        Args:
            llm: LLMインスタンス
            config: 設定オブジェクト
            stance: 立場（"affirmative": 肯定的、"critical": 批判的）
        """
        super().__init__(llm, config)
        self.stance = stance
        
        # 立場に応じた役割説明
        self.role_description = {
            "affirmative": "肯定的解釈を行う議論者",
            "critical": "批判的・慎重な解釈を行う議論者"
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        議論における主張を生成
        
        Args:
            input_data: {
                "query": str,
                "documents": List[Document],
                "opponent_position": str (オプション),
                "round": int
            }
        
        Returns:
            {
                "position": str,  # 主張内容
                "reasoning": str,  # 推論過程
                "citations": List[Dict]  # 引用した法令
            }
        """
        query = input_data.get("query", "")
        documents = input_data.get("documents", [])
        opponent_position = input_data.get("opponent_position", "")
        round_num = input_data.get("round", 1)
        
        # 初回ラウンドか、対抗意見への返答か
        if round_num == 1 or not opponent_position:
            position = self._generate_initial_position(query, documents)
        else:
            position = self._generate_rebuttal(query, documents, opponent_position)
        
        return position
    
    def _generate_initial_position(
        self,
        query: str,
        documents: List[Any]
    ) -> Dict[str, Any]:
        """
        初回ラウンドの主張を生成
        
        Args:
            query: 法律質問
            documents: 検索された法令文書
        
        Returns:
            主張、推論、引用を含む辞書
        """
        context = self._format_documents(documents)
        
        # 立場に応じたプロンプト
        if self.stance == "affirmative":
            stance_instruction = """
あなたは肯定的解釈を行う議論者です。
質問に対して、法令の条文を積極的に適用し、質問者の意図を支持する方向で解釈してください。
ただし、法的根拠のない主張は避け、条文に基づいた解釈を行ってください。
"""
        else:
            stance_instruction = """
あなたは批判的・慎重な解釈を行う議論者です。
質問に対して、法令の適用範囲や例外規定を慎重に検討し、単純な適用には疑問を投げかけてください。
見落とされがちな例外や制約条件に注目してください。
"""
        
        prompt = f"""{stance_instruction}

【質問】
{query}

【関連法令】
{context}

上記の法令文書に基づいて、質問に対するあなたの立場からの解釈を示してください。

以下の形式で回答してください：

主張：
（あなたの解釈と結論）

推論：
（なぜそのように解釈したか、どの条文を根拠としたか）

引用条文：
（使用した法令名と条文番号のリスト）
"""
        
        response = self._safe_llm_invoke(prompt)
        if not response:
            return {
                "position": "応答生成に失敗しました",
                "reasoning": "",
                "citations": []
            }
        
        # 応答をパース
        parsed = self._parse_position_response(response, documents)
        return parsed
    
    def _generate_rebuttal(
        self,
        query: str,
        documents: List[Any],
        opponent_position: str
    ) -> Dict[str, Any]:
        """
        相手の主張に対する反論または補強を生成
        
        Args:
            query: 法律質問
            documents: 検索された法令文書
            opponent_position: 相手の主張
        
        Returns:
            主張、推論、引用を含む辞書
        """
        context = self._format_documents(documents)
        
        if self.stance == "affirmative":
            stance_instruction = """
あなたは肯定的解釈を行う議論者です。
相手の批判的な主張に対して、それでもなお法令が適用される理由を示してください。
"""
        else:
            stance_instruction = """
あなたは批判的・慎重な解釈を行う議論者です。
相手の肯定的な主張に対して、見落とされている問題点や例外規定を指摘してください。
"""
        
        prompt = f"""{stance_instruction}

【質問】
{query}

【相手の主張】
{opponent_position}

【関連法令】
{context}

相手の主張を踏まえて、あなたの立場からの見解を示してください。
相手の主張の問題点を指摘するか、または補強する形で回答してください。

以下の形式で回答してください：

主張：
（相手の主張に対するあなたの見解）

推論：
（なぜそのように考えるか、どの条文を根拠としたか）

引用条文：
（使用した法令名と条文番号のリスト）
"""
        
        response = self._safe_llm_invoke(prompt)
        if not response:
            return {
                "position": "応答生成に失敗しました",
                "reasoning": "",
                "citations": []
            }
        
        parsed = self._parse_position_response(response, documents)
        return parsed
    
    def _parse_position_response(
        self,
        response: str,
        documents: List[Any]
    ) -> Dict[str, Any]:
        """
        LLMの応答から主張、推論、引用をパース
        
        Args:
            response: LLMの応答
            documents: 検索された文書（引用抽出用）
        
        Returns:
            パース結果
        """
        import re
        
        # セクションを抽出
        position_match = re.search(r'主張[：:]\s*\n*(.*?)(?=推論[：:]|引用|$)', response, re.DOTALL)
        reasoning_match = re.search(r'推論[：:]\s*\n*(.*?)(?=引用|$)', response, re.DOTALL)
        citations_match = re.search(r'引用条文[：:]\s*\n*(.*?)$', response, re.DOTALL)
        
        position = position_match.group(1).strip() if position_match else response
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        citations_text = citations_match.group(1).strip() if citations_match else ""
        
        # 引用を構造化
        citations = self._extract_citations(citations_text, documents)
        
        return {
            "position": position,
            "reasoning": reasoning,
            "citations": citations
        }
    
    def _extract_citations(
        self,
        citations_text: str,
        documents: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        引用テキストから構造化された引用リストを生成
        
        Args:
            citations_text: 引用テキスト
            documents: 検索された文書
        
        Returns:
            引用のリスト
        """
        citations = []
        
        # 文書から法令情報を抽出
        for doc in documents:
            meta = doc.metadata
            law_title = meta.get("law_title", "")
            article = meta.get("article", "")
            
            # テキストに法令名が含まれているかチェック
            if law_title in citations_text:
                citation = {
                    "law_title": law_title,
                    "article": article,
                    "paragraph": meta.get("paragraph", ""),
                    "item": meta.get("item", "")
                }
                
                # 重複チェック
                if citation not in citations:
                    citations.append(citation)
        
        return citations

