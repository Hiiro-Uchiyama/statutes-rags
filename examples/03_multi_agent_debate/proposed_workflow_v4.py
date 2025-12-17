"""
提案手法 v4: 2エージェント構成（RetrieverAgent + IntegratorAgent）

Q1-30の低精度対策:
- RetrieverAgent: 検索 + 初期分析（数値・条件の抽出）
- IntegratorAgent: サポート + 確認 + 最終判定

特徴:
- 数値/条件を明示的に抽出・比較
- 不足条文の追加検索
- シンプルな2エージェント構成
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)


@dataclass 
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 120
    num_ctx: int = 16000
    top_k: int = 30
    context_docs: int = 15


class ProposedWorkflowV4:
    """提案手法v4: 2エージェント構成"""
    
    def __init__(
        self,
        retriever,
        config: Optional[WorkflowConfig] = None
    ):
        self.retriever = retriever
        self.config = config or WorkflowConfig()
        
        self.llm = Ollama(
            model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx
        )
    
    def query(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """メインクエリ処理"""
        
        # Step 1: RetrieverAgent - 検索 + 初期分析
        retriever_result = self._retriever_agent(question, choices)
        
        # Step 2: IntegratorAgent - 確認 + 追加検索 + 最終判定
        final_result = self._integrator_agent(question, choices, retriever_result)
        
        return final_result
    
    def _retriever_agent(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """RetrieverAgent: 検索 + 初期分析"""
        from app.utils.number_normalizer import normalize_article_numbers
        
        # 検索
        full_query = question + " " + " ".join(choices)
        normalized = normalize_article_numbers(full_query, to_kanji=True)
        docs = self.retriever.retrieve(normalized, top_k=self.config.top_k)
        
        # コンテキスト構築
        context = self._build_context(docs)
        
        # 質問タイプ判定（詳細化）
        if "誤っている" in question or "誤り" in question:
            question_type = "incorrect"
            type_instruction = "この問題は「誤っているもの」を選ぶ問題です。条文と不一致の選択肢を探してください。"
        elif "正しい" in question or "適切" in question:
            question_type = "correct"
            type_instruction = "この問題は「正しいもの」を選ぶ問題です。条文と一致する選択肢を探してください。"
        elif "抵触" in question or "該当" in question or "違反" in question:
            question_type = "matching"
            type_instruction = "この問題は条文に該当/抵触するものを選ぶ問題です。条文の要件に合致する選択肢を探してください。"
        elif "条文" in question and ("教えて" in question or "示す" in question):
            question_type = "citation"
            type_instruction = "この問題は該当する条文を特定する問題です。質問の内容を規定している条文を探してください。"
        elif "場合" in question or "とき" in question:
            question_type = "condition"
            type_instruction = "この問題は条件に合致するものを選ぶ問題です。条文の条件と選択肢を照合してください。"
        else:
            question_type = "general"
            type_instruction = "質問の意図を理解し、条文に基づいて最も適切な選択肢を選んでください。"
        
        # 初期分析プロンプト
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令検索の専門家（RetrieverAgent）です。
以下の質問に対して、検索した法令条文を分析してください。

【質問】
{question}

{type_instruction}

【選択肢】
{choices_text}

【検索した法令条文】
{context}

【分析タスク】
1. 各選択肢に関連する条文を特定
2. 選択肢に含まれる数値・期間・条件を抽出
3. 条文の対応する数値・期間・条件を抽出
4. 両者を比較

【出力形式】
## 選択肢分析

選択肢a:
- 関連条文: [条文名]
- 選択肢の主張: [数値/期間/条件]
- 条文の規定: [数値/期間/条件]
- 判定: [一致/不一致/要確認]

選択肢b:
...（同様に）

選択肢c:
...

選択肢d:
...

## 初期判断
- 質問タイプ: [正しいもの/誤っているもの]
- 暫定回答: [a/b/c/d]
- 確信度: [高/中/低]
- 追加確認が必要な条文: [あれば記載]

/no_think"""

        try:
            response = self.llm.invoke(prompt)
            return {
                "context": context,
                "docs": docs,
                "question_type": question_type,
                "analysis": response,
                "initial_answer": self._extract_answer_from_analysis(response)
            }
        except Exception as e:
            logger.error(f"RetrieverAgent failed: {e}")
            return {
                "context": context,
                "docs": docs,
                "question_type": question_type,
                "analysis": f"Error: {e}",
                "initial_answer": "a"
            }
    
    def _integrator_agent(
        self,
        question: str,
        choices: List[str],
        retriever_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """IntegratorAgent: 確認 + 追加検索 + 最終判定"""
        
        # 追加検索が必要かチェック
        additional_context = ""
        if "追加確認が必要な条文" in retriever_result.get("analysis", ""):
            additional_context = self._additional_search(retriever_result)
        
        # 最終判定プロンプト
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        question_type = retriever_result.get("question_type", "general")
        if question_type == "incorrect":
            type_instruction = "【重要】「誤っているもの」を選んでください。条文と不一致の選択肢です。"
        elif question_type == "correct":
            type_instruction = "【重要】「正しいもの」を選んでください。条文と一致する選択肢です。"
        elif question_type == "matching":
            type_instruction = "【重要】条文に該当/抵触するものを選んでください。条文の要件に合致する選択肢です。"
        elif question_type == "citation":
            type_instruction = "【重要】該当する条文を特定してください。質問の内容を規定している条文に対応する選択肢を選んでください。"
        elif question_type == "condition":
            type_instruction = "【重要】条件に合致するものを選んでください。条文の条件と選択肢を照合してください。"
        else:
            type_instruction = "【重要】質問の意図を理解し、条文に基づいて最も適切な選択肢を選んでください。"
        
        additional_section = ""
        if additional_context:
            additional_section = f"\n【追加で検索した条文】\n{additional_context}"
        
        prompt = f"""あなたは法令判定の専門家（IntegratorAgent）です。
RetrieverAgentの分析を確認し、最終判定を行ってください。

【質問】
{question}

{type_instruction}

【選択肢】
{choices_text}

【RetrieverAgentの分析】
{retriever_result.get('analysis', '')}
{additional_section}

【最終判定タスク】
1. RetrieverAgentの分析を確認
2. 数値・期間・条件の比較結果を検証
3. 質問タイプに応じて回答を決定

【確認ポイント】
- 選択肢の数値と条文の数値は完全に一致しているか？
- 「三月」と「六月」、「一年」と「二年」など、期間の違いに注意
- 「のみ」「限り」などの限定表現に注意

回答は a, b, c, d のいずれか1文字のみで答えてください。
最終回答:"""

        try:
            response = self.llm.invoke(prompt)
            answer = self._extract_answer(response)
            return {
                "answer": answer,
                "retriever_analysis": retriever_result.get("analysis", ""),
                "additional_context": additional_context,
                "final_response": response
            }
        except Exception as e:
            logger.error(f"IntegratorAgent failed: {e}")
            return {
                "answer": retriever_result.get("initial_answer", "a"),
                "retriever_analysis": retriever_result.get("analysis", ""),
                "additional_context": "",
                "final_response": f"Error: {e}"
            }
    
    def _additional_search(self, retriever_result: Dict[str, Any]) -> str:
        """追加検索"""
        from app.utils.number_normalizer import normalize_article_numbers
        
        analysis = retriever_result.get("analysis", "")
        
        # 追加確認が必要な条文を抽出
        additional_queries = []
        if "追加確認が必要な条文:" in analysis:
            lines = analysis.split("\n")
            for i, line in enumerate(lines):
                if "追加確認が必要な条文:" in line:
                    # 次の行から条文名を抽出
                    for j in range(i+1, min(i+5, len(lines))):
                        if lines[j].strip().startswith("-"):
                            query = lines[j].strip()[1:].strip()
                            if query and query != "なし":
                                additional_queries.append(query)
        
        if not additional_queries:
            return ""
        
        # 追加検索実行
        all_docs = []
        for query in additional_queries[:2]:  # 最大2クエリ
            normalized = normalize_article_numbers(query, to_kanji=True)
            docs = self.retriever.retrieve(normalized, top_k=10)
            all_docs.extend(docs)
        
        # 重複除去
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return self._build_context(unique_docs, max_docs=5)
    
    def _build_context(self, docs: List, max_docs: int = None) -> str:
        """コンテキスト構築"""
        max_docs = max_docs or self.config.context_docs
        parts = []
        for i, doc in enumerate(docs[:max_docs]):
            law = doc.metadata.get("law_title", "")
            article = doc.metadata.get("article_title", "")
            text = doc.page_content[:600]
            parts.append(f"[{i+1}] {law} {article}\n{text}")
        return "\n\n".join(parts)
    
    def _extract_answer_from_analysis(self, analysis: str) -> str:
        """分析結果から暫定回答を抽出"""
        if "暫定回答:" in analysis:
            lines = analysis.split("\n")
            for line in lines:
                if "暫定回答:" in line:
                    for c in ['a', 'b', 'c', 'd']:
                        if c in line.lower():
                            return c
        return 'a'
    
    def _extract_answer(self, response: str) -> str:
        """回答を抽出"""
        response_lower = response.lower().strip()
        for char in ['a', 'b', 'c', 'd']:
            if char in response_lower[:50]:
                return char
        return 'a'

