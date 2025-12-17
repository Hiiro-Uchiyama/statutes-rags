#!/usr/bin/env python3
"""
提案手法 v4.1: v4 + Rerankerのみ

v4の2エージェント構成にRerankerのみを追加
"""
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_community.llms import Ollama

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

from app.retrieval.reranker import Reranker
from app.utils.number_normalizer import normalize_article_numbers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 180
    num_ctx: int = 16000
    top_k: int = 30
    use_reranker: bool = True
    reranker_top_k: int = 20


class ProposedWorkflowV4Reranker:
    """
    v4.1: v4 + Rerankerのみ
    
    v4の2エージェント構成（RetrieverAgent + IntegratorAgent）に
    Rerankerのみを追加して検索精度を向上
    """
    
    def __init__(
        self,
        retriever,
        config: WorkflowConfig = None
    ):
        self.retriever = retriever
        self.config = config or WorkflowConfig()
        
        # LLM
        self.llm = Ollama(
            model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx
        )
        
        # Reranker
        if self.config.use_reranker:
            try:
                self.reranker = Reranker(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device="cuda"
                )
                logger.info("Reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None
        else:
            self.reranker = None
    
    def query(
        self,
        question: str,
        choices: List[str]
    ) -> Dict[str, Any]:
        """
        質問に回答
        """
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # Step 1: 検索
        docs = self.retriever.retrieve(full_query, top_k=self.config.top_k)
        
        # Step 2: Reranker適用
        if self.reranker:
            ranked_docs = self.reranker.rerank(
                full_query, docs, 
                top_k=self.config.reranker_top_k
            )
            context = "\n\n".join([d.content for d in ranked_docs])
        else:
            context = self._build_context(docs)
        
        # Step 3: RetrieverAgent分析
        retriever_analysis = self._retriever_agent_analyze(question, choices, context)
        
        # Step 4: IntegratorAgent統合
        answer = self._integrator_agent_decide(
            question, choices, context, retriever_analysis
        )
        
        return {
            'answer': answer,
            'method': 'v4.1_reranker'
        }
    
    def _build_context(self, docs: List[Any]) -> str:
        """ドキュメントからコンテキストを構築"""
        texts = []
        for doc in docs[:20]:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif isinstance(doc, dict):
                texts.append(doc.get('text', doc.get('content', str(doc))))
            else:
                texts.append(str(doc))
        return "\n\n".join(texts)
    
    def _retriever_agent_analyze(
        self,
        question: str,
        choices: List[str],
        context: str
    ) -> Dict[str, Any]:
        """RetrieverAgentの分析"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        # 質問タイプを判定
        question_type = self._detect_question_type(question)
        
        prompt = f"""あなたは法令QAの分析エージェントです。

【検索された条文】
{context[:5000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

以下を分析してください:
1. 各選択肢に関連する条文を特定
2. 重要な数値・条件を抽出
3. 選択肢と条文の一致/不一致を確認
4. 暫定的な回答と根拠

分析:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            'analysis': response,
            'question_type': question_type
        }
    
    def _integrator_agent_decide(
        self,
        question: str,
        choices: List[str],
        context: str,
        retriever_analysis: Dict[str, Any]
    ) -> str:
        """IntegratorAgentの最終判断"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        question_type = retriever_analysis.get('question_type', '不明')
        
        prompt = f"""あなたは法令QAの最終判断エージェントです。

【RetrieverAgentの分析】
{retriever_analysis.get('analysis', '')[:3000]}

【条文（参考）】
{context[:2000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

上記の分析を統合し、最終回答を決定してください。

判断ポイント:
- 質問タイプが「{question_type}」なので、それに応じた回答を選択
- 条文との整合性を確認
- 数値・条件の一致/不一致を考慮

回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _detect_question_type(self, question: str) -> str:
        """質問タイプを検出"""
        if '誤っている' in question or '誤り' in question:
            return '誤り選択（誤っているものを選ぶ）'
        elif '正しい' in question:
            return '正しい選択（正しいものを選ぶ）'
        elif '組み合わせ' in question or '組合せ' in question:
            return '組み合わせ'
        else:
            return '単純選択'
    
    def _extract_answer(self, response: str) -> str:
        """LLM応答から回答を抽出"""
        response = response.strip().lower()
        
        # 最後の行から抽出
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().lower()
            match = re.search(r'(?:回答|答え|answer)[:\s]*([abcd])', line)
            if match:
                return match.group(1)
            if line in ['a', 'b', 'c', 'd']:
                return line
            match = re.search(r'^[\(\[]?([abcd])[\)\]\.。]?$', line)
            if match:
                return match.group(1)
        
        matches = re.findall(r'\b([abcd])\b', response)
        if matches:
            return matches[-1]
        
        return 'a'

