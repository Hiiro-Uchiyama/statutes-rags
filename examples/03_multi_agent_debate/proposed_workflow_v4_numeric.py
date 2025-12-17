#!/usr/bin/env python3
"""
提案手法 v4.2: v4 + 数値比較のみ

v4の2エージェント構成に数値比較ロジックを追加
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

from app.utils.numeric_extractor import NumericExtractor, NumericComparator
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


class ProposedWorkflowV4Numeric:
    """
    v4.2: v4 + 数値比較のみ
    
    v4の2エージェント構成に数値比較ロジックを追加
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
        
        # 数値比較
        self.numeric_extractor = NumericExtractor()
        self.numeric_comparator = NumericComparator(self.numeric_extractor)
    
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
        context = self._build_context(docs)
        
        # Step 2: 数値比較情報を生成
        numeric_info = self._generate_numeric_comparison(choices, context)
        
        # Step 3: RetrieverAgent分析
        retriever_analysis = self._retriever_agent_analyze(
            question, choices, context, numeric_info
        )
        
        # Step 4: IntegratorAgent統合
        answer = self._integrator_agent_decide(
            question, choices, context, retriever_analysis, numeric_info
        )
        
        return {
            'answer': answer,
            'method': 'v4.2_numeric',
            'numeric_info': numeric_info
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
    
    def _generate_numeric_comparison(
        self,
        choices: List[str],
        context: str
    ) -> str:
        """各選択肢の数値比較情報を生成"""
        info_parts = []
        
        for i, choice in enumerate(choices):
            comparison = self.numeric_comparator.generate_comparison_summary(choice, context)
            # 意味のある比較結果のみ追加
            if "一致" in comparison or "不一致" in comparison:
                info_parts.append(f"選択肢{chr(97+i)}:\n{comparison}")
        
        if info_parts:
            return "\n\n".join(info_parts)
        return "数値比較情報なし"
    
    def _retriever_agent_analyze(
        self,
        question: str,
        choices: List[str],
        context: str,
        numeric_info: str
    ) -> Dict[str, Any]:
        """RetrieverAgentの分析"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        # 質問タイプを判定
        question_type = self._detect_question_type(question)
        
        prompt = f"""あなたは法令QAの分析エージェントです。

【検索された条文】
{context[:4500]}

【数値比較情報】
{numeric_info if numeric_info != "数値比較情報なし" else "（期間や割合の数値比較は特になし）"}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

以下を分析してください:
1. 各選択肢に関連する条文を特定
2. 数値（期間、割合等）の一致/不一致を確認
3. 選択肢と条文の整合性を確認
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
        retriever_analysis: Dict[str, Any],
        numeric_info: str
    ) -> str:
        """IntegratorAgentの最終判断"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        question_type = retriever_analysis.get('question_type', '不明')
        
        prompt = f"""あなたは法令QAの最終判断エージェントです。

【RetrieverAgentの分析】
{retriever_analysis.get('analysis', '')[:2500]}

【数値比較結果】
{numeric_info if numeric_info != "数値比較情報なし" else "なし"}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

上記の分析を統合し、最終回答を決定してください。

判断ポイント:
- 質問タイプが「{question_type}」なので、それに応じた回答を選択
- 数値の一致/不一致を重視
- 条文との整合性を確認

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

