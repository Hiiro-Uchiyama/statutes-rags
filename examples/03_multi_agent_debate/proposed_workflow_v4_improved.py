#!/usr/bin/env python3
"""
提案手法 v4 改善版: プロンプト強化

v4の2エージェント構成を維持しつつ、
「正しい選択」「その他」パターンへの対応を強化
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


class ProposedWorkflowV4Improved:
    """
    v4 改善版: プロンプト強化
    
    改善点:
    1. 質問タイプ別の詳細な回答指針
    2. 「正しい選択」パターンへの明示的対応
    3. 条文と選択肢の1対1マッチングを強調
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
    
    def query(
        self,
        question: str,
        choices: List[str]
    ) -> Dict[str, Any]:
        """質問に回答"""
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # Step 1: 検索
        docs = self.retriever.retrieve(full_query, top_k=self.config.top_k)
        context = self._build_context(docs)
        
        # Step 2: 質問タイプを詳細に判定
        question_type, answer_guide = self._analyze_question_type(question)
        
        # Step 3: RetrieverAgent分析
        retriever_analysis = self._retriever_agent_analyze(
            question, choices, context, question_type, answer_guide
        )
        
        # Step 4: IntegratorAgent統合
        answer = self._integrator_agent_decide(
            question, choices, context, retriever_analysis, question_type, answer_guide
        )
        
        return {
            'answer': answer,
            'method': 'v4_improved',
            'question_type': question_type
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
    
    def _analyze_question_type(self, question: str) -> tuple:
        """質問タイプと回答指針を判定"""
        
        if '誤っている' in question or '誤り' in question:
            return (
                '誤り選択',
                '条文と一致しない選択肢、または条文に反する内容を含む選択肢を選ぶ'
            )
        
        elif '正しい' in question:
            return (
                '正しい選択',
                '条文の内容と完全に一致する選択肢を選ぶ。部分的に正しくても、一部でも誤りがあれば正しくない'
            )
        
        elif '組み合わせ' in question or '組合せ' in question:
            return (
                '組み合わせ',
                '複数の条件の組み合わせが全て正しいものを選ぶ'
            )
        
        elif '該当' in question or '当てはまる' in question:
            return (
                '該当選択',
                '条文の規定に該当する事項を選ぶ'
            )
        
        elif '教えて' in question or 'として' in question:
            return (
                '定義・規定確認',
                '条文に明記されている定義や規定を正確に選ぶ'
            )
        
        else:
            return (
                '一般選択',
                '条文の内容に最も合致する選択肢を選ぶ'
            )
    
    def _retriever_agent_analyze(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        answer_guide: str
    ) -> Dict[str, Any]:
        """RetrieverAgentの分析（強化版）"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの分析エージェントです。各選択肢を条文と照合して分析してください。

【検索された条文】
{context[:5000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}
【回答指針】: {answer_guide}

【分析手順】
1. 各選択肢について、関連する条文を特定
2. 選択肢の内容と条文を1対1で比較
3. 一致/不一致を明確に判定
4. 質問タイプに応じた回答を導出

【各選択肢の分析】（以下の形式で記載）
選択肢a: [条文との一致/不一致] - [根拠]
選択肢b: [条文との一致/不一致] - [根拠]
選択肢c: [条文との一致/不一致] - [根拠]
選択肢d: [条文との一致/不一致] - [根拠]

【暫定回答】: [a/b/c/d]
【根拠】: [理由]

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
        question_type: str,
        answer_guide: str
    ) -> str:
        """IntegratorAgentの最終判断（強化版）"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの最終判断エージェントです。

【RetrieverAgentの分析】
{retriever_analysis.get('analysis', '')[:3000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}
【回答指針】: {answer_guide}

【最終判断のポイント】
- 質問タイプが「{question_type}」なので、「{answer_guide}」
- RetrieverAgentの分析結果を確認
- 条文との整合性を最終確認

回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
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

