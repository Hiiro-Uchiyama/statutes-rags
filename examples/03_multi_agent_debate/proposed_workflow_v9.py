#!/usr/bin/env python3
"""
提案手法 v9b: 構造化法令QAシステム（検証Agentなし）

v7の知見を活かし、以下を追加:
1. 構造化法令DB - 数値情報、参照関係を構造化
2. v7方式の分析+統合（検証Agentなし）

目標: 80%以上の正解率
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_community.llms import Ollama

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

from app.utils.number_normalizer import normalize_article_numbers
from structured_law_db import StructuredLawDB, get_db, reset_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """問題難易度"""
    SIMPLE = "simple"       # 単一条文、明確なキーワード → 簡略パス
    MODERATE = "moderate"   # 通常 → 分析+検証
    COMPLEX = "complex"     # 複数条文、数値比較、組み合わせ → 強化処理


@dataclass
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 180
    num_ctx: int = 16000
    top_k: int = 30


class ProposedWorkflowV9:
    """
    v9b: 構造化法令DB + v7方式統合（検証Agentなし）
    
    アーキテクチャ:
    1. 難易度判断Agent（v7継承）
    2. 検索 + 構造化DB構築
    3. 分析Agent（構造化データ活用）
    4. 統合（検証Agentなし、分析結果から直接決定）
    """
    
    # 複雑問題のキーワード
    COMPLEXITY_KEYWORDS = {
        'high': ['政令で定める', '施行令', '内閣府令', '準用', '組み合わせ', '組合せ'],
        'medium': ['届出', '届け出', '報告', '開示', '訂正', '変更']
    }
    
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
        
        # 検索
        results = self.retriever.retrieve(full_query, top_k=self.config.top_k)
        context = "\n\n".join([doc.page_content for doc in results])
        
        # 構造化DB構築
        reset_db()
        db = get_db()
        db.build_from_documents([{'content': doc.page_content, 'metadata': doc.metadata} for doc in results])
        
        # 質問タイプ判定
        question_type = self._detect_question_type(question)
        
        # 難易度判定
        difficulty = self._assess_difficulty(question, choices)
        
        # 数値対比表
        numbers_table = db.format_numbers_table(choices)
        
        # 難易度に応じた処理
        if difficulty == DifficultyLevel.SIMPLE:
            answer = self._simple_path(question, choices, context, question_type)
            method = 'v9b_simple'
        else:
            # MODERATE/COMPLEX: 分析Agent + 統合（検証Agentなし）
            analysis = self._analyze_with_structure(question, choices, context, question_type, numbers_table, db)
            answer = self._integrate_analysis(question, choices, analysis, question_type)
            method = f'v9b_{difficulty.value}'
        
        return {
            'answer': answer,
            'method': method,
            'difficulty': difficulty.value,
            'question_type': question_type
        }
    
    def _detect_question_type(self, question: str) -> str:
        """質問タイプを判定"""
        if '誤' in question:
            return '誤り選択'
        elif '正しい' in question:
            return '正しい選択'
        elif '組み合わせ' in question or '組合せ' in question:
            return '組み合わせ'
        else:
            return '単純選択'
    
    def _assess_difficulty(self, question: str, choices: List[str]) -> DifficultyLevel:
        """LLMによる難易度判断Agent"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QA問題の難易度を判定するエージェントです。

【問題】
{question}

【選択肢】
{choices_text}

【判定基準】
- SIMPLE: 条文を1つ読めば直接答えられる単純な問題
- MODERATE: 複数の条文を比較したり、細かい条件の違いを判断する必要がある問題
- COMPLEX: 複雑な参照関係（政令、施行令等）や、複数の法的概念の組み合わせが必要な問題

【判定のポイント】
1. 選択肢が似ている場合 → MODERATE以上
2. 条文番号の違いを問う場合 → MODERATE以上
3. 「政令で定める」「施行令」等の参照がある場合 → COMPLEX
4. 組み合わせ問題 → COMPLEX

上記の問題の難易度を判定し、SIMPLE、MODERATE、COMPLEXのいずれか1つのみを回答してください。

難易度:"""

        response = self.llm.invoke(prompt)
        response_text = response.upper() if isinstance(response, str) else str(response).upper()
        
        # 回答から難易度を抽出
        if 'COMPLEX' in response_text:
            return DifficultyLevel.COMPLEX
        elif 'MODERATE' in response_text:
            return DifficultyLevel.MODERATE
        else:
            return DifficultyLevel.SIMPLE
    
    def _simple_path(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str
    ) -> str:
        """簡略パス: 単純問題用の直接回答"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        # 質問タイプ別の指示
        if question_type == '誤り選択':
            type_instruction = "【重要】この問題は「誤っているもの」を選ぶ問題です。条文の内容と異なる（間違っている）選択肢を選んでください。"
        elif question_type == '正しい選択':
            type_instruction = "【重要】この問題は「正しいもの」を選ぶ問題です。条文の内容と一致する選択肢を選んでください。"
        else:
            type_instruction = "【重要】条文の内容と各選択肢を照合し、条文に最も合致する選択肢を選んでください。"
        
        prompt = f"""あなたは法令QAの専門家です。

【検索された条文】
{context[:4000]}

【問題】
{question}

【選択肢】
{choices_text}

{type_instruction}

各選択肢を条文と照合し、回答を決定してください。
最後に a, b, c, d のいずれか1文字のみで回答してください。

回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _analyze_with_structure(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        numbers_table: str,
        db: StructuredLawDB
    ) -> str:
        """分析Agent: 構造化データを活用した分析（v9b-Lベースライン）"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        # 質問タイプ別の指示
        if question_type == '誤り選択':
            type_instruction = "この問題は「誤っているもの」を選ぶ問題です。条文と異なる選択肢を見つけてください。"
        elif question_type == '正しい選択':
            type_instruction = "この問題は「正しいもの」を選ぶ問題です。条文と一致する選択肢を見つけてください。"
        else:
            type_instruction = "条文に基づいて最も適切な選択肢を判断してください。"
        
        structured_context = db.get_structured_context()[:2500 if difficulty == DifficultyLevel.COMPLEX else 1800]
        few_shot = """# 数値差: 条文=30日以内, 選択肢=20日以内 → ×
# 法令種別: 条文=施行令, 選択肢=本法と記載 → ×
# 主体: 条文=取締役のみ, 選択肢=取締役等 → ×
# complex: 本法と施行令を併読し双方に一致 → ○"""
        
        prompt = f"""あなたは法令QAの分析エージェントです。
構造化された条文情報を参考に、各選択肢を分析してください。

【検索された条文】
{context[:4000]}

{numbers_table}

【簡易サマリ（上位抜粋）】
{structured_context}

【Few-shot例（カテゴリ別、簡潔サンプル）】
{few_shot}

【問題】
{question}

【選択肢】
{choices_text}

【注意】{type_instruction}

【重要な照合ポイント】
- 条文番号：「法律第○条」と「施行令第○条」は異なる。正確に区別すること
- 対象範囲：「〜のみ」「〜を含む」「〜を除く」等の限定詞に注目
- 主体：「内国会社」「外国会社」「取締役」「役員」等の違いを明確に

【分析手順】
各選択肢について以下を判定してください：
1. 関連する条文番号（法律/施行令/施行規則を明記）
2. 選択肢のキーワードと条文の対応
3. 条文との一致/不一致と具体的根拠

分析:"""
        
        analysis = self.llm.invoke(prompt)
        return analysis
    
    def _integrate_analysis(
        self,
        question: str,
        choices: List[str],
        analysis: str,
        question_type: str
    ) -> str:
        """統合: 分析結果から最終回答を決定（v9b-Lベースライン）"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        # 質問タイプ別の指示
        if question_type == '誤り選択':
            type_instruction = "【重要】この問題は「誤っているもの」を選ぶ問題です。条文と異なる（間違っている）選択肢を選んでください。"
        elif question_type == '正しい選択':
            type_instruction = "【重要】この問題は「正しいもの」を選ぶ問題です。条文と一致する選択肢を選んでください。"
        else:
            type_instruction = "条文に最も合致する選択肢を選んでください。"
        
        prompt = f"""あなたは法令QAの専門家です。
以下の【問題】と【選択肢】、そして各選択肢に対する【分析結果】を基に、最終的な回答を決定してください。

【問題】
{question}

【選択肢】
{choices_text}

{type_instruction}

【統合の方針】
- 数値・期間条件（〇日以内、〇月以内、割合、金額など）が条文と完全に一致しているか
- 対象範囲・主体（内国会社/外国会社、取締役等/その他など）が条文の記載と一致しているか
- 条文番号や法令種別（法律第○条か、施行令第○条・施行規則第○条か）が正しく対応しているか

これら3点で条文と整合している選択肢を優先し、明確に矛盾している選択肢を除外してください。

【分析結果】
{analysis[:2000]}

【最終判断】
上記の分析結果と「数値・期間」「対象範囲・主体」「条文・法令種別」の3点を踏まえ、最も適切な選択肢（a, b, c, dのいずれか1つ）を選び、その理由を簡潔に述べてください。
最後に a, b, c, d のいずれか1文字のみで回答してください。

回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _extract_answer(self, response: str) -> str:
        """回答を抽出"""
        response_lower = response.lower().strip()
        
        # 末尾から検索
        for char in reversed(response_lower):
            if char in 'abcd':
                return char
        
        # 先頭から検索
        for char in response_lower:
            if char in 'abcd':
                return char
        
        return 'a'  # デフォルト
