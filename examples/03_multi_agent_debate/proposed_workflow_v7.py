#!/usr/bin/env python3
"""
提案手法 v7: 3つの改善を統合

改善点:
1. 選択肢独立判断 - 各選択肢を個別に条文と照合し、選択肢間の干渉を防ぐ
2. 難易度事前判定 - 単純問題は簡略パス、複雑問題のみ2エージェント処理
3. 数値対比表 - 数値を抽出して対比表として提示

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """問題難易度"""
    SIMPLE = "simple"       # 単一条文、明確なキーワード → 簡略パス
    MODERATE = "moderate"   # 通常 → 2エージェント
    COMPLEX = "complex"     # 複数条文、数値比較、組み合わせ → 強化処理


@dataclass
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 180
    num_ctx: int = 16000
    top_k: int = 30


@dataclass 
class ChoiceAnalysis:
    """選択肢ごとの分析結果"""
    choice_id: str          # a, b, c, d
    choice_text: str
    related_articles: List[str] = field(default_factory=list)
    numbers_in_choice: List[str] = field(default_factory=list)
    numbers_in_articles: List[str] = field(default_factory=list)
    judgment: str = ""      # 一致/不一致/不明
    confidence: str = ""    # 高/中/低


class ProposedWorkflowV7:
    """
    v7: 選択肢独立判断 + 難易度事前判定 + 数値対比表
    """
    
    # 数値パターン
    NUMBER_PATTERNS = [
        r'\d+[日月年週]',                      # 10日、3月、1年
        r'[一二三四五六七八九十百千]+[日月年週]',  # 十日、三月
        r'\d+分の\d+',                         # 2分の1
        r'[一二三四五六七八九十]+分の[一二三四五六七八九十]+',
        r'\d+パーセント|\d+%',                 # 50%
        r'\d+円|\d+万円',                      # 金額
        r'[一二三四五六七八九十百千万億]+円',
        r'\d+人',                              # 人数
        r'過半数|三分の二|四分の三',
    ]
    
    # 複雑問題のキーワード
    COMPLEXITY_KEYWORDS = {
        'high': ['政令で定める', '施行令', '内閣府令', '準用', '組み合わせ', '組合せ'],
        'medium': ['誤っている', '正しくない', '以内', '以上', '未満', '超える'],
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
        
        # Step 1: 検索
        docs = self.retriever.retrieve(full_query, top_k=self.config.top_k)
        context = self._build_context(docs)
        
        # Step 2: 質問タイプ判定
        question_type = self._detect_question_type(question)
        
        # Step 3: 難易度事前判定
        difficulty = self._assess_difficulty(question, choices)
        
        # Step 4: 数値抽出
        numbers_table = self._extract_numbers(question, choices, context)
        
        # Step 5: 難易度に応じた処理
        if difficulty == DifficultyLevel.SIMPLE:
            # 簡略パス: 1回のLLM呼び出しで直接回答
            answer = self._simple_path(question, choices, context, question_type)
            method = 'v7_simple'
        elif difficulty == DifficultyLevel.MODERATE:
            # v4スタイル: RetrieverAgent分析 → IntegratorAgent判断
            answer = self._moderate_path(question, choices, context, question_type, numbers_table)
            method = 'v7_moderate'
        else:
            # 選択肢独立判断 → 統合判断（複雑問題のみ）
            choice_analyses = self._analyze_choices_independently(
                question, choices, context, question_type, numbers_table
            )
            answer = self._integrate_choice_analyses(
                question, choices, context, question_type, choice_analyses, numbers_table
            )
            method = 'v7_full'
        
        return {
            'answer': answer,
            'method': method,
            'difficulty': difficulty.value,
            'question_type': question_type,
            'numbers_table': numbers_table
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
    
    def _detect_question_type(self, question: str) -> str:
        """質問タイプを検出"""
        if '誤っている' in question or '誤り' in question or '正しくない' in question:
            return '誤り選択'
        elif '正しい' in question:
            return '正しい選択'
        elif '組み合わせ' in question or '組合せ' in question:
            return '組み合わせ'
        else:
            return '単純選択'
    
    def _assess_difficulty(self, question: str, choices: List[str]) -> DifficultyLevel:
        """
        難易度事前判定
        
        単純問題 → SIMPLE: 1エージェントで直接回答
        複雑問題 → COMPLEX: 選択肢独立判断 + 統合
        """
        full_text = question + " " + " ".join(choices)
        
        # 複雑度スコア
        complexity_score = 0
        
        # 高複雑度キーワード
        for kw in self.COMPLEXITY_KEYWORDS['high']:
            if kw in full_text:
                complexity_score += 2
        
        # 中複雑度キーワード
        for kw in self.COMPLEXITY_KEYWORDS['medium']:
            if kw in full_text:
                complexity_score += 1
        
        # 数値が多い場合は複雑
        numbers = self._extract_all_numbers(full_text)
        if len(numbers) >= 3:
            complexity_score += 2
        elif len(numbers) >= 1:
            complexity_score += 1
        
        # 組み合わせ問題は常に複雑
        if '組み合わせ' in question or '組合せ' in question:
            complexity_score += 3
        
        # 判定（より多くの問題をSIMPLEに分類して過剰処理を防ぐ）
        if complexity_score >= 5:
            return DifficultyLevel.COMPLEX
        elif complexity_score >= 3:
            return DifficultyLevel.MODERATE
        else:
            return DifficultyLevel.SIMPLE
    
    def _extract_all_numbers(self, text: str) -> List[str]:
        """テキストから全ての数値表現を抽出"""
        numbers = []
        for pattern in self.NUMBER_PATTERNS:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        return numbers
    
    def _extract_numbers(
        self,
        question: str,
        choices: List[str],
        context: str
    ) -> Dict[str, Any]:
        """
        数値対比表の作成
        
        選択肢の数値と条文の数値を対比
        """
        table = {
            'question_numbers': self._extract_all_numbers(question),
            'choice_numbers': {},
            'context_numbers': self._extract_all_numbers(context)[:20]  # 上限
        }
        
        for i, choice in enumerate(choices):
            choice_id = chr(97 + i)
            table['choice_numbers'][choice_id] = self._extract_all_numbers(choice)
        
        return table
    
    def _simple_path(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str
    ) -> str:
        """
        簡略パス: 単純問題用の直接回答
        
        劣化を防ぐため、過剰な分析を省略
        """
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
    
    def _moderate_path(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        numbers_table: Dict[str, Any]
    ) -> str:
        """
        中間パス: v4スタイルの2エージェント処理
        
        RetrieverAgent分析 → IntegratorAgent判断
        """
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        numbers_display = self._format_numbers_table(numbers_table)
        
        # 質問タイプ別の指示
        if question_type == '誤り選択':
            type_instruction = "この問題は「誤っているもの」を選ぶ問題です。条文と異なる選択肢を見つけてください。"
        elif question_type == '正しい選択':
            type_instruction = "この問題は「正しいもの」を選ぶ問題です。条文と一致する選択肢を見つけてください。"
        else:
            type_instruction = "条文に基づいて最も適切な選択肢を判断してください。"
        
        # RetrieverAgent: 分析
        analysis_prompt = f"""あなたは法令QAの分析エージェントです。
検索された条文を参考に、各選択肢を分析してください。

【検索された条文】
{context[:5000]}

【数値対比表】
{numbers_display}

【問題】
{question}

【選択肢】
{choices_text}

【注意】{type_instruction}

【分析手順】
各選択肢について以下を判定してください：
- a: 条文との一致/不一致、根拠
- b: 条文との一致/不一致、根拠
- c: 条文との一致/不一致、根拠
- d: 条文との一致/不一致、根拠

分析:"""
        
        analysis = self.llm.invoke(analysis_prompt)
        
        # IntegratorAgent: 最終判断
        decision_prompt = f"""あなたは法令QAの最終判断エージェントです。

【分析結果】
{analysis[:3000]}

【問題】
{question}

【選択肢】
{choices_text}

【重要】{type_instruction}

分析結果を踏まえ、最終回答を決定してください。
最後に a, b, c, d のいずれか1文字のみで回答してください。

最終回答:"""
        
        response = self.llm.invoke(decision_prompt)
        return self._extract_answer(response)
    
    def _analyze_choices_independently(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        numbers_table: Dict[str, Any]
    ) -> List[ChoiceAnalysis]:
        """
        選択肢独立判断
        
        各選択肢を個別に条文と照合し、判断を記録
        選択肢間の干渉を防ぐ
        """
        # 数値対比表を文字列化
        numbers_display = self._format_numbers_table(numbers_table)
        
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの分析エージェントです。
各選択肢を**独立して**条文と照合し、判断してください。

【検索された条文】
{context[:4500]}

【数値対比表】
{numbers_display}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【分析ルール】
1. 各選択肢を独立して評価（他の選択肢と比較しない）
2. 条文の数値と選択肢の数値を厳密に照合
3. 判断は「一致」「不一致」「不明」の3種類

【各選択肢の独立分析】

選択肢a:
  関連条文: [条文番号と該当部分を引用]
  数値照合: [選択肢の数値] vs [条文の数値]
  判断: [一致/不一致/不明]
  確信度: [高/中/低]

選択肢b:
  関連条文: [条文番号と該当部分を引用]
  数値照合: [選択肢の数値] vs [条文の数値]
  判断: [一致/不一致/不明]
  確信度: [高/中/低]

選択肢c:
  関連条文: [条文番号と該当部分を引用]
  数値照合: [選択肢の数値] vs [条文の数値]
  判断: [一致/不一致/不明]
  確信度: [高/中/低]

選択肢d:
  関連条文: [条文番号と該当部分を引用]
  数値照合: [選択肢の数値] vs [条文の数値]
  判断: [一致/不一致/不明]
  確信度: [高/中/低]

分析:"""
        
        response = self.llm.invoke(prompt)
        
        # 分析結果をパース
        analyses = self._parse_choice_analyses(response, choices)
        
        return analyses
    
    def _format_numbers_table(self, numbers_table: Dict[str, Any]) -> str:
        """数値対比表を読みやすい形式に"""
        lines = []
        
        # 選択肢の数値
        lines.append("【選択肢の数値】")
        for choice_id, nums in numbers_table.get('choice_numbers', {}).items():
            if nums:
                lines.append(f"  {choice_id}: {', '.join(nums)}")
            else:
                lines.append(f"  {choice_id}: (数値なし)")
        
        # 条文の数値（抜粋）
        context_nums = numbers_table.get('context_numbers', [])
        if context_nums:
            lines.append("【条文に含まれる数値（抜粋）】")
            lines.append(f"  {', '.join(context_nums[:10])}")
        
        return "\n".join(lines)
    
    def _parse_choice_analyses(
        self,
        response: str,
        choices: List[str]
    ) -> List[ChoiceAnalysis]:
        """LLM応答から選択肢分析をパース"""
        analyses = []
        
        for i, choice in enumerate(choices):
            choice_id = chr(97 + i)
            analysis = ChoiceAnalysis(
                choice_id=choice_id,
                choice_text=choice
            )
            
            # 該当セクションを探す
            pattern = rf'選択肢{choice_id}[:\s]*(.*?)(?=選択肢[abcd]|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                section = match.group(1)
                
                # 判断を抽出
                judgment_match = re.search(r'判断[:\s]*(一致|不一致|不明)', section)
                if judgment_match:
                    analysis.judgment = judgment_match.group(1)
                
                # 確信度を抽出
                conf_match = re.search(r'確信度[:\s]*(高|中|低)', section)
                if conf_match:
                    analysis.confidence = conf_match.group(1)
            
            analyses.append(analysis)
        
        return analyses
    
    def _integrate_choice_analyses(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        analyses: List[ChoiceAnalysis],
        numbers_table: Dict[str, Any]
    ) -> str:
        """
        選択肢分析の統合判断
        
        各選択肢の独立分析結果を統合し、最終回答を決定
        """
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        # 分析サマリを作成
        analysis_summary = []
        for a in analyses:
            summary = f"選択肢{a.choice_id}: 判断={a.judgment or '不明'}, 確信度={a.confidence or '不明'}"
            analysis_summary.append(summary)
        
        prompt = f"""あなたは法令QAの最終判断エージェントです。

【各選択肢の独立分析結果】
{chr(10).join(analysis_summary)}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【最終判断ルール】
- 質問タイプが「{question_type}」であることを確認
- 誤り選択 → 「不一致」と判断された選択肢を選ぶ
- 正しい選択 → 「一致」と判断された選択肢を選ぶ
- 確信度が高い判断を優先

各選択肢の独立分析を統合し、最も適切な選択肢を選んでください。
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
            match = re.search(r'(?:回答|答え|answer|最終回答)[:\s]*([abcd])', line)
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


# テスト用
if __name__ == "__main__":
    import json
    
    sys.path.insert(0, str(project_root / "app"))
    
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    
    # 初期化
    vector = VectorRetriever(
        index_path=str(project_root / "data/faiss_index_xml_v2/vector"),
        embedding_model="intfloat/multilingual-e5-large"
    )
    bm25 = BM25Retriever(index_path=str(project_root / "data/faiss_index_xml_v2/bm25"))
    bm25.load_index()
    retriever = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
    
    workflow = ProposedWorkflowV7(retriever=retriever)
    
    # テストデータ
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r') as f:
        data = json.load(f)
    
    # Q1-5でテスト
    print("=" * 60)
    print("v7 テスト: 選択肢独立判断 + 難易度事前判定 + 数値対比表")
    print("=" * 60)
    
    correct = 0
    for i in range(5):
        sample = data['samples'][i]
        choices = []
        for line in sample['選択肢'].split("\n"):
            line = line.strip()
            if line and line[0] in "abcd" and " " in line:
                choices.append(line[2:].strip())
        
        result = workflow.query(sample['問題文'], choices)
        correct_answer = sample['output'].strip().lower()
        is_correct = result['answer'] == correct_answer
        
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"Q{i+1}: {status} ({result['answer']}/{correct_answer}) "
              f"難易度={result['difficulty']} 方式={result['method']}")
    
    print(f"\n正解: {correct}/5 ({correct/5*100:.0f}%)")
