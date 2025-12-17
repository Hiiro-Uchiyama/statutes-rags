#!/usr/bin/env python3
"""
提案手法 v8: v7からの改善

改善点:
1. complex処理を廃止 → v4スタイル（moderate）に統一
   - 選択肢独立判断は逆効果（33.3%）だったため削除
2. moderate精度向上
   - 数値対比表の活用強化
   - 段階的分析プロンプトの改善
   - 選択肢ごとの詳細照合

v7結果: 75.7% (106/140)
目標: 80%以上
"""
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
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
    SIMPLE = "simple"       # 単純問題 → 1エージェント
    MODERATE = "moderate"   # 中〜高難易度 → 2エージェント強化版


@dataclass
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 180
    num_ctx: int = 16000
    top_k: int = 30


class ProposedWorkflowV8:
    """
    v8: 難易度事前判定 + 数値対比表強化
    
    - complex処理を廃止（選択肢独立判断は逆効果）
    - moderate処理を強化（数値照合重視）
    """
    
    # 数値パターン
    NUMBER_PATTERNS = [
        r'\d+[日月年週]',
        r'[一二三四五六七八九十百千]+[日月年週]',
        r'\d+分の\d+',
        r'[一二三四五六七八九十]+分の[一二三四五六七八九十]+',
        r'\d+パーセント|\d+%',
        r'\d+円|\d+万円',
        r'[一二三四五六七八九十百千万億]+円',
        r'\d+人',
        r'過半数|三分の二|四分の三',
        r'\d+条',
        r'第[一二三四五六七八九十百]+条',
    ]
    
    # 複雑度キーワード
    COMPLEXITY_KEYWORDS = {
        'high': ['政令で定める', '施行令', '内閣府令', '準用', '組み合わせ', '組合せ'],
        'medium': ['誤っている', '正しくない', '以内', '以上', '未満', '超える', 'ただし'],
    }
    
    def __init__(
        self,
        retriever,
        config: WorkflowConfig = None
    ):
        self.retriever = retriever
        self.config = config or WorkflowConfig()
        
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
        
        # Step 3: 難易度事前判定（2段階に簡略化）
        difficulty = self._assess_difficulty(question, choices)
        
        # Step 4: 数値抽出
        numbers_table = self._extract_numbers(question, choices, context)
        
        # Step 5: 難易度に応じた処理
        if difficulty == DifficultyLevel.SIMPLE:
            answer = self._simple_path(question, choices, context, question_type)
            method = 'v8_simple'
        else:
            # MODERATE: 強化版2エージェント処理
            answer = self._enhanced_moderate_path(
                question, choices, context, question_type, numbers_table
            )
            method = 'v8_moderate'
        
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
        難易度事前判定（2段階に簡略化）
        
        SIMPLE: 単純問題 → 1エージェント
        MODERATE: それ以外 → 2エージェント強化版
        """
        full_text = question + " " + " ".join(choices)
        
        complexity_score = 0
        
        # 高複雑度キーワード
        for kw in self.COMPLEXITY_KEYWORDS['high']:
            if kw in full_text:
                complexity_score += 2
        
        # 中複雑度キーワード
        for kw in self.COMPLEXITY_KEYWORDS['medium']:
            if kw in full_text:
                complexity_score += 1
        
        # 数値が多い場合
        numbers = self._extract_all_numbers(full_text)
        if len(numbers) >= 2:
            complexity_score += 1
        
        # SIMPLEの閾値を厳しくして、より多くの問題をMODERATEに
        if complexity_score >= 2:
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
        """数値対比表の作成"""
        table = {
            'question_numbers': self._extract_all_numbers(question),
            'choice_numbers': {},
            'context_numbers': self._extract_all_numbers(context)[:20]
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
        """簡略パス: 単純問題用"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの専門家です。
以下の問題に回答してください。

【検索された条文】
{context[:4000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

条文を参照し、最も適切な選択肢を選んでください。
回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _enhanced_moderate_path(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        numbers_table: Dict[str, Any]
    ) -> str:
        """
        強化版moderate処理
        
        改善点:
        1. 各選択肢を明示的に条文と照合
        2. 数値対比表を効果的に活用
        3. 質問タイプに応じた判断基準を明確化
        """
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        numbers_display = self._format_numbers_table(numbers_table)
        
        # RetrieverAgent: 選択肢ごとの詳細分析
        analysis_prompt = f"""あなたは法令QAの分析エージェントです。
各選択肢を条文と照合し、詳細に分析してください。

【検索された条文】
{context[:5000]}

【数値対比表】
{numbers_display}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【分析指示】
各選択肢について以下を分析してください：

**選択肢a**: {choices[0] if len(choices) > 0 else ''}
- 関連条文: どの条文が関係するか
- 条文との照合: 選択肢の記述が条文と一致するか（○一致/×不一致/△不明）
- 数値確認: 数値がある場合、条文の数値と一致するか

**選択肢b**: {choices[1] if len(choices) > 1 else ''}
- 関連条文: どの条文が関係するか
- 条文との照合: 選択肢の記述が条文と一致するか（○一致/×不一致/△不明）
- 数値確認: 数値がある場合、条文の数値と一致するか

**選択肢c**: {choices[2] if len(choices) > 2 else ''}
- 関連条文: どの条文が関係するか
- 条文との照合: 選択肢の記述が条文と一致するか（○一致/×不一致/△不明）
- 数値確認: 数値がある場合、条文の数値と一致するか

**選択肢d**: {choices[3] if len(choices) > 3 else ''}
- 関連条文: どの条文が関係するか
- 条文との照合: 選択肢の記述が条文と一致するか（○一致/×不一致/△不明）
- 数値確認: 数値がある場合、条文の数値と一致するか

**暫定結論**: 質問タイプ「{question_type}」を考慮した暫定回答と根拠

分析:"""
        
        analysis = self.llm.invoke(analysis_prompt)
        
        # IntegratorAgent: 最終判断（質問タイプに応じた判断基準）
        if question_type == '誤り選択':
            judgment_instruction = """【判断基準】
この問題は「誤っているもの」を選ぶ問題です。
- 条文と「×不一致」の選択肢が正解候補です
- 条文の内容と異なる記述、数値の誤り、要件の欠落などを探してください
- 全て正しそうに見える場合は、最も細かい違いがある選択肢を選んでください"""
        elif question_type == '正しい選択':
            judgment_instruction = """【判断基準】
この問題は「正しいもの」を選ぶ問題です。
- 条文と「○一致」の選択肢が正解候補です
- 条文の内容と完全に一致する記述を探してください"""
        elif question_type == '組み合わせ':
            judgment_instruction = """【判断基準】
この問題は「組み合わせ」問題です。
- 各記述（ア、イ、ウなど）の正誤を判断し、正しい組み合わせを選んでください"""
        else:
            judgment_instruction = """【判断基準】
条文に最も適合する選択肢を選んでください。"""
        
        decision_prompt = f"""あなたは法令QAの最終判断エージェントです。

【RetrieverAgentの分析】
{analysis[:3500]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

{judgment_instruction}

【最終確認】
1. 分析結果を確認
2. 質問タイプ「{question_type}」に応じた正解を特定
3. 最も確実な選択肢を選択

回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = self.llm.invoke(decision_prompt)
        return self._extract_answer(response)
    
    def _format_numbers_table(self, numbers_table: Dict[str, Any]) -> str:
        """数値対比表を読みやすい形式に"""
        lines = []
        
        lines.append("【選択肢の数値】")
        for choice_id, nums in numbers_table.get('choice_numbers', {}).items():
            if nums:
                lines.append(f"  {choice_id}: {', '.join(nums)}")
            else:
                lines.append(f"  {choice_id}: (数値なし)")
        
        context_nums = numbers_table.get('context_numbers', [])
        if context_nums:
            lines.append("【条文に含まれる数値（抜粋）】")
            lines.append(f"  {', '.join(context_nums[:10])}")
        
        return "\n".join(lines)
    
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
    
    vector = VectorRetriever(
        index_path=str(project_root / "data/faiss_index_xml_v2/vector"),
        embedding_model="intfloat/multilingual-e5-large"
    )
    bm25 = BM25Retriever(index_path=str(project_root / "data/faiss_index_xml_v2/bm25"))
    bm25.load_index()
    retriever = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
    
    workflow = ProposedWorkflowV8(retriever=retriever)
    
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("v8 テスト: 難易度事前判定 + 数値対比表強化")
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
