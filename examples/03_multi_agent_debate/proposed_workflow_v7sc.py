#!/usr/bin/env python3
"""
提案手法 v7sc: Self-Consistency（多数決）によるマルチエージェント推論

アーキテクチャ:
- 複数の独立したエージェントが同じ質問を処理
- 各エージェントの回答を収集
- 多数決で最終回答を決定
- 温度パラメータで多様性を確保

v7オリジナル: 75.7% (106/140)
目標: 80%以上
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 180
    num_ctx: int = 16000
    top_k: int = 30
    # Self-Consistency設定
    num_agents: int = 3  # エージェント数
    temperature: float = 0.3  # 多様性のための温度（低めに設定）


class ProposedWorkflowV7SC:
    """
    v7sc: Self-Consistency（多数決）によるマルチエージェント推論
    
    - 複数エージェントによる並列推論
    - 多数決による最終回答決定
    - 不一致時の信頼度考慮
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
        
        # 複数エージェント用LLM（温度を変えて多様性を確保）
        self.agents = []
        for i in range(self.config.num_agents):
            # 各エージェントに少し異なる温度を設定
            temp = self.config.temperature + (i * 0.1)
            agent = Ollama(
                model=self.config.llm_model,
                timeout=self.config.timeout,
                num_ctx=self.config.num_ctx,
                temperature=temp
            )
            self.agents.append(agent)
        
        # 基本LLM（温度0、決定論的）
        self.llm = Ollama(
            model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx,
            temperature=0.0
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
        # Simple問題のみSelf-Consistency、それ以外は通常処理
        if difficulty == DifficultyLevel.SIMPLE:
            # Self-Consistency推論（マルチエージェント）
            agent_answers = self._multi_agent_inference(
                question, choices, context, question_type, difficulty, numbers_table
            )
            final_answer, confidence, vote_details = self._aggregate_answers(agent_answers)
        else:
            # Moderate/Complex: 通常の単一エージェント処理
            if difficulty == DifficultyLevel.MODERATE:
                final_answer = self._moderate_path(question, choices, context, question_type, numbers_table, self.llm)
            else:
                final_answer = self._complex_path(question, choices, context, question_type, numbers_table, self.llm)
            agent_answers = [{'agent_id': 0, 'answer': final_answer, 'status': 'success'}]
            confidence = 1.0
            vote_details = {final_answer: 1}
        
        return {
            'answer': final_answer,
            'method': f'v7sc_{difficulty.value}',
            'difficulty': difficulty.value,
            'question_type': question_type,
            'numbers_table': numbers_table,
            'agent_answers': agent_answers,
            'confidence': confidence,
            'vote_details': vote_details
        }
    
    def _multi_agent_inference(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        difficulty: DifficultyLevel,
        numbers_table: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """複数エージェントによる並列推論"""
        
        def run_agent(agent_id: int, agent: Ollama) -> Dict[str, Any]:
            """単一エージェントの推論を実行"""
            try:
                if difficulty == DifficultyLevel.SIMPLE:
                    answer = self._simple_path(question, choices, context, question_type, agent)
                elif difficulty == DifficultyLevel.MODERATE:
                    answer = self._moderate_path(question, choices, context, question_type, numbers_table, agent)
                else:
                    answer = self._complex_path(question, choices, context, question_type, numbers_table, agent)
                
                return {
                    'agent_id': agent_id,
                    'answer': answer,
                    'status': 'success'
                }
            except Exception as e:
                logger.warning(f"Agent {agent_id} failed: {e}")
                return {
                    'agent_id': agent_id,
                    'answer': None,
                    'status': 'error',
                    'error': str(e)
                }
        
        # 並列実行
        answers = []
        with ThreadPoolExecutor(max_workers=self.config.num_agents) as executor:
            futures = {
                executor.submit(run_agent, i, agent): i 
                for i, agent in enumerate(self.agents)
            }
            for future in as_completed(futures):
                result = future.result()
                answers.append(result)
        
        # agent_idでソート
        answers.sort(key=lambda x: x['agent_id'])
        return answers
    
    def _aggregate_answers(
        self,
        agent_answers: List[Dict[str, Any]]
    ) -> Tuple[str, float, Dict[str, int]]:
        """多数決で最終回答を決定"""
        # 有効な回答を収集
        valid_answers = [
            a['answer'] for a in agent_answers 
            if a['status'] == 'success' and a['answer'] in ['a', 'b', 'c', 'd']
        ]
        
        if not valid_answers:
            return 'a', 0.0, {}
        
        # 投票カウント
        vote_counter = Counter(valid_answers)
        vote_details = dict(vote_counter)
        
        # 最多得票を取得
        most_common = vote_counter.most_common(1)[0]
        final_answer = most_common[0]
        vote_count = most_common[1]
        
        # 信頼度 = 得票率
        confidence = vote_count / len(valid_answers)
        
        return final_answer, confidence, vote_details
    
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
        """難易度事前判定"""
        full_text = question + " " + " ".join(choices)
        
        complexity_score = 0
        
        for kw in self.COMPLEXITY_KEYWORDS['high']:
            if kw in full_text:
                complexity_score += 2
        
        for kw in self.COMPLEXITY_KEYWORDS['medium']:
            if kw in full_text:
                complexity_score += 1
        
        numbers = self._extract_all_numbers(full_text)
        if len(numbers) >= 3:
            complexity_score += 1
        
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
        question_type: str,
        agent: Ollama
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
        
        response = agent.invoke(prompt)
        return self._extract_answer(response)
    
    def _moderate_path(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        numbers_table: Dict[str, Any],
        agent: Ollama
    ) -> str:
        """中程度パス: 2エージェント処理"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        numbers_display = self._format_numbers_table(numbers_table)
        
        # RetrieverAgent分析
        analysis_prompt = f"""あなたは法令QAの分析エージェントです。
問題と条文を分析してください。

【検索された条文】
{context[:5000]}

【数値対比表】
{numbers_display}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

各選択肢の正誤を条文と照合して分析してください。
数値がある場合は特に注意して確認してください。

分析:"""
        
        analysis = agent.invoke(analysis_prompt)
        
        # IntegratorAgent判断
        decision_prompt = f"""あなたは法令QAの最終判断エージェントです。

【分析結果】
{analysis[:3000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}
（「誤り選択」の場合は誤っている選択肢を、「正しい選択」の場合は正しい選択肢を選んでください）

分析結果を踏まえ、最も適切な回答を選んでください。
回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = agent.invoke(decision_prompt)
        return self._extract_answer(response)
    
    def _complex_path(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str,
        numbers_table: Dict[str, Any],
        agent: Ollama
    ) -> str:
        """複雑パス: 選択肢独立分析"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        numbers_display = self._format_numbers_table(numbers_table)
        
        # 各選択肢を独立分析
        choice_analyses = []
        for i, choice in enumerate(choices):
            choice_id = chr(97 + i)
            
            analysis_prompt = f"""あなたは法令QAの分析エージェントです。
以下の選択肢を条文と照合して分析してください。

【検索された条文】
{context[:3500]}

【問題】
{question}

【分析対象の選択肢】
{choice_id}. {choice}

【関連数値】
選択肢の数値: {numbers_table.get('choice_numbers', {}).get(choice_id, [])}
条文の数値: {numbers_table.get('context_numbers', [])[:10]}

この選択肢は条文と一致しますか？
- 一致: 条文の内容と合っている
- 不一致: 条文の内容と異なる
- 不明: 判断に必要な情報が不足

判定結果（一致/不一致/不明）と根拠を述べてください:"""
            
            analysis = agent.invoke(analysis_prompt)
            choice_analyses.append({
                'choice_id': choice_id,
                'choice': choice,
                'analysis': analysis
            })
        
        # 統合判断
        analyses_summary = "\n\n".join([
            f"【選択肢{a['choice_id']}】{a['choice']}\n分析: {a['analysis'][:500]}"
            for a in choice_analyses
        ])
        
        integration_prompt = f"""あなたは法令QAの最終判断エージェントです。

【各選択肢の分析結果】
{analyses_summary}

【問題】
{question}

【質問タイプ】: {question_type}
（「誤り選択」の場合は「不一致」の選択肢を、「正しい選択」の場合は「一致」の選択肢を選んでください）

各選択肢の分析を踏まえ、最終回答を決定してください。
回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = agent.invoke(integration_prompt)
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
    
    # 3エージェントでテスト
    config = WorkflowConfig(num_agents=3, temperature=0.7)
    workflow = ProposedWorkflowV7SC(retriever=retriever, config=config)
    
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("v7sc テスト: Self-Consistency（多数決）マルチエージェント")
    print(f"エージェント数: {config.num_agents}")
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
              f"信頼度={result['confidence']:.0%} 投票={result['vote_details']}")
    
    print(f"\n正解: {correct}/5 ({correct/5*100:.0f}%)")
