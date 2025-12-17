#!/usr/bin/env python3
"""
提案手法 v5: 適応的マルチエージェントワークフロー

改善点:
1. Rerankerによる検索精度向上
2. 問題複雑さに応じた処理分岐
3. 数値抽出・比較の専用ロジック
4. 関連条文の自動追加検索
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

from app.retrieval.reranker import Reranker, HybridReranker
from app.utils.complexity_analyzer import ComplexityAnalyzer, AdaptiveProcessor, ComplexityLevel
from app.utils.numeric_extractor import NumericExtractor, NumericComparator
from app.utils.related_article_finder import RelatedArticleFinder
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
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ProposedWorkflowV5:
    """
    適応的マルチエージェントワークフロー v5
    
    問題の複雑さに応じて処理を分岐:
    - SIMPLE: シンプルRAG
    - MODERATE: 強化RAG（Reranker + 数値比較）
    - COMPLEX: フルマルチエージェント
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
        
        # コンポーネント初期化
        self.complexity_analyzer = ComplexityAnalyzer()
        self.adaptive_processor = AdaptiveProcessor(self.complexity_analyzer)
        self.numeric_extractor = NumericExtractor()
        self.numeric_comparator = NumericComparator(self.numeric_extractor)
        self.related_finder = RelatedArticleFinder(retriever)
        
        # Reranker（オプション）
        if self.config.use_reranker:
            try:
                self.reranker = Reranker(
                    model_name=self.config.reranker_model,
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
        
        Args:
            question: 問題文
            choices: 選択肢リスト
            
        Returns:
            回答結果
        """
        # 1. 複雑さ分析
        proc_config = self.adaptive_processor.get_processing_config(question, choices)
        complexity = proc_config['complexity']
        
        logger.info(f"Complexity: {complexity.level.value} (score: {complexity.score:.2f})")
        
        # 2. 複雑さに応じた処理
        if complexity.level == ComplexityLevel.SIMPLE:
            return self._process_simple(question, choices, proc_config)
        elif complexity.level == ComplexityLevel.MODERATE:
            return self._process_moderate(question, choices, proc_config)
        else:
            return self._process_complex(question, choices, proc_config)
    
    def _process_simple(
        self,
        question: str,
        choices: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """シンプルRAG処理"""
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # 検索
        docs = self.retriever.retrieve(full_query, top_k=config['top_k'])
        
        # コンテキスト構築
        context = self._build_context(docs)
        
        # LLM回答
        answer = self._generate_answer_simple(question, choices, context)
        
        return {
            'answer': answer,
            'strategy': 'simple_rag',
            'complexity': complexity_to_dict(config['complexity'])
        }
    
    def _process_moderate(
        self,
        question: str,
        choices: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """強化RAG処理（Reranker + 数値比較）"""
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # 検索
        docs = self.retriever.retrieve(full_query, top_k=config['top_k'])
        
        # Reranker適用
        if self.reranker and config.get('use_reranker', True):
            ranked_docs = self.reranker.rerank(full_query, docs, top_k=20)
            context = "\n\n".join([d.content for d in ranked_docs])
        else:
            context = self._build_context(docs)
        
        # 数値比較情報を追加
        numeric_info = ""
        if config.get('use_numeric_comparison', True):
            for i, choice in enumerate(choices):
                comparison = self.numeric_comparator.generate_comparison_summary(choice, context)
                if "不一致" in comparison or "一致" in comparison:
                    numeric_info += f"\n選択肢{chr(97+i)}の数値比較:\n{comparison}\n"
        
        # LLM回答
        answer = self._generate_answer_enhanced(question, choices, context, numeric_info)
        
        return {
            'answer': answer,
            'strategy': 'enhanced_rag',
            'complexity': complexity_to_dict(config['complexity'])
        }
    
    def _process_complex(
        self,
        question: str,
        choices: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """フルマルチエージェント処理"""
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # Step 1: 初期検索
        docs = self.retriever.retrieve(full_query, top_k=config['top_k'])
        
        # Reranker適用
        if self.reranker and config.get('use_reranker', True):
            ranked_docs = self.reranker.rerank(full_query, docs, top_k=25)
            context = "\n\n".join([d.content for d in ranked_docs])
        else:
            context = self._build_context(docs)
        
        # Step 2: 関連条文の追加検索
        if config.get('use_related_search', True):
            # 法令名を抽出
            law_match = re.search(r'([一-龥ぁ-んァ-ン]+法)', question)
            base_law = law_match.group(1) if law_match else ""
            
            expanded_context = self.related_finder.expand_context_with_related(
                context,
                base_law,
                max_related=3
            )
        else:
            expanded_context = context
        
        # Step 3: 数値比較情報
        numeric_info = ""
        if config.get('use_numeric_comparison', True):
            for i, choice in enumerate(choices):
                comparison = self.numeric_comparator.generate_comparison_summary(choice, expanded_context)
                if "不一致" in comparison or "一致" in comparison:
                    numeric_info += f"\n選択肢{chr(97+i)}:\n{comparison}\n"
        
        # Step 4: RetrieverAgent分析
        retriever_analysis = self._retriever_agent_analyze(question, choices, expanded_context)
        
        # Step 5: IntegratorAgent統合
        answer = self._integrator_agent_decide(
            question, choices, expanded_context,
            retriever_analysis, numeric_info
        )
        
        return {
            'answer': answer,
            'strategy': 'full_multi_agent',
            'complexity': complexity_to_dict(config['complexity']),
            'retriever_analysis': retriever_analysis
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
    
    def _generate_answer_simple(
        self,
        question: str,
        choices: List[str],
        context: str
    ) -> str:
        """シンプルRAGの回答生成"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""以下の法令条文を参照して、4択問題に回答してください。

【条文】
{context[:6000]}

【問題】
{question}

【選択肢】
{choices_text}

条文の内容と照らし合わせて、最も適切な選択肢を1つ選んでください。
回答は a, b, c, d のいずれか1文字のみで答えてください。

回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _generate_answer_enhanced(
        self,
        question: str,
        choices: List[str],
        context: str,
        numeric_info: str
    ) -> str:
        """強化RAGの回答生成"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""以下の法令条文を参照して、4択問題に回答してください。

【条文】
{context[:5000]}

【数値比較情報】
{numeric_info if numeric_info else "なし"}

【問題】
{question}

【選択肢】
{choices_text}

手順:
1. 各選択肢を条文と比較
2. 数値（期間、割合など）の一致/不一致を確認
3. 最も適切な選択肢を選択

回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

分析と回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _retriever_agent_analyze(
        self,
        question: str,
        choices: List[str],
        context: str
    ) -> Dict[str, Any]:
        """RetrieverAgentの分析"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""法令QAの分析を行います。

【検索された条文】
{context[:4000]}

【問題】
{question}

【選択肢】
{choices_text}

以下を分析してください:
1. 質問タイプ（誤り選択/正しい選択/組み合わせ等）
2. 各選択肢に関連する条文
3. 重要な数値・条件
4. 暫定的な回答と根拠

分析:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            'analysis': response,
            'question_type': self._detect_question_type(question)
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
        
        prompt = f"""【RetrieverAgentの分析】
{retriever_analysis.get('analysis', '')[:2000]}

【数値比較結果】
{numeric_info if numeric_info else "なし"}

【問題】
{question}

【選択肢】
{choices_text}

上記の分析を統合し、最終回答を決定してください。

判断ポイント:
- 条文との整合性
- 数値の一致/不一致
- 質問タイプ（{retriever_analysis.get('question_type', '不明')}）に応じた回答

回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = self.llm.invoke(prompt)
        return self._extract_answer(response)
    
    def _detect_question_type(self, question: str) -> str:
        """質問タイプを検出"""
        if '誤っている' in question or '誤り' in question:
            return '誤り選択'
        elif '正しい' in question:
            return '正しい選択'
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
            # 「回答: a」のようなパターン
            match = re.search(r'(?:回答|答え|answer)[:\s]*([abcd])', line)
            if match:
                return match.group(1)
            # 単独の a, b, c, d
            if line in ['a', 'b', 'c', 'd']:
                return line
            # 「a.」「(a)」などのパターン
            match = re.search(r'^[\(\[]?([abcd])[\)\]\.。]?$', line)
            if match:
                return match.group(1)
        
        # テキスト全体から最後のa/b/c/dを探す
        matches = re.findall(r'\b([abcd])\b', response)
        if matches:
            return matches[-1]
        
        return 'a'


def complexity_to_dict(complexity) -> Dict[str, Any]:
    """ComplexityAnalysisを辞書に変換"""
    return {
        'level': complexity.level.value,
        'score': complexity.score,
        'factors': complexity.factors
    }

