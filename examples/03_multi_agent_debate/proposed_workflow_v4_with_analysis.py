#!/usr/bin/env python3
"""
提案手法 v4 + 事後分析モジュール

v4の判断精度を維持しつつ、事後分析による説明可能性を追加。
"""
import re
import logging
from typing import Dict, Any, List, Optional

from langchain_community.llms import Ollama

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

from app.utils.number_normalizer import normalize_article_numbers
from post_analysis import PostAnalyzer, AnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProposedWorkflowV4WithAnalysis:
    """
    v4 + 事後分析
    
    v4の判断プロセスは変更せず、事後的に以下を分析:
    - 引用された条文の抽出・検証
    - 確信度スコアの算出
    - エラーパターンの検出
    """
    
    def __init__(
        self,
        retriever,
        llm_model: str = "qwen3:8b",
        timeout: int = 180,
        num_ctx: int = 16000,
        top_k: int = 30
    ):
        self.retriever = retriever
        self.top_k = top_k
        
        # LLM
        self.llm = Ollama(
            model=llm_model,
            timeout=timeout,
            num_ctx=num_ctx
        )
        
        # 事後分析器
        self.analyzer = PostAnalyzer()
    
    def query(
        self,
        question: str,
        choices: List[str],
        question_id: int = 0
    ) -> Dict[str, Any]:
        """
        質問に回答し、事後分析を実行
        
        Returns:
            {
                'answer': 最終回答,
                'method': 'v4_with_analysis',
                'analysis': AnalysisResult,
                'llm_response': LLMの生成テキスト,
                'context': 使用したコンテキスト
            }
        """
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # Step 1: 検索
        docs = self.retriever.retrieve(full_query, top_k=self.top_k)
        context = self._build_context(docs)
        
        # Step 2: 質問タイプ判定
        question_type = self._detect_question_type(question)
        
        # Step 3: RetrieverAgent分析
        retriever_analysis = self._retriever_agent_analysis(
            question, choices, context, question_type
        )
        
        # Step 4: IntegratorAgent最終判断
        answer, llm_response = self._integrator_agent_decision(
            question, choices, context, retriever_analysis, question_type
        )
        
        # Step 5: 事後分析（精度に影響なし）
        analysis = self.analyzer.analyze(
            question_id=question_id,
            question=question,
            choices=choices,
            answer=answer,
            llm_response=llm_response,
            context=context
        )
        
        return {
            'answer': answer,
            'method': 'v4_with_analysis',
            'analysis': analysis,
            'llm_response': llm_response,
            'context': context[:1000],  # 保存用に短縮
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
    
    def _retriever_agent_analysis(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str
    ) -> str:
        """RetrieverAgent: 検索結果を分析"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの分析エージェントです。
検索された条文を参考に、質問を分析してください。

【検索された条文】
{context[:5000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【分析を行ってください】
1. 各選択肢に関連する条文を特定してください
2. 重要な数値や条件を抽出してください
3. 選択肢と条文の一致/不一致を確認してください
4. 暫定的な回答と根拠を示してください

分析:"""
        
        return self.llm.invoke(prompt)
    
    def _integrator_agent_decision(
        self,
        question: str,
        choices: List[str],
        context: str,
        retriever_analysis: str,
        question_type: str
    ) -> tuple:
        """IntegratorAgent: 最終判断"""
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの最終判断エージェントです。

【RetrieverAgentの分析】
{retriever_analysis[:3000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【判断方法】
- 質問タイプが「{question_type}」であることを考慮してください
- RetrieverAgentの分析を確認し、最終回答を決定してください
- 回答は a, b, c, d のいずれか1文字のみで最後に答えてください

最終判断:"""
        
        response = self.llm.invoke(prompt)
        answer = self._extract_answer(response)
        
        return answer, response
    
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
    
    def get_analysis_summary(self, results: List[Dict]) -> str:
        """
        複数の結果から分析サマリを生成
        
        Args:
            results: query()の戻り値のリスト
        
        Returns:
            分析レポート
        """
        analysis_results = [r['analysis'] for r in results if 'analysis' in r]
        return self.analyzer.generate_report(analysis_results)


# テスト用
if __name__ == "__main__":
    import json
    
    sys.path.insert(0, str(project_root / "app"))
    
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    
    # 初期化
    vector = VectorRetriever(
        index_path="data/faiss_index_xml_v2/vector",
        embedding_model="intfloat/multilingual-e5-large"
    )
    bm25 = BM25Retriever(index_path="data/faiss_index_xml_v2/bm25")
    bm25.load_index()
    retriever = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
    
    workflow = ProposedWorkflowV4WithAnalysis(retriever=retriever)
    
    # テストデータ
    with open(project_root / 'datasets/lawqa_jp/data/selection.json', 'r') as f:
        data = json.load(f)
    
    # Q1でテスト
    sample = data['samples'][0]
    choices = []
    for line in sample['選択肢'].split("\n"):
        line = line.strip()
        if line and line[0] in "abcd" and " " in line:
            choices.append(line[2:].strip())
    
    result = workflow.query(sample['問題文'], choices, question_id=1)
    
    print(f"回答: {result['answer']}")
    print(f"正解: {sample['output'].strip().lower()}")
    print(f"\n【事後分析】")
    analysis = result['analysis']
    print(f"確信度: {analysis.confidence_score:.2f} ({analysis.confidence_level.value})")
    print(f"引用: {[c.article_ref for c in analysis.citations]}")
    print(f"エラーパターン: {analysis.error_patterns}")
    print(f"警告: {analysis.warnings}")
    print(f"推論品質: {analysis.reasoning_quality}")



