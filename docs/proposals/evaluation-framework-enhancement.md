# 評価フレームワーク強化提案

## 概要

本ドキュメントは、statutes-ragsシステムの評価フレームワークを強化し、より包括的で信頼性の高い性能評価を実現することを提案します。法律ドメイン特有の評価指標と、自動化された継続的評価の仕組みを導入します。

## 現在の評価の課題

### 既存評価の制限

現在の`evaluate_multiple_choice.py`による評価には以下の制限があります:

1. 評価範囲の限定
   - 4択問題の正答率のみ
   - 回答の質的側面を評価していない
   - 引用の正確性を定量的に測定していない

2. 法律ドメイン特有の指標の欠如
   - 法的推論の妥当性
   - 条文引用の正確性
   - 法令間の関連性理解度

3. 継続的評価の不足
   - 本番環境での性能モニタリングがない
   - ユーザーフィードバックの収集・活用がない
   - 時系列でのパフォーマンス追跡がない

## 包括的評価フレームワーク

### 評価の3つの柱

```
┌──────────────────────────────────────────────┐
│          評価フレームワーク                    │
│                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────┐│
│  │  Retrieval │  │ Generation │  │ Domain ││
│  │  Quality   │  │  Quality   │  │Specific││
│  │   評価     │  │   評価     │  │  評価  ││
│  └────────────┘  └────────────┘  └────────┘│
│        ↓              ↓              ↓      │
│  ┌──────────────────────────────────────┐  │
│  │      総合評価スコア算出              │  │
│  └──────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### 1. Retrieval Quality（検索品質）評価

#### 指標

**Recall@K**: 関連する全ての文書のうち、上位K件に含まれる割合

```python
def calculate_recall_at_k(
    retrieved_docs: List[Document],
    relevant_doc_ids: Set[str],
    k: int
) -> float:
    """Recall@Kの計算"""
    retrieved_ids = {doc.metadata['id'] for doc in retrieved_docs[:k]}
    relevant_retrieved = retrieved_ids & relevant_doc_ids
    
    if len(relevant_doc_ids) == 0:
        return 0.0
    
    return len(relevant_retrieved) / len(relevant_doc_ids)
```

**Precision@K**: 上位K件の文書のうち、実際に関連する文書の割合

```python
def calculate_precision_at_k(
    retrieved_docs: List[Document],
    relevant_doc_ids: Set[str],
    k: int
) -> float:
    """Precision@Kの計算"""
    retrieved_ids = {doc.metadata['id'] for doc in retrieved_docs[:k]}
    relevant_retrieved = retrieved_ids & relevant_doc_ids
    
    if k == 0:
        return 0.0
    
    return len(relevant_retrieved) / k
```

**NDCG@K**: 順位を考慮した検索品質

```python
import numpy as np

def calculate_ndcg_at_k(
    retrieved_docs: List[Document],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """NDCG@Kの計算"""
    # DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        doc_id = doc.metadata['id']
        relevance = relevance_scores.get(doc_id, 0.0)
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    # Ideal DCG
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        idcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
```

**MRR (Mean Reciprocal Rank)**: 最初の関連文書が出現する順位の逆数

```python
def calculate_mrr(
    retrieved_docs: List[Document],
    relevant_doc_ids: Set[str]
) -> float:
    """MRRの計算"""
    for i, doc in enumerate(retrieved_docs):
        if doc.metadata['id'] in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0
```

#### LLMベース関連性評価

ラベルデータがない場合、LLMを用いた関連性評価:

```python
class LLMRelevanceJudge:
    """LLMによる関連性評価"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def judge_relevance(
        self,
        query: str,
        document: Document,
        scale: int = 3
    ) -> int:
        """
        文書の関連性を評価
        
        Args:
            query: ユーザークエリ
            document: 評価対象の文書
            scale: 評価スケール（0〜scale）
        
        Returns:
            関連性スコア（0: 無関連、scale: 完全に関連）
        """
        prompt = f"""
        以下のクエリに対して、文書の関連性を{scale}段階で評価してください。
        
        クエリ: {query}
        
        文書:
        {document.page_content}
        
        評価基準:
        - 0: 全く関連性がない
        - 1: わずかに関連
        - 2: 部分的に関連
        - 3: 完全に関連
        
        数字のみで回答してください:
        """
        
        response = self.llm.invoke(prompt).strip()
        
        try:
            score = int(response)
            return min(max(score, 0), scale)
        except ValueError:
            return 0
```

### 2. Generation Quality（生成品質）評価

#### 指標

**Faithfulness**: 生成された回答が検索されたコンテキストに忠実か

```python
class FaithfulnessEvaluator:
    """Faithfulness評価"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def evaluate(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Faithfulnessの評価
        
        Returns:
            スコアと詳細分析
        """
        context_text = "\n\n".join(contexts)
        
        prompt = f"""
        以下の回答が、提供されたコンテキストに基づいているか評価してください。
        
        回答:
        {answer}
        
        コンテキスト:
        {context_text}
        
        評価基準:
        1. 回答の各ステートメントを抽出
        2. 各ステートメントがコンテキストで裏付けられているか確認
        3. 裏付けられたステートメントの割合を計算
        
        以下の形式で回答してください:
        {{
            "total_statements": <総ステートメント数>,
            "supported_statements": <裏付けられたステートメント数>,
            "score": <0.0〜1.0のスコア>,
            "unsupported_claims": [<裏付けのないステートメントのリスト>]
        }}
        """
        
        response = self.llm.invoke(prompt)
        return self._parse_json_response(response)
```

**Answer Relevance**: 回答がクエリに対して適切か

```python
class AnswerRelevanceEvaluator:
    """Answer Relevance評価"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def evaluate(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        回答の関連性を評価
        
        Returns:
            0.0〜1.0のスコア
        """
        prompt = f"""
        以下の質問に対する回答の関連性を0.0〜1.0で評価してください。
        
        質問: {query}
        
        回答: {answer}
        
        評価基準:
        - 1.0: 質問に完全に答えている
        - 0.7: 質問にほぼ答えているが、一部不足
        - 0.5: 部分的にしか答えていない
        - 0.3: ほとんど答えていない
        - 0.0: 全く答えていない
        
        数値のみで回答してください:
        """
        
        response = self.llm.invoke(prompt).strip()
        
        try:
            return float(response)
        except ValueError:
            return 0.0
```

**Correctness**: 回答の正確性（Ground Truthとの比較）

```python
class CorrectnessEvaluator:
    """Correctness評価"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def evaluate(
        self,
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        正確性の評価
        
        Returns:
            スコアと詳細分析
        """
        prompt = f"""
        以下の回答を正解と比較して、正確性を評価してください。
        
        回答: {answer}
        
        正解: {ground_truth}
        
        評価観点:
        1. 事実の正確性（完全一致/部分一致/不一致）
        2. 網羅性（必要な情報が含まれているか）
        3. 過不足（余分な情報や不足している情報）
        
        以下の形式で回答してください:
        {{
            "score": <0.0〜1.0>,
            "factual_accuracy": "<完全一致/部分一致/不一致>",
            "completeness": <0.0〜1.0>,
            "reasoning": "<評価理由>"
        }}
        """
        
        response = self.llm.invoke(prompt)
        return self._parse_json_response(response)
```

### 3. Domain-Specific（法律ドメイン特化）評価

#### 法的引用の正確性

```python
class CitationAccuracyEvaluator:
    """法的引用の正確性評価"""
    
    def __init__(self):
        self.citation_pattern = re.compile(
            r'([^法]+法)(?:第(\d+)条)?(?:第(\d+)項)?(?:第(\d+)号)?'
        )
        
    def evaluate(
        self,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        引用の正確性を評価
        
        Returns:
            正確性スコアと詳細
        """
        # 回答から引用を抽出
        cited_references = self._extract_citations(answer)
        
        # 検索された文書と照合
        available_refs = self._get_available_references(retrieved_docs)
        
        accurate_citations = 0
        inaccurate_citations = []
        
        for citation in cited_references:
            if self._verify_citation(citation, available_refs):
                accurate_citations += 1
            else:
                inaccurate_citations.append(citation)
        
        total_citations = len(cited_references)
        score = accurate_citations / total_citations if total_citations > 0 else 0.0
        
        return {
            "score": score,
            "total_citations": total_citations,
            "accurate_citations": accurate_citations,
            "inaccurate_citations": inaccurate_citations
        }
    
    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        """テキストから引用を抽出"""
        citations = []
        for match in self.citation_pattern.finditer(text):
            citations.append({
                "law": match.group(1),
                "article": match.group(2),
                "paragraph": match.group(3),
                "item": match.group(4)
            })
        return citations
    
    def _verify_citation(
        self,
        citation: Dict[str, str],
        available_refs: Set[tuple]
    ) -> bool:
        """引用が正確かを検証"""
        ref_tuple = (
            citation["law"],
            citation.get("article"),
            citation.get("paragraph"),
            citation.get("item")
        )
        return ref_tuple in available_refs
```

#### 法的推論の妥当性

```python
class LegalReasoningEvaluator:
    """法的推論の妥当性評価"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        法的推論の妥当性を評価
        
        Returns:
            スコアと詳細分析
        """
        prompt = f"""
        以下の法律に関する回答について、法的推論の妥当性を評価してください。
        
        質問: {query}
        
        回答: {answer}
        
        参照条文:
        {chr(10).join(contexts)}
        
        評価観点:
        1. 三段論法の構造（大前提・小前提・結論）が明確か
        2. 適用法令の選択が適切か
        3. 法令間の優先順位が正しいか
        4. 例外規定の考慮があるか
        5. 法的解釈が妥当か
        
        以下の形式で回答してください:
        {{
            "overall_score": <0.0〜1.0>,
            "structure_score": <0.0〜1.0>,
            "law_selection_score": <0.0〜1.0>,
            "interpretation_score": <0.0〜1.0>,
            "issues": [<問題点のリスト>],
            "strengths": [<良い点のリスト>]
        }}
        """
        
        response = self.llm.invoke(prompt)
        return self._parse_json_response(response)
```

#### 法令間の関連性理解度

```python
class LegalRelationshipEvaluator:
    """法令間の関連性理解度評価"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def evaluate(
        self,
        query: str,
        answer: str,
        expected_relationships: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        法令間の関連性理解度を評価
        
        Args:
            expected_relationships: 期待される法令関係
                例: [{"from": "民法", "to": "特別法", "type": "一般法-特別法"}]
        
        Returns:
            スコアと分析
        """
        prompt = f"""
        以下の回答が、法令間の関連性を正しく理解しているか評価してください。
        
        質問: {query}
        
        回答: {answer}
        
        期待される法令関係:
        {json.dumps(expected_relationships, ensure_ascii=False, indent=2)}
        
        評価基準:
        1. 一般法と特別法の関係を理解しているか
        2. 上位法と下位法の関係を理解しているか
        3. 関連する法令を適切に参照しているか
        4. 法令の適用順序が正しいか
        
        以下の形式で回答してください:
        {{
            "score": <0.0〜1.0>,
            "identified_relationships": [<特定された関係のリスト>],
            "missed_relationships": [<見逃された関係のリスト>],
            "incorrect_relationships": [<誤った関係のリスト>]
        }}
        """
        
        response = self.llm.invoke(prompt)
        return self._parse_json_response(response)
```

## 統合評価パイプライン

```python
# app/evaluation/comprehensive_evaluator.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """評価結果"""
    query: str
    answer: str
    ground_truth: Optional[str]
    
    # Retrieval評価
    recall_at_5: float
    precision_at_5: float
    ndcg_at_5: float
    mrr: float
    
    # Generation評価
    faithfulness: float
    answer_relevance: float
    correctness: Optional[float]
    
    # Domain-specific評価
    citation_accuracy: float
    legal_reasoning_score: float
    relationship_understanding: float
    
    # 総合スコア
    overall_score: float
    
    # 詳細情報
    details: Dict[str, Any]


class ComprehensiveEvaluator:
    """包括的評価パイプライン"""
    
    def __init__(
        self,
        llm,
        retrieval_evaluators: Dict[str, Any],
        generation_evaluators: Dict[str, Any],
        domain_evaluators: Dict[str, Any]
    ):
        self.llm = llm
        self.retrieval_evaluators = retrieval_evaluators
        self.generation_evaluators = generation_evaluators
        self.domain_evaluators = domain_evaluators
        
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        ground_truth: Optional[str] = None,
        relevant_doc_ids: Optional[Set[str]] = None
    ) -> EvaluationResult:
        """包括的評価の実行"""
        
        # Retrieval評価
        retrieval_scores = {}
        if relevant_doc_ids:
            retrieval_scores = self._evaluate_retrieval(
                retrieved_docs,
                relevant_doc_ids
            )
        
        # Generation評価
        generation_scores = self._evaluate_generation(
            query,
            answer,
            retrieved_docs,
            ground_truth
        )
        
        # Domain-specific評価
        domain_scores = self._evaluate_domain_specific(
            query,
            answer,
            retrieved_docs
        )
        
        # 総合スコア計算
        overall_score = self._calculate_overall_score(
            retrieval_scores,
            generation_scores,
            domain_scores
        )
        
        return EvaluationResult(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            recall_at_5=retrieval_scores.get("recall@5", 0.0),
            precision_at_5=retrieval_scores.get("precision@5", 0.0),
            ndcg_at_5=retrieval_scores.get("ndcg@5", 0.0),
            mrr=retrieval_scores.get("mrr", 0.0),
            faithfulness=generation_scores["faithfulness"],
            answer_relevance=generation_scores["answer_relevance"],
            correctness=generation_scores.get("correctness"),
            citation_accuracy=domain_scores["citation_accuracy"],
            legal_reasoning_score=domain_scores["legal_reasoning"],
            relationship_understanding=domain_scores["relationship_understanding"],
            overall_score=overall_score,
            details={
                "retrieval": retrieval_scores,
                "generation": generation_scores,
                "domain": domain_scores
            }
        )
    
    def _evaluate_retrieval(
        self,
        retrieved_docs: List[Document],
        relevant_doc_ids: Set[str]
    ) -> Dict[str, float]:
        """Retrieval評価"""
        return {
            "recall@5": calculate_recall_at_k(retrieved_docs, relevant_doc_ids, 5),
            "precision@5": calculate_precision_at_k(retrieved_docs, relevant_doc_ids, 5),
            "ndcg@5": calculate_ndcg_at_k(retrieved_docs, {}, 5),
            "mrr": calculate_mrr(retrieved_docs, relevant_doc_ids)
        }
    
    def _evaluate_generation(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """Generation評価"""
        contexts = [doc.page_content for doc in retrieved_docs]
        
        faithfulness_eval = FaithfulnessEvaluator(self.llm)
        relevance_eval = AnswerRelevanceEvaluator(self.llm)
        
        scores = {
            "faithfulness": faithfulness_eval.evaluate(answer, contexts)["score"],
            "answer_relevance": relevance_eval.evaluate(query, answer)
        }
        
        if ground_truth:
            correctness_eval = CorrectnessEvaluator(self.llm)
            scores["correctness"] = correctness_eval.evaluate(
                answer, ground_truth
            )["score"]
        
        return scores
    
    def _evaluate_domain_specific(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, float]:
        """Domain-specific評価"""
        citation_eval = CitationAccuracyEvaluator()
        reasoning_eval = LegalReasoningEvaluator(self.llm)
        
        citation_result = citation_eval.evaluate(answer, retrieved_docs)
        reasoning_result = reasoning_eval.evaluate(
            query,
            answer,
            [doc.page_content for doc in retrieved_docs]
        )
        
        return {
            "citation_accuracy": citation_result["score"],
            "legal_reasoning": reasoning_result["overall_score"],
            "relationship_understanding": 0.0  # 実装に応じて
        }
    
    def _calculate_overall_score(
        self,
        retrieval_scores: Dict[str, float],
        generation_scores: Dict[str, Any],
        domain_scores: Dict[str, float]
    ) -> float:
        """総合スコアの計算"""
        weights = {
            "retrieval": 0.2,
            "generation": 0.4,
            "domain": 0.4
        }
        
        retrieval_avg = np.mean(list(retrieval_scores.values())) if retrieval_scores else 0.0
        generation_avg = np.mean([
            generation_scores["faithfulness"],
            generation_scores["answer_relevance"]
        ])
        domain_avg = np.mean(list(domain_scores.values()))
        
        return (
            weights["retrieval"] * retrieval_avg +
            weights["generation"] * generation_avg +
            weights["domain"] * domain_avg
        )
```

## 継続的評価システム

### モニタリングダッシュボード

```python
# app/monitoring/dashboard.py
from typing import List
import pandas as pd
import plotly.graph_objects as go

class EvaluationDashboard:
    """評価ダッシュボード"""
    
    def __init__(self, results_db_path: str):
        self.results_db_path = results_db_path
        
    def generate_report(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """期間レポートの生成"""
        results = self._load_results(start_date, end_date)
        
        return {
            "summary": self._generate_summary(results),
            "trends": self._analyze_trends(results),
            "breakdown": self._generate_breakdown(results),
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_summary(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """サマリー統計"""
        df = pd.DataFrame([vars(r) for r in results])
        
        return {
            "avg_overall_score": df["overall_score"].mean(),
            "avg_faithfulness": df["faithfulness"].mean(),
            "avg_citation_accuracy": df["citation_accuracy"].mean(),
            "total_queries": len(results)
        }
    
    def _analyze_trends(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """トレンド分析"""
        # 時系列での性能変化を分析
        pass
    
    def plot_performance_over_time(
        self,
        results: List[EvaluationResult]
    ) -> go.Figure:
        """性能の時系列プロット"""
        df = pd.DataFrame([vars(r) for r in results])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df["overall_score"],
            mode='lines+markers',
            name='Overall Score'
        ))
        fig.add_trace(go.Scatter(
            y=df["faithfulness"],
            mode='lines+markers',
            name='Faithfulness'
        ))
        
        fig.update_layout(
            title="Performance Over Time",
            xaxis_title="Query Index",
            yaxis_title="Score"
        )
        
        return fig
```

## 実装計画

### フェーズ1: 基本評価指標実装（2週間）

1. Retrieval評価指標
   - Recall@K, Precision@K, NDCG, MRR
2. Generation評価指標
   - Faithfulness, Answer Relevance

### フェーズ2: ドメイン特化評価（2週間）

1. 引用正確性評価
2. 法的推論評価
3. 統合評価パイプライン

### フェーズ3: 継続的評価システム（2週間）

1. 評価結果の保存機構
2. ダッシュボード実装
3. アラート機能

### フェーズ4: 最適化（1週間）

1. パフォーマンスチューニング
2. ドキュメント整備
3. CI/CD統合

## 期待される効果

- より包括的な性能評価
- 法律ドメインに特化した品質保証
- 継続的な改善サイクルの確立
- 本番環境での性能モニタリング
- データ駆動の意思決定

## 参考文献

- RAGAS: RAG評価フレームワーク
- LRAGE: Legal RAG Evaluation Tool
- Toloka AI: RAG Evaluation Technical Guide
- NVIDIA: Traditional RAG vs Agentic RAG
