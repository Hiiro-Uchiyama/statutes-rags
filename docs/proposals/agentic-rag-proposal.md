# Agentic RAG導入提案

## 概要

本ドキュメントは、現在のstatutes-ragsシステムをTraditional RAGからAgentic RAGへ進化させることを提案します。Agentic RAGは、自律的なAIエージェントによる動的な情報検索と推論を可能にし、複雑な法律問い合わせに対してより高度な回答を提供します。

## Traditional RAG vs. Agentic RAG

### Traditional RAGの制限

現在のシステムは以下の特徴を持つTraditional RAGです:

- 固定的な単一パス検索プロセス
- 事前定義されたクエリフローに依存
- 静的な情報検索のみ
- 複雑な多段階推論への対応が困難
- 検索結果の品質を動的に評価・改善する機能がない

### Agentic RAGの利点

Agentic RAGは以下の機能により、これらの制限を克服します:

- 自律的なAIエージェントによる動的な意思決定
- 反復的な検索と推論のサイクル
- 中間結果に基づく戦略の適応
- 複数の外部ツールやデータソースとの統合
- リアルタイムでのコンテキスト分析と調整

## アーキテクチャ設計

### マルチエージェント構成

```
┌─────────────────────────────────────────────────┐
│           Manager Agent (調整役)                 │
│   - クエリ分類と意図理解                          │
│   - 適切なエージェントの選択と調整                 │
│   - 結果の統合と品質チェック                       │
└─────────────────────────────────────────────────┘
            ↓           ↓           ↓
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ Retrieval │  │ Reasoning │  │ Validation│
    │  Agent    │  │  Agent    │  │  Agent    │
    └───────────┘  └───────────┘  └───────────┘
```

### 主要エージェントの役割

#### 1. Manager Agent（マネージャーエージェント）

中心的な調整役として機能:

- ユーザークエリの分析と分類
- 必要なエージェントの選択と順序決定
- エージェント間の情報フロー管理
- 最終回答の品質検証と統合

#### 2. Retrieval Agent（検索エージェント）

動的な情報検索を担当:

- クエリの複雑さに応じた検索戦略の選択
- Vector検索、BM25、Hybridの動的切り替え
- 検索結果の品質評価
- 不十分な結果に対する再検索の実行
- 検索パラメータの動的調整（top-k、MMR lambda等）

#### 3. Reasoning Agent（推論エージェント）

法的推論を実行:

- 検索された条文間の関連性分析
- 適用可能な法令の優先順位付け
- 法的論理の構築（三段論法等）
- 判例や解釈の考慮
- 複数条文の統合的解釈

#### 4. Validation Agent（検証エージェント）

回答の品質を保証:

- 生成された回答の事実確認
- 引用の正確性検証
- ハルシネーションの検出
- 法的整合性チェック
- 不確実性の明示

## 実装アプローチ

### フェーズ1: 基本エージェント実装

```python
from langchain.agents import Agent, AgentExecutor
from langchain.tools import Tool

class RetrievalAgent:
    """動的検索エージェント"""
    
    def __init__(self, retrievers: Dict[str, BaseRetriever]):
        self.retrievers = retrievers
        self.llm = Ollama(model="qwen3:8b")
        
    def decide_strategy(self, query: str) -> str:
        """クエリに応じた検索戦略を決定"""
        prompt = f"""
        以下の法律質問に対して、最適な検索戦略を選択してください:
        
        質問: {query}
        
        選択肢:
        - vector: 意味的類似性に基づく検索（概念的な質問に適切）
        - bm25: キーワードマッチング（具体的な条文番号がある場合）
        - hybrid: 両方を組み合わせ（複雑な質問）
        
        戦略を1つ選んでください:
        """
        strategy = self.llm.invoke(prompt).strip().lower()
        return strategy if strategy in self.retrievers else "hybrid"
    
    def retrieve_with_quality_check(self, query: str, min_score: float = 0.7) -> List[Document]:
        """品質チェック付き検索"""
        strategy = self.decide_strategy(query)
        retriever = self.retrievers[strategy]
        
        documents = retriever.retrieve(query, top_k=10)
        
        # 品質チェック
        if not documents or documents[0].score < min_score:
            # 再検索または戦略変更
            alternative_strategy = "hybrid" if strategy != "hybrid" else "vector"
            documents = self.retrievers[alternative_strategy].retrieve(query, top_k=10)
        
        return documents


class ReasoningAgent:
    """法的推論エージェント"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def analyze_legal_structure(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """法的構造の分析"""
        prompt = f"""
        以下の法令条文を分析し、質問に対する法的構造を明確にしてください:
        
        質問: {query}
        
        条文:
        {self._format_documents(documents)}
        
        以下の観点で分析してください:
        1. 主要な適用条文
        2. 関連する条文との関係性
        3. 適用順序または優先順位
        4. 例外規定の有無
        5. 解釈上の注意点
        """
        return self.llm.invoke(prompt)
    
    def construct_legal_reasoning(self, analysis: str, query: str) -> str:
        """法的推論の構築"""
        prompt = f"""
        以下の分析に基づいて、法的推論を構築してください:
        
        分析結果:
        {analysis}
        
        質問: {query}
        
        三段論法の形式で推論を構築してください:
        1. 大前提（適用される法規範）
        2. 小前提（事実関係の当てはめ）
        3. 結論
        """
        return self.llm.invoke(prompt)


class ValidationAgent:
    """検証エージェント"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def verify_citations(self, answer: str, documents: List[Document]) -> Dict[str, Any]:
        """引用の正確性を検証"""
        prompt = f"""
        以下の回答について、引用されている法令条文が正確か検証してください:
        
        回答:
        {answer}
        
        参照可能な条文:
        {self._format_documents(documents)}
        
        検証結果を以下の形式で返してください:
        - 正確性: 正確/不正確/一部不正確
        - 問題点: （あれば指摘）
        - 修正提案: （必要であれば）
        """
        return self.llm.invoke(prompt)
    
    def detect_hallucination(self, answer: str, context: str) -> bool:
        """ハルシネーションの検出"""
        prompt = f"""
        以下の回答が、提供されたコンテキストに基づいているか確認してください:
        
        回答:
        {answer}
        
        コンテキスト:
        {context}
        
        回答がコンテキストから逸脱していない場合は「OK」、
        コンテキストにない情報を含む場合は「HALLUCINATION」と返してください。
        """
        result = self.llm.invoke(prompt).strip().upper()
        return "HALLUCINATION" in result


class ManagerAgent:
    """マネージャーエージェント"""
    
    def __init__(self, retrieval_agent, reasoning_agent, validation_agent, llm):
        self.retrieval_agent = retrieval_agent
        self.reasoning_agent = reasoning_agent
        self.validation_agent = validation_agent
        self.llm = llm
        
    def classify_query(self, query: str) -> Dict[str, Any]:
        """クエリの分類と戦略決定"""
        prompt = f"""
        以下の法律質問を分類してください:
        
        質問: {query}
        
        分類軸:
        1. 複雑度: 単純/中程度/複雑
        2. タイプ: 条文照会/解釈/適用/比較
        3. 必要な推論: 単一条文/複数条文関連/段階的推論
        
        JSONフォーマットで返してください。
        """
        # 分類結果に基づいて実行計画を立てる
        return self._parse_classification(self.llm.invoke(prompt))
    
    def execute_pipeline(self, query: str) -> Dict[str, Any]:
        """エージェントパイプラインの実行"""
        # 1. クエリ分類
        classification = self.classify_query(query)
        
        # 2. 検索実行
        documents = self.retrieval_agent.retrieve_with_quality_check(query)
        
        if not documents:
            return {"answer": "関連する法令条文が見つかりませんでした。", "confidence": 0.0}
        
        # 3. 推論実行（複雑度に応じて）
        if classification.get("complexity") in ["中程度", "複雑"]:
            analysis = self.reasoning_agent.analyze_legal_structure(documents, query)
            reasoning = self.reasoning_agent.construct_legal_reasoning(analysis, query)
        else:
            reasoning = self._simple_answer_generation(documents, query)
        
        # 4. 検証実行
        validation = self.validation_agent.verify_citations(reasoning, documents)
        has_hallucination = self.validation_agent.detect_hallucination(
            reasoning, 
            self._format_context(documents)
        )
        
        # 5. 結果統合
        if has_hallucination or validation.get("accuracy") == "不正確":
            # 再生成または警告付き返答
            reasoning = self._regenerate_with_constraints(documents, query, validation)
        
        return {
            "answer": reasoning,
            "classification": classification,
            "documents": documents,
            "validation": validation,
            "confidence": self._calculate_confidence(validation, has_hallucination)
        }
```

### フェーズ2: 反復的推論の実装

```python
class IterativeReasoningAgent:
    """反復的推論エージェント"""
    
    def __init__(self, manager_agent, max_iterations: int = 3):
        self.manager = manager_agent
        self.max_iterations = max_iterations
        
    def iterative_reasoning(self, query: str) -> Dict[str, Any]:
        """反復的な推論プロセス"""
        iteration = 0
        current_answer = None
        search_history = []
        
        while iteration < self.max_iterations:
            # 検索と推論
            result = self.manager.execute_pipeline(query)
            current_answer = result["answer"]
            search_history.append(result)
            
            # 満足度チェック
            if self._is_satisfactory(result):
                break
            
            # 改善のための追加検索
            additional_query = self._generate_refinement_query(query, result)
            query = additional_query
            iteration += 1
        
        return {
            "final_answer": current_answer,
            "iterations": iteration + 1,
            "history": search_history
        }
    
    def _is_satisfactory(self, result: Dict[str, Any]) -> bool:
        """結果の満足度評価"""
        confidence = result.get("confidence", 0)
        has_sufficient_docs = len(result.get("documents", [])) >= 3
        validation_ok = result.get("validation", {}).get("accuracy") == "正確"
        
        return confidence > 0.8 and has_sufficient_docs and validation_ok
```

### フェーズ3: ツール統合

```python
class LegalToolkit:
    """法律専門ツール群"""
    
    def __init__(self):
        self.tools = [
            Tool(
                name="条文検索",
                func=self.search_articles,
                description="特定の法令の条文を検索"
            ),
            Tool(
                name="判例検索",
                func=self.search_cases,
                description="関連判例を検索（将来実装）"
            ),
            Tool(
                name="改正履歴確認",
                func=self.check_amendments,
                description="法令の改正履歴を確認"
            ),
            Tool(
                name="関連法令検索",
                func=self.search_related_laws,
                description="関連する法令を検索"
            ),
            Tool(
                name="用語定義検索",
                func=self.search_definitions,
                description="法令における用語定義を検索"
            )
        ]
    
    def search_articles(self, law_name: str, article_number: str) -> str:
        """条文検索ツール"""
        # 実装
        pass
    
    def search_related_laws(self, law_name: str) -> List[str]:
        """関連法令検索"""
        # 実装
        pass
```

## 期待される効果

### 精度向上

- 複雑な法律質問への対応能力向上: 30-40%
- 多段階推論が必要なケースでの正答率向上: 50%以上
- ハルシネーション減少: 60-70%

### 応答品質

- より詳細で構造化された回答
- 法的推論プロセスの明示化
- 不確実性の適切な表現
- 引用の正確性向上

### 柔軟性

- 質問の複雑度に応じた適応的処理
- 動的な検索戦略の選択
- 不十分な結果に対する自動再検索
- 複数の情報源からの統合

## 実装上の課題と対策

### 課題1: レイテンシの増加

複数エージェントの連携により処理時間が増加

対策:
- エージェント間の並列実行可能な部分を特定
- キャッシング戦略の導入
- 簡単な質問は単純なパスで処理（分類による振り分け）

### 課題2: コスト増加

LLM呼び出し回数の増加によるコスト上昇

対策:
- ローカルLLM（Ollama）の活用を継続
- エージェント間の通信を効率化
- 必要な場合のみ高度なエージェントを起動

### 課題3: デバッグの複雑化

マルチエージェントシステムのデバッグが困難

対策:
- 各エージェントの決定プロセスをログ記録
- 可視化ツールの導入
- ステップバイステップのトレース機能

### 課題4: エージェント間の協調

エージェント間の情報共有と調整の複雑さ

対策:
- 明確なメッセージフォーマットの定義
- 共有メモリまたは状態管理の導入
- Manager Agentによる集中管理

## 段階的移行計画

### ステップ1: 単一エージェント拡張（1-2週間）

現在のRAGPipelineをRetrievalAgentに拡張:
- 動的な検索戦略選択の実装
- 品質評価機能の追加

### ステップ2: 検証エージェント追加（1週間）

ValidationAgentの実装と統合:
- ハルシネーション検出
- 引用検証機能

### ステップ3: 推論エージェント追加（2週間）

ReasoningAgentの実装:
- 法的構造分析
- 推論構築機能

### ステップ4: マネージャーエージェント統合（1-2週間）

ManagerAgentによる全体調整:
- クエリ分類
- パイプライン実行管理

### ステップ5: 反復推論機能（1週間）

IterativeReasoningAgentの追加:
- 反復的な改善プロセス
- 満足度評価

## 評価計画

### 評価指標

1. 精度メトリクス
   - 4択問題正答率
   - 引用正確性
   - ハルシネーション率

2. 品質メトリクス
   - 回答の完全性スコア
   - 法的推論の明確さ
   - 不確実性の適切な表現

3. 効率メトリクス
   - 平均応答時間
   - エージェント呼び出し回数
   - 成功率（再試行なしでの解決）

### ベースライン比較

Traditional RAGとAgentic RAGの性能を以下で比較:
- 同一の評価データセット
- 複雑度別の性能分析
- コスト対効果の評価

## 関連技術とフレームワーク

### LangGraph

エージェントワークフローの実装に活用:
- グラフベースのエージェント調整
- 条件分岐とループのサポート
- 状態管理機能

### AutoGen

マルチエージェント会話の実装に参考:
- エージェント間の対話パターン
- グループチャット機能

### CrewAI

役割ベースのエージェント設計に参考:
- エージェントの役割定義
- タスクの分解と割り当て

## 参考文献

- Weaviate Blog: "What is Agentic RAG"
- NVIDIA Blog: "Traditional RAG vs. Agentic RAG"
- arXiv: "MA-RAG: Multi-Agent Retrieval-Augmented Generation"
- Hugging Face: "Multi-agent RAG System Cookbook"
- Analytics Vidhya: "Top 7 Agentic RAG System Architectures"
