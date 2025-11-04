# Agent Lightning統合提案

## 概要

本ドキュメントは、statutes-ragsプロジェクトにMicrosoft Agent Lightningを統合し、強化学習と自動プロンプト最適化技術によってRAGパイプラインを自動最適化することを提案するものです。

## Agent Lightningについて

Agent Lightningは、Microsoft Researchが開発したフレームワークで、最小限のコード変更でAIエージェントの訓練と最適化を可能にします。本プロジェクトで既に使用されているLangChainを含む、複数のエージェントフレームワークをサポートしています。

### 主要機能

- LangChain、OpenAI Agent SDK、AutoGenなどをサポートするフレームワーク非依存設計
- 統合に必要なコード変更がほぼゼロ
- 強化学習、自動プロンプト最適化、教師あり学習のサポート
- マルチエージェントシステムにおける選択的最適化
- 活発な開発（最新リリースv0.2.1、2025年10月）
- MITライセンス、GitHub 4.3kスター

### プロジェクトリソース

- リポジトリ: https://github.com/microsoft/agent-lightning
- ドキュメント: https://microsoft.github.io/agent-lightning/
- 論文: https://arxiv.org/abs/2508.03680

## 現在のプロジェクト分析

### 既存アーキテクチャ

statutes-ragsプロジェクトは現在、以下を実装しています:

- LangChain Expression Language（LCEL）を使用したRAGパイプライン
- 複数の検索戦略: Vector、BM25、Hybrid
- 4択法令問題の評価フレームワーク
- Ollama（Qwen2.5:7bモデル）とのLLM統合

### 現在のプロンプト

**RAGパイプラインシステムプロンプト**（`app/retrieval/rag_pipeline.py`）:

```python
template = """あなたは日本の法律に精通した法律アシスタントです。以下の法令条文に基づいて質問に答えてください。

【法令条文】
{context}

【質問】
{question}

【回答】
上記の法令条文に基づいて、正確かつ具体的に回答してください。必ず該当する法令名と条文番号を明記してください。"""
```

**評価用プロンプト**（`scripts/evaluate_multiple_choice.py`）:

```python
prompt = """あなたは日本の法律に精通した法律アシスタントです。以下の法令条文に基づいて、4択問題に答えてください。

【法令条文】
{context}

【問題文】
{question}

【選択肢】
{choices}

【指示】
上記の法令条文に基づいて、選択肢a、b、c、dの中から最も適切なものを1つ選んでください。
回答は必ず「a」「b」「c」「d」のいずれか1文字のみを返してください。説明は不要です。

回答: """
```

## 互換性評価

### 技術的互換性

**高い互換性**: 本プロジェクトは以下の理由によりAgent Lightning統合に適しています:

1. **LangChainの使用**: プロジェクトは既に `langchain>=0.1.0` と `langchain-community>=0.0.10` を使用
2. **LCEL実装**: RAGPipelineはネイティブサポートされているLangChain Expression Languageを使用
3. **構造化された評価**: `evaluate_multiple_choice.py` による既存の評価フレームワークが明確なメトリクスを提供
4. **Python 3.10+**: Pythonバージョン要件を満たしている

### 統合アプローチ

Agent Lightningは既存のコードベースへの最小限の変更で統合可能です:

```python
import agentlightning as agl

class RAGPipeline:
    def query(self, question: str) -> Dict[str, Any]:
        # トラッキングのための初期プロンプト記録
        agl.emit_prompt(question)
        
        # 既存の検索ロジック
        documents = self.retrieve_documents(question)
        context = self.format_context(documents)
        
        # トレース付きの既存LLM呼び出し
        with agl.trace_llm():
            answer = self.chain.invoke({"context": context, "question": question})
        
        # 評価メトリクスに基づく報酬の記録
        reward = self._calculate_reward(answer, ground_truth)
        agl.emit_reward(reward)
        
        return {
            "answer": answer.strip(),
            "citations": self.extract_citations(documents),
            "contexts": self._format_contexts(documents)
        }
```

## 期待される効果

### 自動プロンプト最適化

- 評価パフォーマンスに基づくシステムプロンプトの最適化
- 4択法令問題の精度向上
- 特定の法律ドメインへのプロンプト自動適応

### 強化学習

- 評価データセットの正解/不正解から学習
- 反復訓練による応答品質の継続的改善
- 検索パラメータ（top-k、MMR lambda）の動的最適化

### パフォーマンス改善

Agent Lightningを使用した類似のRAG実装に基づく予測:

- 4択問題での精度向上: 10-20%
- 最適化されたコンテキスト使用によるハルシネーション減少
- 引用精度と法的推論の向上

### 評価統合

既存の評価フレームワークを拡張可能:

- 評価実行から訓練データを自動生成
- 訓練イテレーションごとのパフォーマンス改善の追跡
- ベースライン vs. 訓練済みモデルのパフォーマンス比較

## 実装計画

### フェーズ1: セットアップと基本統合

1. `pyproject.toml` に `agentlightning` を依存関係として追加
2. 開発環境でAgent Lightningをインストールして検証
3. RAGPipelineクラスに基本的なトレース機能を追加
4. トレース収集とストレージ機能の検証

### フェーズ2: 評価統合

1. `evaluate_multiple_choice.py` を修正し、正解性に基づく報酬を記録
2. 4択問題の報酬計算ロジックを実装
3. 既存の評価結果から訓練データセットを作成
4. トレースとリソース管理のためのLightningStoreをセットアップ

### フェーズ3: 訓練設定

1. APO（自動プロンプト最適化）アルゴリズムを設定
2. 訓練パラメータ（学習率、バッチサイズ、イテレーション）を設定
3. RAGPipelineにプロンプト更新メカニズムを実装
4. オフライン最適化用の訓練スクリプトを作成

### フェーズ4: 評価と反復

1. 訓練前のベースライン評価を実行
2. 評価データのサブセットで訓練を実行
3. 訓練済みモデルのパフォーマンスを評価
4. メトリクス比較: 精度、引用品質、応答品質
5. 結果に基づく訓練設定の反復改善

## 必要な依存関係

`pyproject.toml` への追加:

```toml
dependencies = [
    # ... 既存の依存関係
    "agentlightning>=0.2.1",
]
```

Agent Lightningの依存関係は自動的にインストールされます:

- コアトレースとイベント記録
- 訓練アルゴリズム（APO、RL）
- データ管理のためのLightningStore

## リスク評価

### 低リスク

- 最小限のコード変更による非侵襲的統合
- 既存機能を壊すことなく段階的に実装可能
- トレース呼び出しを削除するだけで簡単にロールバック可能
- コア検索ロジックの変更は不要

### 考慮事項

- 訓練には計算リソースが必要（GPU推奨）
- 初期セットアップと設定に時間が必要
- 訓練用の評価データセットが必要
- ハイパーパラメータの調整が必要な場合がある

## 代替アプローチ

### 手動プロンプトエンジニアリング

現在のアプローチでは手動での反復とテストが必要です。Agent Lightningはこのプロセスをデータ駆動の最適化により自動化します。

### 従来のファインチューニング

LLMを直接ファインチューニングする場合、以下が必要となります:

- 大規模な計算リソース
- 大規模な訓練データセット
- 破滅的忘却のリスク
- 一般的な言語能力の喪失

Agent Lightningは基本モデルではなく、エージェントの動作（プロンプト、検索）の最適化に焦点を当てています。

## 結論

Agent Lightning統合は、最小限のリスクと開発工数でstatutes-ragsシステムのパフォーマンスを向上させる有望な方法です。LangChainとの互換性と既存の評価インフラストラクチャにより、本プロジェクトに自然に適合します。

### 推奨される次のステップ

1. 基本的なトレース統合による概念実証の実施
2. パフォーマンスメトリクスを確立するためのベースライン評価の実行
3. 評価データ用の報酬メカニズムの実装
4. 小規模な訓練実験の実行
5. 結果の評価と完全統合の判断

### 成功基準

- 4択問題での精度が10%以上向上
- 法的推論におけるハルシネーションの減少
- 引用精度の向上
- 複雑さの増加を最小限に抑えた保守可能なコードベース

## 参考文献

- Agent Lightning GitHub: https://github.com/microsoft/agent-lightning
- Agent Lightning ドキュメント: https://microsoft.github.io/agent-lightning/
- 研究論文: https://arxiv.org/abs/2508.03680
- LangChain統合ガイド: https://microsoft.github.io/agent-lightning/latest/how-to/
