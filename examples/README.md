# Examples - 拡張実装

本ディレクトリには、基本的なRAGシステムを拡張した4つの実装が含まれています。

## 概要

各実装は、研究・学会発表を見越したMVP（Minimum Viable Product）として設計されています。

```
examples/
├── 01_agentic_rag/          # Agentic RAG実装
├── 02_mcp_egov_agent/       # e-Gov API MCP Agent
├── 03_multi_agent_debate/   # マルチエージェント議論
├── 04_legal_case_generator/ # 法令適用事例生成
├── shared/                  # 共有ユーティリティ
├── tests/                   # テスト
├── IMPLEMENTATION-SPEC.md   # 共通仕様書
└── TECH-RESEARCH.md         # 技術調査レポート
```

## 実装の方向性

### 1. Agentic RAG（01_agentic_rag/）

複数のエージェントが協調して法律質問に回答するシステム。

- Manager Agent: クエリ分類とワークフロー制御
- Retrieval Agent: 動的検索戦略選択
- Reasoning Agent: 法的推論
- Validation Agent: 回答検証

詳細: [01_agentic_rag/README.md](./01_agentic_rag/README.md)

### 2. e-Gov API MCP Agent（02_mcp_egov_agent/）

e-Gov APIを通じて最新の法令データを動的に取得するエージェント。

- リアルタイムな法令データ取得
- ローカルキャッシュとのハイブリッド運用
- LangChain Toolsとしての実装

詳細: [02_mcp_egov_agent/README.md](./02_mcp_egov_agent/README.md)

### 3. Multi-Agent Debate（03_multi_agent_debate/）

複数のエージェントが法的解釈について議論し、合意形成を行うシステム。

- Debater A: 肯定的解釈
- Debater B: 批判的解釈
- Moderator: 議論調整と最終判断

詳細: 未実装（Phase 3で実装予定）

### 4. Legal Case Generator（04_legal_case_generator/）

法令条文から具体的な適用事例を自動生成するシステム。

- Scenario Generator: 事例シナリオ生成
- Legal Checker: 法的整合性検証
- Refiner: 事例の洗練

詳細: [04_legal_case_generator/README.md](./04_legal_case_generator/README.md)

## 共通設計

### 技術スタック

- エージェントフレームワーク: LangGraph
- LLM: Ollama（ローカル）、qwen3:8b
- 埋め込み: intfloat/multilingual-e5-large
- 既存RAGコンポーネントの活用

### ディレクトリ構造

各実装は以下の統一構造を持ちます：

```
XX_implementation_name/
├── __init__.py
├── agents/              # エージェント実装
├── config.py            # 設定管理
├── pipeline.py          # メインパイプライン
├── evaluate.py          # 評価スクリプト
└── docs/
    ├── README.md        # 実装概要
    ├── research.md      # 研究背景と目的
    ├── implementation.md # 実装詳細
    └── evaluation.md    # 評価結果
```

### ドキュメント構成

研究として必要な情報を網羅：

1. **研究背景（research.md）**
   - 動機
   - 先行研究
   - 研究仮説
   - 期待される効果

2. **実装詳細（implementation.md）**
   - アーキテクチャ
   - 主要コンポーネント
   - データフロー
   - 実装の工夫

3. **評価（evaluation.md）**
   - 評価指標
   - 実験設定
   - 結果
   - 考察

## セットアップ

### 自動セットアップ（推奨）

既存のセットアップスクリプトは、**既にexamples用の依存関係を含んでいます**。

```bash
# プロジェクトルートで（初回のみ）
./setup/setup_uv_env.sh
```

これにより、基本的なRAG依存関係とexamples用の依存関係（LangGraph、xmltodict等）が自動的にインストールされます。

### 既存環境への追加インストール

既に基本セットアップが完了している場合：

```bash
# 仮想環境を有効化
source .venv/bin/activate

# Examples用の依存関係のみインストール
./setup/setup_examples.sh

# または直接
uv pip install -e ".[examples]"
```

詳細は `examples/SETUP.md` を参照してください。

### 環境変数の設定

`.env`ファイルに以下を追加：

```bash
# Agentic RAG設定
AGENTIC_MAX_ITERATIONS=3
AGENTIC_CONFIDENCE_THRESHOLD=0.8
AGENTIC_ENABLE_REASONING=true
AGENTIC_ENABLE_VALIDATION=true

# 複雑度判定
AGENTIC_COMPLEXITY_SIMPLE=0.3
AGENTIC_COMPLEXITY_COMPLEX=0.7
```

## 使用方法

### 01_agentic_rag の例

```python
from examples.01_agentic_rag.pipeline import AgenticRAGPipeline
from examples.01_agentic_rag.config import load_config

# 設定のロード
config = load_config()

# パイプラインの初期化
pipeline = AgenticRAGPipeline(config)

# 質問
result = pipeline.query("会社法第26条について教えてください")

print(result["answer"])
print(f"反復回数: {result['metadata']['iterations']}")
print(f"使用エージェント: {result['metadata']['agents_used']}")
```

### 評価の実行

```bash
cd examples/01_agentic_rag
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json
```

## 実装状況

| 実装 | 状態 | 進捗 |
|------|------|------|
| 01_agentic_rag | 完了 | 実装済み、テスト済み |
| 02_mcp_egov_agent | 完了 | 実装済み、テスト済み |
| 03_multi_agent_debate | 完了 | 実装済み、テスト済み |
| 04_legal_case_generator | 完了 | 実装済み、テスト済み |

## 評価方針

### 共通評価指標

- 正答率（4択問題）
- ベースライン（既存RAG）との比較
- 処理時間
- LLM呼び出し回数

### 実装固有の評価

各実装は、固有の評価指標も定義します。

詳細は各実装のドキュメントを参照してください。

## テスト

```bash
# 全テストの実行
cd examples
pytest tests/

# 特定の実装のみ
pytest tests/test_01_agentic_rag.py
```

## 開発ガイドライン

### 新しいエージェントの追加

1. `examples/shared/base_agent.py`を継承
2. `execute()`メソッドを実装
3. 適切なドキュメントを記述

```python
from examples.shared.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def execute(self, input_data):
        # カスタムロジック
        result = self._safe_llm_invoke(prompt)
        return {"output": result}
```

### コーディング規約

- 絵文字は使用しない
- ドキュメント文字列は必須
- ロギングを適切に使用
- エラーハンドリングを忘れずに

## ライセンス

本プロジェクトのライセンスに従います。

## 貢献

各実装は独立しており、個別に開発・改善が可能です。

## 参考資料

- [共通仕様書](./IMPLEMENTATION-SPEC.md)
- [技術調査レポート](./TECH-RESEARCH.md)
- [プロジェクトREADME](../README.md)

