# Agentic RAG - エージェント型RAGシステム

Traditional RAGを拡張し、複数のエージェントが協調して法律質問に回答するAgentic RAGシステムです。

## 目次

- [概要](#概要)
- [研究背景](#研究背景)
- [アーキテクチャ](#アーキテクチャ)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [評価](#評価)
- [実装詳細](#実装詳細)
- [トラブルシューティング](#トラブルシューティング)

## 概要

### 主な特徴

- 複数エージェントによる協調的な質問応答
- 質問の複雑度に応じた動的ワークフロー
- 反復的な検索と推論による精度向上
- 回答の妥当性を自動検証

### エージェント構成

- **Manager Agent**: クエリ分類と全体制御
- **Retrieval Agent**: 動的検索戦略選択
- **Reasoning Agent**: 法的推論（複雑な質問のみ）
- **Validation Agent**: 回答検証（オプション）

### ワークフロー

```
User Query
    ↓
┌─────────────────────┐
│  Manager Agent      │ ← クエリ分類と全体制御
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Retrieval Agent    │ ← 動的検索戦略選択
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Reasoning Agent    │ ← 法的推論（複雑な質問のみ）
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Validation Agent   │ ← 回答検証（オプション）
└─────────────────────┘
    ↓
Final Answer
```

## 研究背景

### Traditional RAGの課題

従来のRAGシステムは単一パスの検索と生成を行いますが、複雑な法律質問では以下の課題があります:

- 一度の検索で必要な情報が得られない
- 複数の条文の関連付けが困難
- 回答の妥当性を検証する機構がない
- 検索戦略を動的に変更できない

### 研究仮説

**主仮説**: 複雑な法律質問に対して、複数のエージェントが協調して反復的に検索と推論を行うAgentic RAGは、単一パスのTraditional RAGよりも高精度な回答を生成できる。

**副仮説**:
1. 質問の複雑度別の効果 - 複雑な質問ほど精度向上が大きい
2. エージェント別の貢献 - 各エージェントが特定の問題で効果を発揮
3. コスト効率 - 精度向上がLLM呼び出し回数の増加を正当化

## アーキテクチャ

### State定義

```python
class AgenticRAGState(TypedDict):
    # 入力
    query: str
    
    # クエリ分析結果
    complexity: str         # "simple" | "medium" | "complex"
    query_type: str         # "lookup" | "interpretation" | "application"
    
    # 検索結果
    documents: List[Any]
    retrieval_strategy: str
    
    # 推論結果
    reasoning: str
    legal_structure: Dict[str, Any]
    
    # 回答生成
    answer: str
    citations: List[Dict[str, Any]]
    
    # メタデータ
    iteration: int
    max_iterations: int
    confidence: float
    agents_used: List[str]
    
    # 制御フラグ
    needs_retry: bool
    is_valid: bool
```

### 複雑度別ワークフロー

質問の複雑度に応じて処理を最適化:

- **Simple**: 既存RAGと同等（高速）
- **Medium**: Retrieval Agent + 簡易推論
- **Complex**: 全エージェントを使用

## セットアップ

### 依存関係のインストール

```bash
# プロジェクトルートに移動
cd statutes-rags

# LangGraphをインストール（uv使用時）
uv pip install "langgraph>=0.2.0,<0.3.0"

# または pip を使用
pip install "langgraph>=0.2.0,<0.3.0"
```

### 環境変数の設定（オプション）

`.env` ファイルに以下を追加:

```bash
# Agentic RAG設定
AGENTIC_MAX_ITERATIONS=3
AGENTIC_CONFIDENCE_THRESHOLD=0.8
AGENTIC_ENABLE_REASONING=true
AGENTIC_ENABLE_VALIDATION=true

# 複雑度判定
AGENTIC_COMPLEXITY_SIMPLE=0.3
AGENTIC_COMPLEXITY_COMPLEX=0.7

# LLM設定（既存の設定を使用）
LLM_MODEL=gpt-oss:20b
LLM_TEMPERATURE=0.1
```

### インデックスの確認

既存のRAGインデックスが構築されていることを確認:

```bash
# プロジェクトルートに移動
cd statutes-rags

# インデックスの存在確認
ls -la data/faiss_index/
# vector/ と bm25/ が存在することを確認
```

インデックスがない場合は構築:

```bash
# プロジェクトルートで
make index

# または直接スクリプトを実行
python scripts/build_index.py
```

## 使用方法

### Pythonスクリプトから使用

```python
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.01_agentic_rag.pipeline import AgenticRAGPipeline
from examples.01_agentic_rag.config import load_config

# 設定のロード
config = load_config()

# パイプラインの初期化
pipeline = AgenticRAGPipeline(config)

# 質問
result = pipeline.query("会社法第26条について教えてください")

# 結果の表示
print("回答:", result["answer"])

print("\n引用:")
for citation in result["citations"]:
    article_str = f"第{citation['article']}条" if citation['article'] else ""
    print(f"  - {citation['law_title']} {article_str}")

print("\nメタデータ:")
print(f"  複雑度: {result['metadata']['complexity']}")
print(f"  反復回数: {result['metadata']['iterations']}")
print(f"  使用エージェント: {', '.join(result['metadata']['agents_used'])}")
print(f"  信頼度: {result['metadata']['confidence']:.2f}")
```

### コマンドラインから使用

```bash
# examples/01_agentic_rag ディレクトリに移動
cd examples/01_agentic_rag

# 単一質問（Pythonワンライナー）
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').parent.parent))

from examples.01_agentic_rag.pipeline import AgenticRAGPipeline
from examples.01_agentic_rag.config import load_config

pipeline = AgenticRAGPipeline(load_config())
result = pipeline.query('民法第1条について教えてください')
print('回答:', result['answer'])
print('信頼度:', result['metadata']['confidence'])
"
```

### 設定のカスタマイズ

#### 反復回数の変更

```bash
# 環境変数で設定
AGENTIC_MAX_ITERATIONS=5 python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json

# またはPythonコードで
from examples.01_agentic_rag.config import AgenticRAGConfig
from examples.01_agentic_rag.pipeline import AgenticRAGPipeline

config = AgenticRAGConfig(max_iterations=5)
pipeline = AgenticRAGPipeline(config)
result = pipeline.query("質問文")
```

#### エージェントの無効化

```bash
# Reasoning Agentを無効化
AGENTIC_ENABLE_REASONING=false python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json

# Validation Agentを無効化
AGENTIC_ENABLE_VALIDATION=false python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json

# 両方とも無効化（Traditional RAGに近い動作）
AGENTIC_ENABLE_REASONING=false AGENTIC_ENABLE_VALIDATION=false python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json
```

## 評価

### デジタル庁4択データでの評価

```bash
# examples/01_agentic_rag ディレクトリに移動
cd examples/01_agentic_rag

# 全問題を評価
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/evaluation_full.json

# 最初の10問のみ評価（テスト用）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/evaluation_test.json \
    --max-questions 10

# 注意: データセットが存在しない場合は、プロジェクトのセットアップガイドに従ってダウンロードしてください
```

### 結果の確認

```bash
# 結果ファイルの内容を確認（jqがインストールされている場合）
cat results/evaluation_full.json | jq '.accuracy, .avg_iterations, .avg_llm_calls'

# サマリーのみ表示（Pythonワンライナー）
python -c "
import json
with open('results/evaluation_full.json') as f:
    data = json.load(f)
print(f'Accuracy: {data.get(\"accuracy\", 0)*100:.2f}%')
print(f'Correct: {data.get(\"correct_count\", 0)}/{data.get(\"total\", 0)}')
print(f'Avg Iterations: {data.get(\"avg_iterations\", 0):.2f}')
print(f'Avg LLM Calls: {data.get(\"avg_llm_calls\", 0):.2f}')
"
```

### 評価指標

- **正答率**: 4択問題での正解率
- **複雑度別正答率**: Simple/Medium/Complexごとの正答率
- **平均反復回数**: エージェントが反復した平均回数
- **LLM呼び出し回数**: 問題あたりの平均LLM呼び出し回数
- **処理時間**: 問題あたりの平均処理時間
- **コスト効率**: (精度向上) / (追加LLM呼び出し回数)

### ベースラインとの比較

```bash
# ベースライン（既存RAG）の評価
# プロジェクトルートに移動
cd statutes-rags
python scripts/evaluate_multiple_choice.py \
    --dataset datasets/lawqa_jp/data/selection.json \
    --output baseline_results.json

# Agentic RAGの評価
cd examples/01_agentic_rag
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/agentic_results.json

# 結果を比較（Pythonワンライナー）
python -c "
import json

with open('../../baseline_results.json') as f:
    baseline = json.load(f)
    
with open('results/agentic_results.json') as f:
    agentic = json.load(f)
    
baseline_acc = baseline.get('accuracy', 0)
agentic_acc = agentic.get('accuracy', 0)
improvement = (agentic_acc - baseline_acc) * 100

print(f'Baseline Accuracy: {baseline_acc * 100:.2f}%')
print(f'Agentic Accuracy: {agentic_acc * 100:.2f}%')
print(f'Improvement: {improvement:+.2f} points')
"
```

## 実装詳細

### 主要コンポーネント

#### 1. Manager Agent

責務:
- クエリの複雑度判定
- ワークフローの決定
- エージェント間の調整

実装例:
```python
class ManagerAgent(BaseAgent):
    def classify_complexity(self, query: str) -> str:
        """
        クエリの複雑度を判定
        Returns: "simple" | "medium" | "complex"
        """
        # LLMを使用して複雑度を判定
```

#### 2. Retrieval Agent

責務:
- 検索戦略の動的選択
- 検索実行
- 検索結果の品質評価

実装例:
```python
class RetrievalAgent(BaseAgent):
    def select_strategy(self, query: str, query_type: str) -> str:
        """
        クエリに応じた検索戦略を選択
        Returns: "vector" | "bm25" | "hybrid"
        """
```

#### 3. Reasoning Agent

責務:
- 複数条文の関連性分析
- 法的推論の構築
- 適用順序・優先順位の判定

#### 4. Validation Agent

責務:
- 回答の妥当性検証
- 引用の正確性チェック
- ハルシネーション検出

### LangGraphワークフロー

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgenticRAGState)

# ノードの追加
workflow.add_node("classify", classify_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("reason", reason_node)
workflow.add_node("validate", validate_node)

# エッジの設定
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_by_complexity, {...})
workflow.add_edge("retrieve", "reason")
workflow.add_conditional_edges("reason", should_continue, {...})
workflow.add_conditional_edges("validate", validation_result, {...})
```

### ディレクトリ構造

```
01_agentic_rag/
├── __init__.py
├── README.md           # このファイル
├── agents/
│   ├── __init__.py
│   ├── manager.py      # Manager Agent
│   ├── retrieval.py    # Retrieval Agent
│   ├── reasoning.py    # Reasoning Agent
│   └── validation.py   # Validation Agent
├── config.py           # 設定管理
├── pipeline.py         # メインパイプライン
├── evaluate.py         # 評価スクリプト
└── requirements.txt
```

## トラブルシューティング

### エラー: "No module named 'langgraph'"

LangGraphがインストールされていません。

```bash
# uvを使用している場合
uv pip install "langgraph>=0.2.0,<0.3.0"

# pipを使用している場合
pip install "langgraph>=0.2.0,<0.3.0"
```

### エラー: "Index not found"

```bash
# プロジェクトルートでインデックスを構築
cd statutes-rags
make index

# または直接ビルドスクリプトを実行
python scripts/build_index.py
```

### エラー: "LLM timeout"

LLMのレスポンスが遅い場合、タイムアウトエラーが発生します。

```bash
# タイムアウトを延長（秒単位）
LLM_TIMEOUT=120 python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json

# または環境変数で設定
export LLM_TIMEOUT=120
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json
```

### パフォーマンスが遅い

処理速度を改善するには、以下のオプションを検討してください:

```bash
# 反復回数を減らす
AGENTIC_MAX_ITERATIONS=1 python evaluate.py ...

# Validation Agentを無効化
AGENTIC_ENABLE_VALIDATION=false python evaluate.py ...

# Reasoning Agentも無効化（最速だが精度低下）
AGENTIC_ENABLE_REASONING=false AGENTIC_ENABLE_VALIDATION=false python evaluate.py ...

# 取得文書数を減らす
AGENTIC_RETRIEVAL_TOP_K=5 python evaluate.py ...

# 複数の設定を組み合わせ
AGENTIC_MAX_ITERATIONS=1 AGENTIC_RETRIEVAL_TOP_K=5 python evaluate.py ...
```

### ログの確認

デバッグや問題解析のために詳細なログを確認できます:

```bash
# 詳細ログを表示してファイルにも保存
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json 2>&1 | tee evaluation.log

# エラーのみ表示
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json 2>&1 | grep ERROR

# Pythonのログレベルを変更（より詳細に）
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" && python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json
```

## テスト

```bash
# examples/01_agentic_rag ディレクトリに移動
cd examples/01_agentic_rag

# 全テストの実行
pytest tests/ -v

# 特定のテストファイルのみ
pytest tests/test_agentic_rag.py -v

# 特定のテストクラスのみ
pytest tests/test_agentic_rag.py::TestManagerAgent -v

# 統合テストを除外（実環境不要なテストのみ実行）
pytest tests/ -v -m "not integration"

# カバレッジ付きで実行
pytest tests/ --cov=examples.01_agentic_rag --cov-report=html
```

## 制約事項

### 実装上の制約

- LLM応答のパースに失敗する可能性（JSON形式が崩れる）
- LLMの推論能力に依存（特に複雑度判定）
- 反復処理により処理時間が増加

### リソース制約

- ローカルLLM（Ollama）の性能制限
- メモリ使用量の増加（状態管理）
- 並列処理の制限

## 今後の拡張

- エージェント選択戦略の最適化
- 反復回数の動的調整
- 複雑度判定アルゴリズムの改善
- 他の法律タスクへの適用

## 参考文献

1. Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629
2. Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366
3. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020

## ライセンス

本プロジェクトのライセンスに従います。

