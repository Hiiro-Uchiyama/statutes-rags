# Multi-Agent Debate - マルチエージェント議論システム

複数のエージェントが法的解釈について議論し、合意形成を行うシステムです。

## 目次

- [概要](#概要)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [評価](#評価)
- [設定](#設定)
- [トラブルシューティング](#トラブルシューティング)

## 概要

異なる立場を持つ2つの議論エージェント（肯定的・批判的）が法律質問について議論を行い、モデレーターが最終的な回答を統合します。

### 主な特徴

- 肯定的解釈と批判的解釈の両面からの分析
- LangGraphによる明示的な議論フロー制御
- 埋め込みベースの合意判定（類似度計算）
- 議論履歴の完全な記録

### エージェント構成

- **Debater A（肯定的）**: 法令の積極的な適用を支持する解釈
- **Debater B（批判的）**: 例外規定や制約条件を慎重に検討する解釈
- **Moderator**: 議論を調整し、最終判断を行う

## セットアップ

### 前提条件

1. プロジェクトルートでの環境セットアップが完了していること
2. Ollamaがインストールされ、`qwen3:8b`モデルが利用可能であること
3. FAISSインデックスが構築されていること

### インストール

#### 方法A: 既存環境が整っている場合

プロジェクトルートで仮想環境が既にセットアップされている場合:

```bash
cd /path/to/statutes-rags

# 仮想環境を有効化（既にセットアップ済みの場合）
source .venv/bin/activate

# examples用の追加依存関係をインストール
uv pip install -e ".[examples]"
```

#### 方法B: 新規セットアップの場合

```bash
cd /path/to/statutes-rags

# uvを使った環境セットアップ
./setup/setup_uv_env.sh

# 仮想環境を有効化
source .venv/bin/activate

# examples用の依存関係をインストール
uv pip install langgraph langsmith
```

### 動作確認テスト（モック使用）

実際のLLMやデータなしで基本動作を確認:

```bash
cd examples/03_multi_agent_debate

# クイックテストを実行
python tests/test_quick.py
```

すべてのテストがパスすれば、実装は正常です。

## 使用方法

### Pythonスクリプトから

```python
import sys
from pathlib import Path

# パスを追加（数字で始まるディレクトリ名のため）
sys.path.insert(0, str(Path("examples/03_multi_agent_debate")))

from workflow import DebateWorkflow
from config import load_config

# 設定のロード
config = load_config()

# ワークフローの初期化
workflow = DebateWorkflow(config)

# 質問を実行
result = workflow.query("会社法第26条について教えてください")

# 結果の表示
print("回答:", result["answer"])
print("ラウンド数:", result["metadata"]["rounds"])
print("合意スコア:", result["metadata"]["agreement_score"])

# 議論の履歴を確認
for history in result["metadata"]["debate_history"]:
    print(f"\nラウンド {history['round']}:")
    print("Debater A:", history["debater_a"]["position"][:100], "...")
    print("Debater B:", history["debater_b"]["position"][:100], "...")
```

### 基本的な質問の例

```python
workflow = DebateWorkflow(load_config())

result = workflow.query("""
民法第1条の基本原則について説明してください。

a. 信義誠実の原則
b. 権利濫用の禁止
c. 公共の福祉
d. すべて

正しいものを選んでください。
""")

print(result["answer"])
```

### 複雑な解釈を要する質問

```python
result = workflow.query("""
個人情報保護法において、第三者提供の同意が不要となるケースは
どのような場合ですか？
""")

# 議論の過程を確認
for i, history in enumerate(result["metadata"]["debate_history"], 1):
    print(f"\n=== ラウンド {i} ===")
    print("肯定的解釈:")
    print(history["debater_a"]["position"])
    print("\n批判的解釈:")
    print(history["debater_b"]["position"])

print(f"\n=== 最終回答 ===")
print(result["answer"])
```

### カスタム設定

```python
import sys
from pathlib import Path

# パスを追加（数字で始まるディレクトリ名のため）
sys.path.insert(0, str(Path("examples/03_multi_agent_debate")))

from config import MultiAgentDebateConfig
from workflow import DebateWorkflow

# カスタム設定
config = MultiAgentDebateConfig(
    max_debate_rounds=5,          # より多くのラウンド
    agreement_threshold=0.9,       # より高い合意閾値
    llm_temperature=0.0,          # より決定的な応答
    retrieval_top_k=15            # より多くの文書を検索
)

workflow = DebateWorkflow(config)
result = workflow.query("あなたの質問")
```

## 評価

### 4択法令問題での評価

デジタル庁の4択法令問題データセットを使用した評価については、[USAGE.md](./USAGE.md)の「ステップ2」以降を参照してください。

### 判例評価

実際の判例データを使用して、マルチエージェント議論システムが判例と同じ結論を出せるかを評価します。

詳細は[USAGE.md](./USAGE.md)の「判例評価」セクションを参照してください。

**クイックスタート:**

```bash
cd examples/03_multi_agent_debate

# 3判例でテスト
python evaluate_precedent.py \
  --precedent-dir data_set/precedent \
  --limit 3 \
  --output results/precedent_test.json
```

### 実際の評価実験（オプション）

実際にLLMとデータを使用した評価を行う場合:

#### Ollamaのセットアップ

```bash
# Ollamaのインストール（まだの場合）
# https://ollama.ai からダウンロード

# Ollamaサーバーを起動
ollama serve

# 別のターミナルでモデルをプル
ollama pull qwen3:8b
```

#### データセットの準備

プロジェクトのREADMEに従ってデータセットを準備:

```bash
# datasets/egov_laws/ に法令XMLファイルを配置
# datasets/lawqa_jp/data/ に評価データを配置

# データ前処理
cd /path/to/statutes-rags
make preprocess

# FAISSインデックス構築
make index
```

#### 評価の実行

```bash
cd examples/03_multi_agent_debate

# 5問で試行（1問あたり30-60秒）
python evaluate.py --limit 5

# 全問題で評価
python evaluate.py

# カスタムデータセットで評価
python evaluate.py --dataset /path/to/dataset.json

# 結果の出力先を指定
python evaluate.py --output results.json
```

### 評価指標

- 正答率（4択問題）
- 平均ラウンド数
- 平均合意スコア
- 合意形成率（高い合意スコアに達した割合）
- 1問あたりの平均処理時間

### 結果の確認

評価結果はJSONファイルとして保存されます:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "metrics": {
    "total_questions": 20,
    "correct_answers": 15,
    "accuracy": 0.75,
    "avg_rounds": 2.3,
    "avg_agreement_score": 0.82,
    "agreement_formation_rate": 0.65,
    "avg_time_per_question": 45.2
  },
  "results": [...]
}
```

## 設定

### 環境変数

`.env`ファイルまたは環境変数で設定可能:

```bash
# 議論設定
DEBATE_MAX_ROUNDS=3
DEBATE_AGREEMENT_THRESHOLD=0.8

# LLM設定
LLM_MODEL=qwen3:8b
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60

# 検索設定
DEBATE_RETRIEVAL_TOP_K=10
VECTOR_STORE_PATH=data/faiss_index
DATA_PATH=data/egov_laws.jsonl

# 埋め込みモデル
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# デバッグ
DEBATE_VERBOSE=false
```

### 主要パラメータの説明

- `max_debate_rounds`: 議論の最大ラウンド数（推奨: 2-4）
- `agreement_threshold`: 合意判定の閾値（0.0-1.0、推奨: 0.7-0.9）
- `llm_temperature`: LLMの温度（0.0-1.0、低いほど一貫性が高い）
- `retrieval_top_k`: 検索する文書数（推奨: 5-15）

## トラブルシューティング

### LangGraphがインストールされていない

```bash
uv pip install langgraph
```

### インポートエラー

プロジェクトルートからPythonを実行しているか確認:

```bash
# プロジェクトルートで
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/03_multi_agent_debate/quick_test.py
```

### Ollamaが起動していない

```bash
# Ollamaを起動
ollama serve

# 別のターミナルでモデルを確認
ollama list
```

### Ollamaに接続できない

```bash
# Ollamaの状態確認
ps aux | grep ollama

# 環境変数を確認
echo $OLLAMA_HOST  # デフォルト: http://localhost:11434

# Ollamaを再起動
ollama serve
```

### FAISSインデックスが見つからない

```bash
# プロジェクトルートで
python scripts/build_index.py
```

### メモリ不足

環境変数で検索文書数を減らす:

```bash
export DEBATE_RETRIEVAL_TOP_K=5
```

### 処理が遅い

最大ラウンド数を減らす:

```bash
export DEBATE_MAX_ROUNDS=2
```

または合意閾値を下げる:

```bash
export DEBATE_AGREEMENT_THRESHOLD=0.7
```

## テスト

```bash
cd examples/03_multi_agent_debate

# クイックテスト（モック使用）
python tests/test_quick.py

# Pytestによる単体テスト
pytest tests/ -v

# 特定のテストファイルのみ
pytest tests/test_multi_agent_debate.py -v

# クイックテストのみ
pytest tests/test_quick.py -v
```

## ディレクトリ構成

```
03_multi_agent_debate/
├── __init__.py
├── README.md                    # このファイル（システム概要）
├── SETUP.md                     # セットアップガイド
├── USAGE.md                     # 使用方法ガイド（ステップバイステップ）
├── config.py                    # 設定管理
├── workflow.py                  # LangGraph議論ワークフロー
├── evaluate.py                  # 4択問題評価スクリプト
├── evaluate_precedent.py        # 判例評価スクリプト
├── precedent_loader.py          # 判例データローダー
├── agents/
│   ├── __init__.py
│   ├── debater.py               # 議論エージェント
│   └── moderator.py             # モデレーター
├── data_set/                    # データセット
│   ├── precedent/               # 判例データ
│   └── law/                     # 法令データ
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # テストフィクスチャ
│   ├── test_multi_agent_debate.py  # 単体テスト
│   └── test_quick.py            # クイックテスト
└── results/                     # 評価結果（.gitignoreで除外）
    └── archive/                 # 過去の結果
```

## 動作確認

### 確認日時
2024年11月7日

### テスト結果
- クイックテスト: 全5項目パス ✓
- 簡易評価（3問）: 実行完了 ✓
  - 議論システム正常動作
  - 合意形成率: 100%
  - 平均合意スコア: 0.921
  - 平均処理時間: 137.8秒/問

### 実際のLLM評価結果（Ollama + qwen3:8b）

#### インデックスロード状況
- **ベクトルインデックス**: 2,802,222件のドキュメント正常ロード
- **BM25インデックス**: 10,000件のドキュメント正常ロード
- **ハイブリッド検索**: 1問あたり10件の関連文書を取得

#### パフォーマンス
- **処理時間**: 約26秒/問（検索最適化後）
  - 修正前（検索無効時）: 108秒/問
  - 改善率: **76%短縮**
- **議論ラウンド数**: 平均1ラウンド
- **合意スコア**: 0.99（非常に高い合意形成）

### 修正履歴

#### 第1段階: インポートエラーの修正
数字で始まるディレクトリ名（`03_multi_agent_debate`）がPythonモジュール名として無効であるため、以下のファイルのインポート方法を修正：

1. **agents/__init__.py**: 相対インポートに変更
2. **workflow.py**: sys.pathに現在のディレクトリを追加
3. **evaluate.py**: sys.pathに現在のディレクトリを追加、データセット形式対応
4. **tests/test_quick.py**: パス設定を修正
5. **tests/test_multi_agent_debate.py**: インポートパスを修正
6. **tests/conftest.py**: インポートパスを修正

#### 第2段階: データセット形式の対応
- `evaluate.py`の`load_dataset`関数を修正：`{'samples': [...]}`形式に対応
- `parse_choices`関数を追加：文字列形式の選択肢をリストに変換

#### 第3段階: Retrieverパスの修正（最重要）
**問題**: 相対パス（`data/faiss_index`）がexamples/03_multi_agent_debateから実行すると解決できない

**解決策**: `workflow.py`の`_initialize_retriever`メソッドで、相対パスをプロジェクトルートからの絶対パスに変換
```python
project_root = Path(__file__).parent.parent.parent
base_path = Path(config.vector_store_path)
if not base_path.is_absolute():
    base_path = project_root / base_path
```

これらの修正により、全てのテストと実際のLLM評価が正常に動作することを確認しました。

## 参考文献

- Du, Y., et al. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate.
- Liang, T., et al. (2023). Encouraging divergent thinking in large language models through multi-agent debate.

## ライセンス

本プロジェクトのライセンスに従います。

## サポート

問題が発生した場合:

1. `quick_test.py` を実行して依存関係をチェック
2. Ollamaの起動を確認
3. ログレベルを上げて詳細確認

## 関連実装

- `01_agentic_rag` - エージェント型RAG（実装済み）
- `02_mcp_egov_agent` - e-Gov API連携（実装済み）
- `04_legal_case_generator` - 事例生成（実装済み）
