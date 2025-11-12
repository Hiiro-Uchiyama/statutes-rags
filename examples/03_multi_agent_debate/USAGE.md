# Multi-Agent Debate - 使用方法ガイド

このガイドは上から順に実行することで、動作確認から評価実験まで完了できるように構成されています。

## 前提条件

セットアップが完了していることを確認してください。未完了の場合は [SETUP.md](./SETUP.md) を参照してください。

### セットアップ確認

```bash
# 1. 仮想環境を有効化
cd /home/toronto02/statutes-rags
source .venv/bin/activate

# 2. Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# Ollamaが起動していない場合
source setup/restore_env.sh

# 3. FAISSインデックスの確認
ls -la data/faiss_index/vector/index.faiss
ls -la data/faiss_index/bm25/index.pkl

# 4. LangGraphの確認
python -c "import langgraph; print('LangGraph OK')"
```

すべての確認が成功したら、次のステップに進んでください。

---

## ステップ1: クイックテスト（モック使用）

実際のLLMやデータを使用せず、基本動作を確認します。

```bash
# examples/03_multi_agent_debateに移動
cd examples/03_multi_agent_debate

# クイックテストを実行
python tests/test_quick.py
```

**期待される出力:**

```
test_config_loading ... ok
test_debater_initialization ... ok
test_moderator_initialization ... ok
test_workflow_initialization ... ok
test_basic_workflow ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.XXXs

OK
```

すべてのテストがパスすれば、実装は正常です。

---

## ステップ2: 簡易評価テスト（3問）

実際のLLMとデータを使用して、3問で動作確認します。

```bash
# examples/03_multi_agent_debateで実行（ステップ1から続けて実行）
python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --limit 3 \
  --output results/test_quick.json
```

**実行時間:** 約5-10分（1問あたり2-3分）

**結果の確認:**

```bash
# 評価結果のサマリーを確認
cat results/test_quick.json | python -m json.tool | head -50

# 精度のみを確認
python -c "
import json
with open('results/test_quick.json') as f:
    data = json.load(f)
    summary = data['summary']
    print(f\"正答率: {summary['correct_answers']}/{summary['total_questions']} = {summary['accuracy']*100:.1f}%\")
    print(f\"平均ラウンド数: {summary['avg_rounds']:.2f}\")
    print(f\"平均合意スコア: {summary['avg_agreement_score']:.2f}\")
"
```

---

## ステップ3: 中規模評価（10問）

10問で評価を実行し、システムのパフォーマンスを確認します。

```bash
# examples/03_multi_agent_debateで実行
python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --limit 10 \
  --output results/eval_10.json
```

**実行時間:** 約20-30分

**結果の確認:**

```bash
cat results/eval_10.json | python -m json.tool | head -50
```

---

## ステップ4: 本番評価（全問題）

全問題で評価を実行します。実行時間が長いため、バックグラウンド実行を推奨します。

```bash
# examples/03_multi_agent_debateで実行

# バックグラウンド実行（推奨）
nohup python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --output results/full_evaluation.json \
  > results/evaluation.log 2>&1 &

# プロセスIDを確認
echo $!

# 進捗確認
tail -f results/evaluation.log

# または、フォアグラウンド実行
python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --output results/full_evaluation.json
```

**実行時間:** 約2-5時間（問題数による）

**結果の確認:**

```bash
# サマリーを確認
cat results/full_evaluation.json | python -m json.tool | head -80

# 詳細な統計を表示
python -c "
import json
with open('results/full_evaluation.json') as f:
    data = json.load(f)
    summary = data['summary']
    print('=== 評価結果サマリー ===')
    print(f\"総問題数: {summary['total_questions']}\")
    print(f\"正答数: {summary['correct_answers']}\")
    print(f\"正答率: {summary['accuracy']*100:.2f}%\")
    print(f\"平均ラウンド数: {summary['avg_rounds']:.2f}\")
    print(f\"平均合意スコア: {summary['avg_agreement_score']:.3f}\")
    print(f\"合意形成率: {summary['agreement_formation_rate']*100:.1f}%\")
    print(f\"平均処理時間: {summary['avg_time_per_question']:.1f}秒/問\")
    if summary.get('errors', 0) > 0:
        print(f\"エラー数: {summary['errors']}\")
"
```

---

## 評価結果の分析

評価完了後、結果を詳しく分析できます。

### 結果ファイルの構造

```json
{
  "timestamp": "2024-11-07T12:00:00",
  "config": {
    "max_debate_rounds": 3,
    "agreement_threshold": 0.8,
    "llm_model": "qwen3:8b"
  },
  "summary": {
    "total_questions": 140,
    "correct_answers": 105,
    "accuracy": 0.75,
    "avg_rounds": 2.3,
    "avg_agreement_score": 0.82,
    "agreement_formation_rate": 0.65,
    "avg_time_per_question": 45.2
  },
  "results": [...]
}
```

### 個別問題の確認

```bash
# 特定の問題を詳しく確認（問題0の場合）
python -c "
import json
with open('results/full_evaluation.json') as f:
    data = json.load(f)
    result = data['results'][0]
    print(f\"質問: {result['question'][:100]}...\")
    print(f\"正解: {result['correct_answer']}\")
    print(f\"予測: {result['predicted_answer']}\")
    print(f\"正誤: {'正解' if result['is_correct'] else '不正解'}\")
    print(f\"\\n=== 議論履歴 ===\")
    for h in result['debate_history']:
        print(f\"\\nラウンド {h['round']}:\")
        print(f\"  Debater A: {h['debater_a']['position']}\")
        print(f\"  Debater B: {h['debater_b']['position']}\")
"
```

---

## 高度な使用方法

### コマンドオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `--dataset` | 評価データセットのパス（必須） | - |
| `--output` | 結果の出力先 | `results/evaluation_{timestamp}.json` |
| `--limit` | 評価する最大問題数 | なし（全問題） |

### 環境変数によるカスタマイズ

Multi-Agent Debateの動作は環境変数でカスタマイズできます：

```bash
# 高速化設定
export DEBATE_MAX_ROUNDS=1
export DEBATE_RETRIEVAL_TOP_K=5
export LLM_MODEL=gpt-oss:7b
export LLM_TEMPERATURE=0.1

# 高精度設定
export DEBATE_MAX_ROUNDS=5
export DEBATE_AGREEMENT_THRESHOLD=0.9
export DEBATE_RETRIEVAL_TOP_K=15
export LLM_TEMPERATURE=0.3

# 評価を実行
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 10
```

主要な環境変数：

| 環境変数 | 説明 | デフォルト値 |
|---------|------|------------|
| `DEBATE_MAX_ROUNDS` | 最大議論ラウンド数 | 3 |
| `DEBATE_AGREEMENT_THRESHOLD` | 合意判定の閾値 | 0.8 |
| `DEBATE_RETRIEVAL_TOP_K` | 検索文書数 | 10 |
| `LLM_MODEL` | 使用するLLMモデル | qwen3:8b |
| `LLM_TEMPERATURE` | LLMの温度パラメータ | 0.1 |
| `LLM_TIMEOUT` | LLMタイムアウト（秒） | 60 |

### Pythonスクリプトからの使用

```python
import sys
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path.cwd().parent.parent))
sys.path.insert(0, str(Path.cwd()))

from workflow import DebateWorkflow
from config import MultiAgentDebateConfig

# カスタム設定
config = MultiAgentDebateConfig(
    max_debate_rounds=5,
    agreement_threshold=0.9,
    llm_temperature=0.0,
    retrieval_top_k=15
)

# ワークフローの初期化
workflow = DebateWorkflow(config)

# 質問を実行
result = workflow.query("会社法第26条について教えてください")

# 結果の表示
print("回答:", result["answer"])
print("ラウンド数:", result["metadata"]["rounds"])
print("合意スコア:", result["metadata"]["agreement_score"])
```

---

## トラブルシューティング

### エラー: "Dataset not found"

```bash
# データセットのパスを確認
ls -la ../../datasets/lawqa_jp/data/selection.json

# 絶対パスで指定
python evaluate.py \
    --dataset /home/toronto02/statutes-rags/datasets/lawqa_jp/data/selection.json \
    --limit 3
```

### エラー: "Index not found" または "Hybrid retrieval returned 0 documents"

```bash
# プロジェクトルートに戻ってインデックスを構築
cd ../../
make index
cd examples/03_multi_agent_debate

# または直接構築
python ../../scripts/build_index.py
```

### エラー: "Connection refused" (Ollama関連)

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# 起動していない場合
ollama serve

# モデルがダウンロードされているか確認
ollama list

# モデルをダウンロード
ollama pull qwen3:8b
```

### 処理が遅い場合

環境変数で設定を調整：

```bash
# 軽量設定
export DEBATE_MAX_ROUNDS=1
export DEBATE_RETRIEVAL_TOP_K=5
export LLM_MODEL=gpt-oss:7b

# 評価を実行
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --limit 3
```

### メモリ不足

```bash
# より軽量なモデルを使用
export LLM_MODEL=gpt-oss:7b
export DEBATE_RETRIEVAL_TOP_K=5
```

詳細なトラブルシューティングは[SETUP.md](./SETUP.md)を参照してください。

## ワンライナーコマンド集

### テスト評価（3問、標準設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 3 \
    --output results/quick_test.json
```

### 本番評価（全問題、標準設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/full_evaluation_$(date +%Y%m%d_%H%M%S).json
```

### デバッグモード（詳細ログ付き）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 1 \
    2>&1 | tee results/debug_$(date +%Y%m%d_%H%M%S).log
```

### 高速テスト（1問のみ、軽量設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 1 \
    --output results/single_test.json
```

## ユニットテストの実行

```bash
# クイックテスト（モックを使用、高速）
python tests/test_quick.py

# すべてのテストを実行
pytest tests/ -v

# 特定のテストのみ実行
pytest tests/test_multi_agent_debate.py::TestDebaterAgent -v

# カバレッジ付きで実行
pytest tests/ --cov=. --cov-report=html
```

## パフォーマンス目安

### システム構成
- **インデックス**: ベクトル 2,802,222件、BM25 10,000件
- **LLMモデル**: qwen3:8b (Ollama)
- **検索方式**: Hybrid (Vector + BM25, RRF融合)

### 処理時間
- **平均処理時間**: 約5.4分/問（324秒/問）
  - 検索: 約2秒
  - 議論: 約5分（モデルとラウンド数に依存）
- **1ラウンドあたり**: 約2-3分
- **3問の評価**: 約10-20分
- **全問題（140問）**: 約12-15時間

### 精度指標
- **平均議論ラウンド数**: 1.0ラウンド
- **平均合意スコア**: 0.95（非常に高い合意形成）
- **合意形成率**: 100%

## Multi-Agent Debateの特徴

### システム構成

1. **Debater Agent A (肯定側)**
   - 質問に対する回答を主張
   - 検索した法令文書を引用して論拠を示す

2. **Debater Agent B (批判的側)**
   - 代替案を提示
   - 肯定側の主張を批判的に検討

3. **Moderator Agent (調停者)**
   - 両者の主張を比較評価
   - 合意スコアを計算
   - 最終的な回答を決定

### 議論フロー

```
Question → Retrieval → Round 1 → Evaluation → Agreement?
                           ↓         ↓            ↓
                       Debater A  Moderator    Yes → Final Answer
                       Debater B     ↓          No  → Round 2 → ...
                                  Agreement
                                    Score
```

---

## 判例評価

マルチエージェント議論システムが、実際の判例と同じ結論を出せるかを評価します。

### 概要

判例データセット（`data_set/precedent/`）から判例を読み込み、事件の要旨を質問として与え、システムが生成した回答と判例の要旨を比較して評価します。

### ステップ1: クイックテスト（3判例）

少数の判例で動作確認します。

```bash
# examples/03_multi_agent_debateで実行
python evaluate_precedent.py \
  --precedent-dir data_set/precedent \
  --limit 3 \
  --output results/precedent_test_quick.json
```

**実行時間:** 約15-30分（1判例あたり5-10分）

**結果の確認:**

```bash
# 評価結果のサマリーを確認
cat results/precedent_test_quick.json | python -m json.tool | head -50

# 類似度のみを確認
python -c "
import json
with open('results/precedent_test_quick.json') as f:
    data = json.load(f)
    metrics = data['metrics']
    print(f\"類似率: {metrics['similar_precedents']}/{metrics['total_precedents']} = {metrics['similarity_rate']*100:.1f}%\")
    print(f\"平均類似度スコア: {metrics['avg_similarity_score']:.3f}\")
    print(f\"平均ラウンド数: {metrics['avg_rounds']:.2f}\")
"
```

### ステップ2: 中規模評価（10判例）

10判例で評価を実行します。

```bash
# examples/03_multi_agent_debateで実行
python evaluate_precedent.py \
  --precedent-dir data_set/precedent \
  --limit 10 \
  --random-seed 42 \
  --output results/precedent_eval_10.json
```

**実行時間:** 約1-2時間

**オプション:**
- `--limit`: 評価する判例数の上限
- `--random-seed`: ランダムシード（指定するとランダムサンプリング）
- `--precedent-dir`: 判例データディレクトリのパス（デフォルト: `data_set/precedent`）

### ステップ3: 本番評価（全判例または多数）

多数の判例で評価を実行します。実行時間が長いため、バックグラウンド実行を推奨します。

```bash
# examples/03_multi_agent_debateで実行

# バックグラウンド実行（推奨）
nohup python evaluate_precedent.py \
  --precedent-dir data_set/precedent \
  --limit 50 \
  --random-seed 42 \
  --output results/precedent_full_evaluation.json \
  > results/precedent_evaluation.log 2>&1 &

# プロセスIDを確認
echo $!

# 進捗確認
tail -f results/precedent_evaluation.log
```

**実行時間:** 判例数に応じて変動（1判例あたり5-10分）

### 評価メトリクス

判例評価では以下のメトリクスを計算します：

- **類似率**: システムの回答が判例要旨と類似していると判定された割合（類似度スコア ≥ 0.7）
- **平均類似度スコア**: 予測回答と判例要旨のコサイン類似度の平均
- **平均ラウンド数**: 議論に要した平均ラウンド数
- **平均合意スコア**: エージェント間の平均合意スコア
- **平均処理時間**: 1判例あたりの平均処理時間

### 結果ファイルの構造

```json
{
  "timestamp": "2024-11-07T12:00:00",
  "config": {
    "max_debate_rounds": 3,
    "agreement_threshold": 0.8,
    "llm_model": "qwen3:8b"
  },
  "metrics": {
    "total_precedents": 10,
    "similar_precedents": 7,
    "similarity_rate": 0.7,
    "avg_similarity_score": 0.75,
    "avg_rounds": 2.3,
    "avg_agreement_score": 0.82,
    "avg_time_per_precedent": 320.5
  },
  "results": [...]
}
```

### 個別判例の確認

```bash
# 特定の判例を詳しく確認（判例0の場合）
python -c "
import json
with open('results/precedent_eval_10.json') as f:
    data = json.load(f)
    result = data['results'][0]
    print(f\"事件名: {result['case_name']}\")
    print(f\"類似度スコア: {result['similarity_score']:.3f}\")
    print(f\"一致判定: {'一致' if result['is_similar'] else '不一致'}\")
    print(f\"\\n=== 質問 ===\")
    print(result['question'][:200])
    print(f\"\\n=== 判例要旨（正解） ===\")
    print(result['correct_answer'][:300])
    print(f\"\\n=== システムの回答 ===\")
    print(result['predicted_answer'][:300])
"
```

### トラブルシューティング

#### エラー: "Precedent directory not found"

```bash
# 判例ディレクトリのパスを確認
ls -la data_set/precedent/

# 絶対パスで指定
python evaluate_precedent.py \
    --precedent-dir /home/toronto02/statutes-rags/examples/03_multi_agent_debate/data_set/precedent \
    --limit 3
```

#### 処理が遅い場合

環境変数で設定を調整：

```bash
# 軽量設定
export DEBATE_MAX_ROUNDS=1
export DEBATE_RETRIEVAL_TOP_K=5
export LLM_MODEL=gpt-oss:7b

# 評価を実行
python evaluate_precedent.py --limit 3
```

#### メモリ不足

```bash
# より軽量なモデルを使用
export LLM_MODEL=gpt-oss:7b
export DEBATE_RETRIEVAL_TOP_K=5
```

### ワンライナーコマンド集

#### テスト評価（3判例、標準設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate_precedent.py \
    --precedent-dir data_set/precedent \
    --limit 3 \
    --output results/precedent_quick_test.json
```

#### 本番評価（50判例、ランダムサンプリング）

```bash
cd examples/03_multi_agent_debate && \
python evaluate_precedent.py \
    --precedent-dir data_set/precedent \
    --limit 50 \
    --random-seed 42 \
    --output results/precedent_evaluation_$(date +%Y%m%d_%H%M%S).json
```

#### デバッグモード（1判例のみ、詳細ログ付き）

```bash
cd examples/03_multi_agent_debate && \
python evaluate_precedent.py \
    --precedent-dir data_set/precedent \
    --limit 1 \
    2>&1 | tee results/precedent_debug_$(date +%Y%m%d_%H%M%S).log
```
