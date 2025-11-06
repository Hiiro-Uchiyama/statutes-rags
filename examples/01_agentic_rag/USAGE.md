# Agentic RAG - 使用ガイド

本ドキュメントでは、Agentic RAGシステムのセットアップから評価まで、**上から順に実行するだけで完了する**手順を記載しています。

最終更新: 2025年11月7日

---

## 目次

1. [前提条件](#1-前提条件)
2. [プロジェクトルートのセットアップ](#2-プロジェクトルートのセットアップ)
3. [Agentic RAG依存関係のインストール](#3-agentic-rag依存関係のインストール)
4. [動作確認テスト](#4-動作確認テスト)
5. [評価実行](#5-評価実行)
6. [結果分析](#6-結果分析)
7. [高度な使用方法](#7-高度な使用方法)
8. [トラブルシューティング](#8-トラブルシューティング)

---

## 1. 前提条件

### 必須環境

- プロジェクトルートのセットアップが完了していること
- 法令データとインデックスが構築されていること
- Ollamaがインストールされ、LLMモデルがダウンロード済みであること

### セットアップ未完了の場合

プロジェクトルートの[USAGE.md](../../USAGE.md)を参照して、まず基本セットアップを完了してください：

```bash
cd /path/to/statutes-rags

# 1. 環境構築
./setup/setup_uv_env.sh
source .venv/bin/activate

# 2. Ollamaセットアップ
./setup/setup_ollama.sh

# 3. データ準備（XMLファイルがある場合）
python scripts/preprocess_egov_xml.py

# 4. インデックス構築
python scripts/build_index.py --index-type vector
```

### セットアップ確認

以下のコマンドで必要なファイルが存在することを確認：

```bash
# プロジェクトルートから実行
ls -lh data/egov_laws.jsonl                          # 法令データ (1.8GB)
ls -lh data/faiss_index/vector/index.faiss          # ベクトルインデックス
ls -lh datasets/lawqa_jp/data/selection.json        # 評価データセット
ollama list | grep qwen3                             # Ollamaモデル確認
```

すべて存在すれば次のステップへ進めます。

---

## 2. プロジェクトルートのセットアップ

### 仮想環境の有効化

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 仮想環境を有効化
source .venv/bin/activate
```

### 環境変数の確認

`.env`ファイルが存在し、以下の設定が含まれていることを確認：

```bash
# .envファイルの確認
cat .env | grep -E "LLM_MODEL|OLLAMA_HOST|VECTOR_STORE_PATH"
```

期待される出力例：
```
LLM_MODEL=qwen3:8b
OLLAMA_HOST=http://localhost:11434
VECTOR_STORE_PATH=data/faiss_index
```

---

## 3. Agentic RAG依存関係のインストール

### LangGraphとLangChain-Ollamaのインストール

Agentic RAGは通常のRAGに加えて、以下の依存関係が必要です：

```bash
# プロジェクトルートから実行
uv pip install "langgraph>=0.2.0,<0.3.0" \
               "langchain>=0.3.0,<0.4.0" \
               "langchain-ollama>=0.2.0" \
               "langsmith>=0.2.0"
```

**実行時間:** 約30秒～1分

### インストール確認

```bash
python -c "import langgraph; import langchain_ollama; print('✓ Dependencies installed')"
```

`✓ Dependencies installed` と表示されればOKです。

---

## 4. 動作確認テスト

### 4.1 簡易動作確認（必須）

Agentic RAGパイプラインが正常に動作するか確認します。

```bash
# examples/01_agentic_rag に移動
cd examples/01_agentic_rag

# 簡易テストを実行（約30秒～1分）
python tests/test_simple.py
```

**期待される出力:**
```
============================================================
Agentic RAG - Simple Test (Minimal Config)
============================================================

1. Loading configuration...
   LLM Model: qwen3:8b
   ...
   ✓ Pipeline initialized successfully

3. Running test query...
   ...
   ✓ Query executed successfully

============================================================
Test passed! ✓
============================================================
```

### 4.2 詳細動作確認（推奨）

より詳細なテストを実行します。

```bash
# 2つのクエリで詳細テスト（約1～2分）
python tests/test_quick.py
```

**期待される出力:**
```
============================================================
Agentic RAG - Quick Test
============================================================
...
All tests passed! ✓
```

### トラブルシューティング（テストが失敗する場合）

#### エラー: "Index not found"
```bash
# プロジェクトルートに戻ってインデックスを確認
cd ../../
ls -la data/faiss_index/vector/

# インデックスがない場合は構築
python scripts/build_index.py --index-type vector
cd examples/01_agentic_rag
```

#### エラー: "Ollama connection failed"
```bash
# Ollamaの状態を確認
curl http://localhost:11434/api/tags

# 応答がない場合はOllamaを起動
cd ../../
./setup/setup_ollama.sh
cd examples/01_agentic_rag
```

---

## 5. 評価実行

テストが成功したら、評価データセットでの性能評価を実行します。

### 5.1 クイック評価（3問、約1～2分）

まず少数の問題で動作確認：

```bash
# examples/01_agentic_rag ディレクトリから実行
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 3 \
    --output results/eval_quick_3q.json
```

**期待される出力:**
```
Question 1: ...
  Predicted: b, Correct: b, Match: True
Question 2: ...
  Predicted: a, Correct: c, Match: False
...
==================================================
Evaluation Summary
==================================================
Accuracy:        66.67%
Correct:         2 / 3
Avg Iterations:  0.00
Avg LLM Calls:   2.00
==================================================
```

### 5.2 小規模評価（10問、約5～8分）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 10 \
    --output results/eval_10q.json
```

### 5.3 中規模評価（50問、約25～40分）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 50 \
    --output results/eval_50q.json
```

### 5.4 完全評価（全140問、約60～90分）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/eval_full_140q.json
```

### 評価設定のカスタマイズ

#### 最小構成（高速、精度は低い）

```bash
# Reasoning/Validation無効化
AGENTIC_ENABLE_REASONING=false \
AGENTIC_ENABLE_VALIDATION=false \
AGENTIC_MAX_ITERATIONS=1 \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 10 \
    --output results/eval_minimal_10q.json
```

#### 完全版（時間かかる、高精度）

```bash
# 全エージェント有効、反復3回
AGENTIC_ENABLE_REASONING=true \
AGENTIC_ENABLE_VALIDATION=true \
AGENTIC_MAX_ITERATIONS=3 \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 20 \
    --output results/eval_full_agentic_20q.json
```

---

## 6. 結果分析

### 6.1 結果サマリーの確認

評価実行後、結果を確認します：

```bash
# JSONの整形表示
cat results/eval_10q.json | python -m json.tool | head -50

# または jq を使用（インストール済みの場合）
cat results/eval_10q.json | jq '.accuracy, .correct_count, .total, .avg_iterations'
```

**出力例:**
```json
0.70
7
10
0.1
```

### 6.2 詳細分析

プロジェクトルートの分析スクリプトを使用：

```bash
# プロジェクトルートに戻る
cd ../../

# 詳細分析を実行
python scripts/analyze_evaluation_results.py \
    examples/01_agentic_rag/results/eval_10q.json
```

**出力内容:**
- 全体サマリー（精度、正解数、誤答数）
- 複雑度別の正答率
- エージェント使用統計
- 誤答例の詳細

### 6.3 ベースラインとの比較

既存RAG（Traditional RAG）と比較評価：

```bash
# プロジェクトルートから実行

# 1. ベースライン評価（既存RAG）
python scripts/evaluate_multiple_choice.py \
    --dataset datasets/lawqa_jp/data/selection.json \
    --samples 20 \
    --output baseline_20q.json

# 2. Agentic RAG評価（最小構成）
cd examples/01_agentic_rag
AGENTIC_ENABLE_REASONING=false \
AGENTIC_ENABLE_VALIDATION=false \
python evaluate.py \
    --max-questions 20 \
    --output results/agentic_minimal_20q.json

# 3. Agentic RAG評価（完全版）
AGENTIC_ENABLE_REASONING=true \
AGENTIC_ENABLE_VALIDATION=true \
python evaluate.py \
    --max-questions 20 \
    --output results/agentic_full_20q.json

# 4. 結果比較（Pythonワンライナー）
cd ../../
python -c "
import json

with open('baseline_20q.json') as f:
    baseline = json.load(f)
with open('examples/01_agentic_rag/results/agentic_minimal_20q.json') as f:
    minimal = json.load(f)
with open('examples/01_agentic_rag/results/agentic_full_20q.json') as f:
    full = json.load(f)

print('=== 精度比較 ===')
print(f'Baseline (Traditional RAG): {baseline[\"summary\"][\"accuracy\"]*100:.2f}%')
print(f'Agentic RAG (Minimal):      {minimal[\"accuracy\"]*100:.2f}%')
print(f'Agentic RAG (Full):         {full[\"accuracy\"]*100:.2f}%')
print()
print(f'Improvement (Minimal): {(minimal[\"accuracy\"]-baseline[\"summary\"][\"accuracy\"])*100:+.2f} points')
print(f'Improvement (Full):    {(full[\"accuracy\"]-baseline[\"summary\"][\"accuracy\"])*100:+.2f} points')
"
```

---

## 7. 高度な使用方法

### 7.1 環境変数による詳細設定

```bash
# 全設定のカスタマイズ例
export AGENTIC_MAX_ITERATIONS=5              # 最大反復回数
export AGENTIC_CONFIDENCE_THRESHOLD=0.85     # 信頼度閾値
export AGENTIC_ENABLE_REASONING=true         # 推論エージェント
export AGENTIC_ENABLE_VALIDATION=true        # 検証エージェント
export AGENTIC_RETRIEVAL_TOP_K=15            # 検索文書数
export LLM_MODEL=qwen3:14b                   # LLMモデル
export LLM_TEMPERATURE=0.05                  # 温度パラメータ
export LLM_TIMEOUT=120                       # タイムアウト（秒）

python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 10 \
    --output results/eval_custom.json
```

### 7.2 異なるLLMモデルでの評価

```bash
# qwen3:14b（より大規模、高精度）
LLM_MODEL=qwen3:14b python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 10 \
    --output results/eval_qwen3_14b.json

# gpt-oss:20b（さらに大規模）
LLM_MODEL=gpt-oss:20b python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 10 \
    --output results/eval_gpt_oss_20b.json
```

### 7.3 バッチ評価スクリプト

複数の設定で一括評価：

```bash
# バッチ評価スクリプトの作成
cat > batch_eval.sh << 'EOF'
#!/bin/bash
set -e

cd /path/to/statutes-rags/examples/01_agentic_rag

echo "=== Batch Evaluation Start ==="

# 1. Minimal config
echo "[1/3] Evaluating: Minimal config"
AGENTIC_ENABLE_REASONING=false \
AGENTIC_ENABLE_VALIDATION=false \
python evaluate.py --max-questions 50 \
    --output results/batch_minimal.json

# 2. Standard config
echo "[2/3] Evaluating: Standard config"
python evaluate.py --max-questions 50 \
    --output results/batch_standard.json

# 3. Full agentic config
echo "[3/3] Evaluating: Full agentic config"
AGENTIC_ENABLE_REASONING=true \
AGENTIC_ENABLE_VALIDATION=true \
AGENTIC_MAX_ITERATIONS=3 \
python evaluate.py --max-questions 50 \
    --output results/batch_full.json

echo "=== Batch Evaluation Complete ==="
EOF

chmod +x batch_eval.sh
./batch_eval.sh
```

---

## 8. トラブルシューティング

### エラー: "Dataset not found"

```bash
# データセットの存在確認
ls -la ../../datasets/lawqa_jp/data/selection.json

# データセットがない場合
# プロジェクトルートのUSAGE.mdを参照してデータを準備
```

### エラー: "Index not found"

```bash
# インデックスの存在確認
ls -la ../../data/faiss_index/vector/

# インデックスがない場合は構築
cd ../../
python scripts/build_index.py --index-type vector
cd examples/01_agentic_rag
```

### エラー: "Ollama connection failed"

```bash
# Ollamaサービスの状態確認
curl http://localhost:11434/api/tags

# Ollamaが起動していない場合
cd ../../
./setup/setup_ollama.sh

# モデルのダウンロード確認
ollama list
```

### エラー: "Graph recursion limit"

LangGraphの再帰制限エラーが発生する場合（通常は発生しません）：

```bash
# 反復回数を減らす
AGENTIC_MAX_ITERATIONS=1 python evaluate.py ...
```

### 処理が遅い場合

```bash
# 高速化オプション
AGENTIC_ENABLE_REASONING=false \
AGENTIC_ENABLE_VALIDATION=false \
AGENTIC_MAX_ITERATIONS=1 \
AGENTIC_RETRIEVAL_TOP_K=5 \
LLM_MODEL=qwen3:8b \
python evaluate.py --max-questions 10
```

### メモリ不足エラー

```bash
# 検索文書数を減らす
export AGENTIC_RETRIEVAL_TOP_K=5

# Rerankerを無効化（プロジェクトルートの.env）
export RERANKER_ENABLED=false
```

### LLMタイムアウトエラー

```bash
# タイムアウトを延長
export LLM_TIMEOUT=180

# または、より軽量なモデルを使用
export LLM_MODEL=qwen3:8b
```

---

## コマンドリファレンス

### よく使うコマンド

```bash
# 仮想環境の有効化
source ../../.venv/bin/activate

# 簡易テスト
python tests/test_simple.py

# クイック評価（3問）
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 3

# 標準評価（10問）
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10

# 結果確認
cat results/eval_10q.json | jq '.accuracy, .correct_count, .total'

# 詳細分析
cd ../../ && python scripts/analyze_evaluation_results.py examples/01_agentic_rag/results/eval_10q.json
```

### 環境変数クイックリファレンス

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `AGENTIC_MAX_ITERATIONS` | 3 | 最大反復回数 |
| `AGENTIC_CONFIDENCE_THRESHOLD` | 0.8 | 信頼度閾値 |
| `AGENTIC_ENABLE_REASONING` | true | Reasoning Agent有効化 |
| `AGENTIC_ENABLE_VALIDATION` | true | Validation Agent有効化 |
| `AGENTIC_RETRIEVAL_TOP_K` | 10 | 検索文書数 |
| `LLM_MODEL` | qwen3:8b | 使用LLMモデル |
| `LLM_TEMPERATURE` | 0.1 | 温度パラメータ |
| `LLM_TIMEOUT` | 60 | タイムアウト（秒） |

---

## 次のステップ

1. **精度向上の実験**
   - パラメータチューニング
   - 異なるLLMモデルの比較
   - プロンプトの最適化

2. **スケールアップ**
   - 完全評価（140問）の実行
   - 複数設定での網羅的評価
   - 統計的有意性の検証

3. **カスタマイズ**
   - エージェントの実装変更
   - 新しい検索戦略の追加
   - 独自の評価指標の導入

---

## 関連ドキュメント

- [README.md](README.md) - Agentic RAGの詳細仕様
- [../../USAGE.md](../../USAGE.md) - プロジェクト全体の使用ガイド
- [../../docs/](../../docs/) - 技術ドキュメント

---

最終更新: 2025年11月7日  
**重要:** 上から順に実行すれば、セットアップから評価まで完了します。
