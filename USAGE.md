# 法令RAGシステム 使用ガイド

本ドキュメントでは、statutes-ragsプロジェクトのセットアップから評価まで、**上から順に実行するだけで完了する**手順を記載しています。

最終更新: 2025年11月7日  
**重要:** RAG実装の改善を適用しました。詳細は[RAG_IMPROVEMENTS_APPLIED.md](docs/RAG_IMPROVEMENTS_APPLIED.md)を参照してください。

---

## 目次

1. [前提条件](#1-前提条件)
2. [セットアップ](#2-セットアップ)
3. [データ準備](#3-データ準備)
4. [インデックス構築](#4-インデックス構築)
5. [評価実行](#5-評価実行)
6. [CLIツール](#6-cliツール)
7. [テスト実行](#7-テスト実行)
8. [結果分析](#8-結果分析)
9. [スクリプト一覧](#9-スクリプト一覧)

---

## 1. 前提条件

### 必須環境

```yaml
OS: Ubuntu Linux (20.04以降推奨)
Python: 3.11以上
GPU: NVIDIA GPU（CUDA対応、VRAMは8GB以上推奨）
メモリ: 16GB以上（32GB推奨）
ディスク: 50GB以上の空き容量
```

### 必要なソフトウェア

- Python 3.11+
- NVIDIA GPU Driver
- CUDA Toolkit (11.8以降)
- uv (Pythonパッケージマネージャー)
- Ollama (ローカルLLM実行環境)

---

## 2. セットアップ

### 2.1 リポジトリのクローン

```bash
git clone https://github.com/your-org/statutes-rags.git
cd statutes-rags
```

### 2.2 環境変数ファイルの作成

```bash
# .env.exampleをコピー
cp .env.example .env

# .envファイルを編集（必要に応じて）
nano .env
```

主要な設定項目:

```bash
# 埋め込みモデル
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# LLM設定
LLM_MODEL=qwen3:14b
LLM_TEMPERATURE=0.1

# Retriever設定
RETRIEVER_TYPE=vector  # vector, bm25, hybrid
RETRIEVER_TOP_K=10

# Reranker設定
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER_TOP_N=3

# Ollama
OLLAMA_HOST=http://localhost:11434
```

### 2.3 uv環境のセットアップ

```bash
# uvと仮想環境のセットアップ
./setup/setup_uv_env.sh
```

このスクリプトは以下を実行します:
- uvのインストール（未インストールの場合）
- 仮想環境の作成（.venv）
- 依存パッケージのインストール

実行後、仮想環境を有効化:

```bash
source .venv/bin/activate
```

### 2.4 Ollamaのセットアップ

```bash
# Ollamaのインストールとモデルのダウンロード
./setup/setup_ollama.sh
```

このスクリプトは以下を実行します:
- Ollamaのインストール（未インストールの場合）
- Ollamaサービスの起動
- qwen3:14bモデルのダウンロード

### 2.5 環境復元（コンテナ再起動後など）

```bash
# 環境変数とサービスの復元
source setup/restore_env.sh
```

---

## 3. データ準備

### 3.1 法令XMLデータの配置

e-Gov法令XMLファイルを配置:

```bash
# データディレクトリの作成
mkdir -p datasets/egov_laws/xml

# XMLファイルを配置（例：共有ディレクトリからコピー）
cp -r /path/to/egov_laws_xml/* datasets/egov_laws/xml/
```

または、e-Gov APIから直接ダウンロード（スクリプトは別途実装が必要）。

### 3.2 評価データセットの配置

デジタル庁公開の4択法令問題データセット:

```bash
# データセットディレクトリの作成
mkdir -p datasets/lawqa_jp/data

# selection.jsonを配置
cp /path/to/selection.json datasets/lawqa_jp/data/
```

### 3.3 XMLからJSONLへの前処理

法令XMLをJSONL形式に変換:

```bash
python scripts/preprocess_egov_xml.py \
    --input-dir datasets/egov_laws/xml \
    --output data/egov_laws.jsonl
```

**実行時間:** 約5-10分  
**出力:** `data/egov_laws.jsonl`（約280万行）

**オプション:**
- `--input-dir`: 入力XMLディレクトリ（デフォルト: `datasets/egov_laws/xml`）
- `--output`: 出力JSONLファイル（デフォルト: `data/egov_laws.jsonl`）

---

## 4. インデックス構築

### 4.1 ベクトルインデックスの構築

FAISSベクトルインデックスを構築:

```bash
python scripts/build_index.py \
    --data data/egov_laws.jsonl \
    --output data/faiss_index \
    --index-type vector
```

**実行時間:** 約30-60分（GPUあり）  
**メモリ使用:** 20-30GB（ピーク時）  
**出力:**
- `data/faiss_index/vector/index.faiss`
- `data/faiss_index/vector/index.pkl`

### 4.2 BM25インデックスの構築（オプション）

```bash
# 警告: メモリ50-60GB必要
python scripts/build_index.py \
    --data data/egov_laws.jsonl \
    --output data/faiss_index \
    --index-type bm25
```

**注意:** BM25インデックスは大量のメモリを消費するため、通常環境では実行困難です。ベクトル検索のみの使用を推奨。

---

## 5. 評価実行

### 5.1 基本評価（ベクトル検索）

最もシンプルな評価方法:

```bash
# 10サンプルでクイックテスト（約2-3分）
./scripts/evaluate.sh 10

# 50サンプルで評価（約10-15分）
./scripts/evaluate.sh 50

# 全140サンプルで評価（約40-50分）
./scripts/evaluate.sh 140
```

**出力:** `results/evaluations/evaluation_results_final.json`

**スクリプトの動作:**
- RETRIEVER_TYPE=vectorに自動設定
- qwen3:14bモデルを使用
- Few-shotプロンプト有効
- Top-K=10（デフォルト）

### 5.2 詳細設定での評価

より細かい制御が必要な場合:

```bash
python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output results/evaluations/my_evaluation.json \
    --samples 140 \
    --llm-model qwen3:14b \
    --top-k 10 \
    --use-reranker \
    --rerank-top-n 3
```

**主要オプション:**
- `--samples`: 評価するサンプル数（デフォルト: 全て）
- `--llm-model`: 使用LLMモデル（デフォルト: .envの設定）
- `--top-k`: 検索文書数（デフォルト: 3）
- `--use-reranker`: Reranker有効化
- `--rerank-top-n`: Reranker後の文書数（デフォルト: 3）
- `--ensemble`: Ensemble推論回数（1=無効、3推奨）
- `--use-cot`: Chain-of-Thought推論を有効化
- `--no-few-shot`: Few-shotプロンプトを無効化
- `--no-rag`: RAG無効、LLMのみで評価

### 5.3 Reranker付き評価

検索精度を向上させる場合:

```bash
python scripts/evaluate_multiple_choice.py \
    --samples 140 \
    --top-k 10 \
    --use-reranker \
    --rerank-top-n 3 \
    --output results/evaluations/eval_with_reranker.json
```

### 5.4 Ensemble評価

複数回推論して多数決を取る:

```bash
python scripts/evaluate_multiple_choice.py \
    --samples 140 \
    --ensemble 3 \
    --output results/evaluations/eval_ensemble.json
```

**注意:** 実行時間は3倍になります。

### 5.5 異なるLLMモデルでの評価

別のOllamaモデルを使用:

```bash
# gemma2:27bで評価（事前にollama pull gemma2:27b必要）
python scripts/evaluate_multiple_choice.py \
    --samples 140 \
    --llm-model gemma2:27b \
    --output results/evaluations/eval_gemma2.json

# llama3.1:70bで評価
python scripts/evaluate_multiple_choice.py \
    --samples 140 \
    --llm-model llama3.1:70b \
    --output results/evaluations/eval_llama31.json
```

### 5.6 RAGAS評価（RAG品質評価）

RAGASメトリクスでの評価:

```bash
python scripts/evaluate_ragas.py \
    --dataset datasets/lawqa_jp/data/selection.json \
    --limit 50 \
    --output results/evaluations/ragas_results.json
```

**注意:** RAGASは時間がかかるため、少数サンプルから開始推奨。

---

## 6. CLIツール

### 6.1 対話型クエリツール

RAGシステムに直接質問:

```bash
python scripts/query_cli.py
```

**使用例:**

```
> 会社法第26条について教えてください

[検索された文書]
1. 会社法 第26条 ...
2. 会社法施行規則 ...

[LLMの回答]
会社法第26条は...

> exit  # 終了
```

**オプション:**
- `--top-k`: 検索文書数（デフォルト: 5）
- `--llm-model`: 使用LLMモデル（デフォルト: .envの設定）

---

## 7. テスト実行

### 7.1 ユニットテストの実行

```bash
# 高速なユニットテストのみ
./scripts/run_tests.sh unit

# 全てのテスト
./scripts/run_tests.sh all

# カバレッジ付きテスト
./scripts/run_tests.sh coverage
```

**テストタイプ:**
- `unit` - ユニットテストのみ（高速、デフォルト）
- `integration` - 統合テスト
- `all` - 全テスト
- `coverage` - カバレッジレポート付き
- `quick` - クイックテスト（遅いテスト除外）
- `slow` - 遅いテストのみ

### 7.2 カバレッジレポートの確認

```bash
# カバレッジテスト実行
./scripts/run_tests.sh coverage

# HTMLレポートを開く
firefox htmlcov/index.html  # または他のブラウザ
```

---

## 8. 結果分析

### 8.1 評価結果の確認

JSON結果を確認:

```bash
# 結果のサマリーを表示
cat results/evaluations/evaluation_results_final.json | python3 -m json.tool | head -30

# 精度のみ抽出
cat results/evaluations/evaluation_results_final.json | jq '.summary.accuracy'
```

### 8.2 詳細分析

評価結果を詳細に分析:

```bash
python scripts/analyze_evaluation_results.py \
    results/evaluations/evaluation_results_final.json
```

**出力内容:**
- 全体サマリー（精度、正解数、誤答数）
- カテゴリ別分析
- エラータイプ分類
- 混同行列
- 誤答例の詳細

---

## 9. スクリプト一覧

### データ処理

| スクリプト | 説明 | 主要オプション |
|-----------|------|---------------|
| `preprocess_egov_xml.py` | XMLからJSONLへ変換 | `--input-dir`, `--output` |
| `build_index.py` | インデックス構築 | `--data`, `--output`, `--index-type` |

### 評価

| スクリプト | 説明 | 主要オプション |
|-----------|------|---------------|
| `evaluate_multiple_choice.py` | 4択評価（メイン） | `--samples`, `--llm-model`, `--top-k`, `--use-reranker` |
| `evaluate.sh` | シンプル評価スクリプト | サンプル数（引数1） |
| `evaluate_ragas.py` | RAGAS評価 | `--dataset`, `--limit` |

### 比較・分析

| スクリプト | 説明 | 主要オプション |
|-----------|------|---------------|
| `analyze_evaluation_results.py` | 評価結果の詳細分析 | 結果ファイルパス |

### ツール

| スクリプト | 説明 | 主要オプション |
|-----------|------|---------------|
| `query_cli.py` | 対話型クエリツール | `--top-k`, `--llm-model` |
| `run_tests.sh` | テスト実行 | テストタイプ（引数1） |

---

## クイックスタートチートシート

### 初回セットアップ

```bash
# 1. 環境構築
./setup/setup_uv_env.sh
source .venv/bin/activate
./setup/setup_ollama.sh

# 2. データ配置
cp /path/to/xml/* datasets/egov_laws/xml/
cp /path/to/selection.json datasets/lawqa_jp/data/

# 3. 前処理とインデックス構築
python scripts/preprocess_egov_xml.py
python scripts/build_index.py --index-type vector

# 4. クイック評価
./scripts/evaluate.sh 10
```

### 日常的な使用

```bash
# 環境復元
source setup/restore_env.sh

# 評価実行
./scripts/evaluate.sh 50

# 結果確認
cat results/evaluations/evaluation_results_final.json | jq '.summary'
```

### 実験的な評価

```bash
# Reranker付き
python scripts/evaluate_multiple_choice.py --use-reranker --samples 140

# Ensemble
python scripts/evaluate_multiple_choice.py --ensemble 3 --samples 140

# 異なるモデル
python scripts/evaluate_multiple_choice.py --llm-model gemma2:27b --samples 140
```

---

## トラブルシューティング

### Ollamaモデルが見つからない

```bash
# モデルの確認
ollama list

# モデルのダウンロード
ollama pull qwen3:14b
```

### インデックスが見つからない

```bash
# インデックスの確認
ls -la data/faiss_index/vector/

# 再構築
python scripts/build_index.py --index-type vector
```

### メモリ不足エラー

```bash
# BM25を無効化（.env）
RETRIEVER_TYPE=vector

# Top-Kを減らす
python scripts/evaluate_multiple_choice.py --top-k 5
```

### GPU使用率が低い

```bash
# GPU状態確認
nvidia-smi

# CUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 関連ドキュメント

詳細な情報は以下のドキュメントを参照:

- [README.md](./README.md) - プロジェクト概要
- [docs/02-SETUP.md](./docs/02-SETUP.md) - 詳細セットアップ
- [docs/03-USAGE.md](./docs/03-USAGE.md) - 使用方法詳細
- [docs/05-ARCHITECTURE.md](./docs/05-ARCHITECTURE.md) - アーキテクチャ
- [docs/EVALUATION_RESULTS_COMPREHENSIVE.md](./docs/appendix/EVALUATION_RESULTS_COMPREHENSIVE.md) - 評価結果総括
- [docs/TECHNICAL_INSIGHTS_SUMMARY.md](./docs/appendix/TECHNICAL_INSIGHTS_SUMMARY.md) - 技術的知見

---

最終更新: 2025年11月6日
