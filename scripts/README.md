# Scripts Directory

本ディレクトリには、RAGシステムのデータ処理、評価、分析を行うスクリプトが格納されています。

---

## スクリプト分類

### データ処理

#### preprocess_egov_xml.py
e-Gov法令XMLファイルをJSONL形式に変換します。

```bash
python scripts/preprocess_egov_xml.py \
    --input-dir datasets/egov_laws/xml \
    --output data/egov_laws.jsonl
```

**実行時間:** 約5-10分  
**入力:** XMLファイル（10,435ファイル）  
**出力:** JSONL（約280万行）

#### build_index.py
JSONL形式のデータからFAISSベクトルインデックスまたはBM25インデックスを構築します。

```bash
# ベクトルインデックス構築
python scripts/build_index.py \
    --data data/egov_laws.jsonl \
    --output data/faiss_index \
    --index-type vector

# BM25インデックス構築（メモリ50-60GB必要）
# 恐らく動作しません
python scripts/build_index.py \
    --data data/egov_laws.jsonl \
    --output data/faiss_index \
    --index-type bm25
```

**実行時間:** 
- Vector: 30-60分（GPU使用時）
- BM25: 10-20分（CPU、メモリ大量消費）

**出力:**
- `data/faiss_index/vector/` - ベクトルインデックス
- `data/faiss_index/bm25/` - BM25インデックス

---

### 評価スクリプト

#### evaluate_multiple_choice.py
デジタル庁4択法令問題データセットでRAGシステムを評価します（メインスクリプト）。

```bash
python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output results/evaluations/evaluation_results.json \
    --samples 140 \
    --llm-model qwen3:14b \
    --top-k 10 \
    --use-reranker \
    --rerank-top-n 3
```

**主要オプション:**
- `--samples`: 評価サンプル数
- `--llm-model`: 使用LLMモデル
- `--top-k`: 検索文書数
- `--use-reranker`: Reranker有効化
- `--rerank-top-n`: Reranker後の文書数
- `--ensemble`: Ensemble推論回数
- `--use-cot`: Chain-of-Thought有効化
- `--no-few-shot`: Few-shot無効化
- `--no-rag`: RAG無効（LLMのみ）

**実行時間:** 40-100分（140サンプル）

#### evaluate.sh
シンプルな評価用シェルスクリプト。

```bash
# 10サンプル評価
./scripts/evaluate.sh 10

# 50サンプル評価
./scripts/evaluate.sh 50

# 全140サンプル評価
./scripts/evaluate.sh 140
```

**機能:**
- Vector-Onlyモードに自動設定
- GPU状態の表示
- 仮想環境の自動有効化

#### evaluate_ragas.py
RAGASメトリクス（Faithfulness, Answer Relevancy, Context Precision）で評価します。

```bash
python scripts/evaluate_ragas.py \
    --dataset datasets/lawqa_jp/data/selection.json \
    --limit 50 \
    --output results/evaluations/ragas_results.json
```

**注意:** RAGASは時間がかかるため、少数サンプルから開始推奨。

---

### 比較・分析

#### analyze_evaluation_results.py
評価結果を詳細に分析します。

```bash
python scripts/analyze_evaluation_results.py \
    results/evaluations/evaluation_results.json
```

**分析内容:**
- カテゴリ別精度
- エラータイプ分類
- 混同行列
- 誤答例の詳細表示

---

### ツール

#### query_cli.py
対話型CLIでRAGシステムに質問できます。

```bash
python scripts/query_cli.py
```

**使用例:**
```
> 会社法第26条について教えてください
[検索結果と回答が表示される]

> exit
```

**オプション:**
- `--top-k`: 検索文書数（デフォルト: 5）
- `--llm-model`: 使用LLMモデル

#### run_tests.sh
pytestを使用してユニットテスト・統合テストを実行します。

```bash
# ユニットテストのみ（高速）
./scripts/run_tests.sh unit

# 全テスト
./scripts/run_tests.sh all

# カバレッジ付き
./scripts/run_tests.sh coverage
```

**テストタイプ:**
- `unit` - ユニットテスト（デフォルト）
- `integration` - 統合テスト
- `all` - 全テスト
- `coverage` - カバレッジレポート
- `quick` - クイックテスト
- `slow` - 遅いテストのみ

---

## 実行順序

### 初回セットアップ

```bash
# 1. データ前処理
python scripts/preprocess_egov_xml.py

# 2. インデックス構築
python scripts/build_index.py --index-type vector

# 3. クイック評価
./scripts/evaluate.sh 10

# 4. 本格評価
./scripts/evaluate.sh 140

# 5. 結果分析
python scripts/analyze_evaluation_results.py \
    results/evaluations/evaluation_results_final.json
```

### 実験的評価

```bash
# Reranker付き評価
python scripts/evaluate_multiple_choice.py \
    --use-reranker --samples 140

# 異なるモデルで評価
python scripts/evaluate_multiple_choice.py \
    --llm-model gemma2:27b --samples 140
```

---

## 出力ファイル

すべての評価結果は `results/evaluations/` に保存されます:

```
results/evaluations/
├── evaluation_results.json          # 標準評価結果
├── evaluation_results_final.json    # evaluate.shの結果
└── ragas_results.json               # RAGAS評価結果
```

ログファイルは `logs/` に保存されます。

---

## ベストプラクティス

### ファイル命名

評価結果には意味のあるファイル名を使用:

```bash
# 実験名を含める
--output results/evaluations/exp_baseline_qwen3.json
--output results/evaluations/exp_reranker_qwen3.json

# 日付を含める
--output results/evaluations/eval_20251106_baseline.json
```

### メモリ管理

- BM25は使用しない（メモリ50-60GB必要）
- Vector検索のみを使用（メモリ4-6GB）
- Top-Kを10以下に抑える

### 評価戦略

1. 少数サンプル（10-20）でクイックテスト
2. 中規模（50）で設定確認
3. 全サンプル（140）で最終評価

---

最終更新: 2025年11月6日
