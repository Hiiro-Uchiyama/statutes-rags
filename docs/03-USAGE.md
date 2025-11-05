# statutes RAG 使用方法ガイド

このドキュメントでは、statutes-ragプロジェクトの各スクリプトとツールの詳細な使用方法を説明します。

## 目次

1. [データ前処理](#データ前処理)
2. [インデックス構築](#インデックス構築)
3. [対話型クエリ](#対話型クエリ)
4. [評価実験](#評価実験)
5. [テスト実行](#テスト実行)
6. [Makefile活用](#makefile活用)
7. [高度な使用方法](#高度な使用方法)

## データ前処理

### `preprocess_egov_xml.py`

e-Gov法令XMLファイルをRAGシステムで使用可能なJSONL形式に変換します。

#### 基本的な使用方法

```bash
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input-dir` | XMLファイルのディレクトリ | 必須 |
| `--output-file` | 出力JSONLファイルパス | 必須 |
| `--limit` | 処理するファイル数の上限 | なし（全て処理） |

#### 使用例

##### 1. 全XMLファイルを処理

```bash
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl
```

**処理時間:** 約5-10分（10,435ファイル）

##### 2. テスト用（最初の100ファイルのみ）

```bash
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/test_egov.jsonl \
  --limit 100
```

**処理時間:** 約30秒

##### 3. Makefileを使用

```bash
# 全ファイル処理
make preprocess

# 最初の100ファイルのみ
make preprocess PREPROCESS_LIMIT=100
```

#### 出力形式

JSONLファイル（1行1JSON）：

```json
{"law_title": "会社法", "law_num": "平成十七年法律第八十六号", "article": "26", "article_caption": "", "article_title": "第26条", "paragraph": "1", "item": null, "text": "株式会社を設立するには、その本店の所在地において設立の登記をしなければならない。"}
{"law_title": "会社法", "law_num": "平成十七年法律第八十六号", "article": "26", "article_caption": "", "article_title": "第26条", "paragraph": "2", "item": null, "text": "株式会社は、前項の登記をすることによって成立する。"}
```

#### 確認方法

```bash
# ファイルサイズ確認
ls -lh data/egov_laws.jsonl

# 行数（ドキュメント数）確認
wc -l data/egov_laws.jsonl

# 最初の1行を整形表示
head -1 data/egov_laws.jsonl | python3 -m json.tool

# 特定の法令を検索
grep "会社法" data/egov_laws.jsonl | head -5
```

## インデックス構築

### `build_index.py`

JSONLファイルから検索用インデックスを構築します。

#### 基本的な使用方法

```bash
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--data-path` | JSONLファイルパス | `.env`から読み込み |
| `--index-path` | インデックス保存先 | `.env`から読み込み |
| `--retriever-type` | `vector`, `bm25`, `hybrid` | `.env`から読み込み |
| `--limit` | 処理するドキュメント数上限 | なし（全て処理） |
| `--batch-size` | バッチサイズ（ドキュメント数） | 10000 |

#### 使用例

##### 1. ハイブリッドインデックス構築（推奨）

```bash
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid
```

**処理時間:** 約20-40分（文書数による）

**生成されるファイル:**
- `data/faiss_index/vector/index.faiss` - FAISSベクトルインデックス
- `data/faiss_index/vector/index.pkl` - メタデータ
- `data/faiss_index/bm25/index.pkl` - BM25インデックス

##### 2. ベクトルインデックスのみ

```bash
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type vector
```

**メリット:** セマンティック検索、類義語に強い
**デメリット:** キーワード完全一致に弱い

##### 3. BM25インデックスのみ

```bash
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type bm25
```

**メリット:** キーワード検索、高速
**デメリット:** セマンティック理解なし

##### 4. テスト用（少数ドキュメント）

```bash
python3 scripts/build_index.py \
  --data-path data/test_egov.jsonl \
  --index-path data/faiss_index_test \
  --retriever-type hybrid \
  --limit 1000
```

**処理時間:** 約2-3分

##### 5. Makefileを使用

```bash
# 全ドキュメント
make index

# 最初の1000ドキュメントのみ
make index INDEX_LIMIT=1000
```

#### 確認方法

```bash
# インデックスファイルの存在確認
ls -lh data/faiss_index/vector/
ls -lh data/faiss_index/bm25/

# ベクトルインデックスのサイズ確認
du -sh data/faiss_index/vector/

# BM25インデックスのサイズ確認
du -sh data/faiss_index/bm25/
```

## 対話型クエリ

### `query_cli.py`

RAGシステムを対話的に使用します。

#### 基本的な使用方法

##### 1. 対話モード

```bash
python3 scripts/query_cli.py --interactive
```

**使用例:**

```
Question: 会社法第26条について教えてください
Searching and generating answer...

================================================================================
ANSWER:
================================================================================
会社法第26条は、株式会社の設立登記に関する規定です。第1項では、株式会社を設立
するには、その本店の所在地において設立の登記をしなければならないと定めています。
第2項では、株式会社は、前項の登記をすることによって成立すると規定しています。

================================================================================
CITATIONS:
================================================================================
[1] 会社法 第26条 第1項
    株式会社を設立するには、その本店の所在地において設立の登記をしなければならない。

[2] 会社法 第26条 第2項
    株式会社は、前項の登記をすることによって成立する。

Question: exit
```

##### 2. 単発クエリ

```bash
python3 scripts/query_cli.py "会社法第26条について教えてください"
```

##### 3. 結果をJSONファイルに保存

```bash
python3 scripts/query_cli.py "会社法第26条について教えてください" \
  --output result.json

# 結果確認
cat result.json | python3 -m json.tool
```

#### 出力形式

```json
{
  "answer": "会社法第26条は...",
  "citations": [
    {
      "law_title": "会社法",
      "article": "26",
      "paragraph": "1",
      "item": null
    }
  ],
  "contexts": [
    {
      "law_title": "会社法",
      "article": "26",
      "paragraph": "1",
      "text": "株式会社を設立するには...",
      "score": 0.92
    }
  ]
}
```

#### よくある質問例

```bash
# 会社法関連
python3 scripts/query_cli.py "株式会社の設立手続きについて教えてください"
python3 scripts/query_cli.py "取締役の責任について教えてください"

# 労働法関連
python3 scripts/query_cli.py "労働基準法における時間外労働の上限は？"
python3 scripts/query_cli.py "有給休暇の付与日数について教えてください"

# 民法関連
python3 scripts/query_cli.py "契約の成立要件について教えてください"
python3 scripts/query_cli.py "不法行為の要件について教えてください"
```

## 評価実験

### `evaluate_multiple_choice.py`

デジタル庁の4択法令データセットでRAGシステムを評価します。

#### 基本的な使用方法

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 3 \
  --llm-model "qwen3:8b"
```

**注:** 軽量モデルを使いたい場合は、`./setup/bin/ollama pull qwen2.5:7b` を実行した上で `--llm-model "qwen2.5:7b"` を指定してください。

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--data` | 4択データセットパス | `datasets/lawqa_jp/data/selection.json` |
| `--output` | 結果の出力パス | `evaluation_results.json` |
| `--samples` | 評価するサンプル数 | 全て（140問） |
| `--top-k` | 検索する文書数 | 5 |
| `--llm-model` | 使用するLLMモデル名 | `.env`から読み込み |
| `--no-rag` | RAG無効（LLMのみ） | False |

#### 使用例

##### 1. 小規模テスト（3問で動作確認）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 3 \
  --llm-model "qwen3:8b"
```

**実行時間:** 約3-5分（ハードウェア性能に依存）  
**用途:** 動作確認、デバッグ

##### 2. 中規模評価（20問）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 20 \
  --llm-model "qwen3:8b" \
  --output evaluation_results_20.json
```

**実行時間:** 約20-30分（ハードウェア性能に依存）
**用途:** 開発中の性能確認

##### 3. 全データ評価（140問）

```bash
# バックグラウンド実行推奨
nohup python3 scripts/evaluate_multiple_choice.py \
  --llm-model "qwen3:8b" \
  --output evaluation_results_full.json \
  > evaluation.log 2>&1 &

# 進捗確認
tail -f evaluation.log

# プロセス確認
ps aux | grep evaluate_multiple_choice
```

**実行時間:** 約90-120分
**用途:** 最終評価、論文・レポート

##### 4. Top-Kパラメータの比較

```bash
# Top-K=3
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --top-k 3 --output eval_k3.json

# Top-K=5
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --top-k 5 --output eval_k5.json

# Top-K=10
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --top-k 10 --output eval_k10.json
```

**用途:** 最適なTop-Kを発見

##### 5. RAG有無の比較

```bash
# RAG有効
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --llm-model "qwen3:8b" \
  --output eval_rag.json

# RAG無効（LLMのみ）
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --llm-model "qwen3:8b" \
  --no-rag --output eval_no_rag.json
```

**用途:** RAGの効果測定

##### 6. 異なるLLMモデルの比較

```bash
# qwen3:8b（約13GB、高精度）
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --llm-model "qwen3:8b" \
  --output eval_20b.json

# qwen2.5:7b（約4.4GB、軽量モデル）
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --llm-model "qwen2.5:7b" \
  --output eval_7b.json

# qwen2.5:3b（約1.9GB、さらに軽量）
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 --llm-model "qwen2.5:3b" \
  --output eval_3b.json
```

**用途:** モデルサイズと精度のトレードオフ分析

#### 結果の確認

```bash
# 結果サマリー表示
cat evaluation_results.json | python3 -m json.tool | head -30

# 精度のみ表示
cat evaluation_results.json | python3 -c \
  "import json, sys; d=json.load(sys.stdin); print(f\"Accuracy: {d['summary']['accuracy']*100:.2f}%\")"

# 不正解ケースのみ表示
cat evaluation_results.json | python3 -c \
  "import json, sys; d=json.load(sys.stdin); \
  [print(f\"Q: {r['question'][:50]}...\nCorrect: {r['correct_answer']}, Predicted: {r['predicted_answer']}\n\") \
  for r in d['results'] if not r['is_correct']]"
```

#### 出力形式

```json
{
  "config": {
    "rag_enabled": true,
    "retriever_type": "hybrid",
    "llm_model": "qwen3:8b",
    "top_k": 5,
    "total_samples": 3
  },
  "summary": {
    "accuracy": 0.6667,
    "correct_count": 2,
    "total_count": 3
  },
  "results": [
    {
      "question": "金融商品取引法施行令第2条の12に定める取得勧誘...",
      "choices": "a ... b ... c ... d ...",
      "correct_answer": "c",
      "predicted_answer": "c",
      "is_correct": true,
      "response": "c",
      "retrieved_docs_count": 5,
      "file_name": "金商法_第2章_選択式...",
      "references": ["https://laws.e-gov.go.jp/..."]
    }
  ]
}
```

#### 重要な注意事項

**温度パラメータについて:**

4択評価では、スクリプト内で`temperature=0.0`が設定されています（デフォルトの`0.1`ではなく）。これは以下の理由によります：

- 4択問題では決定的な回答が求められる
- 温度を0にすることで、LLMの出力が一貫性を持つ
- 再現性が向上し、評価結果の比較が容易になる

この設定は`scripts/evaluate_multiple_choice.py`の202行目で行われています。

### `evaluate_ragas.py`

RAGASフレームワークを使用した高度な評価（現在は参考実装）。

```bash
python3 scripts/evaluate_ragas.py \
  --dataset datasets/lawqa_jp/data/selection.json \
  --output data/evaluation_report.json \
  --limit 10
```

**評価指標:**
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

## テスト実行

### `run_tests.sh`

pytestを使用したテストスイートの実行。

#### 基本的な使用方法

```bash
# ユニットテストのみ（高速）
./scripts/run_tests.sh unit

# 全テスト
./scripts/run_tests.sh all

# インテグレーションテスト
./scripts/run_tests.sh integration

# カバレッジ付き
./scripts/run_tests.sh coverage

# 高速テストのみ
./scripts/run_tests.sh quick
```

#### Makefileを使用

```bash
make test          # ユニットテストのみ
make test-all      # 全テスト
make test-coverage # カバレッジ付き
```

#### 個別テストファイルの実行

```bash
# 特定のテストファイル
pytest tests/test_config.py -v

# 特定のテスト関数
pytest tests/test_config.py::test_load_config -v

# マーカーでフィルタ
pytest tests/ -m "not slow" -v
```

#### カバレッジレポート

```bash
./scripts/run_tests.sh coverage

# HTMLレポート生成
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Makefile活用

### 利用可能なターゲット

```bash
# ヘルプ表示
make help

# セットアップ
make setup-uv      # uv環境セットアップ
make install       # 依存パッケージインストール

# データ処理
make preprocess    # XML→JSONL前処理
make index         # インデックス構築
make all           # preprocess + index

# 実行
make qa            # 対話型CLI起動
make query Q="質問文"  # 単発クエリ

# 評価
make eval                   # RAGAS評価
make eval-multiple-choice   # 4択法令データ評価

# テスト
make test          # ユニットテスト
make test-all      # 全テスト
make test-coverage # カバレッジ付き

# クリーンアップ
make clean         # 生成ファイル削除

# 開発
make dev-setup     # 開発環境セットアップ
make lint          # リント実行
make format        # コード整形
```

### 実験用ターゲット

```bash
# 小規模実験
make experiment-small
# 10ファイル前処理 → 100ドキュメントインデックス → 5サンプル評価

# 中規模実験
make experiment-medium
# 100ファイル前処理 → 1000ドキュメントインデックス → 20サンプル評価
```

### 変数のカスタマイズ

```bash
# 前処理のファイル数制限
make preprocess PREPROCESS_LIMIT=100

# インデックスのドキュメント数制限
make index INDEX_LIMIT=1000

# 評価のサンプル数制限
make eval EVAL_LIMIT=50
```

## 高度な使用方法

### 1. カスタムプロンプトの使用

`app/retrieval/rag_pipeline.py`を編集：

```python
self.prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""あなたは法律専門家です。
    
【参照法令】
{context}

【質問】
{question}

【回答指示】
- 該当条文を明示
- 分かりやすく説明
- 判例があれば言及

回答:"""
)
```

### 2. 環境変数で設定変更

```bash
# LLMモデルを一時的に変更
LLM_MODEL=qwen3:8b python3 scripts/query_cli.py --interactive

# Top-Kを変更
RETRIEVER_TOP_K=20 python3 scripts/query_cli.py --interactive

# MMRを無効化
USE_MMR=false python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index
```

### 3. Rerankerの有効化

`.env`を編集：

```bash
RERANKER_ENABLED=true
RERANKER_TOP_N=5
```

再度実行：

```bash
python3 scripts/query_cli.py --interactive
```

**効果:** 検索精度が向上（ただし処理時間が増加）

### 4. バッチ処理

複数の質問を一括処理：

```bash
# questions.txtを作成
cat > questions.txt << EOF
会社法第26条について教えてください
労働基準法における時間外労働の上限は？
契約の成立要件について教えてください
EOF

# バッチ処理
while read question; do
  echo "Question: $question"
  python3 scripts/query_cli.py "$question"
  echo "---"
done < questions.txt > batch_results.txt
```

### 5. カスタム評価スクリプト

```python
# scripts/my_evaluation.py
import sys
sys.path.insert(0, ".")

from app.core.rag_config import load_config
from app.retrieval.rag_pipeline import RAGPipeline
# ... カスタム評価ロジック
```

### 6. 検索結果のデバッグ

```python
from app.retrieval.bm25_retriever import BM25Retriever

retriever = BM25Retriever(index_path="data/faiss_index/bm25")
retriever.load_index()

documents = retriever.retrieve("会社法第26条", top_k=5)
for i, doc in enumerate(documents, 1):
    print(f"[{i}] Score: {doc.score:.4f}")
    print(f"    Law: {doc.metadata.get('law_title')}")
    print(f"    Article: {doc.metadata.get('article')}")
    print(f"    Text: {doc.page_content[:100]}...")
```

## トラブルシューティング

### スクリプト実行時のエラー

#### 1. `ModuleNotFoundError: No module named 'app'`

```bash
# 解決方法: パッケージをインストール
pip install -e .
```

#### 2. `FileNotFoundError: data/egov_laws.jsonl`

```bash
# 解決方法: 前処理を実行
make preprocess
```

#### 3. `Ollama call failed with status code 404`

```bash
# 解決方法: Ollamaサーバーとモデルを確認
curl http://localhost:11434/api/tags
cd setup && ./bin/ollama list
```

#### 4. トークナイザーが利用できない

```bash
# 解決方法: トークナイザーをインストール
pip install sudachipy sudachidict-core janome

# または既存環境を更新
pip install -e . --upgrade
```

#### 5. `OutOfMemoryError`

```bash
# 解決方法: バッチサイズを削減
python3 scripts/build_index.py \
  --batch-size 50 \
  --data-path data/egov_laws.jsonl
```

詳細なアーキテクチャは`05-ARCHITECTURE.md`、セットアップ手順は`02-SETUP.md`を参照してください。

---

最終更新: 2024-11-04
