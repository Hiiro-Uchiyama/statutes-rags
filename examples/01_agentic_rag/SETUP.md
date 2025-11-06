# Agentic RAG - セットアップガイド

## クイックスタート（コピー＆ペースト用）

以下のコマンドを順番に実行するだけで、セットアップから評価まで完了します。

### ステップ1: 前提条件の確認

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 必要なファイルの確認
ls -lh data/egov_laws.jsonl
ls -lh data/faiss_index/vector/index.faiss
ls -lh datasets/lawqa_jp/data/selection.json
ollama list | grep qwen3
```

すべて存在すれば次へ。存在しない場合は[プロジェクトルートのUSAGE.md](../../USAGE.md)を参照。

### ステップ2: 仮想環境の有効化

```bash
source .venv/bin/activate
```

### ステップ3: 依存関係のインストール

```bash
uv pip install "langgraph>=0.2.0,<0.3.0" \
               "langchain>=0.3.0,<0.4.0" \
               "langchain-ollama>=0.2.0" \
               "langsmith>=0.2.0"
```

### ステップ4: 動作確認

```bash
cd examples/01_agentic_rag
python tests/test_simple.py
```

### ステップ5: クイック評価（3問）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 3 \
    --output results/eval_quick_3q.json
```

### ステップ6: 結果確認

```bash
cat results/eval_quick_3q.json | jq '.accuracy, .correct_count, .total'
```

---

## 完了！

これで基本的な使用準備が整いました。

### 次のステップ

#### 小規模評価（10問、約5～8分）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 10 \
    --output results/eval_10q.json
```

#### 標準評価（50問、約25～40分）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 50 \
    --output results/eval_50q.json
```

#### 完全評価（全140問、約60～90分）

```bash
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/eval_full_140q.json
```

---

## 詳細情報

- **詳細な使用方法**: [USAGE.md](USAGE.md)
- **技術仕様**: [README.md](README.md)
- **プロジェクト全体**: [../../USAGE.md](../../USAGE.md)

---

最終更新: 2025年11月7日
