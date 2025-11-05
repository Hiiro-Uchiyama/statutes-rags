# Agentic RAG - 使用方法ガイド

## クイックスタート

### 1. セットアップ

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 依存関係をインストール（初回のみ）
uv pip install -r examples/01_agentic_rag/requirements.txt

# または pip を使用
pip install -r examples/01_agentic_rag/requirements.txt
```

### 2. インデックスの確認

```bash
# インデックスが存在することを確認
ls -la data/faiss_index/

# インデックスがない場合は構築
make index
# または
python scripts/build_index.py
```

### 3. 評価の実行

#### 基本的な使用方法

```bash
# examples/01_agentic_rag ディレクトリに移動
cd examples/01_agentic_rag

# テスト実行（最初の3問のみ）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --max-questions 3 \
    --output results/test_eval.json
```

#### 全問題の評価

```bash
# 全問題を評価（時間がかかります）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/full_eval.json
```

#### 結果の確認

```bash
# 評価結果のサマリーを確認
cat results/test_eval.json | python -m json.tool | head -30

# または、評価実行時に自動で表示されます
```

## コマンドオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `--dataset` | 評価データセットのパス（必須） | - |
| `--output` | 結果の出力先 | `results/evaluation_{timestamp}.json` |
| `--max-questions` | 評価する最大問題数 | なし（全問題） |

## 環境変数による設定

### LLMモデルの変更

```bash
# より軽量なモデルを使用（高速）
export LLM_MODEL=gpt-oss:7b
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10

# より大規模なモデルを使用（高精度）
export LLM_MODEL=qwen3:8b
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10
```

### エージェント設定の変更

```bash
# Reasoning Agentを無効化（高速化）
export AGENTIC_ENABLE_REASONING=false
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10

# Validation Agentを無効化（高速化）
export AGENTIC_ENABLE_VALIDATION=false
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10

# 反復回数を減らす（高速化）
export AGENTIC_MAX_ITERATIONS=1
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10
```

### タイムアウトの設定

```bash
# LLMのタイムアウトを延長（大規模モデル使用時）
export LLM_TIMEOUT=120
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10
```

## 評価結果の形式

評価結果は以下の形式で出力されます：

```json
{
  "accuracy": 0.75,
  "correct_count": 3,
  "total": 4,
  "avg_iterations": 0.5,
  "avg_llm_calls": 4.25,
  "results": [
    {
      "question_index": 0,
      "question": "金融商品取引法第24条に...",
      "choices": ["a ...", "b ...", "c ...", "d ..."],
      "correct_answer": "b",
      "predicted_answer": "b",
      "is_correct": true,
      "raw_answer": "回答の全文...",
      "citations": [...],
      "metadata": {
        "complexity": "simple",
        "iterations": 0,
        "confidence": 0.85,
        "agents_used": ["manager", "retrieval", "validation"]
      }
    }
  ],
  "timestamp": "2025-11-06T01:30:00.123456"
}
```

## トラブルシューティング

### エラー: "Dataset not found"

```bash
# データセットのパスを確認
ls -la ../../datasets/lawqa_jp/data/selection.json

# データセットがない場合はダウンロード
# （プロジェクトのセットアップガイドを参照）
```

### エラー: "Index not found"

```bash
# プロジェクトルートに戻ってインデックスを構築
cd ../../
make index
cd examples/01_agentic_rag
```

### 処理が遅い場合

```bash
# 複数の高速化オプションを組み合わせ
export LLM_MODEL=gpt-oss:7b
export AGENTIC_ENABLE_REASONING=false
export AGENTIC_MAX_ITERATIONS=1
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 5
```

### LLMのタイムアウト

```bash
# タイムアウトを延長
export LLM_TIMEOUT=180
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 10
```

## ワンライナーコマンド集

### テスト評価（3問、高速設定）

```bash
cd examples/01_agentic_rag && \
AGENTIC_ENABLE_REASONING=false AGENTIC_MAX_ITERATIONS=1 \
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 3 --output results/quick_test.json
```

### 本番評価（全問題、標準設定）

```bash
cd examples/01_agentic_rag && \
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --output results/full_evaluation_$(date +%Y%m%d_%H%M%S).json
```

### デバッグモード（詳細ログ付き）

```bash
cd examples/01_agentic_rag && \
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 1 2>&1 | tee debug.log
```

## ユニットテストの実行

```bash
# すべてのテストを実行
pytest tests/ -v

# 特定のテストのみ実行
pytest tests/test_agentic_rag.py::TestManagerAgent -v

# カバレッジ付きで実行
pytest tests/ --cov=. --cov-report=html
```

## 次のステップ

- 詳細な実装情報: [README.md](README.md)
- 動作確認結果: README.md の最後のセクション
- プロジェクト全体のドキュメント: [../../docs/](../../docs/)
