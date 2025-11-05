# 4択法令データを用いたRAG評価ガイド

## 概要

このガイドでは、デジタル庁が提供する4択法令データセット（lawqa_jp）を使用してRAGシステムを評価する方法を説明します。

## データセット情報

- **提供元**: デジタル庁
- **内容**: 日本の法令に関する4択問題
- **サンプル数**: 140問
- **場所**: `datasets/lawqa_jp/data/selection.json`

各問題には以下の情報が含まれます：
- 問題文
- 4つの選択肢（a, b, c, d）
- 正解
- 法令のコンテキスト
- 参照法令のURL

## 評価スクリプト

### スクリプト: `scripts/evaluate_multiple_choice.py`

4択問題形式でRAGシステムを評価する専用スクリプトです。

### 主な機能

1. **RAG機能の評価**: 法令検索 + LLM回答の精度を測定
2. **ハイブリッド検索**: ベクトル検索とBM25検索の組み合わせ
3. **詳細な結果保存**: 各問題の正誤、検索された文書数、LLM応答を記録
4. **柔軟な設定**: サンプル数、LLMモデル、検索文書数（Top-K）をカスタマイズ可能

## 使用方法

### 基本的な使い方

```bash
cd /home/jovyan/work/statutes-rag
source .venv/bin/activate

# 3サンプルで動作確認
python3 scripts/evaluate_multiple_choice.py \
  --samples 3 \
  --top-k 5 \
  --llm-model "qwen3:8b" \
  --output evaluation_results_3.json
```

### オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--data` | 4択データセットのパス | `datasets/lawqa_jp/data/selection.json` |
| `--output` | 評価結果の出力パス | `evaluation_results.json` |
| `--samples` | 評価するサンプル数 | 全て（140問） |
| `--top-k` | 検索する文書数 | 5 |
| `--llm-model` | 使用するLLMモデル名 | `.env`から読み込み |
| `--no-rag` | RAGを使用せずLLMのみで評価 | False |

### 実行例

#### 例1: 小規模テスト（3問）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 3 \
  --llm-model "qwen3:8b"
```

**実行時間**: 約2分（qwen3:8b使用時）

#### 例2: 中規模評価（20問）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 20 \
  --top-k 10 \
  --llm-model "qwen3:8b" \
  --output evaluation_20samples.json
```

**実行時間**: 約13-20分（qwen3:8b使用時、1問あたり40秒程度）

#### 例3: 全データ評価（140問）

```bash
# バックグラウンド実行を推奨
nohup python3 scripts/evaluate_multiple_choice.py \
  --llm-model "qwen3:8b" \
  --output evaluation_full.json \
  > evaluation.log 2>&1 &
```

**実行時間**: 約90-120分（qwen3:8b使用時）

#### 例4: RAG無効化（LLMのみ）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 \
  --no-rag \
  --llm-model "qwen3:8b" \
  --output evaluation_llm_only.json
```

RAGの効果を測定するため、LLMのみの性能と比較できます。

### Top-Kパラメータの調整

検索文書数（Top-K）を変えて精度への影響を測定：

```bash
# Top-K=3
python3 scripts/evaluate_multiple_choice.py --samples 10 --top-k 3 --llm-model "qwen3:8b"

# Top-K=5（デフォルト）
python3 scripts/evaluate_multiple_choice.py --samples 10 --top-k 5 --llm-model "qwen3:8b"

# Top-K=10
python3 scripts/evaluate_multiple_choice.py --samples 10 --top-k 10 --llm-model "qwen3:8b"
```

## 評価結果の見方

### コンソール出力例

```
==================================================
EVALUATION RESULTS
==================================================
Total Samples: 3
Correct: 2
Incorrect: 1
Accuracy: 66.67%
==================================================

INCORRECT CASES:
==================================================

[Case 1]
Question: 金融商品取引法施行令第2条の12に定める取得勧誘...
Correct: b
Predicted: c
```

### JSON出力形式

評価結果は以下の形式でJSONファイルに保存されます：

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
      "question": "問題文...",
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

## 初期評価結果（3サンプル）

### 設定
- **LLMモデル**: qwen3:8b
- **Retriever**: Hybrid（Vector + BM25）
- **Top-K**: 5
- **Temperature**: 0.0（決定的な回答）

### 結果
- **正解数**: 2/3
- **精度**: 66.67%
- **平均実行時間**: 約40秒/問

### 観察
1. **成功ケース**: 法令条文の詳細な解釈を要する問題で正解
2. **成功ケース**: 複数の条文参照が必要な複雑な問題でも正解
3. **失敗ケース**: 細かい条件の違いを見極める問題で誤答（選択肢bを選ぶべきところcを選択）

## パフォーマンス比較

### LLMモデルによる速度の違い

| モデル | サイズ | 推論速度（概算） | 精度期待値 |
|--------|--------|-----------------|-----------|
| qwen3:8b | 13GB | 40秒/問 | 高 |
| qwen2.5:7b | 4.4GB | 10-15秒/問 | 中 |
| qwen2.5:3b | 1.9GB | 5-10秒/問 | 低〜中 |

※推論速度はGPU性能に依存します（NVIDIA A100 MIG使用時の目安）

## トラブルシューティング

### 問題1: トークナイザーの警告が表示される

**修正済み**: SudachiPyがデフォルトでインストールされるようになりました。利用できない場合は自動的にJanome、n-gram、または簡易トークナイザーにフォールバックします。詳細は`docs/supplemental/tokenizer-guide.md`を参照してください。

**影響**: BM25検索の精度がわずかに低下する可能性がありますが、ハイブリッド検索（Vector + BM25）により実用上は問題ありません。

### 問題2: Ollamaモデルが見つからない

```
Ollama call failed with status code 404. Maybe your model is not found
```

**対処法**: モデルが存在するか確認：

```bash
cd /home/jovyan/work/statutes-rags/setup
./bin/ollama list
```

qwen3:8bがリストにない場合は、Ollamaサーバーを再起動：

```bash
pkill ollama
cd setup && ./bin/ollama serve > ollama.log 2>&1 &
```

### 問題3: 評価が遅すぎる

**対処法**:
1. サンプル数を減らす（`--samples 10`）
2. より小さいモデルを使用（事前に `./setup/bin/ollama pull qwen2.5:7b` を実行し、`--llm-model "qwen2.5:7b"` を指定）
3. バックグラウンド実行（`nohup ... &`）

## 評価指標の分析

### 正解率（Accuracy）

基本的な指標：正解数 / 総問題数

### 推奨される追加分析

1. **問題の難易度別の精度**
   - 参照法令数が多い問題
   - 外部法令参照を含む問題
   - 選択肢が類似している問題

2. **検索品質の分析**
   - 正解時の平均検索文書数
   - 不正解時の検索文書数との比較

3. **法令分野別の精度**
   - 金融商品取引法
   - 会社法
   - その他の法令

## 次のステップ

### 精度向上のための施策

1. **Top-Kの最適化**: 3, 5, 10で比較実験
2. **Rerankerの導入**: Cross-encoderで検索結果を再スコアリング
3. **プロンプトの改善**: より具体的な指示を追加
4. **チャンクサイズの調整**: より大きいコンテキストを提供

### 評価の拡張

1. **複数モデルの比較**: qwen3:8b vs qwen2.5:7b vs 他のモデル
2. **RAG vs Non-RAG比較**: RAGの効果を定量的に測定
3. **他のデータセットでの評価**: 将来的に法律試験問題データセットなどを追加予定

## まとめ

この評価スクリプトを使用することで、RAGシステムの性能を客観的に測定し、改善点を特定できます。初期結果（66.67%）は良好であり、Top-Kやプロンプトの調整により更なる改善が期待できます。

---

最終更新: 2025-10-30
