# Evaluation Results

本ディレクトリには、RAGシステムの評価結果が格納されます。

## ファイル命名規則

評価結果は以下の命名規則で保存されます:

### デフォルトファイル名

- `evaluation_results.json` - 標準の4択評価結果
- `mcp_benchmark_results.json` - MCPエージェント評価結果
- `ragas_evaluation.json` - RAGAS評価結果

### カスタムファイル名

`--output` オプションで任意のファイル名を指定可能:

```bash
python scripts/evaluate_multiple_choice.py \
    --output results/evaluations/my_experiment.json
```

## ファイル形式

評価結果は JSON 形式で保存されます。

### evaluation_results.json の構造

```json
{
  "config": {
    "rag_enabled": true,
    "retriever_type": "vector",
    "llm_model": "qwen3:14b",
    "top_k": 10,
    ...
  },
  "summary": {
    "accuracy": 0.6214,
    "correct_count": 87,
    "total_count": 140
  },
  "results": [
    {
      "question": "...",
      "predicted_answer": "a",
      "correct_answer": "b",
      "is_correct": false,
      ...
    }
  ]
}
```

## 結果の管理

### 結果の保存

実験ごとに意味のあるファイル名で保存することを推奨:

```bash
# 実験名を含める
--output results/evaluations/experiment_baseline_qwen3.json
--output results/evaluations/experiment_cot_gemini.json

# 日付を含める
--output results/evaluations/eval_20251106_baseline.json
```

### 結果の比較

複数の結果ファイルを比較する場合:

```bash
python scripts/compare_benchmark_results.py \
    results/evaluations/experiment1.json \
    results/evaluations/experiment2.json
```

## 注意事項

- すべての `.json` ファイルは Git 管理から除外されています
- 重要な実験結果は `docs/` にドキュメント化することを推奨
- 古い結果ファイルは定期的に整理してください
