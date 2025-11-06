# Results Directory

本ディレクトリには、評価や分析の実行結果が格納されます。

## ディレクトリ構造

```
results/
├── evaluations/     # 評価結果ファイル（JSON形式）
└── README.md        # 本ファイル
```

## evaluations/

評価スクリプトの実行結果が自動的に格納されます。

デフォルトの出力ファイル名:
- `evaluation_results.json` - 標準の4択評価結果
- `mcp_benchmark_results.json` - MCPエージェント評価結果
- `ragas_evaluation.json` - RAGAS評価結果

## 注意事項

- 結果ファイルは `.gitignore` により Git 管理から除外されています
- 評価スクリプトは自動的にこのディレクトリを作成します
- 古い結果ファイルは手動で削除する必要があります
