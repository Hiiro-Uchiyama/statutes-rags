# 補足資料

本ディレクトリは、statutes-ragシステムの開発・運用に関する補足情報を提供するドキュメントを格納しています。

## 重要なドキュメント

### システム制約と設定

- **[memory_issue_analysis.md](memory_issue_analysis.md)** - **BM25が使用できない理由**の詳細分析（必読）
  - 280万件のデータセットで50-60GBのメモリが必要
  - Vector-Onlyモードの推奨理由
  - 技術的制約の説明

- **[final_evaluation_report.md](final_evaluation_report.md)** - RAG評価の最終報告
  - Vector-Onlyモードでの評価結果（精度50%）
  - GPU使用状況の確認結果
  - システム性能の評価

- **[investigation_report.md](investigation_report.md)** - 停止問題の調査報告
  - メモリ問題の発見過程
  - 原因特定の詳細

### 実装ガイド

- **[evaluation-guide.md](evaluation-guide.md)** - 評価実験の詳細ガイド
  - デジタル庁提供の4択法令データセット（lawqa_jp）の使用方法
  - 評価スクリプトの使用方法
  - パラメータ設定とカスタマイズ
  - 結果の解釈方法

- **[tokenizer-guide.md](tokenizer-guide.md)** - トークナイザーの詳細ガイド
  - 各トークナイザー（SudachiPy、Janome、n-gram等）の特徴と比較
  - インストール方法（管理者権限不要）
  - 使用方法とベストプラクティス
  - パフォーマンスベンチマーク
  - トラブルシューティング

- **[code-fix-summary.md](code-fix-summary.md)** - コード修正のサマリー
  - 実施した修正の一覧
  - 修正理由の説明

## クイックスタート

**初めての方へ**: まず [memory_issue_analysis.md](memory_issue_analysis.md) を読んで、なぜVector-Onlyモード（`RETRIEVER_TYPE=vector`）を使用するのかを理解してください。その後、[final_evaluation_report.md](final_evaluation_report.md) で期待される性能を確認してください。

## 使用方法

各ドキュメントは、特定のトピックに関する詳細な情報や実施例を提供しています。プロジェクトのセットアップや実行中に疑問が生じた場合は、関連するドキュメントを参照してください。
