# Legal RAG データセット

全4種のデータセット

## データセット一覧

| データセット | サイズ | 用途 | ステータス |
|------------|--------|------|-----------|
| e-Gov法令一括 | 264MB | コーパス | ダウンロード済み |
| lawqa_jp | 4.9MB | 評価 | ダウンロード済み |
| 民法Instruction | 128KB | 評価 | ダウンロード済み |
| 刑法試験問題 | 472KB | 評価 | ダウンロード済み |

## 1. e-Gov法令一括（XML）
- 提供元: 総務省 e-Gov
- 内容: 日本の全法令
- 用途: メインコーパス
- ファイル: egov_laws/egov_laws_all.zip

## 2. lawqa_jp
- 提供元: デジタル庁
- 内容: 4択法令QA
- 用途: RAG評価
- ファイル: lawqa_jp/data/selection.json

## 3. 民法Instruction
- 提供元: APTO-001 (HF)
- 内容: 103件の民法Q&A
- 用途: Few-shot/Fine-tuning
- ファイル: civil_law_instructions/data/train-00000-of-00001.parquet

## 4. 刑法試験問題
- 提供元: nguyenthanhasia (HF)
- 内容: 5年分の司法試験問題
- 用途: 高難度評価
- ファイル: criminal_law_exams/all_criminal_law_exams.json

詳細はdocs/DATASETS.mdを参照
