# datasets/ ディレクトリ

このディレクトリには、Legal RAGシステムで使用する元データセットが格納されます。

**重要:** このディレクトリのファイルは手動でダウンロード・配置する必要があります。

## ディレクトリ構造

```
datasets/
├── egov_laws/                     # e-Gov法令XMLデータ（要ダウンロード）
│   ├── README.md                  # データセット詳細
│   ├── egov_laws_all.zip          # 圧縮アーカイブ（配置する場合）
│   └── *.xml                      # 展開後のXMLファイル（約10,435件）
└── lawqa_jp/                      # 法令QAデータセット（4択、要ダウンロード）
    ├── README.md                  # データセット詳細
    ├── LICENSE.md                 # ライセンス情報
    └── data/                      # データファイル一式
        ├── law_list.json
        ├── selection.json
        ├── selection.csv
        ├── selection_randomized.json
        └── selection_with_reference_randomized.json

# 追加のデータセットを利用する場合は、任意で別ディレクトリを作成して配置してください
```

## データセット一覧

| データセット | サイズ | ファイル数/問題数 | 用途 | 提供元 |
|------------|--------|-----------------|------|--------|
| e-Gov法令一括 | 2GB (展開後) | 10,435ファイル | メインコーパス | 総務省 e-Gov |
| lawqa_jp | 4.9MB | 5ファイル | RAG評価（4択） | デジタル庁 |

> 任意の追加データセット（司法試験QAなど）はリポジトリには含まれていません。必要に応じて各自で取得し、適切なディレクトリを作成して配置してください。

## 1. e-Gov法令一括（XML）

### 概要

日本の全法令XMLファイルを含むデータセット。Legal RAGシステムのメインコーパスとして使用します。

**詳細:**
- **提供元**: 総務省 e-Gov法令検索
- **ファイル数**: 約10,435件のXMLファイル
- **圧縮サイズ**: 264MB（egov_laws_all.zip）
- **展開後サイズ**: 約2GB
- **用途**: メインコーパス（検索対象となる法令データ）
- **ライセンス**: 政府標準利用規約

### ダウンロード方法

```bash
# ディレクトリに移動
cd datasets/egov_laws

# e-Gov法令検索からダウンロード
# URL: https://elaws.e-gov.go.jp/download/
# 「法令XMLデータ一括ダウンロード」からzipファイルを取得

# ダウンロードしたファイルを配置
mv ~/Downloads/egov_laws_all.zip .
```

### データの展開

```bash
# zipファイルを展開
cd datasets/egov_laws
unzip egov_laws_all.zip

# または Python を使用
python3 -m zipfile -e egov_laws_all.zip .

# 展開されたファイルを確認
find . -name "*.xml" | wc -l
# 期待値: 約10,435ファイル
```

### XML形式の例

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Law Era="Reiwa" Lang="ja" LawType="Act" Num="89" Year="05">
  <LawNum>平成十七年法律第八十六号</LawNum>
  <LawBody>
    <LawTitle>会社法</LawTitle>
    <MainProvision>
      <Article Num="1">
        <ArticleCaption>（趣旨）</ArticleCaption>
        <ArticleTitle>第一条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence>
            <Sentence>この法律は、...</Sentence>
          </ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
  </LawBody>
</Law>
```

### 次のステップ

XMLファイルの展開が完了したら、前処理スクリプトでJSONL形式に変換します：

```bash
# プロジェクトルートに移動
cd ../..

# XMLからJSONLへ変換（約5-10分）
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl

# または Makefile を使用
make preprocess
```

詳細は [egov_laws/README.md](egov_laws/README.md) を参照してください。

## 2. lawqa_jp

### 概要

法令分野に関する4択形式の多肢選択問題を収録したデータセット。RAGシステムの評価に使用します。

**詳細:**
- **提供元**: デジタル庁
- **問題数**: 複数の法令QA問題（4択形式）
- **サイズ**: 約4.9MB
- **用途**: RAG評価、法的知識の理解度測定
- **ライセンス**: 公共データ利用規約（第1.0版）

### ダウンロード方法

```bash
# ディレクトリに移動
cd datasets/lawqa_jp

# GitHubから取得（例）
# または手動でダウンロードしてdata/ディレクトリに配置
```

### データ形式

#### selection.json（メインファイル）

```json
{
  "ファイル名": "金商法_第2章_選択式_関連法令_問題番号57",
  "回答オーダーマップ番号": "1",
  "コンテキスト": "## 金融商品取引法\n### 第5条\n...",
  "指示": "以下の問題文に対する回答を，選択肢a，b，c，dの中から１つ選んでください．",
  "問題文": "金融商品取引法第5条第6項により，...",
  "選択肢": "a ～\nb ～\nc ～\nd ～",
  "output": "c",
  "references": [
    "https://laws.e-gov.go.jp/law/323AC0000000025"
  ]
}
```

#### ファイル一覧

| ファイル名 | 説明 |
|-----------|------|
| `law_list.json` | 設問で参照されている法令の一覧 |
| `selection.json` | QA本体（メイン評価データ） |
| `selection.csv` | CSV形式バージョン |
| `selection_randomized.json` | 選択肢順をランダム化したデータ |
| `selection_with_reference_randomized.json` | 外部法令参照を含む問題のみ抽出 |

### 評価スクリプトでの使用

```bash
# 多肢選択式QA評価を実行
python3 scripts/evaluate_multiple_choice.py \
  --data-path datasets/lawqa_jp/data/selection.json \
  --index-path data/faiss_index \
  --limit 20

# または Makefile を使用
make eval-mc
```

詳細は [lawqa_jp/README.md](lawqa_jp/README.md) を参照してください。

## 3. japanese-bar-exam-qa（任意、デフォルトでは未配置）

### 概要

日本の司法試験（2015-2024年）から抽出されたTrue/False形式の質問応答データセット。高難度の法的推論評価に使用します。

**詳細:**
- **提供元**: HuggingFace (nguyenthanhasia)
- **問題数**: 2,846問
- **期間**: 2015-2024年（10年分）
- **分野**: 刑法（19.3%）、憲法（18.9%）、民法（61.8%）
- **形式**: バイナリ分類（True/False）
- **用途**: 高難度評価、法的推論の理解度測定
- **ライセンス**: CC BY 4.0

### データ構成

**データ分割:**
- Train: 2,276問（80%）
- Validation: 284問（10%）
- Test: 286問（10%）

**回答分布:**
- False: 1,502問（52.8%）
- True: 1,344問（47.2%）

**質問タイプ:**
- CSQ（単一選択）: 2,080問（73.1%）
- MRQ（複数回答）: 448問（15.7%）
- TFQ（True/False）: 318問（11.2%）

### ダウンロード方法

```bash
# HuggingFace datasetsライブラリを使用
pip install datasets

# Python経由でダウンロード
python3 << 'EOF'
from datasets import load_dataset
dataset = load_dataset("nguyenthanhasia/japanese-bar-exam-qa")
dataset.save_to_disk("datasets/japanese_bar_exam_qa")
EOF
```

### データ形式の例

```json
{
  "id": "令和３年-民法-24-3",
  "year": "令和３年",
  "subject": "Civil Law",
  "subject_jp": "民法",
  "theme": "贈与",
  "question_type": "CSQ",
  "instruction": "判例の趣旨に照らし",
  "question": "受贈者は,贈与契約が書面によらない場合であっても,履行の終わっていない部分について贈与契約を解除することができない。",
  "label": "N",
  "answer": "False"
}
```

### 特徴

**判例参照指示（48.2%の問題に含まれる）:**
- "判例の趣旨に照らし"（1,040問）
- "判例の立場に従って検討"（109問）
- "判例の立場に従って検討し"（55問）

### 参照リンク

- HuggingFace: https://huggingface.co/datasets/nguyenthanhasia/japanese-bar-exam-qa
- 論文引用:
```bibtex
@misc{japanese_bar_exam_qa_2025,
  title        = {Japanese Bar Examination QA Dataset},
  author       = {Fumihito Nishino and Nguyen Ha Thanh and Ken Satoh},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/datasets/nguyenthanhasia/japanese-bar-exam-qa}},
  note         = {Publisher: Hugging Face}
}
```

## 4. japan-law（任意、デフォルトでは未配置）

### 概要

日本の法令に関する追加コーパス。

**詳細:**
- **提供元**: HuggingFace (y2lan)
- **用途**: 法令コーパスの補完
- **参照リンク**: https://huggingface.co/datasets/y2lan/japan-law

### ダウンロード方法

```bash
# HuggingFace datasetsライブラリを使用
from datasets import load_dataset
dataset = load_dataset("y2lan/japan-law")
dataset.save_to_disk("datasets/japan_law")
```

## 5. japanese-law-analysis（任意、デフォルトでは未配置）

### 概要

法令分析のためのデータセット。

**詳細:**
- **提供元**: GitHub (japanese-law-analysis)
- **用途**: 法令の詳細分析
- **参照リンク**: https://github.com/japanese-law-analysis/data_set

### ダウンロード方法

```bash
# GitHubからクローン
cd datasets
git clone https://github.com/japanese-law-analysis/data_set japanese_law_analysis
```

## その他の参考リソース

### e-Gov法令検索

公式の法令データベース。最新の法令情報を確認できます。

- **URL**: https://laws.e-gov.go.jp/
- **用途**: 最新法令の確認、法令検索

## ディスク容量

データセットのダウンロード・展開には以下の容量が必要です：

### 必須データセット

#### e-Gov法令一括
- 圧縮ファイル（egov_laws_all.zip）: 264MB
- 展開後のXMLファイル: 約2GB
- **小計**: 約2.3GB

#### lawqa_jp
- 全データファイル: 約4.9MB

### オプションデータセット

#### japanese-bar-exam-qa
- データファイル: 約10-50MB（推定）

#### japan-law
- データファイル: サイズ可変

#### japanese-law-analysis
- データファイル: サイズ可変

### 合計
- **必須データセット**: 約2.4GB
- **全データセット**: 約3-5GB（オプション含む）

## .gitignore

データセットの大部分は `.gitignore` で除外されています：

```
# Datasets
datasets/egov_laws/*.xml
datasets/egov_laws/*.zip
```

**追跡されるファイル:**
- READMEファイル
- LICENSEファイル
- lawqa_jpデータファイル（サイズが小さいため）

## トラブルシューティング

### XMLファイルが見つからない

```
FileNotFoundError: datasets/egov_laws/*.xml
```

**解決方法:**

1. zipファイルが配置されているか確認
```bash
ls -lh datasets/egov_laws/egov_laws_all.zip
```

2. zipファイルを展開
```bash
cd datasets/egov_laws
unzip egov_laws_all.zip
```

### ディスク容量が不足する場合

展開前にディスク容量を確認：

```bash
# 現在のディスク使用状況
df -h .

# 必要な空き容量: 最低3GB推奨
```

### lawqa_jpデータが見つからない

```
FileNotFoundError: datasets/lawqa_jp/data/selection.json
```

**解決方法:**

1. データディレクトリの確認
```bash
ls -lh datasets/lawqa_jp/data/
```

2. ファイルが存在しない場合は手動でダウンロード
```bash
# データセット提供元から取得
# 参考: https://www.digital.go.jp/
```

### 前処理が失敗する

XMLファイルの展開が完了していることを確認：

```bash
# XMLファイル数を確認
find datasets/egov_laws -name "*.xml" | wc -l
# 期待値: 約10,435

# 前処理を再実行
make preprocess
```

### HuggingFaceデータセットのダウンロードエラー

```python
# datasetsライブラリが必要
pip install datasets

# 認証が必要な場合
from huggingface_hub import login
login()
```

## データセットの活用方法

### 評価用データセットとしての使用

このプロジェクトでは、複数のデータセットを評価に使用できます：

#### 1. lawqa_jpによる4択評価

```bash
# 基本的な評価
python3 scripts/evaluate_multiple_choice.py \
  --data datasets/lawqa_jp/data/selection.json \
  --limit 20

# Makefileを使用
make eval-multiple-choice
```

#### 2. japanese-bar-exam-qaによる高難度評価

```python
# カスタム評価スクリプトの例
from datasets import load_dataset

dataset = load_dataset("nguyenthanhasia/japanese-bar-exam-qa")
test_data = dataset["test"]

# RAGシステムで各問題を評価
for item in test_data:
    question = item["question"]
    instruction = item["instruction"]
    correct_answer = item["answer"]
    # ... RAG評価ロジック
```

#### 3. 複数データセットによる包括評価

```bash
# 異なる難易度・形式での評価を組み合わせ
# 1. 基本評価（lawqa_jp）
make eval-mc

# 2. 高難度評価（司法試験データ）
python3 scripts/evaluate_bar_exam.py  # 別途実装

# 3. RAGAS評価
make eval-ragas
```

### コーパス拡張

e-Gov法令以外のコーパスを追加することで、検索精度を向上できます：

```python
# japan-lawデータセットを追加コーパスとして使用
from datasets import load_dataset

# メインコーパス: e-Gov法令
# 補完コーパス: japan-law、japanese-law-analysis
# これらを組み合わせてインデックスを構築
```

## 関連ドキュメント

- [egov_laws/README.md](egov_laws/README.md) - e-Gov法令データセット詳細
- [lawqa_jp/README.md](lawqa_jp/README.md) - 法令QAデータセット詳細
- [../docs/02-SETUP.md](../docs/02-SETUP.md) - セットアップガイド
- [../docs/03-USAGE.md](../docs/03-USAGE.md) - スクリプトの使用方法
- [../data/README.md](../data/README.md) - 前処理済みデータの説明

## データセットのライセンス

各データセットには異なるライセンスが適用されます：

### e-Gov法令一括
- **ライセンス**: 政府標準利用規約
- **詳細**: https://elaws.e-gov.go.jp/
- **利用範囲**: 政府が公開する法令データ

### lawqa_jp
- **ライセンス**: 公共データ利用規約（第1.0版）
- **詳細**: https://www.digital.go.jp/resources/open_data/public_data_license_v1.0
- **提供元**: デジタル庁

### japanese-bar-exam-qa
- **ライセンス**: CC BY 4.0
- **詳細**: https://creativecommons.org/licenses/by/4.0/
- **出典**: 日本の法務省が実施する司法試験問題

### japan-law
- **ライセンス**: データセット提供元のライセンスに従う
- **詳細**: HuggingFaceページを参照

### japanese-law-analysis
- **ライセンス**: GitHubリポジトリのライセンスに従う
- **詳細**: https://github.com/japanese-law-analysis/data_set

## 重要な注意事項

### 法的助言について
- これらのデータセットは研究・教育目的で提供されています
- 問題文および選択肢は複数のLLMにより作成されている場合があり、法的助言を目的としたものではありません
- 実際の法的問題については、必ず専門家にご相談ください

### 法令の最新性
- 法令は将来的に改正される可能性があります
- 利用にあたっては、e-Gov法令検索（https://laws.e-gov.go.jp/）で最新の法令をご確認ください
- データセットの作成時点と現在で法令内容が異なる場合があります

### 引用と帰属
- データセットを使用する際は、適切な引用と帰属を行ってください
- 論文等で使用する場合は、各データセットの推奨する引用形式に従ってください
