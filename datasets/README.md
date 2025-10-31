# datasets/ ディレクトリ

このディレクトリには、プロジェクトで使用する各種データセットを配置します。

**重要:** データセットファイルは大容量のため、Gitリポジトリには含まれていません（一部ドキュメントファイルを除く）。使用前に個別に入手・配置してください。

## ディレクトリ構造

```
datasets/
├── egov_laws/              # e-Gov法令XMLファイル（必須）
│   ├── *.xml               # 10,435ファイル、約2GB
│   └── egov_laws_all.zip   # アーカイブ（264MB）
├── lawqa_jp/               # デジタル庁 4択法令データ（必須）
│   ├── README.md           # データセット説明（追跡済み）
│   ├── LICENSE.md          # ライセンス情報（追跡済み）
│   └── data/               # 実データファイル
│       ├── selection.json  # 4択問題（140問）
│       └── ...
├── civil_law_instructions/ # 民法QAデータ（オプション）
└── criminal_law_exams/     # 刑法試験問題（オプション）
```

## データセット詳細

### 1. egov_laws/（必須）

- **概要:** e-Gov法令XMLファイル（日本の全法令）
- **内容:** 10,435の法令XMLファイル
- **データソース:** [e-Gov法令検索API](https://www.e-gov.go.jp/elaws/)
- **サイズ:** 約2GB（解凍後）

**入手方法:**

```bash
mkdir -p datasets/egov_laws
cd datasets/egov_laws
# egov_laws_all.zip を配置
unzip egov_laws_all.zip
```

### 2. lawqa_jp/（必須）

- **概要:** デジタル庁の4択法令データ（140問）
- **データソース:** [デジタル庁 調査研究](https://www.digital.go.jp/news/382c3937-f43c-4452-ae27-2ea7bb66ec75)
- **ライセンス:** 公共データ利用規約（第1.0版）
- **サイズ:** 約5-10MB

詳細は [lawqa_jp/README.md](lawqa_jp/README.md) を参照。

**入手方法:**

```bash
mkdir -p datasets/lawqa_jp/data
# selection.json などを配置
```

## セットアップ手順

```bash
# 1. egov_laws配置
mkdir -p datasets/egov_laws
cd datasets/egov_laws && unzip egov_laws_all.zip

# 2. lawqa_jp配置
mkdir -p datasets/lawqa_jp/data
# selection.json を配置

# 3. 確認
find datasets/egov_laws -name "*.xml" | wc -l  # 10,435
ls datasets/lawqa_jp/data/
```

詳細は [../docs/02-SETUP.md](../docs/02-SETUP.md) を参照。
