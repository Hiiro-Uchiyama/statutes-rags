# e-Gov法令XMLデータセット

## 概要

このディレクトリには、e-Gov法令検索から取得した日本の全法令XMLファイルが含まれています。

- **ファイル数**: 約10,435件
- **総サイズ**: 約2GB (展開後)
- **圧縮ファイル**: egov_laws_all.zip (264MB)
- **提供元**: 総務省 e-Gov法令検索
- **用途**: Legal RAGシステムのメインコーパス

## データの展開方法

### 前提条件

- `egov_laws_all.zip` がこのディレクトリに配置されていること
- unzipコマンドが利用可能であること

### 展開手順

```bash
# このディレクトリに移動
cd datasets/egov_laws

# zipファイルを展開
unzip egov_laws_all.zip

# 展開されたファイルを確認
ls -lh

# XMLファイルの数を確認
find . -name "*.xml" | wc -l
# 期待値: 約10,435ファイル
```

### 展開後のディレクトリ構造

```
datasets/egov_laws/
├── README.md                  # このファイル
├── egov_laws_all.zip          # 圧縮アーカイブ (264MB)
├── 32AC9000000001_*.xml       # 法令XMLファイル
├── 32AC9000000002_*.xml
├── ...
└── (約10,435個のXMLファイル)
```

## 展開後の次のステップ

XMLファイルの展開が完了したら、以下のコマンドで前処理を実行してください：

```bash
# プロジェクトルートディレクトリに移動
cd /home/jovyan/work/statutes-rags

# 仮想環境を有効化
source .venv/bin/activate

# XMLからJSONLへ変換
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl

# または Makefile を使用
make preprocess
```

詳細は `docs/02-SETUP.md` の「データ準備」セクションを参照してください。

## トラブルシューティング

### unzipコマンドが見つからない場合

```bash
# Debian/Ubuntu
sudo apt-get install unzip

# macOS
brew install unzip

# または、Python を使用
python3 -m zipfile -e egov_laws_all.zip .
```

### 展開中にエラーが発生する場合

```bash
# zipファイルの整合性を確認
unzip -t egov_laws_all.zip

# 上書き確認をスキップして強制展開
unzip -o egov_laws_all.zip
```

### ディスク容量が不足する場合

展開後のサイズは約2GBです。事前にディスク容量を確認してください：

```bash
# 現在のディスク使用状況を確認
df -h .

# 必要な空き容量: 最低3GB推奨
```

## ライセンス

e-Gov法令検索で公開されている法令データは、政府標準利用規約に基づいて利用できます。

詳細: https://elaws.e-gov.go.jp/
