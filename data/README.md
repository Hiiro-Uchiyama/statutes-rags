# data/ ディレクトリ

このディレクトリには、前処理済みデータとインデックスファイルが格納されます。

**重要:** このディレクトリのファイルは自動生成されるため、Gitリポジトリには含まれていません。

## ディレクトリ構造

```
data/
├── egov_laws.jsonl          # 前処理済み法令データ（約1.8GB）
├── egov_laws_1k.jsonl       # テスト用（最初の1,000件）
├── egov_laws_5k.jsonl       # 小規模テスト用（最初の5,000件）
├── egov_laws_10k.jsonl      # 中規模テスト用（最初の10,000件）
├── test_egov.jsonl          # 開発用テストデータ
└── faiss_index/             # 検索用インデックス
    ├── vector/              # ベクトル検索インデックス
    │   ├── index.faiss      # FAISSインデックスファイル
    │   └── index.pkl        # メタデータとドキュメント情報
    └── bm25/                # BM25検索インデックス
        └── index.pkl        # BM25インデックスファイル
```

## ファイル生成方法

### 1. 前処理済みJSONLファイルの生成

```bash
# 全XMLファイルを前処理（約5-10分）
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl

# または Makefile を使用
make preprocess
```

### 2. 検索インデックスの構築

```bash
# ハイブリッド検索用インデックス構築（約20-40分）
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid

# または Makefile を使用
make index
```

## ファイル説明

### egov_laws.jsonl

前処理済みの法令データファイル（JSONL形式）。各行が1つの法令条文を表します。

**形式:**

```json
{
  "law_id": "325AC0000000089",
  "law_name": "会社法",
  "article_id": "第1条",
  "article_title": "趣旨",
  "content": "この法律は、会社の設立、組織、運営及び管理について...",
  "full_text": "会社法 第1条 趣旨\nこの法律は...",
  "metadata": {
    "source": "https://laws.e-gov.go.jp/law/325AC0000000089"
  }
}
```

**サイズ:** 約1.8GB（全10,435法令から生成される数万件の条文データ）

### faiss_index/

検索用のインデックスファイル。RAGシステムが類似文書を高速検索するために使用します。

#### vector/

**index.faiss**: FAISSベクトルインデックス
- ベクトル検索用のインデックス
- 768次元の埋め込みベクトル（nomic-embed-text モデル）
- サイズ: 約500MB-1GB（文書数による）

**index.pkl**: メタデータ
- 各ベクトルに対応するドキュメント情報
- 法令名、条文番号、本文などを含む

#### bm25/

**index.pkl**: BM25インデックス
- キーワードベース検索用のインデックス
- MeCabによる日本語トークナイズ
- サイズ: 約100-200MB

## テスト用データ

開発・テスト用に小規模なデータセットを生成できます：

```bash
# 1,000件のみ
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws_1k.jsonl \
  --limit 1000

# テスト用インデックス構築（約2-5分）
python3 scripts/build_index.py \
  --data-path data/egov_laws_1k.jsonl \
  --index-path data/faiss_index_1k \
  --retriever-type hybrid
```

## ディスク容量

完全なデータセットとインデックスには以下の容量が必要です：

- `egov_laws.jsonl`: 約1.8GB
- `faiss_index/vector/`: 約500MB-1GB
- `faiss_index/bm25/`: 約100-200MB
- **合計**: 約2.5-3GB

テスト用（1,000件）の場合：約50-100MB

## .gitignore

このディレクトリの内容は `.gitignore` で除外されています：

```
# Data
data/
```

**例外:** このREADMEファイルのみ追跡されています。

## トラブルシューティング

### ファイルが生成されない

1. `datasets/egov_laws/` にXMLファイルが配置されているか確認
2. 前処理スクリプトを実行したか確認
3. ディスク容量が十分か確認（最低3GB必要）

### インデックスの読み込みエラー

```
FileNotFoundError: data/faiss_index/vector/index.faiss
```

解決方法：インデックスを再構築

```bash
make index
```

### メモリ不足エラー

小規模データセットでテスト：

```bash
# 1,000件のみでインデックス構築
python3 scripts/build_index.py \
  --data-path data/egov_laws_1k.jsonl \
  --index-path data/faiss_index_1k \
  --retriever-type hybrid
```

## 関連ドキュメント

- [SETUP.md](../docs/02-SETUP.md) - セットアップガイド
- [USAGE.md](../docs/03-USAGE.md) - スクリプトの使用方法
- [ARCHITECTURE.md](../docs/05-ARCHITECTURE.md) - システムアーキテクチャ
