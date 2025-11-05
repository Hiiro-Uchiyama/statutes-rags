# BM25が使用できない理由：メモリ制約の詳細分析

## 結論（要約）

**BM25キーワード検索とHybridモードは、280万件の大規模データセットでは使用できません。**

- **メモリ要件**: 50-60GB
- **利用可能メモリ**: 62GB（システム使用分を除くと実質50GB以下）
- **代替手段**: Vector-Onlyモード（FAISSベクトル検索）
- **Vector-Only精度**: 50%達成（実用レベル）

**推奨設定**: `.env`ファイルで `RETRIEVER_TYPE=vector` を設定してください。

---

## 問題の詳細

### 発生した問題

BM25インデックスの再構築を試みましたが、以下の対策を実施してもシステムが停止（OOM）しました：

1. tokenized_corpusの保存・ロード機能の実装
2. バッチ処理による段階的な構築
3. n-gramトークナイザーへの変更

## 根本原因

### BM25アーキテクチャの制約

BM25 Retrieverは以下の構造を持ちます：

```python
class BM25Retriever:
    def __init__(self):
        self.documents = []              # 全ドキュメント（1.8GB）
        self.tokenized_corpus = []       # 全トークン配列（推定40GB以上）
        self.bm25 = None                 # BM25Okapi インデックス（1.1GB）
```

### メモリ消費の内訳

280万件のドキュメント処理時：
- `documents`: 1.8GB（pickle保存サイズ）
- `tokenized_corpus`: **40-50GB**（メモリ上のサイズ）
- `bm25`: 1.1GB（pickle保存サイズ）

**合計: 約50-60GBのメモリが必要**

### なぜバッチ処理でも失敗するか

```python
# バッチごとに実行
for batch in batches:
    retriever.add_documents(batch, rebuild_index=False)
    # ↓ この処理が問題
    self.tokenized_corpus.extend(new_tokens)
```

- バッチを分けても、`tokenized_corpus`は**累積**される
- 最終的に280万件全てのトークンがメモリに蓄積
- システムメモリ62GBを超えてOOM (Out of Memory)

## 試した対策と結果

### ✗ 対策1: tokenized_corpusの保存・ロード
- **結果**: 再構築時にメモリ爆発が発生
- **理由**: 新規構築時は全件をメモリに保持する必要がある

### ✗ 対策2: バッチ処理
- **結果**: 190万件付近でメモリ不足
- **理由**: バッチを分けても最終的に全件蓄積される

### ✗ 対策3: n-gramトークナイザー
- **結果**: Sudachiの制限は回避できたが、メモリ問題は解決せず
- **理由**: トークナイザーの種類ではなく、データ量が問題

## 技術的な制約

### rank-bm25ライブラリの設計

```python
class BM25Okapi:
    def __init__(self, corpus):
        # corpus = 全ドキュメントのトークン配列が必要
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(len, corpus)) / self.corpus_size
        # ... 全ドキュメントの統計情報を計算
```

BM25Okapiは**全コーパスの統計情報**が必要なため、分割構築は不可能。

## 可能な解決策

### オプションA: Vector-Onlyで評価 ✓ 推奨
- BM25を使用せず、FAISSベクトル検索のみで評価
- メモリ使用量: 約20GB（安定動作確認済み）
- 精度: Vector検索のみ（すでに50%達成）

```bash
export RETRIEVER_TYPE=vector
python scripts/evaluate_multiple_choice.py --samples 100
```

### オプションB: BM25サブセットインデックス
- 重要な法令のみ（例：10万件）でBM25インデックスを構築
- メモリ使用量: 約5-10GB
- 精度: 完全なデータセットより低下の可能性

### オプションC: 外部検索エンジンの使用
- Elasticsearch、Meilisearch等を使用
- メモリ: 別プロセスで管理
- 実装コスト: 高い

### オプションD: BM25の諦め（Hybrid → Vector）
- 環境変数で`RETRIEVER_TYPE=vector`に固定
- コードは維持（将来的な対応の余地）
- 実用上は問題なし

## 推奨アクション

### 即時対応（評価を完了するため）

**Vector-Onlyモードで評価を実行**

```bash
# .env または環境変数で設定
RETRIEVER_TYPE=vector

# 評価実行
python scripts/evaluate_multiple_choice.py \
    --data datasets/lawqa_jp/data/selection.json \
    --output evaluation_results.json \
    --samples 100 \
    --top-k 10
```

### 中期対応（メモリ効率改善）

1. **BM25Retrieverの分割インデックス対応**
   - 複数の小さなインデックスに分割
   - 検索時に並列クエリ→マージ
   - 実装コスト: 中

2. **代替BM25ライブラリの検討**
   - `pyserini`（Luceneベース）
   - ディスクベースのインデックス
   - 実装コスト: 高

### 長期対応（アーキテクチャ変更）

- Elasticsearch/OpenSearchへの移行
- 分散検索システムの導入
- GPU加速ベクトル検索への完全移行

## 結論

**BM25の全件（280万件）インデックス再構築は、現在のシステム構成では不可能**

理由：
- メモリ要件: 50-60GB
- 利用可能メモリ: 62GB（システム使用分を除くと実質50GB以下）
- rank-bm25の設計上、全コーパスのメモリ保持が必須

**推奨**: Vector-Onlyモードで評価を継続
