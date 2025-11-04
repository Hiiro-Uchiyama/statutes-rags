# RAGコード修正サマリー

修正日: 2025年11月4日

## 修正概要

RAG実装のコードレビューを実施し、6つの重要な問題を特定・修正しました。

## 修正項目

### 1. HybridRetrieverのスコア正規化の実装

**問題点:**
- FAISSのスコア（距離、小さいほど良い）とBM25のスコア（類似度、大きいほど良い）をそのまま加算していた
- 異なるスケールのスコアを統合すると、正しいランキングができない

**修正内容:**
- Reciprocal Rank Fusion (RRF) 方式を実装
- Min-Max正規化を用いた重み付き統合方式も実装
- `fusion_method` パラメータで統合方式を選択可能に
- ドキュメント重複判定をハッシュベースに改善（メモリ効率向上）

**変更ファイル:**
- `app/retrieval/hybrid_retriever.py`

**主な変更:**
```python
# RRF方式の追加
def _rrf_fusion(self, vector_results, bm25_results) -> List[Document]:
    """Reciprocal Rank Fusion (RRF) でスコアを統合"""
    # ランクベースのスコアリングで異なるスケールを統一

# スコア正規化の追加
def _normalize_scores(self, documents) -> List[Document]:
    """Min-Max正規化でスコアを0-1の範囲に正規化"""

# ハッシュベースのドキュメントID生成
def _get_doc_id(self, doc: Document) -> str:
    """ドキュメントの一意なIDを生成（ハッシュベース）"""
```

### 2. VectorRetrieverのMMRスコア計算の修正

**問題点:**
- MMR使用時にスコアが0.0で固定されていた
- ハイブリッド検索で正しくスコアリングできない

**修正内容:**
- MMR実行前に類似度検索でスコアを取得
- 元のスコアとMMRの順位スコアを組み合わせて最終スコアを計算
- スコアマップを使用して効率的にマッピング

**変更ファイル:**
- `app/retrieval/vector_retriever.py`

**主な変更:**
```python
if self.use_mmr:
    # MMR使用時：まず類似度検索で候補を取得
    fetch_k = top_k * 3
    candidates_with_scores = self.vector_store.similarity_search_with_score(
        query, k=fetch_k
    )
    
    # MMRで再ランキング
    docs = self.vector_store.max_marginal_relevance_search(
        query, k=top_k, lambda_mult=self.mmr_lambda, fetch_k=fetch_k
    )
    
    # スコアを組み合わせ
    combined_score = original_score * 0.7 + mmr_score * 0.3
```

### 3. BM25RetrieverのMeCabパス設定の改善

**問題点:**
- MeCabのパスがハードコードされていた
- 環境依存性が高く、異なる環境で動作しない可能性があった
- プロジェクトパスと異なるパスが記載されていた

**修正内容:**
- 環境変数 `MECABRC` を優先的に使用
- プロジェクトルートからの相対パスを動的に計算
- macOS Homebrew、Linux標準パスにも対応
- 複数のフォールバックパスを追加

**変更ファイル:**
- `app/retrieval/bm25_retriever.py`

**主な変更:**
```python
# 環境変数から優先的にパスを取得
if 'MECABRC' not in os.environ:
    # プロジェクトルートからの相対パスを構築
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mecab_paths = [
        os.path.join(project_root, "setup/lib/mecab/etc/mecabrc"),
        "/usr/local/etc/mecabrc",
        "/etc/mecabrc",
        "/opt/homebrew/etc/mecabrc",  # macOS Homebrew
    ]
```

### 4. RAG設定のパスを相対パスに修正

**問題点:**
- データパスとベクトルストアパスが特定環境にハードコードされていた
- ポータビリティに欠ける

**修正内容:**
- プロジェクトルートを動的に取得する関数を追加
- 相対パスを絶対パスに変換するヘルパー関数を実装
- 環境変数が優先、なければプロジェクトルートからの相対パスを使用

**変更ファイル:**
- `app/core/rag_config.py`

**主な変更:**
```python
def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    return Path(__file__).parent.parent.parent

def get_default_path(relative_path: str) -> str:
    """プロジェクトルートからの相対パスを絶対パスに変換"""
    project_root = get_project_root()
    return str(project_root / relative_path)

# 使用例
vector_store_path: str = Field(
    default=os.getenv("VECTOR_STORE_PATH", get_default_path("data/faiss_index"))
)
```

### 5. エラーハンドリングの強化

**問題点:**
- インデックスのロード/保存時のエラーハンドリングが不十分
- RAGパイプラインでエラーが発生した際の処理が不明確

**修正内容:**
- すべてのファイルI/O操作にtry-exceptブロックを追加
- エラーメッセージを明確に出力
- エラー発生時の状態を適切にリセット
- RAGパイプラインでエラー情報を返却するように改善

**変更ファイル:**
- `app/retrieval/vector_retriever.py`
- `app/retrieval/bm25_retriever.py`
- `app/retrieval/rag_pipeline.py`

**主な変更:**
```python
# VectorRetriever, BM25Retrieverの例
try:
    # インデックス保存/ロード処理
    self.vector_store.save_local(str(index_path))
    print(f"Index saved to {index_path}")
except Exception as e:
    print(f"Error saving index to {self.index_path}: {e}")
    raise

# RAGPipelineの例
try:
    # 検索と回答生成
    documents = self.retrieve_documents(question)
    answer = self.chain.invoke({"context": context, "question": question})
    return {...}
except Exception as e:
    print(f"Error during query processing: {e}")
    return {
        "answer": f"エラーが発生しました: {str(e)}",
        "error": str(e)
    }
```

### 6. ドキュメント重複判定の改善

**問題点:**
- `page_content`全文を辞書のキーにしていた
- メモリ効率が悪く、ハッシュ衝突の可能性があった

**修正内容:**
- メタデータとコンテンツの一部からハッシュIDを生成
- MD5ハッシュを使用して一意なIDを作成
- メモリ使用量を大幅に削減

**変更ファイル:**
- `app/retrieval/hybrid_retriever.py`

**主な変更:**
```python
def _get_doc_id(self, doc: Document) -> str:
    """ドキュメントの一意なIDを生成（ハッシュベース）"""
    meta = doc.metadata
    id_parts = [
        meta.get("law_title", ""),
        str(meta.get("article", "")),
        str(meta.get("paragraph", "")),
        str(meta.get("item", "")),
        doc.page_content[:100]  # 最初の100文字も含める
    ]
    id_string = "|".join(id_parts)
    return hashlib.md5(id_string.encode()).hexdigest()
```

## 影響範囲

### 既存機能への影響
- **後方互換性**: すべての修正は後方互換性を保持
- **API変更**: HybridRetrieverのコンストラクタに新しいオプションパラメータを追加（デフォルト値あり）
- **動作変更**: ハイブリッド検索のスコア計算がより正確になり、検索精度が向上する可能性

### 新機能
- RRFベースのハイブリッド検索
- より柔軟なMeCabパス設定
- 改善されたエラーメッセージとハンドリング

## 使用例

### HybridRetrieverの使用（RRF方式）

```python
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever

# Retrieverの初期化
vector_retriever = VectorRetriever(
    embedding_model="intfloat/multilingual-e5-large",
    index_path="data/faiss_index"
)
bm25_retriever = BM25Retriever(index_path="data/bm25_index")

# RRF方式のハイブリッド検索
hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    fusion_method="rrf",  # RRF方式
    vector_weight=0.5,
    bm25_weight=0.5,
    rrf_k=60
)

# 重み付き方式のハイブリッド検索
hybrid_retriever_weighted = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    fusion_method="weighted",  # 重み付き方式
    vector_weight=0.6,
    bm25_weight=0.4
)

results = hybrid_retriever.retrieve("個人情報保護法について", top_k=10)
```

### 環境変数の設定

```bash
# MeCabのパスを指定（オプション）
export MECABRC=/path/to/mecabrc

# データパスを指定（オプション）
export VECTOR_STORE_PATH=/custom/path/to/faiss_index
export DATA_PATH=/custom/path/to/data.jsonl
```
