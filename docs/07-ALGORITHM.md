# アルゴリズム詳細ガイド

本ドキュメントでは、法令RAGシステムで使用されているアルゴリズムの詳細を説明します。

## 目次

1. [システムアーキテクチャ概要](#システムアーキテクチャ概要)
2. [文書前処理とチャンキング](#文書前処理とチャンキング)
3. [ベクトル検索アルゴリズム](#ベクトル検索アルゴリズム)
4. [BM25検索アルゴリズム](#bm25検索アルゴリズム)
5. [ハイブリッド検索とスコア統合](#ハイブリッド検索とスコア統合)
6. [Rerankerアルゴリズム](#rerankerアルゴリズム)
7. [RAGパイプライン全体フロー](#ragパイプライン全体フロー)
8. [パラメータチューニングガイド](#パラメータチューニングガイド)

---

## システムアーキテクチャ概要

### 全体構成

```
┌─────────────────┐
│  クエリ入力     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Retriever (文書検索)           │
│  ┌─────────────────────────────┐   │
│  │  Vector Retriever (FAISS)   │   │
│  │  - Embedding生成            │   │
│  │  - 類似度検索               │   │
│  │  - MMR (多様性考慮)         │   │
│  └─────────────────────────────┘   │
│            または                    │
│  ┌─────────────────────────────┐   │
│  │  BM25 Retriever             │   │
│  │  - トークン化               │   │
│  │  - BM25スコア計算           │   │
│  └─────────────────────────────┘   │
│            または                    │
│  ┌─────────────────────────────┐   │
│  │  Hybrid Retriever           │   │
│  │  - Vector + BM25            │   │
│  │  - RRFスコア統合            │   │
│  └─────────────────────────────┘   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│      Reranker (オプション)          │
│  - Cross-Encoder再ランキング        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│      LLM (回答生成)                 │
│  - コンテキスト整形                 │
│  - プロンプト生成                   │
│  - LLM呼び出し (Ollama)             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────┐
│  回答 + 引用    │
└─────────────────┘
```

---

## 文書前処理とチャンキング

### XMLパース処理

e-Gov法令XMLファイルから構造化データを抽出します。

**アルゴリズム:**

```python
def parse_law_xml(xml_path):
    """
    1. XMLファイルを解析
    2. 法令メタデータ（法令名、法令番号）を抽出
    3. 各条文（Article）を走査
    4. 各条文を項（Paragraph）、号（Item）単位に分割
    5. チャンクとして保存
    """
    
    # チャンクの構造:
    {
        "law_title": "法令名",
        "law_num": "法令番号",
        "article": "条番号",
        "article_caption": "条見出し",
        "article_title": "条タイトル",
        "paragraph": "項番号",
        "item": "号番号",
        "text": "本文テキスト"
    }
```

**特徴:**
- 法令の階層構造（条→項→号）を保持
- メタデータを完全に保存
- 検索・引用時に正確な条文参照が可能

---

## ベクトル検索アルゴリズム

### 概要

ベクトル検索は、テキストを高次元ベクトル空間に埋め込み、類似度に基づいて検索を行います。

### 実装: `VectorRetriever`

**使用技術:**
- **埋め込みモデル:** HuggingFace Transformers (`intfloat/multilingual-e5-large`)
- **ベクトルストア:** FAISS (Facebook AI Similarity Search)
- **類似度計算:** コサイン類似度（FAISSのL2距離から変換）

### ベクトル化プロセス

```python
def embed_text(text):
    """
    1. テキストをトークン化
    2. Transformerモデルで埋め込みベクトル生成
    3. 正規化（L2ノルム）
    
    入力: "博物館の定義は何ですか？"
    出力: [0.023, -0.145, 0.089, ..., 0.231] (1024次元)
    """
```

### 検索アルゴリズム

#### 標準検索（類似度ベース）

```python
def similarity_search(query, top_k=10):
    """
    1. クエリをベクトル化
    2. FAISSで最近傍探索（k-NN）
    3. L2距離を類似度スコアに変換
       similarity = 1.0 / (1.0 + distance)
    4. スコア降順でソート
    """
```

**スコア計算:**

```
FAISS距離: d (小さいほど類似)
類似度スコア: s = 1.0 / (1.0 + d)

例:
d = 0.0  → s = 1.0  (完全一致)
d = 1.0  → s = 0.5
d = 4.0  → s = 0.2
```

#### MMR検索（多様性考慮）

**MMR (Maximal Marginal Relevance)** は、関連性と多様性のバランスを取る検索手法です。

```python
def mmr_search(query, top_k=10, lambda_mult=0.5):
    """
    1. クエリに類似した候補をfetch_k個取得
    2. MMRアルゴリズムで再ランキング
    
    MMRスコア = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected))
    
    λ = 1.0: 関連性のみ重視
    λ = 0.5: 関連性と多様性のバランス
    λ = 0.0: 多様性のみ重視
    """
```

**MMRアルゴリズムの詳細:**

```
選択済み文書集合: S = {}
候補文書集合: C = {doc1, doc2, ..., docN}

while |S| < top_k:
    for each doc in C:
        relevance = Sim(query, doc)
        diversity = max(Sim(doc, s) for s in S) if S else 0
        mmr_score = λ * relevance - (1-λ) * diversity
    
    best_doc = argmax(mmr_score)
    S = S ∪ {best_doc}
    C = C - {best_doc}
```

**本実装の簡略版:**

FAISSのMMR実装を使用し、結果に順位ベースのスコアを付与:

```python
# MMRで多様性を考慮した検索（FAISSが順序を決定）
docs = vector_store.max_marginal_relevance_search(
    query, k=top_k, lambda_mult=mmr_lambda, fetch_k=fetch_k
)

# 順位ベースのシンプルなスコアリング
for rank, doc in enumerate(docs, start=1):
    rank_score = 1.0 / rank

例:
1位: 1.0 / 1 = 1.0
2位: 1.0 / 2 = 0.5
3位: 1.0 / 3 = 0.333
4位: 1.0 / 4 = 0.25
```

**利点:**
- シンプルで理解しやすい
- MMRの順序をそのまま尊重
- 予測可能なスコアリング

---

## BM25検索アルゴリズム

> **⚠️ 重要**: BM25検索は、280万件の大規模データセットで50-60GBのメモリを必要とし、現在のシステム構成（62GBメモリ）では**使用できません**。Vector-Onlyモード（FAISSベクトル検索）を使用してください。詳細は [docs/supplemental/memory_issue_analysis.md](supplemental/memory_issue_analysis.md) を参照してください。

### 概要

BM25 (Best Matching 25) は、キーワードベースの情報検索アルゴリズムで、TF-IDFの改良版です。

**注意**: 以下の説明は、システムアーキテクチャの理解のために記載していますが、現在の構成では実行できません。

### 実装: `BM25Retriever`

**使用ライブラリ:** `rank-bm25`

### BM25スコア計算式

```
BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

パラメータ:
- D: ドキュメント
- Q: クエリ
- qi: クエリ内の各トークン
- f(qi, D): ドキュメントD内のqiの出現頻度
- |D|: ドキュメントDの長さ（トークン数）
- avgdl: コーパス内の平均ドキュメント長
- k1: 用語頻度の飽和パラメータ（デフォルト: 1.5）
- b: 文書長正規化パラメータ（デフォルト: 0.75）

IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
- N: コーパス内の総ドキュメント数
- n(qi): qiを含むドキュメント数
```

### トークン化処理

日本語テキストをトークン化するための複数の方式をサポート:

#### 1. **Sudachi（推奨）**

```python
def tokenize_sudachi(text):
    """
    形態素解析ベースのトークン化
    
    入力: "博物館は資料を収集する機関です。"
    出力: ["博物館", "は", "資料", "を", "収集", "する", "機関", "です", "。"]
    
    特徴:
    - 管理者権限不要
    - 高精度な分かち書き
    - 辞書ベース
    """
```

#### 2. **Janome**

```python
def tokenize_janome(text):
    """
    純Pythonの形態素解析
    
    入力: "個人情報保護法第27条"
    出力: ["個人", "情報", "保護", "法", "第", "27", "条"]
    
    特徴:
    - 軽量
    - 管理者権限不要
    - やや精度が低い
    """
```

#### 3. **N-gram（辞書不要）**

```python
def tokenize_ngram(text):
    """
    2-gram + 3-gram + 完全一致
    
    入力: "博物館"
    出力: ["博物", "物館", "博物館", "博", "物", "館"]
    
    特徴:
    - 辞書不要
    - ロバスト
    - 部分一致に強い
    """
```

### BM25検索アルゴリズム

```python
def bm25_search(query, top_k=10):
    """
    1. クエリをトークン化
       例: "博物館の定義" → ["博物館", "の", "定義"]
    
    2. 各ドキュメントに対してBM25スコアを計算
    
    3. スコア降順でソート
    
    4. 上位top_k件を返す
    """
```

---

## ハイブリッド検索とスコア統合

> **⚠️ 重要**: Hybrid検索（VectorとBM25の組み合わせ）は、BM25が使用できないため、現在のシステム構成では**使用できません**。Vector-Onlyモード（`RETRIEVER_TYPE=vector`）を使用してください。詳細は [docs/supplemental/memory_issue_analysis.md](supplemental/memory_issue_analysis.md) を参照してください。

### 概要

ベクトル検索とBM25検索を組み合わせ、それぞれの長所を活かします。

**注意**: 以下の説明は、システムアーキテクチャの理解のために記載していますが、現在の構成では実行できません。

**それぞれの特性:**

| 検索手法 | 得意分野 | 苦手分野 |
|---------|----------|----------|
| ベクトル検索 | 意味的類似性<br>言い換え表現<br>文脈理解 | 正確なキーワード一致<br>固有名詞<br>数字 |
| BM25検索 | キーワード一致<br>固有名詞<br>数字<br>法令番号 | 意味的類似性<br>言い換え表現 |

### 実装: `HybridRetriever`

### スコア統合方式

#### 1. **RRF (Reciprocal Rank Fusion)** - 推奨

標準的な手法で、重み付けなしで各検索手法を平等に扱います。

**アルゴリズム:**

```python
def rrf_fusion(vector_results, bm25_results, k=60):
    """
    RRF公式:
    score(doc) = Σ 1 / (k + rank_i(doc))
    
    k: ランクの影響を調整するパラメータ（デフォルト: 60）
    rank_i(doc): i番目の検索結果におけるdocの順位
    """
    
    scores = {}
    
    # ベクトル検索の順位スコア
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = get_doc_id(doc)
        scores[doc_id] = 1.0 / (k + rank)
    
    # BM25検索の順位スコア（加算）
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = get_doc_id(doc)
        if doc_id in scores:
            scores[doc_id] += 1.0 / (k + rank)
        else:
            scores[doc_id] = 1.0 / (k + rank)
    
    # スコア降順でソート
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**スコア例:**

```
ドキュメントA:
  Vector: 1位 → 1/(60+1) = 0.0164
  BM25:   3位 → 1/(60+3) = 0.0159
  合計: 0.0323

ドキュメントB:
  Vector: 2位 → 1/(60+2) = 0.0161
  BM25:   1位 → 1/(60+1) = 0.0164
  合計: 0.0325 ← 高スコア（両方で上位）

ドキュメントC:
  Vector: 1位 → 1/(60+1) = 0.0164
  BM25:   なし → 0
  合計: 0.0164 ← 片方のみ
```

#### 2. **Weighted RRF**

検索手法ごとに重みを設定する拡張版。

```python
def weighted_rrf_fusion(vector_results, bm25_results, 
                        vector_weight=0.5, bm25_weight=0.5, k=60):
    """
    重み付きRRF:
    score(doc) = w_v * (1/(k + rank_v)) + w_b * (1/(k + rank_b))
    """
    
    scores = {}
    
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = get_doc_id(doc)
        scores[doc_id] = vector_weight * (1.0 / (k + rank))
    
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = get_doc_id(doc)
        if doc_id in scores:
            scores[doc_id] += bm25_weight * (1.0 / (k + rank))
        else:
            scores[doc_id] = bm25_weight * (1.0 / (k + rank))
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**使い分け:**
- `vector_weight > bm25_weight`: 意味的検索を重視
- `vector_weight < bm25_weight`: キーワード検索を重視
- `vector_weight = bm25_weight`: 標準RRFと同等

#### 3. **Weighted (正規化後の重み付き加算)**

生スコアを正規化してから重み付き加算。

```python
def weighted_fusion(vector_results, bm25_results,
                    vector_weight=0.5, bm25_weight=0.5):
    """
    1. Min-Max正規化
       norm_score = (score - min) / (max - min)
    
    2. 重み付き加算
       final_score = w_v * norm_v + w_b * norm_b
    """
    
    # 正規化
    vector_scores_norm = normalize_scores(vector_results)
    bm25_scores_norm = normalize_scores(bm25_results)
    
    # 統合
    scores = {}
    for doc_id, score in vector_scores_norm.items():
        scores[doc_id] = vector_weight * score
    
    for doc_id, score in bm25_scores_norm.items():
        if doc_id in scores:
            scores[doc_id] += bm25_weight * score
        else:
            scores[doc_id] = bm25_weight * score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### fetch_kの最適化

ハイブリッド検索では、各Retrieverから `fetch_k = top_k * fetch_k_multiplier` 個の候補を取得します。

**理由:**
- 統合後に重複が排除される
- 片方にしか現れない文書を確保
- より多様な結果を得る

**推奨値:**
- `fetch_k_multiplier = 2`: 標準（top_k=10なら20件ずつ取得）
- `fetch_k_multiplier = 3`: より多様性重視

---

## Rerankerアルゴリズム

### 概要

Rerankerは、Retrieverで取得した候補を再ランキングして精度を向上させます。

### 実装: `CrossEncoderReranker`

**使用モデル:** `cross-encoder/ms-marco-MiniLM-L-12-v2`

### Cross-Encoderの仕組み

**Bi-Encoder（Retriever）との違い:**

```
Bi-Encoder (Retriever):
  query → [encoder] → vec_q
  doc   → [encoder] → vec_d
  score = cosine(vec_q, vec_d)
  
  特徴: 高速、スケーラブル
  欠点: クエリと文書の相互作用を考慮できない

Cross-Encoder (Reranker):
  [CLS] query [SEP] doc [SEP] → [encoder] → score
  
  特徴: クエリと文書の相互作用を考慮、高精度
  欠点: 遅い（候補ごとに推論が必要）
```

### Reranking アルゴリズム

```python
def rerank(query, documents, top_n=5):
    """
    1. クエリと各文書をペアにする
       pairs = [
           [query, doc1.text],
           [query, doc2.text],
           ...
       ]
    
    2. Cross-Encoderで関連性スコアを計算
       scores = model.predict(pairs)
       # 出力: [-2.3, 1.5, 0.8, -0.5, ...]
       # 高いほど関連性が高い
    
    3. スコア降順でソート
    
    4. 上位top_n件を返す
    """
```

**使用例:**

```
入力（Retrieverから）:
  1. スコア: 0.85, 文書: "博物館法第2条..."
  2. スコア: 0.82, 文書: "個人情報保護法第27条..."
  3. スコア: 0.80, 文書: "博物館の定義は..."

Reranking後:
  1. スコア: 1.5,  文書: "博物館の定義は..."  ← 最も関連性が高い
  2. スコア: 0.8,  文書: "博物館法第2条..."
  3. スコア: -0.5, 文書: "個人情報保護法第27条..."  ← 関連性が低い
```

---

## RAGパイプライン全体フロー

### 実装: `RAGPipeline`

### フローチャート

```
1. クエリ入力
   ↓
2. Retriever.retrieve(query, top_k=10)
   ├─ Vector検索
   ├─ BM25検索
   └─ Hybrid統合
   ↓
3. [オプション] Reranker.rerank(query, docs, top_n=5)
   ↓
4. コンテキスト整形
   ↓
5. プロンプト生成
   ↓
6. LLM呼び出し (Ollama)
   ↓
7. 回答 + 引用情報の返却
```

### 詳細アルゴリズム

#### ステップ1: 文書検索

```python
def retrieve_documents(query):
    """
    documents = retriever.retrieve(query, top_k=10)
    
    if reranker:
        documents = reranker.rerank(query, documents, top_n=5)
    
    return documents
    """
```

#### ステップ2: コンテキスト整形

```python
def format_context(documents):
    """
    各文書を以下の形式で整形:
    
    [1] 博物館法 第2条 第1項
    博物館は、歴史、芸術、民俗、産業、自然科学等に関する資料を
    収集し、保管し、展示して教育的配慮の下に一般公衆の利用に
    供し、その教養、調査研究、レクリエーション等に資するために
    必要な事業を行う機関をいう。
    
    [2] 博物館法 第3条
    ...
    
    最大長制限:
    - max_context_length文字まで
    - 超過する場合は警告を出して切り捨て
    """
```

#### ステップ3: プロンプト生成

```python
prompt_template = """
あなたは日本の法律に精通した法律アシスタントです。
以下の法令条文に基づいて質問に答えてください。

【法令条文】
{context}

【質問】
{question}

【回答】
上記の法令条文に基づいて、正確かつ具体的に回答してください。
必ず該当する法令名と条文番号を明記してください。
"""
```

#### ステップ4: LLM呼び出し

```python
def query(question):
    """
    1. 文書検索
    2. コンテキスト整形
    3. LLM呼び出し
       - タイムアウト: 60秒（設定可能）
       - エラーハンドリング
    4. 結果返却
    """
    
    try:
        documents = retrieve_documents(question)
        
        if not documents:
            return {
                "answer": "関連する法令条文が見つかりませんでした。",
                "citations": [],
                "contexts": []
            }
        
        context = format_context(documents)
        answer = llm.invoke({"context": context, "question": question})
        citations = extract_citations(documents)
        
        return {
            "answer": answer,
            "citations": citations,
            "contexts": [format_doc(d) for d in documents]
        }
    
    except TimeoutError:
        return {"answer": "LLMのリクエストがタイムアウトしました。", ...}
    except Exception as e:
        return {"answer": f"エラーが発生しました: {e}", ...}
```

#### ステップ5: 引用情報の抽出

```python
def extract_citations(documents):
    """
    重複を排除して引用情報を抽出
    
    入力: [
        {law_title: "博物館法", article: "2", paragraph: "1"},
        {law_title: "博物館法", article: "2", paragraph: "1"},  # 重複
        {law_title: "博物館法", article: "3", paragraph: "1"}
    ]
    
    出力: [
        {law_title: "博物館法", article: "2", paragraph: "1"},
        {law_title: "博物館法", article: "3", paragraph: "1"}
    ]
    """
```

---

## パラメータチューニングガイド

### 環境変数による設定

`.env`ファイルまたは環境変数で設定できます。

```bash
# Retriever設定
RETRIEVER_TYPE=hybrid          # vector, bm25, hybrid
RETRIEVER_TOP_K=10             # 検索する文書数

# Vector検索設定
USE_MMR=true                   # MMR使用
MMR_LAMBDA=0.5                 # MMR多様性パラメータ (0-1)

# Hybrid検索設定
FUSION_METHOD=rrf              # rrf, weighted_rrf, weighted
VECTOR_WEIGHT=0.5              # ベクトル検索の重み
BM25_WEIGHT=0.5                # BM25検索の重み
RRF_K=60                       # RRFパラメータ
FETCH_K_MULTIPLIER=2           # 候補取得倍率

# BM25設定
BM25_TOKENIZER=auto            # auto, sudachi, janome, mecab, ngram

# Reranker設定
RERANKER_ENABLED=false         # Reranker使用
RERANKER_TOP_N=5               # Reranker後の文書数

# LLM設定
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
LLM_TEMPERATURE=0.1
```

### 推奨設定

#### 1. **高精度重視（4択問題など）**

```bash
RETRIEVER_TYPE=hybrid
RETRIEVER_TOP_K=10
USE_MMR=true
MMR_LAMBDA=0.3                # 多様性よりも関連性重視
FUSION_METHOD=rrf
FETCH_K_MULTIPLIER=3          # より多くの候補
RERANKER_ENABLED=true
RERANKER_TOP_N=5
LLM_TEMPERATURE=0.0           # 決定的な回答
```

#### 2. **バランス型（一般的な質問応答）**

```bash
RETRIEVER_TYPE=hybrid
RETRIEVER_TOP_K=10
USE_MMR=true
MMR_LAMBDA=0.5
FUSION_METHOD=rrf
FETCH_K_MULTIPLIER=2
RERANKER_ENABLED=false
LLM_TEMPERATURE=0.1
```

#### 3. **高速重視**

```bash
RETRIEVER_TYPE=bm25           # または vector
RETRIEVER_TOP_K=5
USE_MMR=false
RERANKER_ENABLED=false
LLM_TEMPERATURE=0.1
```

### パラメータの影響

| パラメータ | 値の範囲 | 影響 |
|-----------|---------|------|
| `RETRIEVER_TOP_K` | 5-20 | 大きいほど多様だが、ノイズも増える |
| `MMR_LAMBDA` | 0.0-1.0 | 0に近いほど多様性重視、1に近いほど関連性重視 |
| `VECTOR_WEIGHT` | 0.0-1.0 | 意味的検索の重要度 |
| `BM25_WEIGHT` | 0.0-1.0 | キーワード検索の重要度 |
| `RRF_K` | 10-100 | 大きいほどランクの影響が小さい |
| `FETCH_K_MULTIPLIER` | 1-5 | 大きいほど多様だが計算コスト増 |
| `LLM_TEMPERATURE` | 0.0-1.0 | 0に近いほど決定的、1に近いほどクリエイティブ |

### デバッグとログ

ログレベルを`DEBUG`に設定すると、詳細な情報が表示されます:

```bash
export LOG_LEVEL=DEBUG
python scripts/query_cli.py "博物館とは"
```

**出力例:**

```
DEBUG - MMR retrieval: top_k=10, fetch_k=20, lambda=0.5
DEBUG - MMR returned 10 documents
DEBUG - Hybrid retrieval for query: '博物館とは', top_k=10, fetch_k=20
DEBUG - RRF fusion: 15 unique documents from 20 vector + 20 BM25 results
INFO - Hybrid retrieval returned 10 documents (method: rrf)
DEBUG - Context length: 2345 characters
INFO - LLM response received successfully
```

---

## まとめ

本システムは、以下の高度なアルゴリズムを組み合わせています:

1. **多様な検索手法**: Vector、BM25、Hybrid
2. **多様性考慮**: MMRアルゴリズム
3. **スコア統合**: RRF（Reciprocal Rank Fusion）
4. **再ランキング**: Cross-Encoder
5. **柔軟な設定**: 環境変数によるパラメータ調整

これらのアルゴリズムにより、法令という専門的なドメインにおいても、高精度な情報検索と回答生成が可能になっています。

---

## 参考文献

- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal rank fusion outperforms condorcet and individual rank learning methods"
- Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
- Carbonell, J., & Goldstein, J. (1998). "The use of MMR, diversity-based reranking for reordering documents and producing summaries"
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

---

関連ドキュメント:
- [05-ARCHITECTURE.md](05-ARCHITECTURE.md) - システムアーキテクチャ
- [03-USAGE.md](03-USAGE.md) - 使用方法
- [supplemental/tokenizer-guide.md](supplemental/tokenizer-guide.md) - トークナイザーガイド
- [supplemental/evaluation-guide.md](supplemental/evaluation-guide.md) - 評価ガイド

---

最終更新: 2024-11-04

