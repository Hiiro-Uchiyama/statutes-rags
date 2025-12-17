# 実装詳細ドキュメント

法令QAシステムの検索品質改善に関する実装修正の詳細記録。

---

## 1. XMLパーサーの修正

### 1.1 問題

e-Gov法令XMLの構造が複雑（Chapter > Section > Subsection > Article）で、ネストされた条文が抽出できていなかった。

**症状**: 金融商品取引法で216条文しか抽出されない（実際は1,732条文）

### 1.2 原因

元のパーサーは直接の子要素のみを処理していた：

```python
# 修正前（scripts/parse_egov_xml.py）
for chapter in main_prov.findall("Chapter"):
    result["chapters"].append(parse_section(chapter, "Chapter"))
```

### 1.3 修正内容

XPathの `//` を使用して全ての子孫要素を再帰的に取得：

```python
# 修正後
def parse_xml_file(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    result = {
        "law_num": "",
        "law_title": "",
        "law_id": xml_path.stem.split("_")[0],
        "chapters": [],
        "articles": []
    }
    
    # LawNum, LawTitle の抽出（省略）
    
    # 全てのArticleを再帰的に抽出
    for art in root.findall(".//Article"):
        result["articles"].append(parse_article(art))
    
    return result
```

### 1.4 効果

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| 総条文数 | 1,669 | **5,045** |
| 金融商品取引法 | 216 | **1,732** |
| コンテキスト一致率 | 47.1% | **80.0%** |

---

## 2. 数字正規化（Number Normalizer）

### 2.1 問題

クエリとインデックスで条文番号の表記が異なる：
- クエリ: 「第21条」（アラビア数字）
- インデックス: 「第二十一条」（漢数字）

### 2.2 実装

**ファイル**: `app/utils/number_normalizer.py`

```python
def arabic_to_kanji_number(num: int) -> str:
    """
    アラビア数字を漢数字に変換
    
    Examples:
        21 → 二十一
        164 → 百六十四
        1234 → 千二百三十四
    """
    if num == 0:
        return '〇'
    
    result = ''
    
    # 千の位
    if num >= 1000:
        thousands = num // 1000
        if thousands == 1:
            result += '千'
        else:
            result += ARABIC_TO_KANJI[str(thousands)] + '千'
        num %= 1000
    
    # 百の位、十の位、一の位も同様に処理
    # ...
    
    return result


def normalize_article_numbers(text: str, to_kanji: bool = True) -> str:
    """
    テキスト内の条文番号を正規化
    
    Examples:
        「第21条」→「第二十一条」
        「第27条の5」→「第二十七条の五」
    """
    if to_kanji:
        patterns = [
            (r'第(\d+)条', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}条'),
            (r'第(\d+)項', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}項'),
            (r'第(\d+)号', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}号'),
            # ...
        ]
        
        for pattern, replacer in patterns:
            text = re.sub(pattern, replacer, text)
        
        # 「の38」のような接続も変換
        text = re.sub(r'の(\d+)', 
                     lambda m: f'の{arabic_to_kanji_number(int(m.group(1)))}', text)
    
    return text
```

### 2.3 効果

| テストケース | 変換結果 |
|--------------|----------|
| 第21条 | 第二十一条 |
| 第27条の5 | 第二十七条の五 |
| 第23条の2の15 | 第二十三条の二の十五 |

---

## 3. Query Processor

### 3.1 目的

検索クエリの前処理を一元化し、数字正規化と条文参照抽出を統合。

### 3.2 実装

**ファイル**: `app/retrieval/query_processor.py`

```python
class QueryProcessor:
    """検索クエリの前処理"""
    
    def process(self, query: str) -> dict:
        """
        クエリを処理して検索に最適化
        
        Returns:
            {
                "original": 元のクエリ,
                "normalized": 正規化されたクエリ,
                "article_refs": 抽出された条文参照のリスト
            }
        """
        # 数字正規化
        normalized = normalize_article_numbers(query, to_kanji=True)
        
        # 条文参照抽出
        article_refs = extract_article_references(query)
        
        return {
            "original": query,
            "normalized": normalized,
            "article_refs": article_refs
        }
```

### 3.3 使用例

```python
processor = QueryProcessor()
result = processor.process("金融商品取引法第27条の5の規定により...")

# result = {
#     "original": "金融商品取引法第27条の5の規定により...",
#     "normalized": "金融商品取引法第二十七条の五の規定により...",
#     "article_refs": ["第二十七条の五"]
# }
```

---

## 4. Hybrid Retriever

### 4.1 目的

Vector検索（意味的類似性）とBM25検索（キーワードマッチング）を組み合わせて検索精度を向上。

### 4.2 実装

**ファイル**: `app/retrieval/hybrid_retriever.py`

```python
class HybridRetriever:
    """Vector + BM25 のハイブリッド検索"""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        ハイブリッド検索を実行
        
        1. Vector検索とBM25検索を並列実行
        2. RRF (Reciprocal Rank Fusion) でスコア統合
        3. 上位k件を返却
        """
        # Vector検索
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k*2)
        
        # BM25検索
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)
        
        # RRFでスコア統合
        combined = self._rrf_fusion(vector_results, bm25_results)
        
        return combined[:top_k]
    
    def _rrf_fusion(self, results1, results2, k=60):
        """Reciprocal Rank Fusion"""
        scores = {}
        
        for rank, doc in enumerate(results1):
            doc_id = hash(doc.page_content[:100])
            scores[doc_id] = scores.get(doc_id, 0) + self.vector_weight / (k + rank)
        
        for rank, doc in enumerate(results2):
            doc_id = hash(doc.page_content[:100])
            scores[doc_id] = scores.get(doc_id, 0) + self.bm25_weight / (k + rank)
        
        # スコア順にソート
        # ...
```

---

## 5. Multi-Query Retriever（実装済み、未使用）

### 5.1 目的

複数条文を参照する問題に対応するため、問題文から条文参照を抽出し、各条文で個別検索を実行。

### 5.2 実装

**ファイル**: `app/retrieval/multi_query_retriever.py`

```python
class MultiQueryRetriever:
    """複数クエリ検索"""
    
    def extract_article_references(self, question: str, choices: str = "") -> List[ArticleReference]:
        """問題文と選択肢から条文参照を抽出"""
        full_text = question + " " + choices
        references = []
        
        pattern = r'第(\d+)条(?:の(\d+))?'
        for match in re.finditer(pattern, full_text):
            article_num = match.group(1)
            suffix = match.group(2) or ""
            references.append(ArticleReference(
                law_name=self._extract_law_name(full_text),
                article_num=article_num,
                article_suffix=suffix
            ))
        
        return references
    
    def retrieve(self, question: str, choices: str = "", top_k: int = None) -> List:
        """
        複数クエリ検索を実行
        
        1. メインクエリ（問題文+選択肢）で検索
        2. 各条文参照で個別検索
        3. 結果を統合して重複排除
        """
        refs = self.extract_article_references(question, choices)
        
        # メインクエリで検索
        main_results = self.base_retriever.retrieve(main_query, top_k=self.per_article_top_k * 2)
        all_results.extend(main_results)
        
        # 各条文で個別検索
        for ref in refs[:5]:
            article_query = f"{ref.law_name} {ref.full_article}"
            article_results = self.base_retriever.retrieve(article_query, top_k=self.per_article_top_k)
            all_results.extend(article_results)
        
        # 重複排除してスコア順に返却
        return self._deduplicate_and_sort(all_results)[:top_k]
```

---

## 6. インデックス構成

### 6.1 XMLインデックスv2

| 項目 | 値 |
|------|-----|
| パス | `data/faiss_index_xml_v2/` |
| データソース | e-Gov法令XML（9法令） |
| 条文数 | 5,045 |
| Vector | FAISS (intfloat/multilingual-e5-large) |
| BM25 | rank_bm25 + sudachi tokenizer |

### 6.2 対象法令

| 法令ID | 法令名 | 条文数 |
|--------|--------|--------|
| 323AC0000000025 | 金融商品取引法 | 1,732 |
| 335AC0000000145 | 医薬品医療機器等法 | 245 |
| 336M50000100001 | 医薬品医療機器等法施行規則 | 646 |
| 340CO0000000321 | 金融商品取引法施行令 | 413 |
| 403AC0000000090 | 借地借家法 | 26 |
| 419M60000002052 | 有価証券の取引等の規制に関する内閣府令 | 84 |
| 419M60000002059 | 証券情報等の提供又は公表に関する内閣府令 | 19 |
| 420M60000002078 | 金融商品取引業等に関する内閣府令 | 8 |
| 429M60000002054 | 重要情報の公表に関する内閣府令 | 12 |

---

## 7. 検索品質の改善効果

### 7.1 コンテキスト一致率

| 段階 | 一致率 | 改善 |
|------|--------|------|
| Markdownインデックス | 7.9% | - |
| XMLインデックスv1 (1,669条文) | 47.1% | +39pt |
| **XMLインデックスv2 (5,045条文)** | **80.0%** | **+33pt** |

### 7.2 条文取得率（複数法令問題）

XMLパーサー修正後、複数法令問題（15問）で必要な条文の取得率を検証：

| 検索条件 | 取得率 |
|----------|--------|
| TOP-30 Hybrid検索 | **92.3%** (24/26条文) |

### 7.3 残る課題

| 問題タイプ | 問題数 | 対処 |
|------------|--------|------|
| ガイドライン参照 | 13問 | データ追加が必要 |
| 条文一部欠落 | 2問 | top_k増加で対応可能 |

---

## 8. ファイル一覧

| パス | 説明 |
|------|------|
| `app/utils/number_normalizer.py` | 数字正規化ユーティリティ |
| `app/retrieval/query_processor.py` | クエリ前処理 |
| `app/retrieval/vector_retriever.py` | Vector検索 |
| `app/retrieval/bm25_retriever.py` | BM25検索 |
| `app/retrieval/hybrid_retriever.py` | Hybrid検索 |
| `app/retrieval/multi_query_retriever.py` | 複数クエリ検索 |
| `scripts/parse_egov_xml.py` | XMLパーサー |
| `data/lawqa_xml_chunks_v2.jsonl` | XMLチャンクデータ |
| `data/faiss_index_xml_v2/` | インデックス（Vector + BM25） |

---

## 9. 実験スクリプト

| スクリプト | 説明 |
|------------|------|
| `run_baseline_xml_v2.py` | XMLインデックスv2でのベースラインRAGテスト |
| `run_proposed_v2.py` | 提案手法（マルチエージェント）のテスト |

---

*最終更新: 2025/12/03*

