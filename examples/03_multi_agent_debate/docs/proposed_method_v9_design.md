# 提案手法 v9: 構造化法令QAシステム（設計構想）

## 概要

v9は、v7の知見「Less is More + 難易度判断エージェント」を基盤に、
**条文の構造化**と**難易度別マルチパス**を組み合わせ、80%以上の精度を目指す発展版です。v9b-L 現行実装では「検証Agent」は撤廃し、分析+統合のシンプル2段で意思決定します。

### 改善の流れ（v4→v7→v9→v9b-L）
- v4: 単一パス + RAG で 73.6% 前後。
- v7: ルールベース難易度判定で SIMPLE/ MODERATE を分岐し 76.4%。
- v9: 構造化DB + 検証Agentを追加するも過剰検証で 67.9%。
- v9b: 検証Agentを外し、構造化DB + v7方式統合で 75.0%。
- v9b-L: 難易度判定をLLM化し、構造化サマリ/数値対比を分析に注入。最高時点 75.7%（8b）。直近の改変後は 70% 前後で停滞し、Few-shot強化と検索精緻化が課題。

### 核心的主張

```
┌─────────────────────────────────────────────────────────────┐
│  1. シンプルが最高（v7で実証済み）                          │
│  2. 難易度判断エージェントで過剰処理を回避（v7→v9b-L）     │
│  3. 日本語法令の構造化（数値・参照）で精度向上（v9系列）    │
│  4. 検証Agentを外し、分析+統合の2段で軽量化（v9b-L）        │
└─────────────────────────────────────────────────────────────┘
```

---

## 新規性と学会貢献ポイント

### 1. 難易度判断エージェントの有効性（v7で実証）

> **主張**: LLMに「難しさを判断するエージェント」を追加するだけで性能向上する

```python
# 従来手法
def answer(question):
    return complex_multi_agent_process(question)  # 全問題に同じ処理

# 提案手法
def answer(question):
    difficulty = DifficultyJudgeAgent.assess(question)
    if difficulty == SIMPLE:
        return simple_process(question)   # シンプルな問題にはシンプルな処理
    else:
        return complex_process(question)  # 複雑な問題にのみ多段階処理
```

**効果**: v4 (73.6%) → v7 (76.4%)、過剰処理による精度低下を防止

### 2. 日本語法令データの構造化（v9の新規性）

日本語法令には、英語法令にはない固有の課題があり、**専用の構造化**が必要：

| 課題 | 構造化による対策 |
|------|-----------------|
| **漢数字表記** | 正規化済みの数値を構造化保存 |
| **複雑な参照関係** | 参照グラフとして構造化 |
| **数値条件の厳密性** | 期間・割合・金額を分類して保存 |
| **条文間の関係性** | 関連条文リンクを構造化 |

### 3. 構造化データを用いた軽量な分析+統合（Hallucination抑制）

```
検索(Hybrid) → 構造化DB構築 → 難易度判定(LLM) → 
  SIMPLE: 直接回答
  MODERATE/COMPLEX: 分析Agent（数値・主体・法令種別を○/△/×評価）
                   → 統合（検証Agentなしで最終回答）
```

---

## アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│                  v9b-L（現行ベースライン）                   │
├──────────────────────────────────────────────────────────────┤
│ 質問 + 選択肢                                                 │
│    │ 正規化                                                   │
│    v                                                          │
│ Hybrid検索 (Vector+BM25, RRF/weighted RRF)                    │
│    │ 上位k                                                     │
│    v                                                          │
│ 構造化法令DBビルド（数値・参照抽出）                         │
│    │ numbers_table / structured_summary                       │
│    v                                                          │
│ 難易度判定Agent (LLM) → SIMPLE / MODERATE / COMPLEX           │
│    ├─ SIMPLE: 短プロンプトで直接回答                         │
│    └─ MODERATE/COMPLEX:                                       │
│         分析Agent（数値・主体・法令種別を○/△/×評価）        │
│         └→ 統合（検証Agentなしで最終回答を1回で決定）        │
│                                                              │
│ 結果 + ログ (JSON, log)                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 構造化法令DB

### データ構造

```python
@dataclass
class StructuredArticle:
    """構造化された条文"""
    article_id: str           # 第二十七条の二十三の三
    article_number: str       # 27-23-3（正規化済み）
    text: str                 # 条文本文
    law_type: str             # 本法/施行令/施行規則
    
    # 数値情報（構造化）
    numbers: NumberInfo
    
    # 参照関係
    references: List[Reference]
    referenced_by: List[str]  # この条文を参照している条文
    
    # 関連条文
    related_articles: List[str]

@dataclass
class NumberInfo:
    """条文内の数値情報"""
    periods: List[str]        # ["六十日以内", "三十日以内"]
    ratios: List[str]         # ["三分の二以上", "過半数"]
    amounts: List[str]        # ["一億円以上", "五億円"]
    counts: List[str]         # ["三人以上", "五人以下"]

@dataclass
class Reference:
    """参照関係"""
    ref_type: str             # 施行令/内閣府令/他法令
    target_article: str       # 参照先条文ID
    context: str              # 参照文脈（「政令で定めるところにより」等）
```

### 構築方法

```python
def build_structured_db(raw_articles: List[str]) -> Dict[str, StructuredArticle]:
    """法令XMLから構造化DBを構築"""
    db = {}
    
    for article in raw_articles:
        # 1. 条文番号の抽出と正規化
        article_id = extract_article_id(article)
        
        # 2. 数値情報の抽出
        numbers = extract_numbers(article)
        
        # 3. 参照関係の抽出
        references = extract_references(article)
        
        # 4. 構造化オブジェクトの作成
        db[article_id] = StructuredArticle(
            article_id=article_id,
            text=article,
            numbers=numbers,
            references=references,
            ...
        )
    
    # 5. 逆参照の構築
    build_reverse_references(db)
    
    return db
```

---

## 現行アルゴリズム詳細（v9b-L）

### 1. 難易度判定Agent（LLMベース）

```python
def assess_difficulty(question, choices):
    prompt = f"""
    あなたは法令QAの難易度を判定します。
    基準: SIMPLE / MODERATE / COMPLEX
    {question}
    {choices_text}
    難易度のみ答えてください。
    """
    label = llm.invoke(prompt)
    return parse_label(label)
```

- 役割: SIMPLE は短プロンプトで即答、MODERATE/COMPLEX は構造化分析パスへ送る。

### 2. 検索 + 構造化

```python
results = hybrid.retrieve(query_norm, top_k=k, fusion=rrf/weighted_rrf)
structured_db.build_from_documents(results)   # 数値・参照を抽出
numbers_table = structured_db.format_numbers_table(choices)
structured_summary = structured_db.get_structured_context()
```

- Hybrid (Vector+BM25, RRF/weighted RRF) で回収。
- StructuredLawDB が期間・割合・金額・人数と参照（施行令/規則/条文）を抽出し、分析プロンプトに渡す。

### 3. 分析Agent（MODERATE/COMPLEX）

```python
prompt = f"""
【検索条文】{context_snip}
{numbers_table}
【構造化サマリ】{structured_summary}
各選択肢について:
- 条文番号/法令種別 = ...
- 数値・期間の一致 = ○/△/×
- 対象範囲・主体 = ○/△/×
- 結論 = 一致/不一致/不明, 根拠 = ...
"""
analysis = llm.invoke(prompt)
```

- 数値・主体・法令種別の3観点で○/△/×を明示させ、根拠も短く残す。

### 4. 統合（検証Agentなし）

```python
prompt = f"""
分析結果を踏まえ、数値・期間 / 対象範囲・主体 / 法令種別が
最も整合する選択肢を a/b/c/d で1つだけ出力。
"""
answer = llm.invoke(prompt)
```

- 分析の要約（1500字トリム）を入力し、単一プロンプトで最終決定。検証Agentは用いない。

---

## 今後の改善候補（v9b-Lを起点）

- Few-shotのカテゴリ別強化（数値・期間 / 主体 / 法令種別）。
- 検索前処理の精緻化（法令名・条番号正規化、BM25寄りRRFのチューニング）。
- 軽量チェックリスト検証（数値/法令種別/主体のYes/Noを最終確認）を追加検討。
- complex向けのリコール改善: top_k拡大とnum_ctx延伸の併用。

---

## 参考

- v7ドキュメント: `docs/proposed_method_v7.md`
- v4ドキュメント: `docs/proposed_method_v4.md`
