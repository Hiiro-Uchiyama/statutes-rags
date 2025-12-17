# 研究貢献: 法令4択QAに対するマルチエージェントアプローチ

## 新規性の方向性

### 制約条件

- 大規模LLM・Fine-tuningに依存しない
- 法令ドメインに特化した独自性
- 実用的な精度向上を伴う

---

## 提案1: Citation-Grounded Multi-Agent Reasoning (CG-MAR)

### 概要

法令QAにおいて、**全ての推論ステップに条文引用を必須とする**マルチエージェントフレームワーク。

### 新規性

1. **引用強制推論 (Citation-Forced Reasoning)**
   - 各選択肢の判断に対して、根拠となる条文を明示的に要求
   - 引用なしの判断を無効とする制約

2. **引用妥当性検証エージェント (Citation Validator Agent)**
   - 引用された条文が実際に存在するか検証
   - 引用内容と判断の整合性を確認

3. **法令構造認識 (Legal Structure Awareness)**
   - 法律→施行令→規則の階層を認識
   - 参照関係を自動追跡

### アーキテクチャ

```
[質問]
   │
   v
[RetrieverAgent] ─── 条文検索 + 参照関係抽出
   │
   v
[ReasonerAgent] ─── 選択肢ごとの判断 + 条文引用
   │
   ├── 選択肢a: 判断 + 引用条文
   ├── 選択肢b: 判断 + 引用条文
   ├── 選択肢c: 判断 + 引用条文
   └── 選択肢d: 判断 + 引用条文
   │
   v
[CitationValidator] ─── 引用の妥当性検証
   │
   v
[IntegratorAgent] ─── 検証済み判断の統合
   │
   v
[最終回答]
```

### 学術的貢献

- 法令QAにおけるHallucination（幻覚）の抑制
- 判断根拠の透明性・追跡可能性
- 引用ベースの信頼性評価メトリクス

---

## 提案2: Hierarchical Legal Reference Tracking (HLRT)

### 概要

法令の**階層的参照関係**を追跡し、複数条文問題を体系的に解決する手法。

### 新規性

1. **参照グラフの動的構築**
   - 「政令で定める」→ 施行令への自動展開
   - 条文間の参照関係をグラフ化

2. **階層的検索戦略**
   - 第1層: 直接関連条文
   - 第2層: 参照先条文（施行令等）
   - 第3層: 間接参照条文

3. **参照完全性検証**
   - 必要な参照が全て解決されているか確認
   - 未解決参照がある場合は追加検索

### アーキテクチャ

```
[質問]
   │
   v
[PrimarySearch] ─── 直接関連条文の検索
   │
   v
[ReferenceExtractor] ─── 参照パターンの抽出
   │                     - 「政令で定める」
   │                     - 「第X条の規定」
   │                     - 「施行令第X条」
   │
   v
[SecondarySearch] ─── 参照先条文の検索
   │
   v
[ReferenceGraph] ─── 参照関係のグラフ化
   │
   v
[CompletenessCheck] ─── 参照完全性の検証
   │
   v
[ReasoningAgent] ─── 統合コンテキストでの推論
   │
   v
[最終回答]
```

### 学術的貢献

- 法令の階層構造を活用した検索戦略
- 参照関係の自動追跡メカニズム
- 複数条文問題への体系的アプローチ

---

## 提案3: Contrastive Choice Verification (CCV)

### 概要

4択QAに特化した**対照的選択肢検証**フレームワーク。

### 新規性

1. **選択肢間の対照分析**
   - 選択肢同士の違いを明示的に抽出
   - 差分に焦点を当てた検証

2. **消去法的推論の体系化**
   - 明らかに誤りの選択肢を先に除外
   - 残った選択肢の詳細比較

3. **質問タイプ適応的検証**
   - 「誤り選択」: 不一致を探す
   - 「正しい選択」: 完全一致を確認
   - 「組み合わせ」: 複合条件の検証

### アーキテクチャ

```
[質問 + 選択肢]
   │
   v
[QuestionTypeClassifier] ─── 質問タイプの判定
   │
   v
[ChoiceDiffExtractor] ─── 選択肢間の差分抽出
   │
   ├── a vs b: 差分1
   ├── a vs c: 差分2
   ├── ...
   │
   v
[FocusedRetrieval] ─── 差分に関連する条文を重点検索
   │
   v
[ContrastiveVerifier] ─── 対照的検証
   │
   ├── Round 1: 明らかな誤りの除外
   ├── Round 2: 残り選択肢の詳細比較
   │
   v
[最終回答]
```

### 学術的貢献

- 4択QAに特化したアーキテクチャ
- 選択肢間の差分に基づく効率的検証
- 消去法的推論の形式化

---

## 推奨: 提案の組み合わせ

### 統合アーキテクチャ: CLMR (Citation-grounded Legal Multi-agent Reasoning)

```
┌─────────────────────────────────────────────────┐
│  CLMR: Citation-grounded Legal Multi-agent      │
│         Reasoning Framework                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  [Stage 1: Question Analysis]                   │
│      QuestionTypeClassifier                      │
│      ChoiceDiffExtractor                         │
│                                                  │
│  [Stage 2: Hierarchical Retrieval]              │
│      PrimarySearch → ReferenceExtractor          │
│      → SecondarySearch → ReferenceGraph          │
│                                                  │
│  [Stage 3: Citation-Grounded Reasoning]         │
│      ReasonerAgent (with forced citation)        │
│      CitationValidator                           │
│                                                  │
│  [Stage 4: Contrastive Verification]            │
│      ContrastiveVerifier                         │
│      IntegratorAgent                             │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 主要な貢献（論文向け）

1. **法令QA特化のマルチエージェントフレームワーク**
   - 法令の構造的特性を活用
   - 4択形式に最適化

2. **Citation-Grounded推論**
   - 全判断に条文引用を要求
   - Hallucinationの抑制

3. **階層的参照追跡**
   - 法律→施行令→規則の自動展開
   - 複数条文問題への対応

4. **対照的選択肢検証**
   - 選択肢間差分の活用
   - 効率的な絞り込み

---

## 実装計画

### Phase 1: 複数条文問題への対応（提案2の実装）

```python
# ReferenceExtractor
class ReferenceExtractor:
    def extract_references(self, text):
        # 「政令で定める」→ 施行令を検索
        # 「第X条の規定」→ 同法の条文を検索
        pass

# SecondarySearch
class SecondarySearch:
    def search_references(self, references):
        # 参照先を検索
        pass
```

### Phase 2: Citation-Grounded推論（提案1の実装）

```python
# CitationValidator
class CitationValidator:
    def validate_citation(self, citation, context):
        # 引用が実際に存在するか
        # 引用内容と判断が整合するか
        pass
```

### Phase 3: 対照的検証（提案3の実装）

```python
# ChoiceDiffExtractor
class ChoiceDiffExtractor:
    def extract_diffs(self, choices):
        # 選択肢間の差分を抽出
        pass

# ContrastiveVerifier
class ContrastiveVerifier:
    def verify_contrastively(self, choices, context):
        # 対照的検証
        pass
```

---

## 期待される効果

| 改善項目 | 現状 | 期待 | 根拠 |
|----------|------|------|------|
| 複数条文問題 | 低精度 | +10-15% | 参照追跡 |
| Hallucination | 発生あり | 大幅減少 | 引用強制 |
| 判断透明性 | 低 | 高 | 引用記録 |

---

## 論文構成案

### タイトル案

- "CLMR: Citation-grounded Legal Multi-agent Reasoning for Statutory QA"
- "Hierarchical Reference Tracking for Multi-article Legal Question Answering"

### 構成

1. Introduction: 法令QAの課題
2. Related Work: RAG, Multi-agent, Legal NLP
3. Proposed Method: CLMR Framework
4. Experiments: 法令4択QAデータセット
5. Results: ベースライン比較、アブレーション
6. Analysis: 引用妥当性、参照追跡の効果
7. Conclusion

### 強調ポイント

- **ドメイン特化**: 法令の構造的特性を活用
- **解釈可能性**: 引用ベースの判断根拠
- **実用性**: 小規模LLMでも効果的

