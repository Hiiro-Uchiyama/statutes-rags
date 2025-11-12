# Legal Case Generator - 使用ガイド

本ドキュメントでは、Legal Case Generatorシステムのセットアップから評価まで、**上から順に実行するだけで完了する**手順を記載しています。

最終更新: 2025年11月7日

---

## 目次

1. [前提条件](#1-前提条件)
2. [プロジェクトルートのセットアップ](#2-プロジェクトルートのセットアップ)
3. [Legal Case Generator依存関係の確認](#3-legal-case-generator依存関係の確認)
4. [動作確認テスト](#4-動作確認テスト)
5. [単一法令での事例生成](#5-単一法令での事例生成)
6. [評価実行（複数法令）](#6-評価実行複数法令)
7. [結果分析](#7-結果分析)
8. [高度な使用方法](#8-高度な使用方法)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [4択問題からの事例生成（MCQモード）](#10-4択問題からの事例生成mcqモード)

---

## 1. 前提条件

### 必須環境

- プロジェクトルートのセットアップが完了していること
- 法令データとインデックスが構築されていること
- Ollamaがインストールされ、LLMモデルがダウンロード済みであること
- LangGraphとLangChain-Ollamaがインストール済みであること

### セットアップ未完了の場合

プロジェクトルートの[README.md](../../README.md)を参照して、まず基本セットアップを完了してください：

```bash
cd /path/to/statutes-rags

# 1. 環境構築（LangGraph含む全依存関係がインストールされます）
./setup/setup_uv_env.sh
source .venv/bin/activate

# 2. Ollamaセットアップ
./setup/setup_ollama.sh

# 3. データ準備（XMLファイルがある場合）
python scripts/preprocess_egov_xml.py

# 4. インデックス構築
python scripts/build_index.py --index-type vector
```

### セットアップ確認

以下のコマンドで必要なファイルとサービスが存在することを確認：

```bash
# プロジェクトルートから実行
ls -lh data/egov_laws.jsonl                          # 法令データ (1.8GB)
ls -lh data/faiss_index/vector/index.faiss          # ベクトルインデックス
ls -lh datasets/lawqa_jp/data/selection.json        # 評価データセット
ollama list | grep qwen3                             # Ollamaモデル確認
```

すべて存在すれば次のステップへ進めます。

---

## 2. プロジェクトルートのセットアップ

### 仮想環境の有効化

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 仮想環境を有効化
source .venv/bin/activate
```

### 環境変数の確認

`.env`ファイルが存在し、以下の設定が含まれていることを確認：

```bash
# .envファイルの確認
cat .env | grep -E "LLM_MODEL|OLLAMA_HOST"
```

期待される出力例：
```
LLM_MODEL=qwen3:8b
OLLAMA_HOST=http://localhost:11434
```

---

## 3. Legal Case Generator依存関係の確認

### LangGraphとLangChain-Ollamaの確認

プロジェクトルートで `./setup/setup_uv_env.sh` を実行済みであれば、LangGraphを含む全ての依存関係は既にインストール済みです。

```bash
# プロジェクトルートから実行
python3 -c "import langgraph; import langchain_ollama; print('✓ Dependencies installed')"
```

`✓ Dependencies installed` と表示されればOKです。

### 依存関係が不足している場合

```bash
# プロジェクトルートから実行
uv pip install -e ".[examples]"
```

**実行時間:** 約30秒～1分

---

## 4. 動作確認テスト

### 4.1 簡易動作確認（必須）

Legal Case Generatorパイプラインが正常に動作するか確認します。

```bash
# examples/04_legal_case_generator に移動
cd examples/04_legal_case_generator

# 簡易テストを実行（約10秒）
python3 tests/test_legal_case_generator.py
```

**期待される出力:**
```
============================================================
Legal Case Generator - Import and Configuration Test
============================================================

1. Testing imports...
   ✓ Config module imported successfully
   ✓ Pipeline module imported successfully
   ✓ All agent modules imported successfully

2. Testing configuration...
   ✓ Configuration loaded successfully
   LLM Model: qwen3:8b
   ...

============================================================
Test passed! ✓
============================================================
```

### トラブルシューティング（テストが失敗する場合）

#### エラー: "Module not found: langgraph"
```bash
# プロジェクトルートに戻って依存関係をインストール
cd ../../
uv pip install -e ".[examples]"
cd examples/04_legal_case_generator
```

#### エラー: "Ollama connection failed"
```bash
# Ollamaの状態を確認
curl http://localhost:11434/api/tags

# 応答がない場合はOllamaを起動
cd ../../
./setup/setup_ollama.sh
cd examples/04_legal_case_generator
```

---

## 5. 単一法令での事例生成

テストが成功したら、実際に法令条文から事例を生成してみます。

### 5.1 基本的な使い方

```bash
# examples/04_legal_case_generator ディレクトリから実行
python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。" \
  --output results/company_law_case.json
```

**実行時間:** 約2～5分（3つの事例タイプを生成）

**期待される出力:**
```
Generating cases for 会社法 Article 26...
  Processing case type: applicable
    Generated scenario (Score: 0.95, Valid: True)
  Processing case type: non_applicable
    Generated scenario (Score: 0.92, Valid: True)
  Processing case type: boundary
    Generated scenario (Score: 0.88, Valid: True)

✓ Generated 3 cases successfully
Results saved to: results/company_law_case.json
```

### 5.2 結果の確認

```bash
# JSONを整形して表示
cat results/company_law_case.json | python3 -m json.tool | head -50

# 生成された事例の概要を確認
cat results/company_law_case.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'法令: {r[\"law_info\"][\"law_title\"]} 第{r[\"law_info\"][\"article\"]}条')
print(f'生成事例数: {len(r[\"cases\"])}')
for case in r['cases']:
    print(f'  - {case[\"case_type\"]}: スコア {case[\"validation_score\"]:.2f}, 反復 {case[\"iterations\"]}回')
"
```

### 5.3 他の法令でも試す

```bash
# 民法の例
python3 pipeline.py \
  --law-number "明治二十九年法律第八十九号" \
  --law-title "民法" \
  --article "96" \
  --article-content "詐欺又は強迫による意思表示は、取り消すことができる。" \
  --output results/civil_law_case.json
```

---

## 6. 評価実行（複数法令）

単一法令での生成が成功したら、複数の法令で評価を実行します。

### 6.1 デフォルトサンプルで評価（3法令、約5～10分）

```bash
# examples/04_legal_case_generator ディレクトリから実行
python3 evaluate.py --output results/evaluation_results.json
```

**デフォルトのテストケース:**
- 会社法 第26条（株主名簿）
- 民法 第96条（詐欺・強迫）
- 個人情報保護法 第27条

**実行時間:** 約5～10分（9つの事例を生成）

**期待される出力:**
```
============================================================
Legal Case Generator - Evaluation
============================================================
Evaluating 3 test cases...

[1/3] 会社法 Article 26
  Processing case type: applicable
    ✓ Generated (Score: 0.95, Valid: True, Iterations: 0)
  Processing case type: non_applicable
    ✓ Generated (Score: 0.92, Valid: True, Iterations: 0)
  Processing case type: boundary
    ✓ Generated (Score: 0.88, Valid: True, Iterations: 1)
  Completed in 89.5s

[2/3] 民法 Article 96
  ...

[3/3] 個人情報保護法 Article 27
  ...

============================================================
Evaluation Summary
============================================================
Total Test Cases:       3
Total Cases Generated:  9
Success Count:          9
Success Rate:           100.0%
Total Time:             283.3s
Average Time per Case:  31.5s
Average Iterations:     0.44
============================================================
```

### 6.2 カスタムテストケースで評価

独自のテストケースを作成して評価することもできます。

```bash
# テストケースファイルを作成
cat > results/my_test_cases.json << 'EOF'
[
  {
    "law_number": "平成十七年法律第八十七号",
    "law_title": "会社法",
    "article": "26",
    "article_content": "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。"
  },
  {
    "law_number": "明治二十九年法律第八十九号",
    "law_title": "民法",
    "article": "96",
    "article_content": "詐欺又は強迫による意思表示は、取り消すことができる。"
  }
]
EOF

# カスタム評価を実行
python3 evaluate.py \
  --test-cases results/my_test_cases.json \
  --output results/my_evaluation.json
```

### 6.3 ワンコマンド評価（推奨）

全ての確認と評価を自動実行するスクリプトも用意されています：

```bash
./run_evaluation.sh
```

このスクリプトは以下を実行します：
1. Ollamaの起動確認
2. LLMモデルの確認
3. 基本テストの実行
4. 評価の実行（3法令、9事例）
5. 結果サマリーの表示

---

## 7. 結果分析

### 7.1 結果サマリーの確認

評価実行後、結果を確認します：

```bash
# 評価サマリーを表示
cat results/evaluation_results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(json.dumps(r['summary'], ensure_ascii=False, indent=2))
"
```

**出力例:**
```json
{
  "total_test_cases": 3,
  "total_cases_generated": 9,
  "success_count": 9,
  "success_rate": 1.0,
  "total_time": 283.3,
  "average_time_per_case": 31.5,
  "average_iterations": 0.44
}
```

### 7.2 個別事例の確認

生成された各事例の詳細を確認：

```bash
# 全体を整形表示
cat results/evaluation_results.json | python3 -m json.tool | less

# 各法令ごとの結果を確認
cat results/evaluation_results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
for result in r['results']:
    print(f\"\n法令: {result['law_title']} 第{result['article']}条\")
    print(f\"実行時間: {result['execution_time']:.1f}秒\")
    print(f\"生成事例数: {len(result['cases'])}\")
    for case in result['cases']:
        print(f\"  - {case['case_type']}: スコア {case['validation_score']:.2f}\")
"
```

### 7.3 検証スコアの分布確認

```bash
cat results/evaluation_results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
scores = []
for result in r['results']:
    for case in result['cases']:
        scores.append(case['validation_score'])
print(f'平均スコア: {sum(scores)/len(scores):.3f}')
print(f'最小スコア: {min(scores):.3f}')
print(f'最大スコア: {max(scores):.3f}')
print(f'スコア分布:')
for i, s in enumerate(scores):
    print(f'  事例{i+1}: {s:.3f}')
"
```

---

## 8. 高度な使用方法

### 8.1 環境変数による詳細設定

```bash
# 生成パラメータのカスタマイズ
export LEGAL_CASE_MIN_LENGTH=100
export LEGAL_CASE_MAX_LENGTH=500
export LEGAL_CASE_MAX_ITERATIONS=3
export LEGAL_CASE_VALIDATION_THRESHOLD=0.8

# LLM設定
export LLM_MODEL="qwen3:8b"
export LLM_TEMPERATURE=0.3
export LLM_TIMEOUT=180

# 事例タイプの選択（特定のタイプのみ生成）
export LEGAL_CASE_GEN_APPLICABLE=true
export LEGAL_CASE_GEN_NON_APPLICABLE=true
export LEGAL_CASE_GEN_BOUNDARY=true

python3 evaluate.py --output results/custom_eval.json
```

### 8.2 特定の事例タイプのみ生成

```bash
# 適用事例のみ生成
export LEGAL_CASE_GEN_APPLICABLE=true
export LEGAL_CASE_GEN_NON_APPLICABLE=false
export LEGAL_CASE_GEN_BOUNDARY=false

python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し..." \
  --output results/applicable_only.json
```

### 8.3 高品質モード（反復回数増加）

```bash
# より高品質な事例を生成（時間がかかります）
export LEGAL_CASE_MAX_ITERATIONS=3
export LLM_TEMPERATURE=0.2
export LEGAL_CASE_VALIDATION_THRESHOLD=0.8

python3 pipeline.py \
  --law-number "明治二十九年法律第八十九号" \
  --law-title "民法" \
  --article "96" \
  --article-content "詐欺又は強迫による意思表示は、取り消すことができる。" \
  --output results/high_quality.json
```

### 8.4 高速モード（反復なし）

```bash
# 検証の反復をスキップして高速化
export LEGAL_CASE_MAX_ITERATIONS=0
export LLM_TIMEOUT=60

python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し..." \
  --output results/fast_mode.json
```

### 8.5 異なるLLMモデルでの評価

```bash
# qwen3:14b（より大規模、高精度）
LLM_MODEL=qwen3:14b python3 evaluate.py \
  --output results/eval_qwen3_14b.json

# gpt-oss:20b（さらに大規模）
LLM_MODEL=gpt-oss:20b python3 evaluate.py \
  --output results/eval_gpt_oss_20b.json
```

### 8.6 人手評価用テンプレート生成

```bash
# 生成された事例を人手で評価するためのテンプレートを作成
python3 evaluate.py \
  --output results/eval_results.json \
  --generate-template results/human_eval_template.json
```

---

## 9. トラブルシューティング

### エラー: "Dataset not found"

```bash
# データセットの存在確認
ls -la ../../datasets/lawqa_jp/data/selection.json

# データセットがない場合
# プロジェクトルートのREADME.mdを参照してデータを準備
```

### エラー: "Index not found"

```bash
# インデックスの存在確認
ls -la ../../data/faiss_index/vector/

# インデックスがない場合は構築
cd ../../
python scripts/build_index.py --index-type vector
cd examples/04_legal_case_generator
```

### エラー: "Ollama connection failed"

```bash
# Ollamaサービスの状態確認
curl http://localhost:11434/api/tags

# Ollamaが起動していない場合
cd ../../
./setup/setup_ollama.sh

# モデルのダウンロード確認
ollama list
```

### エラー: "Module not found: langgraph"

```bash
# 依存関係をインストール
cd ../../
uv pip install -e ".[examples]"
cd examples/04_legal_case_generator
```

### タイムアウトエラー

```bash
# タイムアウトを延長
export LLM_TIMEOUT=180

python3 evaluate.py --output results/eval.json
```

### メモリ不足エラー

```bash
# 反復回数を減らす
export LEGAL_CASE_MAX_ITERATIONS=1

# または軽量モデルを使用
export LLM_MODEL="qwen3:8b"
```

### 処理が遅い場合

```bash
# 高速化オプション
export LEGAL_CASE_MAX_ITERATIONS=0
export LLM_TIMEOUT=60

python3 evaluate.py --output results/eval_fast.json
```

---

## コマンドリファレンス

### pipeline.py - 事例生成

**必須引数**:
```bash
--law-number    法令番号（例: "平成十七年法律第八十七号"）
--law-title     法令名（例: "会社法"）
--article       条文番号（例: "26"）
--article-content 条文内容（全文）
```

**オプション引数**:
```bash
--output        出力ファイルパス（省略時は標準出力）
```

**実行例**:
```bash
python3 pipeline.py \
  --law-number "明治二十九年法律第八十九号" \
  --law-title "民法" \
  --article "96" \
  --article-content "詐欺又は強迫による意思表示は、取り消すことができる。" \
  --output civil_law_case.json
```

### evaluate.py - 評価実行

**オプション引数**:
```bash
--test-cases         カスタムテストケースのJSONファイル
--output             評価結果の出力ファイル（デフォルト: 標準出力）
--generate-template  人手評価用テンプレートの生成
```

**実行例:**
```bash
# デフォルトサンプルで評価
python3 evaluate.py --output results/eval_results.json

# カスタムテストケースで評価
python3 evaluate.py \
  --test-cases results/my_test_cases.json \
  --output results/my_eval.json

# 人手評価テンプレート生成
python3 evaluate.py \
  --output results/eval_results.json \
  --generate-template results/human_eval_template.json
```

### よく使うコマンド

```bash
# 仮想環境の有効化
source ../../.venv/bin/activate

# 簡易テスト
python3 tests/test_legal_case_generator.py

# 単一法令で事例生成
python3 pipeline.py \
  --law-number "..." \
  --law-title "..." \
  --article "..." \
  --article-content "..." \
  --output results/case.json

# デフォルト評価
python3 evaluate.py --output results/eval.json

# ワンコマンド評価
./run_evaluation.sh

# 結果サマリー表示
cat results/eval.json | python3 -c "import sys, json; r=json.load(sys.stdin); print(json.dumps(r['summary'], ensure_ascii=False, indent=2))"
```

### 環境変数クイックリファレンス

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `LEGAL_CASE_MIN_LENGTH` | 100 | 事例の最小文字数 |
| `LEGAL_CASE_MAX_LENGTH` | 500 | 事例の最大文字数 |
| `LEGAL_CASE_MAX_ITERATIONS` | 2 | 最大反復回数 |
| `LEGAL_CASE_VALIDATION_THRESHOLD` | 0.7 | 検証スコアの閾値 |
| `LEGAL_CASE_GEN_APPLICABLE` | true | 適用事例を生成 |
| `LEGAL_CASE_GEN_NON_APPLICABLE` | true | 非適用事例を生成 |
| `LEGAL_CASE_GEN_BOUNDARY` | true | 境界事例を生成 |
| `LLM_MODEL` | qwen3:8b | 使用LLMモデル |
| `LLM_TEMPERATURE` | 0.3 | 温度パラメータ |
| `LLM_TIMEOUT` | 120 | タイムアウト（秒） |
| `OLLAMA_HOST` | http://localhost:11434 | Ollamaホスト |

---

## 10. 4択問題からの事例生成（MCQモード）

LangGraphワークフローにMCQモードが追加され、4択問題（LAW QA）を入力に正解肢を裏付ける約500文字の事例を生成できます。  
入力データには `datasets/lawqa_jp/data/selection.json` などの4択問題データセットを利用します。

### 10.1 単一問題を処理する

```bash
cd examples/04_legal_case_generator

python3 pipeline.py mcq \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --index 0 \
  --count 1 \
  --output results/mcq_case_000.json
```

- `--index`: 生成を開始する問題番号（0始まり）
- `--count`: 連続して処理する件数（省略時1件）
- `--output`: 出力ファイル。省略時は標準出力にJSONを表示

### 10.2 出力例

```json
{
  "dataset": ".../selection.json",
  "start_index": 0,
  "count": 1,
  "results": [
    {
      "question_id": "金商法_第2章_選択式_関連法令_問題番号57",
      "question": "金融商品取引法第5条第6項により...",
      "choices": {
        "a": "第5条第1項の届出書に類する書類であって...",
        "b": "外国において開示予定の参照書類であって...",
        "c": "第5条第1項の届出書に類する書類であって、英語で記載されているもの",
        "d": "外国において開示が行われている参照書類であって、日本語で記載されているもの"
      },
      "correct_choice": "c",
      "scenario": "（約500文字の具体事例。正解肢や条文番号を直接示さず、条文要件と事実で結論を描写する）",
      "character_count": 502,
      "is_valid": true,
      "validation_score": 0.92,
      "feedback": [],
      "iterations": 0,
      "agents_used": ["mcq_parser", "mcq_case_generator", "mcq_checker"]
    }
  ]
}
```

### 10.3 動作のポイント

- 文字数は既定で **460〜540文字**（環境変数 `MCQ_CASE_MIN_LENGTH` / `MCQ_CASE_MAX_LENGTH` で調整可能）
- シナリオ本文で「選択肢」「正解」「回答」など問題形式を示唆する語を使用しない
- 「第〇条」「第〇項」「第〇号」等の条文番号・項号を直接引用せず、制度趣旨を言い換えて説明する
- 正解肢の文面を逐語的に引用せず、条文要件を満たす具体的事実で裏付ける
- 整合性チェックに失敗した場合は `MCQRefinerAgent` がフィードバックを基に再生成
- `--count` を増やすと連続した複数問題をまとめて処理可能
- `results[*].feedback` に失敗時の改善ヒントが格納されます

---

## 出力フォーマット

### 生成結果 (pipeline.py)

```json
{
  "law_info": {
    "law_number": "平成十七年法律第八十七号",
    "law_title": "会社法",
    "article": "26",
    "article_content": "株式会社は、株主名簿を作成し..."
  },
  "cases": [
    {
      "case_type": "applicable",
      "scenario": "（適用される具体的な事例）",
      "legal_analysis": "（法的分析）",
      "educational_point": "（学習ポイント）",
      "is_valid": true,
      "validation_score": 0.95,
      "iterations": 0,
      "agents_used": ["scenario_generator", "legal_checker"]
    },
    {
      "case_type": "non_applicable",
      "...": "..."
    },
    {
      "case_type": "boundary",
      "...": "..."
    }
  ]
}
```

### 評価結果 (evaluate.py)

```json
{
  "summary": {
    "total_test_cases": 3,
    "total_cases_generated": 9,
    "success_count": 9,
    "success_rate": 1.0,
    "total_time": 283.3,
    "average_time_per_case": 31.5,
    "average_iterations": 0.44
  },
  "results": [
    {
      "law_title": "会社法",
      "article": "26",
      "cases": [...],
      "execution_time": 89.5
    }
  ]
}
```

---

## パフォーマンス目安

- **生成時間**: 平均31.5秒/事例
  - 適用事例: 3-10秒
  - 非適用事例: 5-15秒
  - 境界事例: 10-40秒

- **検証スコア**: 平均0.937（閾値0.7）

- **成功率**: 100%（標準設定）

---

## 次のステップ

1. **基本動作の確認**
   - テストの実行
   - 単一法令での事例生成
   - 結果の確認

2. **評価実験の実行**
   - デフォルトサンプルでの評価
   - カスタムテストケースでの評価
   - 結果分析

3. **パラメータチューニング**
   - 異なる温度パラメータの試行
   - 反復回数の調整
   - 異なるLLMモデルでの比較

4. **人手評価**
   - 人手評価テンプレートの生成
   - 生成された事例の質的評価
   - フィードバックの収集

## 関連ドキュメント

- [README.md](README.md) - プロジェクト概要と詳細ガイド
- [docs/testing_report.md](docs/testing_report.md) - 詳細なテストレポート
- [docs/verification_results.md](docs/verification_results.md) - 動作確認結果

## 注意事項

⚠️ **重要**: 生成される事例は教育目的の参考資料です
- 法的判断の根拠として使用しないでください
- 専門家による検証は行われていません
- LLMの出力は非決定的（同じ入力でも異なる結果）
- 実際の法的問題には専門家に相談してください

---

最終更新: 2025年11月7日  
**重要:** 上から順に実行すれば、セットアップから評価まで完了します。
