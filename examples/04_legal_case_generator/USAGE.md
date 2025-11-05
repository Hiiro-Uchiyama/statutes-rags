# Legal Case Generator - 使い方

法令条文から具体的な適用事例を自動生成するシステムの使用ガイド。

## クイックスタート

### 0. ワンコマンド評価（推奨）

全ての確認と評価を自動実行：

```bash
cd examples/04_legal_case_generator
./run_evaluation.sh
```

このスクリプトは以下を実行します：
1. Ollamaの起動確認
2. LLMモデルの確認
3. 基本テストの実行
4. 評価の実行（3法令、9事例、約5-10分）
5. 結果サマリーの表示

### 1. 基本的な使い方（1つの法令で事例生成）

```bash
cd examples/04_legal_case_generator

python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。" \
  --output result.json

# 結果の確認
cat result.json | python3 -m json.tool
```

**出力**: 3種類の事例（適用・非適用・境界）がJSON形式で保存されます。

### 2. 評価の実行（複数の法令でテスト）

```bash
# デフォルトのサンプルケース（3法令）で評価
python3 evaluate.py --output evaluation_results.json

# 結果のサマリー表示
cat evaluation_results.json | python3 -c "import sys, json; r=json.load(sys.stdin); print(json.dumps(r['summary'], ensure_ascii=False, indent=2))"
```

**サンプルケース**:
- 会社法 第26条（株主名簿）
- 民法 第96条（詐欺・強迫）
- 個人情報保護法 第27条

### 3. テストの実行

```bash
# インポートと基本機能のテスト
python3 tests/test_legal_case_generator.py
```

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
--test-cases    カスタムテストケースのJSONファイル
--output        評価結果の出力ファイル（デフォルト: 標準出力）
--generate-template  人手評価用テンプレートの生成
```

**デフォルトサンプルで評価**:
```bash
python3 evaluate.py --output eval_results.json
```

**カスタムテストケースで評価**:
```bash
# テストケースファイル作成
cat > my_test_cases.json << 'EOF'
[
  {
    "law_number": "平成十七年法律第八十七号",
    "law_title": "会社法",
    "article": "26",
    "article_content": "株式会社は、株主名簿を作成し..."
  }
]
EOF

# 評価実行
python3 evaluate.py --test-cases my_test_cases.json --output my_eval.json
```

**人手評価テンプレート生成**:
```bash
python3 evaluate.py \
  --output eval_results.json \
  --generate-template human_eval_template.json
```

## 環境変数による設定

### 生成パラメータ

```bash
# 事例の長さ制限
export LEGAL_CASE_MIN_LENGTH=100
export LEGAL_CASE_MAX_LENGTH=500

# 最大反復回数（洗練プロセス）
export LEGAL_CASE_MAX_ITERATIONS=2

# 事例タイプの選択
export LEGAL_CASE_GEN_APPLICABLE=true      # 適用事例
export LEGAL_CASE_GEN_NON_APPLICABLE=true  # 非適用事例
export LEGAL_CASE_GEN_BOUNDARY=true        # 境界事例
```

### LLM設定

```bash
# モデル選択
export LLM_MODEL="gpt-oss:20b"

# 温度パラメータ（0.0-1.0）
export LLM_TEMPERATURE=0.3

# タイムアウト（秒）
export LLM_TIMEOUT=120
```

### 検証設定

```bash
# 検証スコアの閾値（0.0-1.0）
export LEGAL_CASE_VALIDATION_THRESHOLD=0.7
```

### Ollama設定

```bash
# Ollamaホスト
export OLLAMA_HOST="http://localhost:11434"
```

## 実行例

### 例1: 適用事例のみ生成

```bash
export LEGAL_CASE_GEN_APPLICABLE=true
export LEGAL_CASE_GEN_NON_APPLICABLE=false
export LEGAL_CASE_GEN_BOUNDARY=false

python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し..." \
  --output applicable_only.json
```

### 例2: 高品質モード（反復回数増加）

```bash
export LEGAL_CASE_MAX_ITERATIONS=3
export LLM_TEMPERATURE=0.2
export LEGAL_CASE_VALIDATION_THRESHOLD=0.8

python3 pipeline.py \
  --law-number "明治二十九年法律第八十九号" \
  --law-title "民法" \
  --article "96" \
  --article-content "詐欺又は強迫による意思表示は、取り消すことができる。" \
  --output high_quality.json
```

### 例3: 高速モード（反復なし）

```bash
export LEGAL_CASE_MAX_ITERATIONS=0
export LLM_TIMEOUT=60

python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し..." \
  --output fast_mode.json
```

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

## トラブルシューティング

### Ollamaに接続できない

```bash
# Ollamaの起動確認
curl http://localhost:11434/api/tags

# 起動していない場合
ollama serve
```

### モジュールインポートエラー

**エラー**: `SyntaxError: invalid decimal literal`

**原因**: Python 3.xでは数字で始まるモジュール名を直接インポートできない

**解決済み**: 全てのファイルで `importlib` による動的インポートに修正済み

### タイムアウトエラー

```bash
# タイムアウトを延長
export LLM_TIMEOUT=180

python3 pipeline.py ...
```

### メモリ不足

```bash
# 反復回数を減らす
export LEGAL_CASE_MAX_ITERATIONS=1

# または軽量モデルを使用
export LLM_MODEL="smaller-model"
```

## パフォーマンス目安

- **生成時間**: 平均31.5秒/事例
  - 適用事例: 3-10秒
  - 非適用事例: 5-15秒
  - 境界事例: 10-40秒

- **検証スコア**: 平均0.937（閾値0.7）

- **成功率**: 100%（標準設定）

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
