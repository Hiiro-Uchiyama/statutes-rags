# Legal Case Generator - 法的事例生成システム

法令条文から具体的な適用事例を自動生成する教育的ツールです。

## ドキュメント

- **[USAGE.md](USAGE.md)** - 使い方ガイド（コマンドリファレンス）
- **[docs/testing_report.md](docs/testing_report.md)** - 詳細テストレポート
- **[docs/verification_results.md](docs/verification_results.md)** - 動作確認結果

## 目次

- [概要](#概要)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [評価](#評価)
- [設定](#設定)
- [トラブルシューティング](#トラブルシューティング)

## 概要

ユーザーが指定した法令条文について、以下の3種類の事例を自動生成します:

1. **適用事例**: 法令が明確に適用されるケース
2. **非適用事例**: 法令が適用されないケース
3. **境界事例**: 判断が分かれる可能性があるケース

### 特徴

- LangGraphベースのマルチエージェントシステム
- 法的整合性の自動検証
- 教育的価値を重視した事例生成
- 反復的な洗練プロセス

### エージェント構成

- **Scenario Generator**: 初期事例シナリオを生成
- **Legal Checker**: 法的整合性を検証
- **Refiner**: フィードバックに基づき事例を洗練

### ワークフロー

```
法令条文の取得
    ↓
Scenario Generator（事例生成）
    ↓
Legal Checker（整合性検証）
    ↓
整合性OK? → No → Refiner（修正）
    ↓ Yes           ↓
    ↓         再検証必要?
    ↓               ↓
    ↓←──────────────┘
    ↓
最終事例の整形
```

## セットアップ

### 前提条件の確認

既に以下がセットアップ済みであることを前提とします:

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 既存のセットアップスクリプトを実行済み
# ./setup/setup_uv_env.sh  # Python環境 + 全依存関係（LangGraph含む）
# ./setup/setup_ollama.sh  # Ollama + qwen3:8b

# uvとvenvが設定済みか確認
source .venv/bin/activate
which python3
# -> .venv/bin/python3 が表示されればOK

# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags
```

**重要**: `setup/setup_uv_env.sh` を実行済みであれば、LangGraphを含む全ての依存関係は自動的にインストール済みです。追加のインストール作業は不要です。

### 依存関係の追加インストール（必要な場合のみ）

環境構築を個別に行う場合は以下を実行：

```bash
cd /path/to/statutes-rags

# examplesグループを含めてインストール（LangGraphを含む）
uv pip install -e ".[examples]"
```

**注**: LangGraphは`pyproject.toml`の`optional-dependencies.examples`に含まれています。

### Ollamaの起動確認

```bash
# Ollamaが起動していることを確認
curl http://localhost:11434/api/tags

# qwen3:8bモデルがあることを確認
ollama list | grep gpt-oss
```

## 使用方法

### コマンドライン使用

```bash
cd examples/04_legal_case_generator

# 単一の法令条文について事例を生成
python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。" \
  --output result.json

# 結果の確認
cat result.json | python3 -m json.tool
```

### Pythonスクリプトでの使用

```python
import sys
from pathlib import Path
import importlib

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 動的インポート（ディレクトリ名に数字が含まれるため）
pipeline_module = importlib.import_module("examples.04_legal_case_generator.pipeline")
config_module = importlib.import_module("examples.04_legal_case_generator.config")

LegalCaseGenerator = pipeline_module.LegalCaseGenerator
load_config = config_module.load_config

# 設定のロード
config = load_config()

# ジェネレータの初期化
generator = LegalCaseGenerator(config)

# 事例生成
result = generator.generate_cases(
    law_number="平成十七年法律第八十七号",
    law_title="会社法",
    article="26",
    article_content="株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。"
)

# 結果の確認
print(f"生成された事例数: {len(result['cases'])}")
for case in result['cases']:
    print(f"\n事例タイプ: {case['case_type']}")
    print(f"シナリオ: {case['scenario'][:100]}...")
    print(f"検証スコア: {case['validation_score']:.2f}")
```

### 結果の読み方

生成された結果には以下が含まれます:

```json
{
  "law_info": {
    "law_number": "平成十七年法律第八十七号",
    "law_title": "会社法",
    "article": "26",
    "article_content": "..."
  },
  "cases": [
    {
      "case_type": "applicable",
      "scenario": "（適用される具体的な事例）",
      "legal_analysis": "（なぜ適用されるかの分析）",
      "educational_point": "（この事例から学べること）",
      "is_valid": true,
      "validation_score": 0.85,
      "iterations": 1
    },
    {
      "case_type": "non_applicable",
      "..."
    },
    {
      "case_type": "boundary",
      "..."
    }
  ]
}
```

## 評価

### 自動評価

```bash
cd examples/04_legal_case_generator

# デフォルトのサンプルケースで評価
python3 evaluate.py --output evaluation_results.json

# カスタムテストケースで評価
python3 evaluate.py \
  --test-cases my_test_cases.json \
  --output evaluation_results.json

# 人手評価用テンプレートも生成
python3 evaluate.py \
  --output evaluation_results.json \
  --generate-template human_evaluation_template.json
```

### テストケースのフォーマット

`my_test_cases.json`:
```json
[
  {
    "law_number": "平成十七年法律第八十七号",
    "law_title": "会社法",
    "article": "26",
    "article_content": "株式会社は、株主名簿を作成し..."
  },
  {
    "law_number": "明治二十九年法律第八十九号",
    "law_title": "民法",
    "article": "96",
    "article_content": "詐欺又は強迫による意思表示は..."
  }
]
```

### 評価結果の確認

```bash
# サマリーを表示（評価スクリプト実行時に自動表示）
cat evaluation_results.json | python3 -c "import sys, json; r=json.load(sys.stdin); print(json.dumps(r['summary'], ensure_ascii=False, indent=2))"

# 詳細結果の確認
cat evaluation_results.json | python3 -m json.tool
```

評価結果:

```json
{
  "summary": {
    "total_test_cases": 3,
    "total_cases_generated": 9,
    "success_count": 9,
    "success_rate": 1.0,
    "total_time": 125.3,
    "average_time_per_case": 13.9,
    "average_iterations": 1.2
  },
  "results": [...]
}
```

## 設定

環境変数で設定をカスタマイズできます:

```bash
# 生成設定
# 注: 現在は3つの事例タイプ（適用・非適用・境界）が固定で生成されます
# export LEGAL_CASE_MAX_CASES=3
export LEGAL_CASE_MIN_LENGTH=100
export LEGAL_CASE_MAX_LENGTH=500
export LEGAL_CASE_MAX_ITERATIONS=2

# 事例タイプの制御
export LEGAL_CASE_GEN_APPLICABLE=true
export LEGAL_CASE_GEN_NON_APPLICABLE=true
export LEGAL_CASE_GEN_BOUNDARY=true

# LLM設定
export LLM_MODEL="qwen3:8b"
export LLM_TEMPERATURE=0.3
export LLM_TIMEOUT=120

# 検証設定
export LEGAL_CASE_VALIDATION_THRESHOLD=0.7

# Ollamaホスト
export OLLAMA_HOST="http://localhost:11434"
```

### 単一事例タイプのみ生成

```bash
# 適用事例のみ生成
export LEGAL_CASE_GEN_APPLICABLE=true
export LEGAL_CASE_GEN_NON_APPLICABLE=false
export LEGAL_CASE_GEN_BOUNDARY=false

python3 pipeline.py ...
```

### カスタム設定での実行

```python
config_module = importlib.import_module("examples.04_legal_case_generator.config")
LegalCaseConfig = config_module.LegalCaseConfig

# カスタム設定
config = LegalCaseConfig(
    max_iterations=3,
    llm_temperature=0.4,
    generate_boundary=False  # 境界事例は生成しない
)

generator = LegalCaseGenerator(config)
```

## トラブルシューティング

### Ollamaに接続できない

```bash
# Ollamaの起動を確認
ollama serve

# 別のターミナルで確認
curl http://localhost:11434/api/tags
```

### モジュールが見つからない

```bash
# プロジェクトルートから実行していることを確認
pwd
# /path/to/statutes-rags であるべき

# Pythonパスの確認
python3 -c "import sys; print('\n'.join(sys.path))"
```

### LangGraphがインストールできない

```bash
# 再試行
uv pip install --upgrade langgraph

# または直接pipを使用
pip install langgraph
```

### Ollamaが応答しない

```bash
# Ollamaの再起動
cd setup
./setup_ollama.sh

# または手動起動
./bin/ollama serve > ollama.log 2>&1 &
```

### メモリ不足エラー

```bash
# LLMのタイムアウトを増やす
export LLM_TIMEOUT=180

# または反復回数を減らす
export LEGAL_CASE_MAX_ITERATIONS=1
```

### 生成が遅い

- `LLM_TIMEOUT`を増やす（デフォルト: 120秒）
- `LEGAL_CASE_MAX_ITERATIONS`を減らす（デフォルト: 2）
- より軽量なモデルを使用する

### 生成品質が低い

- `LLM_TEMPERATURE`を調整（0.1-0.5の範囲）
- `LEGAL_CASE_MAX_ITERATIONS`を増やす
- プロンプトを調整（`agents/*.py`を編集）

## 高度な使用方法

### プロンプトの調整

より良い事例を生成するためにプロンプトを調整できます:

```bash
# エディタで開く
vim agents/scenario.py

# _create_prompt メソッドを編集
# 例: より具体的な指示を追加、文字数制限を調整など
```

### 検証基準の調整

```bash
# Legal Checkerの検証基準を調整
vim agents/legal_checker.py

# _create_validation_prompt メソッドを編集
# 例: 検証項目の追加、スコアリング基準の変更など
```

### 評価指標の追加

```bash
# 評価スクリプトにカスタム指標を追加
vim evaluate.py

# LegalCaseEvaluator クラスに新しいメソッドを追加
```

## 実験例

### 実験例1: 複数の法令で評価

```bash
# 複数の法令を含むテストケースを作成
cat > multi_law_test.json << 'EOF'
[
  {"law_number": "...", "law_title": "会社法", "article": "26", ...},
  {"law_number": "...", "law_title": "民法", "article": "96", ...},
  {"law_number": "...", "law_title": "個人情報保護法", "article": "27", ...}
]
EOF

# 評価実行
python3 evaluate.py --test-cases multi_law_test.json --output multi_eval.json
```

### 実験例2: 温度パラメータの最適化

```bash
# 異なる温度で複数回生成して比較
for temp in 0.1 0.3 0.5; do
  export LLM_TEMPERATURE=$temp
  python3 pipeline.py ... --output result_temp_${temp}.json
done

# 結果を比較
```

### 実験例3: プロンプトエンジニアリング

1. `agents/scenario.py`のプロンプトを調整
2. 同じ法令で再生成
3. `validation_score`の変化を観察
4. より高スコアのプロンプトを採用

## ディレクトリ構成

```
04_legal_case_generator/
├── __init__.py
├── README.md                     # このファイル
├── config.py                     # 設定管理
├── pipeline.py                   # メインパイプライン
├── evaluate.py                   # 評価スクリプト
├── agents/
│   ├── __init__.py
│   ├── scenario.py               # シナリオ生成エージェント
│   ├── legal_checker.py          # 法的整合性検証エージェント
│   └── refiner.py                # 洗練エージェント
└── tests/
    ├── __init__.py
    ├── conftest.py               # pytestフィクスチャ
    └── test_legal_case_generator.py  # テストスクリプト
```

## テスト

```bash
cd examples/04_legal_case_generator

# 簡易テスト（インポートと設定の確認）
python tests/test_legal_case_generator.py

# より包括的なテストはevaluate.pyを使用
python evaluate.py --output test_evaluation.json
```

**注**: 現在は簡易的なインポートテストのみ実装されています。包括的な機能テストは`evaluate.py`を使用してください。

## 次のステップ

1. サンプル法令で事例を生成
2. 生成された事例を確認
3. 必要に応じてプロンプトを調整
4. 評価を実行して品質を確認
5. 人手評価テンプレートで詳細評価

## 動作確認済み

**テスト実行日**: 2025-11-06

### 基本動作テスト

```bash
python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し..." \
  --output result.json
```

**結果**:
- ✓ 3種類の事例（適用・非適用・境界）を正常に生成
- ✓ 検証スコア: 0.92-1.00 (全て閾値以上)
- ✓ 実行時間: 約3分

### 総合評価テスト

```bash
python3 evaluate.py --output evaluation_results.json
```

**結果**:
- テストケース数: 3 (会社法、民法、個人情報保護法)
- 生成事例数: 9
- 成功率: **100%**
- 平均生成時間: 31.5秒/事例
- 平均反復回数: 0.44回

### 検証された機能

- ✓ LangGraphベースのマルチエージェントワークフロー
- ✓ シナリオ生成エージェント
- ✓ 法的整合性検証エージェント
- ✓ 事例洗練エージェント（フィードバックループ）
- ✓ 3種類の事例タイプ生成
- ✓ 評価スクリプト

### 既知の問題と対処

#### 1. モジュールインポートエラー (解決済み)

**問題**: Python 3.xでは `04_legal_case_generator` のような数字で始まるモジュール名を直接インポートできない

```python
# ✗ これはSyntaxErrorになる
from examples.04_legal_case_generator.config import load_config
```

**対処**: `importlib` を使用した動的インポートに修正済み

```python
# ✓ 正しい方法
import importlib
config_module = importlib.import_module('examples.04_legal_case_generator.config')
load_config = config_module.load_config
```

**修正済みファイル**:
- `pipeline.py`
- `evaluate.py`
- `tests/conftest.py`
- `tests/test_legal_case_generator.py`

#### 2. LangChain非推奨警告

**警告メッセージ**:
```
LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1
```

**影響**: 動作には問題なし（警告のみ）

**今後の対応**: `langchain-ollama` パッケージへの移行を推奨

### パフォーマンス特性

- **シナリオ生成**: 3-30秒（事例の複雑さによる）
- **法的検証**: 5-40秒
- **洗練処理**: 6-10秒（必要な場合のみ）
- **メモリ使用量**: ~500MB（qwen3:8bモデル使用時）

### 推奨環境

- Python: 3.10以上
- メモリ: 8GB以上
- Ollama: 最新版
- LLMモデル: qwen3:8b (13GB)
