# Legal Case Generator - セットアップガイド

## クイックスタート（コピー＆ペースト用）

以下のコマンドを順番に実行するだけで、セットアップから評価まで完了します。

### ステップ1: 前提条件の確認

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 必要なファイルの確認
ls -lh data/egov_laws.jsonl
ls -lh data/faiss_index/vector/index.faiss
ls -lh datasets/lawqa_jp/data/selection.json
ollama list | grep qwen3
```

すべて存在すれば次へ。存在しない場合は[プロジェクトルートのREADME.md](../../README.md)を参照。

### ステップ2: 仮想環境の有効化

```bash
source .venv/bin/activate
```

### ステップ3: 依存関係の確認

プロジェクトルートで `./setup/setup_uv_env.sh` を実行済みであれば、LangGraphを含む全ての依存関係は既にインストール済みです。

```bash
# 依存関係が正しくインストールされているか確認
python3 -c "import langgraph; import langchain_ollama; print('✓ Dependencies installed')"
```

`✓ Dependencies installed` と表示されればOKです。

**注**: もしエラーが出る場合は、以下を実行：

```bash
cd /path/to/statutes-rags
uv pip install -e ".[examples]"
```

### ステップ4: Ollamaの起動確認

```bash
# Ollamaが起動していることを確認
curl http://localhost:11434/api/tags

# qwen3:8bモデルがあることを確認
ollama list | grep qwen3
```

### ステップ5: 動作確認テスト

```bash
cd examples/04_legal_case_generator

# 簡易テスト（約10秒）
python3 tests/test_legal_case_generator.py
```

### ステップ6: クイック評価（3法令、約5～10分）

```bash
# 評価スクリプトを実行（デフォルトのサンプルケース）
python3 evaluate.py --output results/eval_quick.json
```

### ステップ7: 結果確認

```bash
# 評価サマリーを表示
cat results/eval_quick.json | python3 -c "import sys, json; r=json.load(sys.stdin); print(json.dumps(r['summary'], ensure_ascii=False, indent=2))"
```

---

## 完了！

これで基本的な使用準備が整いました。

### 次のステップ

#### 単一法令での事例生成

```bash
python3 pipeline.py \
  --law-number "平成十七年法律第八十七号" \
  --law-title "会社法" \
  --article "26" \
  --article-content "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。" \
  --output results/company_law_case.json

# 結果確認
cat results/company_law_case.json | python3 -m json.tool
```

#### カスタムテストケースでの評価

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
  --output results/my_eval.json
```

#### ワンコマンド評価（推奨）

全ての確認と評価を自動実行：

```bash
./run_evaluation.sh
```

---

## トラブルシューティング

### エラー: "Index not found"

```bash
# プロジェクトルートに戻ってインデックスを確認
cd ../../
ls -la data/faiss_index/vector/

# インデックスがない場合は構築
python scripts/build_index.py --index-type vector
cd examples/04_legal_case_generator
```

### エラー: "Ollama connection failed"

```bash
# Ollamaの状態を確認
curl http://localhost:11434/api/tags

# 応答がない場合はOllamaを起動
cd ../../
./setup/setup_ollama.sh
cd examples/04_legal_case_generator
```

### エラー: "Module not found: langgraph"

```bash
# プロジェクトルートに戻って依存関係をインストール
cd ../../
uv pip install -e ".[examples]"
cd examples/04_legal_case_generator
```

### タイムアウトエラー

LLMの応答が遅い場合：

```bash
# タイムアウトを延長
export LLM_TIMEOUT=180

python3 evaluate.py --output results/eval.json
```

---

## 詳細情報

- **詳細な使用方法**: [USAGE.md](USAGE.md)
- **技術仕様**: [README.md](README.md)
- **プロジェクト全体**: [../../README.md](../../README.md)

---

最終更新: 2025年11月7日
