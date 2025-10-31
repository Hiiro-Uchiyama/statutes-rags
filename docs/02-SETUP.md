# Legal RAG セットアップガイド

このドキュメントでは、legal-ragプロジェクトの初回セットアップから評価実験まで、ステップバイステップで説明します。

## 目次

1. [前提条件](#前提条件)
2. [環境構築](#環境構築)
3. [データ準備](#データ準備)
4. [インデックス構築](#インデックス構築)
5. [動作確認](#動作確認)
6. [評価実験](#評価実験)
7. [トラブルシューティング](#トラブルシューティング)

## 前提条件

- Python 3.10以上
- Linux環境（Docker環境でも可）
- 最低20GBのディスク空き容量
- GPUがある場合は推奨（なくても動作可能だが遅い）

## 環境構築

### ステップ1: リポジトリのクローン

```bash
cd /home/jovyan/work
git clone <repository-url> legal-rag
cd legal-rag
```

### ステップ2: Python仮想環境のセットアップ

プロジェクトでは高速なパッケージマネージャー`uv`を使用します。

```bash
# セットアップスクリプトを実行
./setup/setup_uv_env.sh
```

このスクリプトは以下を実行します：
- `uv`のインストール（未インストールの場合）
- Python仮想環境`.venv`の作成
- 必要な全依存パッケージのインストール
  - FastAPI, LangChain, FAISS, SentenceTransformers等
  - 開発ツール（pytest, black, ruff等）

実行後、仮想環境を有効化：

```bash
source .venv/bin/activate
```

**確認方法:**

```bash
python3 --version  # Python 3.10以上が表示されること
pip list | grep langchain  # langchainパッケージが表示されること
```

### ステップ3: Ollamaのセットアップ

OllamaはローカルLLMを実行するためのツールです。

```bash
cd setup
./setup_ollama.sh
```

このスクリプトは以下を実行します：
- Ollamaバイナリのダウンロードと展開
- Ollamaサーバーの起動（バックグラウンド）
- 埋め込みモデル`nomic-embed-text`のダウンロード
- LLMモデル`gpt-oss:20b`（約13GB）のダウンロード

**注意:** `gpt-oss:20b`のダウンロードには10-30分かかる場合があります。

**確認方法:**

```bash
# サーバーが起動しているか確認
curl http://localhost:11434/api/tags

# モデルがインストールされているか確認
./bin/ollama list
# 出力例:
# NAME                    ID              SIZE    MODIFIED
# gpt-oss:20b            abc123...       13 GB   2 minutes ago
# nomic-embed-text       def456...       274 MB  3 minutes ago
```

**トラブル:** サーバーが起動していない場合

```bash
# ログを確認
cat setup/ollama.log

# サーバーを再起動
pkill ollama
cd setup && ./bin/ollama serve > ollama.log 2>&1 &
```

### ステップ4: MeCabのセットアップ（オプション）

MeCabは日本語形態素解析器で、BM25検索に使用されます。インストールしなくてもシステムは動作しますが、BM25の精度が低下します。

```bash
cd setup
./setup_mecab.sh
```

このスクリプトは以下を実行します：
- MeCab本体のビルドとインストール（ローカル、sudo不要）
- IPA辞書のインストール
- Python bindingのインストール

**確認方法:**

```bash
source setup/mecab_env.sh
echo "これはテストです。" | setup/bin/mecab -Owakati
# 出力: これ は テスト です 。
```

## データ準備

### ステップ5: データセットの確認

プロジェクトには以下のデータセットが含まれています：

```bash
ls -lh datasets/
```

- `egov_laws/` - e-Gov法令XML（約264MB、10,435ファイル）
- `lawqa_jp/` - デジタル庁の4択法令データ（140問）
- `civil_law_instructions/` - 民法QAデータ
- `criminal_law_exams/` - 刑法試験問題

### ステップ6: XMLからJSONLへの前処理

法令XMLファイルを検索可能なJSONL形式に変換します。

```bash
# プロジェクトルートに戻る
cd /home/jovyan/work/legal-rag

# 仮想環境を有効化（まだの場合）
source .venv/bin/activate

# 前処理を実行（全XMLファイル）
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl
```

**処理時間:** 約5-10分（10,435ファイル）

**または、Makefileを使用:**

```bash
make preprocess
```

**テスト実行（少数ファイルで確認）:**

```bash
# 最初の100ファイルのみ処理
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/test_egov.jsonl \
  --limit 100
```

**確認方法:**

```bash
# ファイルが生成されたか確認
ls -lh data/egov_laws.jsonl

# 行数を確認
wc -l data/egov_laws.jsonl

# 内容をサンプル表示
head -1 data/egov_laws.jsonl | python3 -m json.tool
```

## インデックス構築

### ステップ7: FAISSインデックスの構築

検索用のベクトルインデックスとBM25インデックスを構築します。

```bash
# ハイブリッド検索用インデックスを構築（Vector + BM25）
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid
```

**処理時間:** 
- ベクトル化: 約20-40分（文書数による）
- BM25インデックス: 約5分

**または、Makefileを使用:**

```bash
make index
```

**テスト実行（少数ドキュメントで確認）:**

```bash
# 最初の1000ドキュメントのみ
python3 scripts/build_index.py \
  --data-path data/test_egov.jsonl \
  --index-path data/faiss_index_test \
  --retriever-type hybrid \
  --limit 1000
```

**確認方法:**

```bash
# インデックスが生成されたか確認
ls -lh data/faiss_index/vector/
ls -lh data/faiss_index/bm25/

# 以下のファイルが存在するはず:
# vector/index.faiss
# vector/index.pkl
# bm25/index.pkl
```

**注意:** 既にインデックスが存在する場合（`data/faiss_index/`）、このステップはスキップ可能です。

## 動作確認

### ステップ8: 対話型CLIで動作確認

RAGシステムが正しく動作するか確認します。

```bash
# 対話モードで起動
python3 scripts/query_cli.py --interactive
```

**テスト質問例:**

```
Question: 会社法第26条について教えてください
Question: 労働基準法における時間外労働の制限は？
Question: exit
```

**期待される出力:**

```
ANSWER:
会社法第26条は、株式会社の設立に関する規定です...

CITATIONS:
[1] 会社法 第26条 第1項
...
```

**単発クエリ実行:**

```bash
python3 scripts/query_cli.py "会社法第26条について教えてください"
```

## 評価実験

### ステップ9: 4択法令データでRAG評価

デジタル庁の4択法令データセットを使用してRAGシステムの精度を評価します。

#### 小規模テスト（3問で動作確認）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 3 \
  --llm-model "gpt-oss:20b" \
  --output evaluation_results_3.json
```

**実行時間:** 約2分（1問あたり40秒）

#### 中規模評価（20問）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 20 \
  --llm-model "gpt-oss:20b" \
  --output evaluation_results_20.json
```

**実行時間:** 約13-20分

#### 全データ評価（140問）

```bash
# バックグラウンド実行推奨
nohup python3 scripts/evaluate_multiple_choice.py \
  --llm-model "gpt-oss:20b" \
  --output evaluation_results_full.json \
  > evaluation.log 2>&1 &

# 進捗確認
tail -f evaluation.log
```

**実行時間:** 約90-120分

### 評価結果の確認

```bash
# 結果サマリーを表示
cat evaluation_results_3.json | python3 -m json.tool | head -20

# 精度のみを表示
cat evaluation_results_3.json | python3 -c "import json, sys; data=json.load(sys.stdin); print(f\"Accuracy: {data['summary']['accuracy']*100:.2f}%\")"
```

**期待される出力例:**

```json
{
  "config": {
    "rag_enabled": true,
    "retriever_type": "hybrid",
    "llm_model": "gpt-oss:20b",
    "top_k": 5,
    "total_samples": 3
  },
  "summary": {
    "accuracy": 0.6667,
    "correct_count": 2,
    "total_count": 3
  }
}
```

## テストの実行

### ステップ10: ユニットテストの実行

```bash
# 全テスト実行
./scripts/run_tests.sh all

# ユニットテストのみ（高速）
./scripts/run_tests.sh unit

# カバレッジ付き
./scripts/run_tests.sh coverage
```

**または、Makefileを使用:**

```bash
make test          # ユニットテストのみ
make test-all      # 全テスト
make test-coverage # カバレッジ付き
```

## トラブルシューティング

### 問題1: Ollamaサーバーが応答しない

**症状:**

```
Ollama call failed with status code 404
```

**解決方法:**

```bash
# サーバー状態を確認
curl http://localhost:11434/api/tags

# サーバーを再起動
pkill ollama
cd setup && ./bin/ollama serve > ollama.log 2>&1 &
sleep 5

# モデルが存在するか確認
./bin/ollama list
```

### 問題2: MeCabの警告

**症状:**

```
Failed to initialize MeCab. Using simple tokenizer.
```

**解決方法:**

MeCabはオプショナルです。警告を無視しても動作しますが、インストールする場合：

```bash
cd setup
./setup_mecab.sh
source mecab_env.sh
```

以降のシェルセッションでは常に以下を実行：

```bash
source setup/mecab_env.sh
```

### 問題3: メモリ不足エラー

**症状:**

```
OutOfMemoryError: Cannot allocate memory
```

**解決方法:**

1. より小さいLLMモデルを使用

```bash
# .envファイルを編集
nano .env

# LLM_MODELを変更
LLM_MODEL=qwen2.5:7b  # 13GB -> 4.4GB
```

2. インデックス構築時のバッチサイズを削減

```bash
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --batch-size 50  # デフォルト100から削減
```

### 問題4: FAISSインデックスが読み込めない

**症状:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/faiss_index/vector/index.faiss'
```

**解決方法:**

インデックスを再構築：

```bash
make index
```

または：

```bash
python3 scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid
```

### 問題5: JSONL前処理が失敗する

**症状:**

```
xml.etree.ElementTree.ParseError: not well-formed
```

**解決方法:**

破損したXMLファイルがある可能性があります。`--limit`オプションで範囲を絞って特定：

```bash
# 最初の100ファイル
python3 scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws_test.jsonl \
  --limit 100
```

エラーが出たファイルをスキップして続行します（スクリプトは自動的にスキップします）。

## 次のステップ

セットアップが完了したら、以下のドキュメントを参照してください：

- `ARCHITECTURE.md` - コードベースの構造とモジュール説明
- `USAGE.md` - 各スクリプトの詳細な使用方法
- `EVALUATION_GUIDE.md` - 評価実験の詳細ガイド

## クイックリファレンス

### 環境起動コマンド

```bash
# 仮想環境を有効化
source .venv/bin/activate

# MeCab環境を設定（使用する場合）
source setup/mecab_env.sh

# Ollamaサーバーを起動（停止している場合）
cd setup && ./bin/ollama serve > ollama.log 2>&1 &
```

### よく使うコマンド

```bash
# 対話型CLI起動
python3 scripts/query_cli.py --interactive

# 3問で評価テスト
python3 scripts/evaluate_multiple_choice.py --samples 3 --llm-model "gpt-oss:20b"

# テスト実行
./scripts/run_tests.sh unit

# コード整形
black app/ scripts/ tests/
ruff check app/ scripts/ tests/
```

## まとめ

このガイドに従えば、以下が完了します：

1. Python環境のセットアップ
2. Ollama LLMサーバーの起動
3. 法令データの前処理
4. ベクトルインデックスの構築
5. RAGシステムの動作確認
6. 評価実験の実行

問題が発生した場合は、トラブルシューティングセクションを参照してください。
