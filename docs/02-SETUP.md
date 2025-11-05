# statutes RAG セットアップガイド

このドキュメントでは、statutes-ragプロジェクトの初回セットアップから評価実験まで、ステップバイステップで説明します。

## 目次

1. [前提条件](#前提条件)
2. [コンテナ環境での永続化と復元について](#コンテナ環境での永続化と復元について)  
3. [環境構築](#環境構築)
4. [データ準備](#データ準備)
5. [インデックス構築](#インデックス構築)
6. [コンテナ再起動後の環境復元](#コンテナ再起動後の環境復元)
7. [動作確認](#動作確認)
8. [評価実験](#評価実験)
9. [トラブルシューティング](#トラブルシューティング)
10. [クイックリファレンス](#クイックリファレンス)
11. [まとめ](#まとめ)

## 前提条件

* Python 3.10以上  
* Linux環境（Dockerコンテナを想定）  
* **永続ボリュームが `~/work` にマウントされていること**  
* 最低20GBのディスク空き容量（永続ボリューム内）  
* GPU利用を強く推奨


## コンテナ環境での永続化と復元について

このプロジェクトは、コンテナが再起動しても `~/work` ディレクトリのみが永続化される環境を想定しています。

1. 初回セットアップ（`setup_*.sh` スクリプト）:  
   `setup/setup_uv_env.sh` や `setup/setup_ollama.sh` などのスクリプトは、必要なツール（UV, Ollama）とPython環境（.venv）、Ollamaモデルを永続ボリューム `~/work 配下` にインストールします。これは初回のみ実行します。  
2. 再起動後の復元（`restore_env.sh` スクリプト）:  
   コンテナが再起動すると、インストールされたツールへのパス（$PATH）や環境変数は失われます。  
   `setup/restore_env.sh` スクリプトは、これら全ての環境設定を一度に復元し、Ollamaサーバーを自動起動するために提供されています。

## 環境構築

**重要:** 以下のステップは、プロジェクトのセットアップが完了していない場合に**一度だけ**実行してください。

### ステップ1: リポジトリのクローン

```bash
cd /home/jovyan/work
git clone <repository-url> statutes-rags
cd statutes-rags
```

### ステップ2: Python仮想環境のセットアップ（初回のみ）

プロジェクトでは高速なパッケージマネージャーuvを使用します。
```
# プロジェクトルート `/work/statutes-rags` で実行  
# sourceコマンドで実行します  
source setup/setup_uv_env.sh
```
このスクリプトは以下を実行します：

* uv を永続ボリューム `/work/tools/uv` にインストール  
* Python仮想環境 .venv をプロジェクトルートに作成（永続化）  
* 必要な全依存パッケージ（CUDA 12.1用PyTorchを含む）を .venv にインストール  
* uv へのパスを ~/.bashrc に永続化

<!-- このスクリプトは以下を実行します：
- `uv`のインストール（未インストールの場合）
- Python仮想環境`.venv`の作成
- 必要な全依存パッケージのインストール
  - FastAPI, LangChain, FAISS, SentenceTransformers等
  - 開発ツール（pytest, black, ruff等） -->

実行後、仮想環境を有効化：

```bash
source .venv/bin/activate
```

**確認方法:**

```bash
python3 --version  # Python 3.10以上が表示されること
uv pip list | grep langchain  # langchainパッケージが表示されること
```

### ステップ3: Ollamaのセットアップ

OllamaはローカルLLMを実行するためのツールです。

```bash
cd setup
./setup_ollama.sh
```

* Ollamaバイナリを永続ボリューム (setup/bin) にダウンロード  
* Ollamaサーバーの**初回起動**  
* 永続ボリューム `~/work/.ollama-models` へのモデルのダウンロード  
  * qwen3:8b（LLMモデル、約13GB）
  
**注:** 埋め込みモデル（intfloat/multilingual-e5-large）はHuggingFace経由で自動ダウンロードされるため、Ollamaでのダウンロードは不要です。

**注意:** qwen3:8bのダウンロードには10-30分かかる場合があります。

**確認方法:**

```bash
# サーバーが起動しているか確認
curl http://localhost:11434/api/tags

# モデルがインストールされているか確認
./bin/ollama list
# 出力例:
# NAME                    ID              SIZE    MODIFIED
# qwen3:8b            abc123...       13 GB   2 minutes ago
```

**トラブル:** サーバーが起動していない場合

```bash
# ログを確認
cat setup/ollama.log

# サーバーを再起動
pkill ollama
cd setup && ./bin/ollama serve > ollama.log 2>&1 &
```

### ステップ4: 日本語トークナイザー（自動インストール済み）

**SudachiPyとJanomeがデフォルトで自動インストールされます。**

`setup_uv_env.sh`を実行した時点で、以下のトークナイザーが既にインストールされています：

- **SudachiPy**: 高性能な日本語形態素解析器（推奨）
- **sudachidict-core**: SudachiPy用の辞書
- **Janome**: 軽量なフォールバック

管理者権限不要で、MeCabと同等以上の性能を実現します。

BM25Retrieverはデフォルトで `tokenizer="auto"` を使用し、利用可能なトークナイザーを自動選択します（優先順位: SudachiPy > Janome > MeCab > n-gram > simple）。

**確認方法:**

```bash
python -c "from app.retrieval.bm25_retriever import BM25Retriever; r = BM25Retriever(); print(f'トークナイザー: {r.tokenizer_type}')"
# 出力例: トークナイザー: sudachi
```

### ステップ5: 環境変数の設定（オプション）

RAGシステムはデフォルト値が設定されているため、環境変数の設定は**オプション**です。カスタマイズが必要な場合のみ設定してください。

プロジェクトルートに`.env`ファイルを作成して環境変数を設定できます。`.env.example`ファイルをテンプレートとして使用できます：

```bash
cd /home/jovyan/work/statutes-rags

# .env.exampleをコピーして編集（推奨）
cp .env.example .env
nano .env

# または、新規作成
cat > .env << 'EOF'
# LLM設定
LLM_MODEL=qwen3:8b
LLM_TEMPERATURE=0.1

# Retriever設定
RETRIEVER_TYPE=hybrid
RETRIEVER_TOP_K=10

# その他の設定はデフォルト値が使用されます
EOF
```

主要な環境変数：
- `LLM_MODEL`: LLMモデル名（デフォルト: `qwen3:8b`）
- `RETRIEVER_TYPE`: Retrieverタイプ `vector`/`bm25`/`hybrid`（デフォルト: `hybrid`）
- `RETRIEVER_TOP_K`: 検索する文書数（デフォルト: 10）
- `BM25_TOKENIZER`: トークナイザー `auto`/`sudachi`/`janome`/`mecab`/`ngram`/`simple`（デフォルト: `auto`）
- `MMR_FETCH_K_MAX`: MMRで取得する候補の最大数（デフォルト: 50）

全ての環境変数とデフォルト値は`.env.example`または`app/core/rag_config.py`を参照してください。

## データ準備

### ステップ6: データセットの配置（初回のみ）



**重要:** データセットファイルは大容量のため、Gitリポジトリには含まれていません。以下の手順で個別に入手・配置してください。また、この手順は**初回のみ**行ってください。

#### datasets/ ディレクトリ構造

```
datasets/
├── egov_laws/           # e-Gov法令XMLファイル（必須）
│   ├── *.xml            # 10,435ファイル、約264MB（zip圧縮時）
│   └── egov_laws_all.zip  # ダウンロードアーカイブ（264MB）
└── lawqa_jp/            # デジタル庁 4択法令データ（評価用、必須）
    ├── README.md        # データセット説明（リポジトリに含む）
    ├── LICENSE.md       # ライセンス情報（リポジトリに含む）
    └── data/            # 実データファイル（要ダウンロード）
        ├── selection.json
        ├── selection.csv
        └── ...
```

#### データセット入手方法

**1. e-Gov法令XMLファイル（必須）**

```bash
# datasets/egov_laws/ ディレクトリを作成
mkdir -p datasets/egov_laws

# オプション1: e-Gov法令APIから取得
# （詳細な手順は別途提供）

# オプション2: アーカイブファイルがある場合
# datasets/egov_laws/egov_laws_all.zip を配置し、解凍
cd datasets/egov_laws
unzip egov_laws_all.zip
```

**2. lawqa_jp（4択法令データ、必須）**

デジタル庁の公開データを入手：

- データソース: [政府等が保有するデータのAI学習データへの変換に係る調査研究](https://www.digital.go.jp/news/382c3937-f43c-4452-ae27-2ea7bb66ec75)
- ライセンス: 公共データ利用規約（第1.0版）

```bash
# datasets/lawqa_jp/data/ ディレクトリにファイルを配置
mkdir -p datasets/lawqa_jp/data
# selection.json, selection.csv など を配置
```


#### データセットの確認

配置後、以下のコマンドで確認：

```bash
# ディレクトリ構造を確認
ls -lh datasets/

# e-Gov法令XMLファイル数を確認
find datasets/egov_laws -name "*.xml" | wc -l
# 期待値: 10,435ファイル

# lawqa_jpデータを確認
ls -lh datasets/lawqa_jp/data/
```

### ステップ7: XMLからJSONLへの前処理（初回のみ）

法令XMLファイルを検索可能なJSONL形式に変換します。

```bash
# プロジェクトルートに戻る
cd /home/jovyan/work/statutes-rags

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

### ステップ8: FAISSインデックスの構築

検索用のベクトルインデックスとBM25インデックスを構築します。

インデックスの作成には全件（約10,435ファイルから抽出された全条文）を使用すると数時間を要するので、軽く動かす程度なら`--limit` で件数を絞ることを推奨

**少数ドキュメントで実行:**

```bash
# 最初の1000ドキュメントのみ
python3 scripts/build_index.py \
  --data-path data/test_egov.jsonl \
  --index-path data/faiss_index_test \
  --retriever-type hybrid \
  --limit 1000
```

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

`make`を使用する場合、全件データ（約10,435ファイル）を処理するため、長時間になることに注意
```bash
make index
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

## コンテナ再起動後の環境復元

コンテナが再起動（または新しいターミナルセッションを開始）した場合は、**以下のコマンドを1つ実行するだけ**で環境が復元されます。

```
# プロジェクトルート（~/work/statutes-rags）に移動  
cd ~/work/statutes-rags

# setup/restore_env.sh を source で実行  
source setup/restore_env.sh
```

このスクリプトは以下を**自動的**に行います：

1. uv, ollama へのPATHを設定  
2. OLLAMA_MODELS 環境変数を設定  
3. .venv 仮想環境を有効化  
4. Ollamaサーバーが起動していない場合は**自動で起動**  
5. curl を使用してOllama APIが応答するかを**確認**


## 動作確認

### ステップ9: 対話型CLIで動作確認

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

### ステップ10: 4択法令データでRAG評価

デジタル庁の4択法令データセットを使用してRAGシステムの精度を評価します。

**検索モード**: デフォルトでVector-Only（FAISSベクトル検索）を使用します。環境変数 `RETRIEVER_TYPE=vector` が設定されていることを確認してください。

> **重要**: BM25キーワード検索とHybridモードは、280万件のデータセットで50-60GBのメモリを必要とし、現在のシステム構成（62GBメモリ）では使用できません。Vector-Onlyモードで十分な精度（50%）を達成しています。詳細は [docs/supplemental/memory_issue_analysis.md](../supplemental/memory_issue_analysis.md) を参照してください。

#### 簡単な評価実行（推奨）

シェルスクリプトを使用して、1コマンドで評価を実行できます：

```bash
# 10サンプルで評価（約2-3分）
./scripts/evaluate.sh 10

# 50サンプルで評価（約10-15分）
./scripts/evaluate.sh 50

# 100サンプルで評価（約20-30分）
./scripts/evaluate.sh 100
```

このスクリプトは以下を自動的に行います：
- 環境変数の設定（RETRIEVER_TYPE=vector）
- 仮想環境の有効化
- GPU状態の確認
- 評価の実行
- 結果の保存（evaluation_results_final.json）

#### 手動での評価実行

直接Pythonスクリプトを実行する場合：

```bash
# 環境変数を設定
export RETRIEVER_TYPE=vector

# 小規模テスト（10問で動作確認）
python3 scripts/evaluate_multiple_choice.py \
  --samples 10 \
  --llm-model "qwen3:8b" \
  --output evaluation_results_10.json
```

**実行時間:** 約2-3分（ハードウェア性能に依存）

**注:** 軽量モデルを使用したい場合は、事前に `./setup/bin/ollama pull qwen2.5:7b` を実行して `--llm-model "qwen2.5:7b"` に置き換えてください。

#### 中規模評価（50問）

```bash
python3 scripts/evaluate_multiple_choice.py \
  --samples 50 \
  --llm-model "qwen3:8b" \
  --output evaluation_results_50.json
```

**実行時間:** 約10-15分（ハードウェア性能に依存）

#### 全データ評価（140問）

```bash
# バックグラウンド実行推奨
nohup python3 scripts/evaluate_multiple_choice.py \
  --llm-model "qwen3:8b" \
  --output evaluation_results_full.json \
  > evaluation.log 2>&1 &

# 進捗確認
tail -f evaluation.log
```

**実行時間:** 約120分前後（ハードウェア性能に依存）

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
    "llm_model": "qwen3:8b",
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

### ステップ11: ユニットテストの実行

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

### 問題2: トークナイザーに関する警告

**症状:**

```
SudachiPy not available. Using simple tokenizer.
```

**説明:**

BM25検索では日本語トークナイザーが必要です。`tokenizer="auto"`を指定した場合、システムは以下の優先順位で自動的にトークナイザーを選択します：
1. SudachiPy（推奨、高精度、管理者権限不要）
2. Janome（軽量、フォールバック、管理者権限不要）
3. MeCab（レガシーサポート、管理者権限が必要）
4. n-gram（辞書不要）
5. simple（最小限の機能）

**解決方法:**

通常はSudachiPyとJanomeが環境構築時（`setup_uv_env.sh`実行時）に自動でインストールされます。
この警告が出る場合は、手動でインストールしてください：

```bash
pip install sudachipy sudachidict-core janome
```

または、プロジェクト全体を再インストール：

```bash
pip install -e . --upgrade
```

注: 簡易トークナイザー（simple/ngram）でもシステムは動作しますが、検索精度が低下する可能性があります。

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
LLM_MODEL=qwen3:8b  # デフォルトモデル（13GB）
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

- [05-ARCHITECTURE.md](./05-ARCHITECTURE.md) - コードベースの構造とモジュール説明
- [03-USAGE.md](./03-USAGE.md) - 各スクリプトの詳細な使用方法
- [04-TESTING.md](./04-TESTING.md) - テスト実行ガイド

## クイックリファレンス

### 環境起動コマンド

```bash
# 仮想環境を有効化
source .venv/bin/activate

# トークナイザーの確認
python -c "from app.retrieval.bm25_retriever import BM25Retriever; r = BM25Retriever(); print(f'使用中: {r.tokenizer_type}')"

# Ollamaサーバーを起動（停止している場合）
cd setup && ./bin/ollama serve > ollama.log 2>&1 &
```

### よく使うコマンド

```bash
# 対話型CLI起動
python3 scripts/query_cli.py --interactive

# 3問で評価テスト
python3 scripts/evaluate_multiple_choice.py --samples 3 --llm-model "qwen3:8b"

# テスト実行
./scripts/run_tests.sh unit

# コード整形
black app/ scripts/ tests/
ruff check app/ scripts/ tests/
```

問題が発生した場合は、トラブルシューティングセクションを参照してください。

---

最終更新: 2024-11-04
