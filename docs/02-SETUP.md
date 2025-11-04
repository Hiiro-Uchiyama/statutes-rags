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
# プロジェクトルート `/work/statutes-rag` で実行  
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
  * nomic-embed-text（埋め込みモデル）  
  * gpt-oss:20b（LLMモデル、約13GB）

**注意:** gpt-oss:20bのダウンロードには10-30分かかる場合があります。

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

### ~~ステップ4: MeCabのセットアップ（オプション）~~

**重要** 現在はこのステップは飛ばしてください。MeCabはインストールできません、
~~MeCabは日本語形態素解析器で、BM25検索に使用されます。インストールしなくてもシステムは動作しますが、BM25の精度が低下します。~~

```bash
cd setup
./setup_mecab.sh
```

~~このスクリプトは以下を実行します：~~
- ~~MeCab本体のビルドとインストール（ローカル、sudo不要）~~
- ~~IPA辞書のインストール~~
- ~~Python bindingのインストール~~

**確認方法:**

```bash
source setup/mecab_env.sh
echo "これはテストです。" | setup/bin/mecab -Owakati
# 出力: これ は テスト です 。
```

## データ準備

### ステップ5: データセットの配置（初回のみ）



**重要:** データセットファイルは大容量のため、Gitリポジトリには含まれていません。以下の手順で個別に入手・配置してください。また、この手順は**初回のみ**行ってください。

#### datasets/ ディレクトリ構造

```
datasets/
├── egov_laws/           # e-Gov法令XMLファイル（必須）
│   ├── *.xml            # 10,433ファイル、約2GB
│   └── egov_laws_all.zip  # ダウンロードアーカイブ（264MB）
├── lawqa_jp/            # デジタル庁 4択法令データ（評価用、必須）
│   ├── README.md        # データセット説明（リポジトリに含む）
│   ├── LICENSE.md       # ライセンス情報（リポジトリに含む）
│   └── data/            # 実データファイル（要ダウンロード）
│       ├── selection.json
│       ├── selection.csv
│       └── ...
├── civil_law_instructions/  # 民法QAデータ（オプション）
└── criminal_law_exams/      # 刑法試験問題（オプション）
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

**3. その他のデータセット（オプション）**

- `civil_law_instructions/`: HuggingFace等から入手
- `criminal_law_exams/`: 研究用データセット

#### データセットの確認

配置後、以下のコマンドで確認：

```bash
# ディレクトリ構造を確認
ls -lh datasets/

# e-Gov法令XMLファイル数を確認
find datasets/egov_laws -name "*.xml" | wc -l
# 期待値: 10,433ファイル

# lawqa_jpデータを確認
ls -lh datasets/lawqa_jp/data/
```

### ステップ6: XMLからJSONLへの前処理（初回のみ）

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

**処理時間:** 約5-10分（10,433ファイル）

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

インデックスの作成には全件(280万件)を使用すると数十時間を要するので、軽く動かす程度なら`--limit` で件数を絞ることを推奨

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

`make`を使用する場合、全件データ(280万)を処理するため、長時間になることに注意
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

1. uv, ollama, mecab へのPATHを設定  
2. OLLAMA_MODELS 環境変数を設定  
3. .venv 仮想環境を有効化  
4. Ollamaサーバーが起動していない場合は**自動で起動**  
5. curl を使用してOllama APIが応答するかを**確認**


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
cat evaluation_results_3.json | python3 -c "import json, sys; data=json.load(sys.stdin); print(f"Accuracy: {data['summary']['accuracy']*100:.2f}%")"
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
