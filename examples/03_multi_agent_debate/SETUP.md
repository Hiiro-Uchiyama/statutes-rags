# Multi-Agent Debate - セットアップガイド

マルチエージェント討論システムのセットアップガイドです。

## 目次

1. [前提条件](#前提条件)
2. [プロジェクトルートのセットアップ](#プロジェクトルートのセットアップ)
3. [Multi-Agent Debate専用の依存関係](#multi-agent-debate専用の依存関係)
4. [動作確認](#動作確認)
5. [トラブルシューティング](#トラブルシューティング)

## 前提条件

Multi-Agent Debateを使用する前に、プロジェクトルートの基本セットアップが完了している必要があります。

### 必要な環境

- Python 3.10以上
- Ollama（LLMサーバー）
- FAISSインデックス（法令データ）
- データセット（評価用）

## プロジェクトルートのセットアップ

まだプロジェクトルートのセットアップが完了していない場合は、以下の手順を実行してください。

### 1. Python環境のセットアップ

```bash
# プロジェクトルートに移動
cd /home/toronto02/statutes-rags

# 仮想環境のセットアップ（初回のみ）
source setup/setup_uv_env.sh

# 仮想環境を有効化
source .venv/bin/activate
```

### 2. Ollamaのセットアップ

```bash
# Ollamaのインストールと起動（初回のみ）
cd setup
./setup_ollama.sh
cd ..

# または、コンテナ再起動後の復元
source setup/restore_env.sh
```

**確認:**

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# qwen3:8bモデルが利用可能か確認
setup/bin/ollama list | grep qwen3:8b
```

### 3. データセットの準備

```bash
# e-Gov法令XMLファイルを配置
mkdir -p datasets/egov_laws
# ここにXMLファイル（10,435ファイル）を配置

# 評価用データセットを配置
mkdir -p datasets/lawqa_jp/data
# ここにselection.jsonなどを配置
```

詳細は[プロジェクトルートのREADME.md](../../README.md)および[docs/02-SETUP.md](../../docs/02-SETUP.md)を参照してください。

### 4. データ前処理とインデックス構築

```bash
# プロジェクトルートで実行
cd /home/toronto02/statutes-rags

# 仮想環境を有効化（まだの場合）
source .venv/bin/activate

# データ前処理（初回のみ）
python scripts/preprocess_egov_xml.py \
  --input-dir datasets/egov_laws \
  --output-file data/egov_laws.jsonl

# FAISSインデックス構築（初回のみ）
python scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid
```

または、Makefileを使用：

```bash
make preprocess
make index
```

**確認:**

```bash
# インデックスが存在するか確認
ls -la data/faiss_index/vector/
ls -la data/faiss_index/bm25/
```

## Multi-Agent Debate専用の依存関係

プロジェクトルートのセットアップが完了したら、Multi-Agent Debate専用の追加パッケージをインストールします。

```bash
# プロジェクトルートで実行
cd /home/toronto02/statutes-rags

# 仮想環境を有効化
source .venv/bin/activate

# examples用の依存関係をインストール
uv pip install -e ".[examples]"
```

これにより、以下がインストールされます：
- `langgraph` - マルチエージェントワークフロー用
- `langsmith` - トレーシング用（オプション）
- その他の依存関係

**確認:**

```bash
# LangGraphがインストールされているか確認
python -c "import langgraph; print('LangGraph OK')"

# LangSmithがインストールされているか確認
python -c "import langsmith; print('LangSmith OK')"
```

## 動作確認

### クイックテスト（モック使用）

実際のLLMやデータなしで基本動作を確認できます：

```bash
# examples/03_multi_agent_debateに移動
cd examples/03_multi_agent_debate

# クイックテストを実行
python tests/test_quick.py
```

**期待される出力:**

```
test_config_loading ... ok
test_debater_initialization ... ok
test_moderator_initialization ... ok
test_workflow_initialization ... ok
test_basic_workflow ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.XXXs

OK
```

すべてのテストがパスすれば、基本的な実装は正常です。

### ユニットテスト（pytest）

より詳細なテストを実行：

```bash
# examples/03_multi_agent_debateで実行
pytest tests/ -v

# 特定のテストファイルのみ
pytest tests/test_quick.py -v

# カバレッジ付き
pytest tests/ --cov=. --cov-report=html
```

### 簡易評価テスト（実LLM使用）

実際のLLMとデータを使用した簡易評価：

```bash
# examples/03_multi_agent_debateで実行

# 3問で動作確認（約5-10分）
python evaluate.py \
  --dataset ../../datasets/lawqa_jp/data/selection.json \
  --limit 3 \
  --output results/test_quick.json

# 結果の確認
cat results/test_quick.json | python -m json.tool | head -50
```

## トラブルシューティング

### LangGraphがインストールされていない

**エラー:**

```
ModuleNotFoundError: No module named 'langgraph'
```

**解決策:**

```bash
# プロジェクトルートで実行
cd /home/toronto02/statutes-rags
source .venv/bin/activate
uv pip install langgraph langsmith
```

### Ollamaに接続できない

**エラー:**

```
ConnectionError: Cannot connect to Ollama server
```

**解決策:**

```bash
# Ollamaの状態確認
curl http://localhost:11434/api/tags

# Ollamaが停止している場合は起動
cd setup
./bin/ollama serve > ollama.log 2>&1 &
cd ..

# または、restore_env.shを使用
source setup/restore_env.sh
```

### FAISSインデックスが見つからない

**エラー:**

```
FileNotFoundError: data/faiss_index/vector/index.faiss not found
```

**解決策:**

```bash
# プロジェクトルートでインデックスを構築
cd /home/toronto02/statutes-rags
source .venv/bin/activate
python scripts/build_index.py \
  --data-path data/egov_laws.jsonl \
  --index-path data/faiss_index \
  --retriever-type hybrid
```

### データセットが見つからない

**エラー:**

```
FileNotFoundError: datasets/lawqa_jp/data/selection.json not found
```

**解決策:**

データセットを配置してください。詳細は[docs/02-SETUP.md](../../docs/02-SETUP.md)の「データ準備」セクションを参照。

### メモリ不足

**症状:**

処理が遅い、または Out of Memory エラー

**解決策:**

環境変数で設定を調整：

```bash
# 検索文書数を減らす
export DEBATE_RETRIEVAL_TOP_K=5

# 最大ラウンド数を減らす
export DEBATE_MAX_ROUNDS=2

# 軽量モデルを使用
export LLM_MODEL=gpt-oss:7b
```

### インポートエラー

**エラー:**

```
ModuleNotFoundError: No module named 'agents'
```

**原因:**

数字で始まるディレクトリ名（`03_multi_agent_debate`）がPythonモジュール名として無効なため、パス設定が必要です。

**解決策:**

コード内で既に対処済みですが、Pythonを直接実行する場合は以下のようにパスを設定：

```python
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent
sys.path.insert(0, str(debate_dir))
```

## 環境変数設定（オプション）

Multi-Agent Debateの動作は環境変数でカスタマイズできます：

```bash
# プロジェクトルートに.envファイルを作成
cat >> .env << 'EOF'

# Multi-Agent Debate設定
DEBATE_MAX_ROUNDS=3
DEBATE_AGREEMENT_THRESHOLD=0.8
DEBATE_RETRIEVAL_TOP_K=10
DEBATE_VERBOSE=false

# LLM設定
LLM_MODEL=qwen3:8b
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60

# Retriever設定
VECTOR_STORE_PATH=data/faiss_index
DATA_PATH=data/egov_laws.jsonl
EMBEDDING_MODEL=intfloat/multilingual-e5-large

EOF
```

## セットアップ確認チェックリスト

セットアップが完了したら、以下を確認してください：

- [ ] Python 3.10以上がインストールされている
- [ ] 仮想環境（.venv）が作成され、有効化されている
- [ ] Ollamaが起動し、qwen3:8bモデルが利用可能
- [ ] データセット（egov_laws、lawqa_jp）が配置されている
- [ ] FAISSインデックスが構築されている（data/faiss_index/）
- [ ] LangGraphとLangSmithがインストールされている
- [ ] クイックテスト（test_quick.py）がパスする

全て完了したら、[USAGE.md](./USAGE.md)を参照して評価実験を実行できます。

## 次のステップ

- [USAGE.md](./USAGE.md) - 評価実験の実行方法
- [README.md](./README.md) - システムの概要と使用方法
- [../../docs/02-SETUP.md](../../docs/02-SETUP.md) - プロジェクトルートの詳細なセットアップガイド

---

最終更新: 2024-11-07
