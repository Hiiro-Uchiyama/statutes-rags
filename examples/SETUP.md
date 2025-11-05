# Examples セットアップガイド

## セットアップ方法

### 方法1: 初回セットアップ時に全てインストール（推奨）

既存のセットアップスクリプトは、**既にAgentic RAG用の依存関係をインストールします**。

```bash
# プロジェクトルートで
./setup/setup_uv_env.sh
```

これにより以下が自動的にインストールされます：

- 基本的なRAG依存関係
- 開発用ツール（pytest等）
- **LangGraph** (Agentic RAG用)
- **LangSmith** (トレーシング用、オプション)
- **xmltodict** (MCP e-Gov Agent用)

### 方法2: 既存環境に追加インストール

既に基本的なセットアップが完了している場合：

```bash
# 仮想環境を有効化
source .venv/bin/activate

# Examples用の依存関係のみインストール
./setup/setup_examples.sh
```

または直接：

```bash
source .venv/bin/activate
uv pip install -e ".[examples]"
```

### 方法3: 個別にインストール

```bash
source .venv/bin/activate
uv pip install langgraph xmltodict

# オプション: トレーシング用
uv pip install langsmith
```

## インストールの確認

```bash
python -c "
import langgraph
import xmltodict
print('LangGraph version:', langgraph.__version__)
print('xmltodict: OK')
print('All dependencies installed successfully!')
"
```

## トラブルシューティング

### エラー: "No module named 'langgraph'"

```bash
# 仮想環境が有効化されているか確認
echo $VIRTUAL_ENV

# 有効化されていない場合
source .venv/bin/activate

# 再インストール
uv pip install langgraph xmltodict
```

### エラー: "No module named 'examples'"

```bash
# プロジェクトルートで
uv pip install -e ".[examples]"
```

### 既存のsetup_uv_env.shを実行したが依存関係がない

setup_uv_env.shは既にexamples用の依存関係をインストールするように設定されていますが、
古いバージョンを使用している可能性があります。

```bash
# 最新のスクリプトで再セットアップ
./setup/setup_uv_env.sh
```

## セットアップ完了後

### 動作確認

```bash
cd examples
pytest tests/test_01_agentic_rag.py::TestManagerAgent -v
```

### Agentic RAGの実行

```bash
cd examples/01_agentic_rag
python evaluate.py --dataset ../../datasets/lawqa_jp/data/selection.json --max-questions 5
```

詳細は `QUICKSTART.md` を参照してください。

## 各実装の依存関係

### 01_agentic_rag
- langgraph (必須)
- langsmith (オプション)

### 02_mcp_egov_agent
- xmltodict (必須)
- httpx (既に基本依存に含まれる)

### 03_multi_agent_debate
- langgraph (必須)

### 04_legal_case_generator
- langgraph (必須)

## まとめ

- **初回セットアップ**: `./setup/setup_uv_env.sh` だけでOK
- **追加インストール**: `./setup/setup_examples.sh` または `uv pip install -e ".[examples]"`
- **確認**: `python -c "import langgraph; print('OK')"`

全ての依存関係は自動的にインストールされるように設定されています。

