このプロジェクト`statutes-rags`は皆さんの課題の方向性の一つとして`RAG`の簡単な実装例です。

テーマは以下の通りです。
「質問に対して法令根拠を持って来て回答できるサービス」であり、4択法令データ(デジタル庁)を用いて評価を行います。
使用しているモデルは`gpt-oss-20b`です。
ワークフローやRAGの仕組み、枠に囚われず、AIを活用し、上記サービスの精度向上を最大限目指し、最終的には学会発表を行う予定です。

詳細なドキュメントは `docs/` ディレクトリを参照してください：

- [SETUP.md](docs/SETUP.md) - 初回セットアップガイド（環境構築から評価実験まで）
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - コードベースの構造とモジュール説明
- [USAGE.md](docs/USAGE.md) - 各スクリプトの詳細な使用方法
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - 4択評価の詳細ガイド

初回の週は「技術調査や課題の調査」を行います。
そこから、`RAG`や`MCPサーバ`技術検証対象を実務レベルの実装や研究、最新状況を把握しましょう。
実装に生成AIは用いて問題ないので触りたい技術を自由に調べよう。

調査用プロンプトは、`DEEP-RESEARCH-PROMPT.md`です。
各自活用して下さい。

タスクは「質問に対して法令根拠を持って来て回答できるサービス」の実装です。
評価は、4択法令データ(デジタル庁)を用いて行います。

## セットアップ

プロジェクトを初めて使用する場合は、[docs/02-SETUP.md](docs/02-SETUP.md)に従ってセットアップを行ってください。

**重要:** データセットファイルは大容量のため、Gitリポジトリには含まれていません。以下の手順でデータを配置してください：

1. `datasets/egov_laws/` に e-Gov法令XMLファイル（10,435ファイル、約2GB）を配置
2. `datasets/lawqa_jp/data/` にデジタル庁の4択法令データを配置（評価用）

詳細な入手方法は [docs/02-SETUP.md の「データ準備」セクション](docs/02-SETUP.md#データ準備) を参照してください。

### 簡易セットアップ手順

```bash
# 1. Python環境構築
./setup/setup_uv_env.sh
source .venv/bin/activate

# 2. Ollamaセットアップ
cd setup && ./setup_ollama.sh

# 3. データセットを配置（手動）
# datasets/egov_laws/ に法令XMLファイルを配置
# datasets/lawqa_jp/data/ に評価データを配置

# 4. データ前処理
make preprocess

# 5. インデックス構築
make index
```

詳細な手順、トラブルシューティングは[docs/02-SETUP.md](docs/02-SETUP.md)を参照してください。