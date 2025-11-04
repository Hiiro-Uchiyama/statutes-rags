「何らかの質問に対して法令根拠を基に回答できるサービス」を実装し、4択法令問題(デジタル庁)を用いて評価を行います。
デフォルトで使用するLLMモデルは`gpt-oss:20b`です（環境変数で変更可能）。

**注意:** 研究室メンバー（Heart01サーバー利用者）の方は、[README-FIRST.md](README-FIRST.md)も参照してください。

詳細は `docs/` ディレクトリを参照してください
- [docs/02-SETUP.md](docs/02-SETUP.md) - 初回セットアップガイド（環境構築から評価実験まで）
- [docs/03-USAGE.md](docs/03-USAGE.md) - 各スクリプトの詳細な使用方法
- [docs/04-TESTING.md](docs/04-TESTING.md) - テスト実行ガイド
- [docs/05-ARCHITECTURE.md](docs/05-ARCHITECTURE.md) - コードベースの構造とモジュール説明

初回の週は「技術調査や課題の調査」を行います。
`RAG`や`MCPサーバ`技術検証対象を実務レベルの実装や研究、最新状況を把握しましょう。
実装に生成AIは用いて良いので触りたい技術を自由に調べよう。

調査用プロンプトは、`DEEP-RESEARCH-PROMPT.md`です。
各自活用して下さい。

タスクは「質問に対して法令根拠を持って来て回答できるサービス」の実装です。
評価は、4択法令データ(デジタル庁)を用いて行います。

## セットアップ

プロジェクトを初めて使用する場合は、[docs/02-SETUP.md](docs/02-SETUP.md)に従ってセットアップを行ってください。

**重要:** データセットファイルは大容量のため、Gitリポジトリには含まれていません。**初回セットアップ時に手動でダウンロード・配置が必要です。**

必要なデータセット：
1. **e-Gov法令XMLファイル**（必須）
   - `datasets/egov_laws/` に配置
   - 10,435ファイル、約264MB（zip圧縮時）
   
2. **デジタル庁の4択法令データ**（必須、評価用）
   - `datasets/lawqa_jp/data/` に配置
   - selection.json など

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