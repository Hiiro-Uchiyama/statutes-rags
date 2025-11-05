# Multi-Agent Debate - 使用方法ガイド

## クイックスタート

### 1. セットアップ

```bash
# プロジェクトルートに移動
cd /path/to/statutes-rags

# 依存関係をインストール（初回のみ）
uv pip install langchain-community langchain-core langgraph langchain-ollama

# または pip を使用
pip install langchain-community langchain-core langgraph langchain-ollama
```

### 2. Ollamaの起動とモデルのダウンロード

```bash
# Ollamaを起動（別ターミナル）
ollama serve

# qwen3:8b モデルをダウンロード（初回のみ、サイズ大）
ollama pull qwen3:8b

# または、より軽量なモデル
ollama pull gpt-oss:7b
```

### 3. インデックスの確認

```bash
# インデックスが存在することを確認
ls -la data/faiss_index/vector
ls -la data/faiss_index/bm25

# インデックスがない場合は構築
make index
# または
python scripts/build_index.py
```

### 4. 評価の実行

#### 基本的な使用方法

```bash
# examples/03_multi_agent_debate ディレクトリに移動
cd examples/03_multi_agent_debate

# テスト実行（最初の3問のみ）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 3 \
    --output results/test_debate.json
```

#### 全問題の評価

```bash
# 全問題を評価（時間がかかります: 約5分/問）
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/full_debate.json
```

#### 結果の確認

```bash
# 評価結果のサマリーを確認
cat results/test_debate.json | python -m json.tool | head -50

# または、評価実行時に自動で表示されます
```

## コマンドオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `--dataset` | 評価データセットのパス（必須） | - |
| `--output` | 結果の出力先 | `results/evaluation_{timestamp}.json` |
| `--limit` | 評価する最大問題数 | なし（全問題） |
| `--config` | 設定ファイルのパス | `config.yaml` |

## 設定ファイル（config.yaml）

Multi-Agent Debateシステムの動作は `config.yaml` で細かく制御できます：

```yaml
# LLMモデル設定
model_name: "qwen3:8b"  # 使用するOllamaモデル
temperature: 0.3           # 生成の多様性（0.0-1.0）
max_tokens: 2000          # 最大トークン数

# 議論設定
max_debate_rounds: 3      # 最大議論ラウンド数
agreement_threshold: 0.9  # 合意と判定するスコア閾値

# 検索設定
retrieval_top_k: 10       # 取得する文書数
vector_store_path: "data/faiss_index"  # インデックスパス
embedding_model: "intfloat/multilingual-e5-large"
```

### 高速化設定

```yaml
# より高速な評価のための設定例
model_name: "gpt-oss:7b"   # 軽量モデル
max_debate_rounds: 1       # 1ラウンドのみ
retrieval_top_k: 5         # 文書数を削減
temperature: 0.1           # より確実な出力
```

### 高精度設定

```yaml
# より高精度な評価のための設定例
model_name: "qwen3:8b"  # 大規模モデル
max_debate_rounds: 5       # 議論を深める
agreement_threshold: 0.95  # より厳密な合意基準
retrieval_top_k: 15        # より多くの文書を参照
temperature: 0.5           # 適度な多様性
```

## 評価結果の形式

評価結果は以下の形式で出力されます：

```json
{
  "summary": {
    "total_questions": 3,
    "correct_answers": 2,
    "accuracy": 0.6667,
    "avg_rounds": 1.33,
    "avg_agreement_score": 0.94,
    "agreement_formation_rate": 1.0,
    "avg_time_per_question": 45.2,
    "errors": 0
  },
  "results": [
    {
      "question_index": 0,
      "question": "金融商品取引法第5条...",
      "choices": ["a ...", "b ...", "c ...", "d ..."],
      "correct_answer": "c",
      "predicted_answer": "c",
      "is_correct": true,
      "debate_rounds": 1,
      "agreement_score": 0.99,
      "debate_history": [
        {
          "round": 1,
          "debater_a": {
            "position": "c",
            "reasoning": "...",
            "citations": [...]
          },
          "debater_b": {
            "position": "c",
            "reasoning": "...",
            "citations": [...]
          }
        }
      ],
      "elapsed_time": 26.4,
      "error": null
    }
  ]
}
```

## トラブルシューティング

### エラー: "Dataset not found"

```bash
# データセットのパスを確認
ls -la ../../datasets/lawqa_jp/data/selection.json

# 絶対パスで指定
python evaluate.py \
    --dataset /home/toronto02/statutes-rags/datasets/lawqa_jp/data/selection.json \
    --limit 3
```

### エラー: "Index not found" または "Hybrid retrieval returned 0 documents"

```bash
# プロジェクトルートに戻ってインデックスを構築
cd ../../
make index
cd examples/03_multi_agent_debate

# または直接構築
python ../../scripts/build_index.py
```

### エラー: "Connection refused" (Ollama関連)

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# 起動していない場合
ollama serve

# モデルがダウンロードされているか確認
ollama list

# モデルをダウンロード
ollama pull qwen3:8b
```

### 処理が遅い場合

```bash
# 軽量モデルを使用（config.yamlを編集）
# model_name: "gpt-oss:7b"
# max_debate_rounds: 1
# retrieval_top_k: 5

# または、一時的に環境変数で設定
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 3 \
    --config config_fast.yaml
```

### メモリ不足

```bash
# より軽量なモデルを使用
ollama pull gpt-oss:7b

# config.yamlでモデルを変更
# model_name: "gpt-oss:7b"
```

## ワンライナーコマンド集

### テスト評価（3問、標準設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 3 \
    --output results/quick_test.json
```

### 本番評価（全問題、標準設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --output results/full_evaluation_$(date +%Y%m%d_%H%M%S).json
```

### デバッグモード（詳細ログ付き）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 1 \
    2>&1 | tee results/debug_$(date +%Y%m%d_%H%M%S).log
```

### 高速テスト（1問のみ、軽量設定）

```bash
cd examples/03_multi_agent_debate && \
python evaluate.py \
    --dataset ../../datasets/lawqa_jp/data/selection.json \
    --limit 1 \
    --output results/single_test.json
```

## ユニットテストの実行

```bash
# クイックテスト（モックを使用、高速）
python tests/test_quick.py

# すべてのテストを実行
pytest tests/ -v

# 特定のテストのみ実行
pytest tests/test_multi_agent_debate.py::TestDebaterAgent -v

# カバレッジ付きで実行
pytest tests/ --cov=. --cov-report=html
```

## パフォーマンス目安

### システム構成
- **インデックス**: ベクトル 2,802,222件、BM25 10,000件
- **LLMモデル**: qwen3:8b (Ollama)
- **検索方式**: Hybrid (Vector + BM25, RRF融合)

### 処理時間
- **平均処理時間**: 約5.4分/問（324秒/問）
  - 検索: 約2秒
  - 議論: 約5分（モデルとラウンド数に依存）
- **1ラウンドあたり**: 約2-3分
- **3問の評価**: 約10-20分
- **全問題（140問）**: 約12-15時間

### 精度指標
- **平均議論ラウンド数**: 1.0ラウンド
- **平均合意スコア**: 0.95（非常に高い合意形成）
- **合意形成率**: 100%

## Multi-Agent Debateの特徴

### システム構成

1. **Debater Agent A (肯定側)**
   - 質問に対する回答を主張
   - 検索した法令文書を引用して論拠を示す

2. **Debater Agent B (批判的側)**
   - 代替案を提示
   - 肯定側の主張を批判的に検討

3. **Moderator Agent (調停者)**
   - 両者の主張を比較評価
   - 合意スコアを計算
   - 最終的な回答を決定

### 議論フロー

```
Question → Retrieval → Round 1 → Evaluation → Agreement?
                           ↓         ↓            ↓
                       Debater A  Moderator    Yes → Final Answer
                       Debater B     ↓          No  → Round 2 → ...
                                  Agreement
                                    Score
```
