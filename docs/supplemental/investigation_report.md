# RAG推論停止問題 調査報告

## 問題の概要
RAG評価を実行すると、処理が途中で停止し、システムが応答しなくなる問題が発生しました。

## 調査結果

### 1. 症状
- RAG評価スクリプト実行時にシステムが応答しなくなる
- プロセスの停止またはPCの再起動が必要になる
- 処理が長時間（10分以上）進まない

### 2. 原因の特定

#### **メモリ使用量の異常な増加**
停止直前のプロセス状態：
```
toronto+    9541 94.6 89.4 71535408 58832248 pts/0 Dl+ 22:29  10:13
 python scripts/debug_rag_inference.py
```

- **CPU使用率**: 94.6%
- **メモリ使用率**: 89.4%（システムメモリ62GBのうち**58GB**を使用）
- **仮想メモリ**: 約71GB

#### **根本原因: BM25 Retrieverのload_index()処理**

BM25 Retrieverは以下のファイルを保存・ロードします：
- `bm25.pkl`: 1.1GB（BM25インデックス）
- `documents.pkl`: 1.8GB（280万件のドキュメント）
- 合計: 2.8GB

**問題のコード**（`app/retrieval/bm25_retriever.py` 365-370行目）：
```python
# 追加学習に備えてトークナイズ済みコーパスを再構築
if self.documents:
    self.tokenized_corpus = [
        self.tokenize(doc.get("text", ""))
        for doc in self.documents
    ]
```

この処理により：
1. **280万件のドキュメント全てを再トークナイズ**
2. トークナイズ処理でメモリを大量消費（推定40GB以上）
3. Sudachiトークナイザーの制限（最大49,149バイト）を超える文書で警告連発
4. メモリ不足によりシステムが停止

### 3. インデックスサイズ

```bash
# BM25インデックス
data/faiss_index/bm25/
├── bm25.pkl         1.1GB
├── documents.pkl    1.8GB
└── tokenizer_info.pkl  42B
合計: 2.8GB

# FAISSインデックス  
data/faiss_index/vector/
├── index.faiss      11GB
└── index.pkl        2.0GB
合計: 13GB

総インデックスサイズ: 約16GB
```

### 4. トークナイゼーションの問題

**Sudachiトークナイザーの制限**：
```
sudachi tokenization failed: "Error during tokenization": 
Input is too long, it can't be more than 49149 bytes, was 718357.
```

- 最大入力サイズ: 49,149バイト（約48KB）
- 実際の法令文書: 最大718,357バイト（約700KB）
- フォールバック処理で対応しているが、効率が悪い

### 5. GPU使用状況

**正常に動作している部分**：
- ✓ CUDA利用可能（RTX 4090, 24GB VRAM）
- ✓ Embedding Model（HuggingFace）はGPU使用
- ✓ Ollama（qwen3:8b）もGPU使用

**GPU使用率**：
- Embedding Model: 約2.6GB VRAM
- Ollama: 約13GB VRAM（モデルロード時）
- 合計: 約16GB / 24GB（正常範囲）

## 解決策

### 優先度1: メモリ問題の修正

**オプションA: トークナイズ済みコーパスを保存・ロード**
```python
# save_index()
with open(index_path / "tokenized_corpus.pkl", "wb") as f:
    pickle.dump(self.tokenized_corpus, f)

# load_index()  
with open(index_path / "tokenized_corpus.pkl", "rb") as f:
    self.tokenized_corpus = pickle.load(f)
```

**オプションB: 再トークナイズを遅延実行**
```python
# load時は再トークナイズしない
# 必要な時（add_documents）のみトークナイズ
```

### 優先度2: トークナイゼーション処理の改善

1. **長文ドキュメントの分割**
   - 49,149バイト以下のチャンクに分割してトークナイズ
   
2. **代替トークナイザーの使用**
   - n-gramトークナイザーに切り替え（制限なし）
   - または、Janomeを使用（Sudachiより制限が緩い）

### 優先度3: Retriever Typeの変更

評価時のみ、メモリ負荷の低い設定を使用：
- `RETRIEVER_TYPE=vector`（BM25を使用しない）
- または`RETRIEVER_TYPE=bm25`（FAISSを使用しない）

## 推奨アクション

### 即座に実行可能（評価を続けるため）

1. **vector-onlyモードで評価実行**
   ```bash
   # .envファイルで設定
   RETRIEVER_TYPE=vector
   ```

2. **評価サンプル数を制限**
   ```bash
   python scripts/evaluate_multiple_choice.py --samples 10 --top-k 5
   ```

### 恒久的な修正（開発タスク）

1. `bm25_retriever.py`の`load_index()`を修正
   - `tokenized_corpus`をpickleに保存・ロード
   - または再トークナイズを削除

2. トークナイゼーション処理の改善
   - 長文の分割処理を追加
   - またはn-gramトークナイザーをデフォルトに

3. メモリ使用量のモニタリング追加
   - インデックスロード後のメモリ使用量をログ出力

## 結論

**停止原因**: BM25 Retrieverのload_index()時に280万件のドキュメントを再トークナイズし、58GBのメモリを消費してシステムが停止。

**緊急対応**: `RETRIEVER_TYPE=vector`で評価を実行（BM25を無効化）

**恒久対策**: `tokenized_corpus`を保存・ロードする修正を実装
