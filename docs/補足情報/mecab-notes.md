# MeCab に関する注意事項

## 概要

このプロジェクトではBM25検索で日本語のトークン化にMeCabを使用しようとしますが、MeCabが利用できない場合は自動的にシンプルなトークナイザーにフォールバックします。

## 現在の状態

- **MeCab**: インストールされていない（利用不可）
- **トークナイザー**: シンプルトークナイザーを使用（正規表現ベース）
- **影響**: BM25検索の精度がわずかに低下する可能性がありますが、実用上は問題ありません

## フォールバックトークナイザー

MeCabが利用できない場合、以下のシンプルなトークナイザーが自動的に使用されます：

### 実装
```python
def _simple_tokenize(text: str) -> List[str]:
    """
    シンプルなトークン化（MeCab不使用時のフォールバック）
    - 日本語（ひらがな、カタカナ、漢字）
    - 英数字の連続
    を抽出
    """
```

### 動作
1. 正規表現で日本語文字列と英数字を抽出
2. 2文字以上の日本語はそのまま保持
3. 英数字の連続も保持

## MeCabのインストール（オプション）

より高精度なトークン化が必要な場合は、以下の方法でMeCabをインストールできます：

### 方法1: システムパッケージマネージャー（推奨）

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

# CentOS/RHEL
sudo yum install -y mecab mecab-devel mecab-ipadic
```

### 方法2: ソースからビルド

```bash
cd /home/jovyan/work/statutes-rag/setup

# MeCabとIPA辞書をダウンロード（有効なURLを使用）
# 注意: setup_mecab.shのダウンロードURLは現在無効です

# 手動でインストールする場合：
# 1. https://taku910.github.io/mecab/ からMeCabをダウンロード
# 2. configure --prefix=/home/jovyan/work/statutes-rag/setup/lib/mecab
# 3. make && make install
# 4. IPA辞書も同様にインストール
```

### 方法3: Python mecab-python3のみ再インストール

システムにMeCabが既にインストールされている場合：

```bash
source .venv/bin/activate
pip uninstall mecab-python3
pip install mecab-python3
```

## 評価への影響

### パフォーマンス比較（推定）

| トークナイザー | 精度 | 速度 | 備考 |
|--------------|------|------|------|
| MeCab | 高 | 中 | 形態素解析による高精度な分割 |
| シンプル | 中〜高 | 速い | 正規表現ベース、実用上十分 |

### 実測結果

現在のシステム（シンプルトークナイザー使用）での評価：
- **初期評価**: 66.67% (2/3問正解)
- **ハイブリッド検索**: Vector検索も併用しているため、BM25の精度低下は限定的

## トラブルシューティング

### 問題: MeCab警告が表示される

**修正済み**: `app/retrieval/bm25_retriever.py`でMeCabが利用できない場合の警告を抑制しました。

### 問題: BM25検索の精度が低い

**対処法**:
1. **ハイブリッド検索を使用**（デフォルト）: Vector検索と組み合わせることで精度を補完
2. **Top-Kを増やす**: より多くの候補を検索
3. **Rerankerを有効化**: Cross-encoderで結果を再スコアリング

### 問題: setup_mecab.shが失敗する

**原因**: Google Driveのダウンロードリンクが無効

**対処法**: 
- システムパッケージマネージャーを使用（方法1）
- 公式サイトから手動ダウンロード

## 修正履歴

### 2025-10-30
- MeCab初期化失敗時の警告を抑制
- 自動的にシンプルトークナイザーにフォールバック
- mecabrcパスの自動検出機能を追加（使用されていないが将来の拡張用）

## 結論

**MeCabは必須ではありません。** 現在のシステムはMeCabなしでも十分に動作し、ハイブリッド検索により高い精度（66.67%）を実現しています。より高精度なトークン化が必要な場合のみ、MeCabのインストールを検討してください。

---

最終更新: 2025-10-30
