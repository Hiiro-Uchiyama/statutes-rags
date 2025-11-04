# Legal RAG System - テストガイド

## テスト構成

### テストマーカー

pytest markersでテストを分類:

- `unit`: ユニットテスト（高速、外部依存なし）
- `integration`: 統合テスト（外部サービスが必要な場合あり）
- `slow`: 時間がかかるテスト（数分）
- `xmlparse`: XML解析テスト
- `retrieval`: Retrieval系テスト
- `rag`: RAGパイプラインテスト
- `eval`: 評価テスト

### テストファイル

```
tests/
├── conftest.py              # フィクスチャとpytest設定
├── test_preprocessing.py    # XML前処理テスト
├── test_config.py          # 設定管理テスト
├── test_retrieval.py       # Retrievalシステムテスト
└── test_rag_pipeline.py    # RAGパイプラインテスト
```

## セットアップ

### 1. uv環境のセットアップ

```bash
make setup-uv
```

または

```bash
./setup/setup_uv_env.sh
```

### 2. 仮想環境の有効化

```bash
source .venv/bin/activate
```

## テストの実行

### クイックテスト（ユニットテストのみ）

```bash
make test
```

または

```bash
./scripts/run_tests.sh unit
```

### 全テスト

```bash
make test-all
```

### 統合テストのみ

```bash
make test-integration
```

### カバレッジ付きテスト

```bash
make test-coverage
```

### 特定のマーカーで実行

```bash
# ユニットテストのみ
pytest tests/ -v -m unit

# XML解析テストのみ
pytest tests/ -v -m xmlparse

# 遅いテストを除外
pytest tests/ -v -m "not slow"
```

### 特定のテストファイル

```bash
pytest tests/test_config.py -v
```

### 特定のテスト関数

```bash
pytest tests/test_config.py::test_embedding_config_defaults -v
```

## テストの種類

### ユニットテスト（高速）

外部依存なし、モックを使用:

```bash
pytest tests/ -v -m "unit and not slow"
```

- 設定管理
- データモデル
- ユーティリティ関数
- インターフェース

**実行時間**: 数秒

### 統合テスト（中速）

実際のコンポーネント統合:

```bash
pytest tests/ -v -m integration
```

- XML解析とJSONL生成
- BM25 Retriever
- インデックス保存/ロード

**実行時間**: 数秒〜数十秒

### 遅いテスト（低速）

埋め込みモデルのロードを含む:

```bash
pytest tests/ -v -m slow
```

- Vector Retriever（FAISS）
- Hybrid Retriever
- 埋め込み生成

**実行時間**: 数分

## カバレッジ

カバレッジレポートの生成:

```bash
pytest tests/ -v --cov=app --cov=scripts --cov-report=html --cov-report=term
```

HTMLレポートは`htmlcov/index.html`に生成されます。

## フィクスチャ

### 基本フィクスチャ

- `project_root_path`: プロジェクトルートのパス
- `test_data_dir`: テストデータディレクトリ
- `tmp_path`: pytest組み込みの一時ディレクトリ

### データフィクスチャ

- `sample_xml_content`: サンプルXML文字列
- `sample_xml_file`: サンプルXMLファイル
- `sample_jsonl_data`: サンプルJSONLデータ（リスト）
- `sample_jsonl_file`: サンプルJSONLファイル
- `sample_lawqa_data`: サンプル評価データ

### インデックスフィクスチャ

- `temp_index_dir`: 一時インデックスディレクトリ（自動クリーンアップ）

### 設定フィクスチャ

- `mock_config`: テスト用のRAG設定

## テストの追加

### 1. 新しいテストファイルを作成

```python
"""
新機能のテスト
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.my_module import MyClass


@pytest.mark.unit
def test_my_function():
    """機能のテスト"""
    result = MyClass().my_method()
    assert result == expected_value
```

### 2. フィクスチャの活用

```python
@pytest.mark.integration
def test_with_fixture(sample_jsonl_data, temp_index_dir):
    """フィクスチャを使ったテスト"""
    # sample_jsonl_dataとtemp_index_dirが利用可能
    pass
```

### 3. 適切なマーカーを付与

```python
@pytest.mark.unit  # 高速なユニットテスト
@pytest.mark.integration  # 統合テスト
@pytest.mark.slow  # 時間がかかるテスト
@pytest.mark.xmlparse  # 特定カテゴリ
```

## モックの使用

外部依存をモック化:

```python
from unittest.mock import Mock, patch

@pytest.mark.unit
def test_with_mock():
    """モックを使ったテスト"""
    with patch('app.module.ExternalService') as mock_service:
        mock_service.return_value.method.return_value = "mocked"
        # テスト実行
```

## ベストプラクティス

1. **高速なテストを優先**: ユニットテストを多く書く
2. **明確な命名**: `test_<what>_<condition>_<expected>`
3. **1テスト1検証**: 1つのテスト関数で1つのことをテスト
4. **フィクスチャの活用**: 共通のセットアップはフィクスチャに
5. **マーカーの使用**: テストを適切に分類
6. **クリーンアップ**: 一時ファイルは自動削除
7. **モックの活用**: 外部依存を減らす

## トラブルシューティング

### ImportError

```bash
# プロジェクトルートから実行
cd /home/jovyan/work/legal-rag
pytest tests/ -v
```

### トークナイザーエラー

```bash
# SudachiPyとJanomeをインストール（管理者権限不要）
pip install sudachipy sudachidict-core janome

# または環境を更新
pip install -e . --upgrade
```

### 遅いテストをスキップ

```bash
pytest tests/ -v -m "not slow"
```

### 特定のテストのみ実行

```bash
pytest tests/test_config.py -v
```

## CI/CD統合

GitHub Actionsなどでの実行例:

```yaml
- name: Run tests
  run: |
    make setup-uv
    source .venv/bin/activate
    make test-coverage
```

## 参考資料

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
