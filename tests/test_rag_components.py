"""
RAGコンポーネントの基本テスト
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_document_creation():
    """Documentの作成テスト"""
    from app.retrieval.base import Document
    
    doc = Document(
        page_content="テスト本文",
        metadata={"law_title": "テスト法", "article": "1"},
        score=0.95
    )
    assert doc.page_content == "テスト本文"
    assert doc.metadata["law_title"] == "テスト法"
    assert doc.score == 0.95


def test_document_metadata_default_isolation():
    """metadata のデフォルト辞書が共有されないことを確認"""
    from app.retrieval.base import Document

    doc_a = Document(page_content="A")
    doc_b = Document(page_content="B")

    doc_a.metadata["law_title"] = "法A"

    assert "law_title" not in doc_b.metadata
    assert doc_a.metadata["law_title"] == "法A"


def test_config_loading():
    """設定ロードテスト"""
    from app.core.rag_config import load_config
    
    config = load_config()
    assert config.embedding.provider in ["openai", "huggingface", "ollama"]
    assert config.llm.provider in ["openai", "ollama", "anthropic"]
    assert config.retriever.retriever_type in ["vector", "bm25", "hybrid"]
    assert config.chunking.chunk_size > 0
    assert config.retriever.top_k > 0


def test_jsonl_parsing():
    """JSONLファイルのパーステスト"""
    import json
    
    test_jsonl = Path(__file__).parent.parent / "data" / "test_egov.jsonl"
    
    if not test_jsonl.exists():
        return
    
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            assert "law_title" in data
            assert "text" in data
            assert "article" in data


if __name__ == "__main__":
    test_document_creation()
    print("✓ Document creation test passed")
    
    test_config_loading()
    print("✓ Config loading test passed")
    
    test_jsonl_parsing()
    print("✓ JSONL parsing test passed")
    
    print("\nAll tests passed!")
