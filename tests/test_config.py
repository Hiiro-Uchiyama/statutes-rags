"""
設定管理のテスト
"""
import pytest
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag_config import (
    EmbeddingConfig,
    LLMConfig,
    RetrieverConfig,
    RerankerConfig,
    RAGConfig,
    load_config
)


@pytest.mark.unit
def test_embedding_config_defaults():
    """埋め込み設定のデフォルト値テスト"""
    config = EmbeddingConfig()
    assert config.provider == "huggingface"
    assert config.model_name is not None
    assert config.dimension > 0


@pytest.mark.unit
def test_llm_config_defaults():
    """LLM設定のデフォルト値テスト"""
    config = LLMConfig()
    assert config.provider == "ollama"
    assert config.model_name is not None
    assert 0 <= config.temperature <= 1
    assert config.max_tokens > 0




@pytest.mark.unit
def test_retriever_config_defaults():
    """Retriever設定のデフォルト値テスト"""
    config = RetrieverConfig()
    assert config.retriever_type in ["vector", "bm25", "hybrid"]
    assert config.top_k > 0
    assert isinstance(config.use_mmr, bool)
    assert 0 <= config.mmr_lambda <= 1


@pytest.mark.unit
def test_reranker_config_defaults():
    """Reranker設定のデフォルト値テスト"""
    config = RerankerConfig()
    assert isinstance(config.enabled, bool)
    assert config.model_name is not None
    assert config.top_n > 0


@pytest.mark.unit
def test_rag_config_composition():
    """RAG設定の構成テスト"""
    config = RAGConfig()
    
    assert isinstance(config.embedding, EmbeddingConfig)
    assert isinstance(config.llm, LLMConfig)
    assert isinstance(config.retriever, RetrieverConfig)
    assert isinstance(config.reranker, RerankerConfig)
    
    assert config.vector_store_path is not None
    assert config.data_path is not None


@pytest.mark.unit
def test_config_from_env_vars(monkeypatch):
    """環境変数からの設定ロードテスト"""
    import os
    import importlib
    import sys
    
    # 環境変数を設定
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    monkeypatch.setenv("RETRIEVER_TYPE", "hybrid")
    monkeypatch.setenv("RETRIEVER_TOP_K", "15")
    monkeypatch.setenv("USE_MMR", "true")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    
    # モジュールをリロード
    if 'app.core.rag_config' in sys.modules:
        importlib.reload(sys.modules['app.core.rag_config'])
    
    from app.core.rag_config import load_config
    config = load_config()
    
    assert config.embedding.provider == "huggingface"
    assert config.embedding.model_name == "test-model"
    assert config.embedding.dimension == 768
    assert config.llm.provider == "ollama"
    assert config.llm.model_name == "test-llm"
    assert config.llm.temperature == 0.5
    assert config.retriever.retriever_type == "hybrid"
    assert config.retriever.top_k == 15
    assert config.retriever.use_mmr == True
    assert config.reranker.enabled == True


@pytest.mark.unit
def test_config_validation():
    """設定の検証テスト"""
    # 正常な設定
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="huggingface",
            model_name="test-model",
            dimension=768
        )
    )
    
    assert config.embedding.dimension == 768


@pytest.mark.unit
def test_load_config_function():
    """load_config関数のテスト"""
    config = load_config()
    
    # 型チェックの代わりに属性チェック
    assert hasattr(config, 'embedding')
    assert hasattr(config, 'llm')
    assert hasattr(config, 'retriever')
    assert hasattr(config, 'reranker')
    
    assert config.embedding is not None
    assert config.llm is not None
    assert config.retriever is not None


@pytest.mark.unit
def test_retriever_type_enum():
    """Retrieverタイプの列挙テスト"""
    valid_types = ["vector", "bm25", "hybrid"]
    
    for rtype in valid_types:
        config = RetrieverConfig(retriever_type=rtype)
        assert config.retriever_type == rtype


@pytest.mark.unit
def test_mmr_lambda_range():
    """MMR lambdaの範囲テスト"""
    # 有効な値
    for lambda_val in [0.0, 0.5, 1.0]:
        config = RetrieverConfig(mmr_lambda=lambda_val)
        assert config.mmr_lambda == lambda_val
