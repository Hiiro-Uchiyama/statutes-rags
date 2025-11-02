"""
RAGシステムの設定管理
全てのパラメータを環境変数またはデフォルト値で管理
"""
import os
from typing import Literal
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """埋め込みモデル設定"""
    provider: Literal["openai", "huggingface", "ollama"] = Field(
        default=os.getenv("EMBEDDING_PROVIDER", "huggingface")
    )
    model_name: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    )
    dimension: int = Field(
        default=int(os.getenv("EMBEDDING_DIM", "1024"))
    )


class LLMConfig(BaseModel):
    """LLM設定"""
    provider: Literal["openai", "ollama", "anthropic"] = Field(
        default=os.getenv("LLM_PROVIDER", "ollama")
    )
    model_name: str = Field(
        default=os.getenv("LLM_MODEL", "qwen2.5:7b")
    )
    temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    max_tokens: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )


class ChunkingConfig(BaseModel):
    """チャンク分割設定"""
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "50"))
    )


class RetrieverConfig(BaseModel):
    """Retriever設定"""
    retriever_type: Literal["vector", "bm25", "hybrid"] = Field(
        default=os.getenv("RETRIEVER_TYPE", "hybrid")
    )
    top_k: int = Field(
        default=int(os.getenv("RETRIEVER_TOP_K", "10"))
    )
    use_mmr: bool = Field(
        default=os.getenv("USE_MMR", "true").lower() == "true"
    )
    mmr_lambda: float = Field(
        default=float(os.getenv("MMR_LAMBDA", "0.5"))
    )


class RerankerConfig(BaseModel):
    """Reranker設定"""
    enabled: bool = Field(
        default=os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    )
    model_name: str = Field(
        default=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    )
    top_n: int = Field(
        default=int(os.getenv("RERANKER_TOP_N", "5"))
    )


class RAGConfig(BaseModel):
    """RAGシステム全体の設定"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    vector_store_path: str = Field(
        default=os.getenv("VECTOR_STORE_PATH", "/home/jovyan/work/statutes-rags/data/faiss_index")
    )
    data_path: str = Field(
        default=os.getenv("DATA_PATH", "/home/jovyan/work/statutes-rags/data/egov_laws.jsonl")
    )


def load_config() -> RAGConfig:
    """設定をロード"""
    return RAGConfig()
