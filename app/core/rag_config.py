"""
RAGシステムの設定管理
全てのパラメータを環境変数またはデフォルト値で管理
"""
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    # このファイルからプロジェクトルートまでの相対パスを計算
    # app/core/rag_config.py から見て2階層上がプロジェクトルート
    return Path(__file__).parent.parent.parent


def _load_environment_variables() -> None:
    """`.env` が存在する場合は読み込む"""
    project_root = get_project_root()
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        # `.env` が無い環境でも他の手段で設定されていればそのまま使う
        load_dotenv()


_load_environment_variables()


def get_default_path(relative_path: str) -> str:
    """プロジェクトルートからの相対パスを絶対パスに変換"""
    project_root = get_project_root()
    return str(project_root / relative_path)


class EmbeddingConfig(BaseModel):
    """埋め込みモデル設定（現状はHuggingFaceのみサポート）"""
    provider: Literal["huggingface"] = Field(
        default=os.getenv("EMBEDDING_PROVIDER", "huggingface")
    )
    model_name: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    )
    dimension: int = Field(
        default=int(os.getenv("EMBEDDING_DIM", "1024"))
    )


class LLMConfig(BaseModel):
    """LLM設定（現状はOllamaのみサポート）"""
    provider: Literal["ollama"] = Field(
        default=os.getenv("LLM_PROVIDER", "ollama")
    )
    model_name: str = Field(
        default=os.getenv("LLM_MODEL", "gpt-oss:20b")
    )
    temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    max_tokens: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "2048"))
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
    mmr_fetch_k_max: int = Field(
        default=int(os.getenv("MMR_FETCH_K_MAX", "50"))
    )
    # Hybrid Retriever設定
    fusion_method: Literal["rrf", "weighted_rrf", "weighted"] = Field(
        default=os.getenv("FUSION_METHOD", "rrf")
    )
    vector_weight: float = Field(
        default=float(os.getenv("VECTOR_WEIGHT", "0.5"))
    )
    bm25_weight: float = Field(
        default=float(os.getenv("BM25_WEIGHT", "0.5"))
    )
    rrf_k: int = Field(
        default=int(os.getenv("RRF_K", "60"))
    )
    fetch_k_multiplier: int = Field(
        default=int(os.getenv("FETCH_K_MULTIPLIER", "2"))
    )
    # BM25設定
    bm25_tokenizer: Literal["auto", "sudachi", "janome", "mecab", "ngram", "simple"] = Field(
        default=os.getenv("BM25_TOKENIZER", "auto")
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
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    vector_store_path: str = Field(
        default=os.getenv("VECTOR_STORE_PATH", get_default_path("data/faiss_index"))
    )
    data_path: str = Field(
        default=os.getenv("DATA_PATH", get_default_path("data/egov_laws.jsonl"))
    )


def load_config() -> RAGConfig:
    """設定をロード"""
    return RAGConfig()
