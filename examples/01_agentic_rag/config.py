"""
Agentic RAG設定管理
"""
import os
from pydantic import BaseModel, Field


class AgenticRAGConfig(BaseModel):
    """Agentic RAG設定"""
    
    # 反復設定
    max_iterations: int = Field(
        default=int(os.getenv("AGENTIC_MAX_ITERATIONS", "3")),
        description="最大反復回数"
    )
    confidence_threshold: float = Field(
        default=float(os.getenv("AGENTIC_CONFIDENCE_THRESHOLD", "0.8")),
        description="満足度閾値（これを超えたら反復を終了）"
    )
    
    # エージェント有効化
    enable_reasoning: bool = Field(
        default=os.getenv("AGENTIC_ENABLE_REASONING", "true").lower() == "true",
        description="Reasoning Agentを有効化"
    )
    enable_validation: bool = Field(
        default=os.getenv("AGENTIC_ENABLE_VALIDATION", "true").lower() == "true",
        description="Validation Agentを有効化"
    )
    
    # 複雑度閾値
    complexity_simple_threshold: float = Field(
        default=float(os.getenv("AGENTIC_COMPLEXITY_SIMPLE", "0.3")),
        description="単純と判定する閾値"
    )
    complexity_complex_threshold: float = Field(
        default=float(os.getenv("AGENTIC_COMPLEXITY_COMPLEX", "0.7")),
        description="複雑と判定する閾値"
    )
    
    # LLM設定
    llm_model: str = Field(
        default=os.getenv("LLM_MODEL", "qwen3:8b"),
        description="使用するLLMモデル"
    )
    llm_temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        description="LLMの温度パラメータ"
    )
    llm_timeout: int = Field(
        default=int(os.getenv("LLM_TIMEOUT", "60")),
        description="LLMリクエストのタイムアウト（秒）"
    )
    
    # 検索設定
    retrieval_top_k: int = Field(
        default=int(os.getenv("AGENTIC_RETRIEVAL_TOP_K", "10")),
        description="検索時の取得文書数"
    )
    
    # ベクトルストアとデータのパス（プロジェクトルートからの相対パス）
    vector_store_path: str = Field(
        default=os.getenv("VECTOR_STORE_PATH", "data/faiss_index"),
        description="ベクトルストアのパス（プロジェクトルートからの相対パス）"
    )
    data_path: str = Field(
        default=os.getenv("DATA_PATH", "data/egov_laws.jsonl"),
        description="法令データのパス（プロジェクトルートからの相対パス）"
    )
    
    class Config:
        """Pydantic設定"""
        validate_assignment = True


def load_config() -> AgenticRAGConfig:
    """
    設定をロード
    
    環境変数から設定を読み込みます。
    
    Returns:
        AgenticRAGConfig インスタンス
    """
    return AgenticRAGConfig()

