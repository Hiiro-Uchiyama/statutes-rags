"""
Multi-Agent Debate設定管理
"""
import os
from typing import Optional
from pydantic import BaseModel, Field


class MultiAgentDebateConfig(BaseModel):
    """Multi-Agent Debate設定"""
    
    # 議論設定
    max_debate_rounds: int = Field(
        default=int(os.getenv("DEBATE_MAX_ROUNDS", "3")),
        description="最大議論ラウンド数"
    )
    agreement_threshold: float = Field(
        default=float(os.getenv("DEBATE_AGREEMENT_THRESHOLD", "0.8")),
        description="合意判定の閾値（類似度）"
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
        default=int(os.getenv("DEBATE_RETRIEVAL_TOP_K", "10")),
        description="検索時の取得文書数"
    )
    
    # ベクトルストアとデータのパス
    vector_store_path: str = Field(
        default=os.getenv("VECTOR_STORE_PATH", "data/faiss_index"),
        description="ベクトルストアのパス"
    )
    data_path: str = Field(
        default=os.getenv("DATA_PATH", "data/egov_laws.jsonl"),
        description="法令データのパス"
    )
    
    # 埋め込みモデル設定（合意判定用）
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
        description="埋め込みモデル名"
    )
    
    # デバッグ設定
    verbose: bool = Field(
        default=os.getenv("DEBATE_VERBOSE", "false").lower() == "true",
        description="詳細ログを出力するか"
    )
    
    class Config:
        """Pydantic設定"""
        validate_assignment = True


def load_config(config_path: Optional[str] = None) -> MultiAgentDebateConfig:
    """
    設定をロード
    
    Args:
        config_path: 設定ファイルのパス（オプション）
    
    Returns:
        MultiAgentDebateConfig インスタンス
    """
    # 環境変数から設定を読み込み
    return MultiAgentDebateConfig()

