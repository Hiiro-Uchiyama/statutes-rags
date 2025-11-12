"""
Legal Case Generator 設定管理
"""
import os
from typing import Optional
from pydantic import BaseModel, Field


class LegalCaseConfig(BaseModel):
    """Legal Case Generator設定"""
    
    # 生成設定
    # 注: 現在は3つの事例タイプ（適用・非適用・境界）が固定で生成されます
    # max_cases_per_law は将来の拡張用として残されています
    # max_cases_per_law: int = Field(
    #     default=int(os.getenv("LEGAL_CASE_MAX_CASES", "3")),
    #     description="法令あたりの生成事例数"
    # )
    
    min_length: int = Field(
        default=int(os.getenv("LEGAL_CASE_MIN_LENGTH", "100")),
        description="事例の最小文字数"
    )
    max_length: int = Field(
        default=int(os.getenv("LEGAL_CASE_MAX_LENGTH", "500")),
        description="事例の最大文字数"
    )
    max_iterations: int = Field(
        default=int(os.getenv("LEGAL_CASE_MAX_ITERATIONS", "2")),
        description="最大反復回数（洗練プロセス）"
    )
    mcq_target_length: int = Field(
        default=int(os.getenv("MCQ_CASE_TARGET_LENGTH", "500")),
        description="4択問題用シナリオの目標文字数"
    )
    mcq_min_length: int = Field(
        default=int(os.getenv("MCQ_CASE_MIN_LENGTH", "460")),
        description="4択問題用シナリオの最小文字数"
    )
    mcq_max_length: int = Field(
        default=int(os.getenv("MCQ_CASE_MAX_LENGTH", "540")),
        description="4択問題用シナリオの最大文字数"
    )
    mcq_max_iterations: int = Field(
        default=int(os.getenv("MCQ_CASE_MAX_ITERATIONS", "6")),
        description="4択問題シナリオ生成の最大反復回数"
    )
    mcq_validation_threshold: float = Field(
        default=float(os.getenv("MCQ_CASE_VALIDATION_THRESHOLD", "0.75")),
        description="4択問題シナリオの整合性スコア閾値"
    )
    
    # 事例タイプの生成制御
    generate_applicable: bool = Field(
        default=os.getenv("LEGAL_CASE_GEN_APPLICABLE", "true").lower() == "true",
        description="適用事例を生成"
    )
    generate_non_applicable: bool = Field(
        default=os.getenv("LEGAL_CASE_GEN_NON_APPLICABLE", "true").lower() == "true",
        description="非適用事例を生成"
    )
    generate_boundary: bool = Field(
        default=os.getenv("LEGAL_CASE_GEN_BOUNDARY", "true").lower() == "true",
        description="境界事例を生成"
    )
    
    # LLM設定
    llm_model: str = Field(
        default=os.getenv("LLM_MODEL", "qwen3:8b"),
        description="使用するLLMモデル"
    )
    llm_temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        description="LLMの温度パラメータ（事例生成ではやや高めで多様性確保）"
    )
    llm_timeout: int = Field(
        default=int(os.getenv("LLM_TIMEOUT", "300")),
        description="LLMリクエストのタイムアウト（秒）"
    )
    llm_retry_attempts: int = Field(
        default=int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
        description="LLM呼び出し時の再試行回数"
    )
    llm_retry_delay: float = Field(
        default=float(os.getenv("LLM_RETRY_DELAY", "5.0")),
        description="LLM再試行前の待機秒数"
    )
    
    # 検証設定
    validation_threshold: float = Field(
        default=float(os.getenv("LEGAL_CASE_VALIDATION_THRESHOLD", "0.7")),
        description="法的整合性の閾値（これを超えたら合格）"
    )
    
    # 以下は将来の拡張用（現在は未使用）
    # ベクトルストアとデータのパス
    # vector_store_path: str = Field(
    #     default=os.getenv("VECTOR_STORE_PATH", "data/faiss_index"),
    #     description="ベクトルストアのパス（法令条文取得用）"
    # )
    # data_path: str = Field(
    #     default=os.getenv("DATA_PATH", "data/egov_laws.jsonl"),
    #     description="法令データのパス"
    # )
    
    class Config:
        """Pydantic設定"""
        validate_assignment = True


def load_config(config_path: Optional[str] = None) -> LegalCaseConfig:
    """
    設定をロード
    
    Args:
        config_path: 設定ファイルのパス（オプション）
    
    Returns:
        LegalCaseConfig インスタンス
    """
    # 環境変数から設定を読み込み
    return LegalCaseConfig()

