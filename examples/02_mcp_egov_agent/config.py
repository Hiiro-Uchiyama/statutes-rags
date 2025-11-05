"""
MCP e-Gov Agent 設定管理
"""
import os
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field


class MCPEgovConfig(BaseModel):
    """MCP e-Gov Agent 設定"""
    
    # e-Gov API設定
    api_base_url: str = Field(
        default=os.getenv("EGOV_API_BASE_URL", "https://laws.e-gov.go.jp/api/2"),
        description="e-Gov API v2のベースURL"
    )
    api_timeout: int = Field(
        default=int(os.getenv("EGOV_API_TIMEOUT", "30")),
        description="APIタイムアウト（秒）"
    )
    api_max_retries: int = Field(
        default=int(os.getenv("EGOV_API_MAX_RETRIES", "3")),
        description="API最大リトライ回数"
    )
    
    # ハイブリッド戦略設定
    prefer_api: bool = Field(
        default=os.getenv("MCP_PREFER_API", "true").lower() == "true",
        description="API優先モード（Trueの場合、可能な限りAPIを使用）"
    )
    fallback_to_local: bool = Field(
        default=os.getenv("MCP_FALLBACK_TO_LOCAL", "true").lower() == "true",
        description="ローカルフォールバック有効化"
    )
    
    # 動的判断設定
    use_api_for_recent_laws: bool = Field(
        default=os.getenv("MCP_USE_API_FOR_RECENT", "true").lower() == "true",
        description="最近の法令はAPI優先（改正の可能性が高いため）"
    )
    recent_law_threshold_days: int = Field(
        default=int(os.getenv("MCP_RECENT_LAW_DAYS", "90")),
        description="最近の法令と判定する日数（日）"
    )
    
    # キャッシュ設定（将来の拡張用）
    enable_cache: bool = Field(
        default=os.getenv("MCP_ENABLE_CACHE", "false").lower() == "true",
        description="APIレスポンスのキャッシュ有効化"
    )
    cache_ttl: int = Field(
        default=int(os.getenv("MCP_CACHE_TTL", "3600")),
        description="キャッシュの有効期間（秒）"
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
        default=int(os.getenv("MCP_RETRIEVAL_TOP_K", "10")),
        description="検索時の取得文書数"
    )
    rerank_top_n: int = Field(
        default=int(os.getenv("MCP_RERANK_TOP_N", "5")),
        description="リランク後の文書数"
    )
    
    # ローカルデータ設定
    vector_store_path: str = Field(
        default=os.getenv("VECTOR_STORE_PATH", "data/faiss_index"),
        description="ベクトルストアのパス"
    )
    data_path: str = Field(
        default=os.getenv("DATA_PATH", "data/egov_laws.jsonl"),
        description="法令データのパス"
    )
    
    # プロジェクトルート（自動設定）
    project_root: Optional[Path] = Field(
        default=None,
        description="プロジェクトルートディレクトリ"
    )
    
    class Config:
        """Pydantic設定"""
        validate_assignment = True
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """初期化時にプロジェクトルートを設定"""
        super().__init__(**data)
        
        if self.project_root is None:
            # このファイルから3階層上がプロジェクトルート
            current_file = Path(__file__)
            self.project_root = current_file.parent.parent.parent
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """
        相対パスを絶対パスに変換
        
        Args:
            relative_path: プロジェクトルートからの相対パス
        
        Returns:
            絶対パス
        """
        if Path(relative_path).is_absolute():
            return Path(relative_path)
        return self.project_root / relative_path
    
    def validate_paths(self) -> bool:
        """
        設定パスの存在確認
        
        Returns:
            すべてのパスが存在する場合True
        """
        vector_store = self.get_absolute_path(self.vector_store_path)
        data_file = self.get_absolute_path(self.data_path)
        
        if not vector_store.exists():
            raise FileNotFoundError(f"Vector store not found: {vector_store}")
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        return True


def load_config(
    config_path: Optional[str] = None,
    validate: bool = True
) -> MCPEgovConfig:
    """
    設定をロード
    
    Args:
        config_path: 設定ファイルのパス（オプション、将来の拡張用）
        validate: パスの検証を実施するか
    
    Returns:
        MCPEgovConfig インスタンス
    """
    # 環境変数から設定を読み込み
    config = MCPEgovConfig()
    
    # パスの検証（オプション）
    if validate:
        try:
            config.validate_paths()
        except FileNotFoundError as e:
            import logging
            logging.warning(f"Path validation failed: {e}")
            logging.warning("Continuing without validation...")
    
    return config

