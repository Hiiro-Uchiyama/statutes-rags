"""
MCP e-Gov Agent

e-Gov APIとローカルデータを動的に使い分けるエージェント。
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

from examples.shared.base_agent import BaseAgent
from ..config import MCPEgovConfig
from .tools import EgovToolsFactory
from .egov_client import EGovAPIError

logger = logging.getLogger(__name__)


class MCPEgovAgent(BaseAgent):
    """
    MCP e-Gov Agent
    
    質問を分析し、e-Gov APIとローカルデータを動的に選択して検索を行うエージェント。
    """
    
    def __init__(
        self,
        llm,
        config: MCPEgovConfig,
        local_retriever = None
    ):
        """
        Args:
            llm: LLMインスタンス
            config: MCP e-Gov Agent設定
            local_retriever: ローカル検索用のRetriever（オプション）
        """
        super().__init__(llm, config)
        self.config = config
        self.local_retriever = local_retriever
        
        # Toolsの初期化
        self.tools_factory = EgovToolsFactory(config, local_retriever)
        self.tools = self.tools_factory.create_tools()
        self.api_client = self.tools_factory.get_api_client()
        
        logger.info(f"MCPEgovAgent initialized with {len(self.tools)} tools")
    
    def _should_use_api(self, query: str, metadata: Optional[Dict] = None) -> bool:
        """
        APIを使用すべきか判定
        
        Args:
            query: 検索クエリ
            metadata: 追加メタデータ
        
        Returns:
            APIを使用する場合True
        """
        # 基本設定でAPI優先が無効の場合
        if not self.config.prefer_api:
            return False
        
        # API疎通確認（簡易）
        # 注: 毎回チェックするとコストが高いため、実運用では別の方法を検討
        try:
            # APIが使用可能か確認（簡易版: タイムアウトを短く設定）
            # 本実装ではスキップしてTrue/Falseを設定で判定
            pass
        except Exception:
            logger.warning("API health check failed, falling back to local")
            return False
        
        # 最近の法令に関する質問か判定
        if self.config.use_api_for_recent_laws:
            # 質問文から年号や日付を抽出して判定
            if self._is_recent_law_query(query):
                logger.info("Query appears to be about recent laws, using API")
                return True
        
        # デフォルトではAPIを使用
        return True
    
    def _is_recent_law_query(self, query: str) -> bool:
        """
        最近の法令に関する質問か判定
        
        Args:
            query: 検索クエリ
        
        Returns:
            最近の法令に関する質問の場合True
        """
        # 令和の年号を検出
        reiwa_pattern = r'令和(\d+)年'
        match = re.search(reiwa_pattern, query)
        
        if match:
            year = int(match.group(1))
            current_reiwa_year = datetime.now().year - 2018  # 令和元年 = 2019年
            
            # 設定された閾値以内の年号の場合
            threshold_years = self.config.recent_law_threshold_days / 365
            if current_reiwa_year - year < threshold_years:
                return True
        
        # 「最新」「新しい」「改正」などのキーワード
        recent_keywords = ['最新', '新しい', '改正', '施行', '最近']
        for keyword in recent_keywords:
            if keyword in query:
                return True
        
        return False
    
    def _extract_law_number_from_query(self, query: str) -> Optional[str]:
        """
        質問文から法令番号を抽出
        
        Args:
            query: 検索クエリ
        
        Returns:
            法令番号（見つからない場合はNone）
        """
        # 法令番号のパターン（例: 平成二十八年個人情報保護委員会規則第六号）
        patterns = [
            r'(明治|大正|昭和|平成|令和).*?第\d+号',
            r'(明治|大正|昭和|平成|令和).*?法律.*?第\d+号',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(0)
        
        return None
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        エージェントの主要な処理を実行
        
        Args:
            input_data: {
                "query": str,  # 検索クエリ
                "force_api": bool (optional),  # API強制使用
                "force_local": bool (optional),  # ローカル強制使用
            }
        
        Returns:
            {
                "documents": List[Document],  # 取得したドキュメント
                "source": str,  # "api" or "local" or "hybrid"
                "metadata": Dict  # 追加メタデータ
            }
        """
        query = input_data.get("query", "")
        force_api = input_data.get("force_api", False)
        force_local = input_data.get("force_local", False)
        
        if not query:
            return {
                "documents": [],
                "source": "none",
                "metadata": {"error": "Empty query"}
            }
        
        logger.info(f"Executing MCPEgovAgent with query: {query[:100]}...")
        
        # 強制設定の確認
        use_api = force_api or (not force_local and self._should_use_api(query))
        
        documents = []
        source = "none"
        metadata = {
            "query": query,
            "use_api": use_api,
            "timestamp": datetime.now().isoformat()
        }
        
        # API経由で検索
        if use_api:
            try:
                api_docs = self._search_with_api(query)
                documents.extend(api_docs)
                source = "api"
                metadata["api_success"] = True
                logger.info(f"API search returned {len(api_docs)} documents")
            
            except EGovAPIError as e:
                logger.error(f"API search failed: {e}")
                metadata["api_success"] = False
                metadata["api_error"] = str(e)
                
                # フォールバックが有効な場合
                if self.config.fallback_to_local and self.local_retriever:
                    logger.info("Falling back to local search")
                    local_docs = self._search_with_local(query)
                    documents.extend(local_docs)
                    source = "local_fallback"
                    metadata["fallback_used"] = True
        
        # ローカル検索
        else:
            if self.local_retriever:
                local_docs = self._search_with_local(query)
                documents.extend(local_docs)
                source = "local"
                logger.info(f"Local search returned {len(local_docs)} documents")
            else:
                logger.warning("Local retriever not available")
                metadata["error"] = "Local retriever not available"
        
        return {
            "documents": documents,
            "source": source,
            "metadata": metadata
        }
    
    def _search_with_api(self, query: str) -> List[Any]:
        """
        e-Gov APIで検索
        
        Args:
            query: 検索クエリ
        
        Returns:
            ドキュメントのリスト
        """
        from app.retrieval.base import Document
        
        documents = []
        
        # 法令番号が含まれる場合は直接取得を試みる
        law_number = self._extract_law_number_from_query(query)
        if law_number:
            try:
                result = self.api_client.get_law_data(law_number)
                # 結果をDocumentに変換（簡易版）
                doc = self._convert_api_result_to_document(result, "law_data")
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.debug(f"Failed to get law data for {law_number}: {e}")
        
        # キーワード検索
        try:
            result = self.api_client.search_by_keyword(query)
            
            # 結果をDocumentに変換
            for law in result.get("items", [])[:self.config.retrieval_top_k]:
                doc = self._convert_api_result_to_document(law, "keyword_search")
                if doc:
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise
        
        return documents
    
    def _search_with_local(self, query: str) -> List[Any]:
        """
        ローカルデータで検索
        
        Args:
            query: 検索クエリ
        
        Returns:
            ドキュメントのリスト
        """
        if not self.local_retriever:
            return []
        
        try:
            documents = self.local_retriever.retrieve(
                query,
                top_k=self.config.retrieval_top_k
            )
            return documents
        
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []
    
    def _convert_api_result_to_document(
        self,
        api_result: Dict[str, Any],
        result_type: str
    ) -> Optional[Any]:
        """
        API結果をDocumentオブジェクトに変換
        
        Args:
            api_result: APIからの結果
            result_type: 結果タイプ（"keyword_search" or "law_data"）
        
        Returns:
            Documentオブジェクト
        """
        from app.retrieval.base import Document
        
        try:
            if result_type == "keyword_search":
                law_info = api_result.get("law_info", {})
                matched_provisions = api_result.get("matched_provisions", [])
                
                if not matched_provisions:
                    return None
                
                provision = matched_provisions[0]
                
                # メタデータの構築
                metadata = {
                    "law_title": law_info.get("law_title", ""),
                    "law_num": law_info.get("law_num", ""),
                    "article": provision.get("article", {}).get("article_num", ""),
                    "source": "egov_api"
                }
                
                # 本文の抽出
                text = provision.get("text", "")
                
                return Document(
                    page_content=text,
                    metadata=metadata,
                    score=1.0  # APIスコアは仮置き
                )
            
            elif result_type == "law_data":
                law_title = api_result.get("law_title", "")
                law_num = api_result.get("law_num", "")
                
                # 簡易版: 法令の概要を返す
                metadata = {
                    "law_title": law_title,
                    "law_num": law_num,
                    "source": "egov_api"
                }
                
                # 本文の簡易抽出
                law_body = api_result.get("law_body", {})
                text = f"{law_title}\n{law_num}"
                
                return Document(
                    page_content=text,
                    metadata=metadata,
                    score=1.0
                )
        
        except Exception as e:
            logger.error(f"Failed to convert API result to Document: {e}")
            return None

