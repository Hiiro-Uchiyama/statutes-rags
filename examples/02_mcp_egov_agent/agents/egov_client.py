"""
e-Gov API v2 クライアント

e-Gov法令API Version 2との通信を管理するクライアントクラス。
API仕様: https://laws.e-gov.go.jp/api/2/swagger-ui
"""
import logging
import time
from typing import Dict, Any, List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class EGovAPIError(Exception):
    """e-Gov API エラー"""
    pass


class EGovAPIClient:
    """
    e-Gov API v2 クライアント
    
    法令データの検索、取得を行うためのクライアント。
    タイムアウト、リトライ、エラーハンドリングを含む。
    """
    
    def __init__(
        self,
        base_url: str = "https://laws.e-gov.go.jp/api/2",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Args:
            base_url: APIベースURL
            timeout: タイムアウト秒数
            max_retries: 最大リトライ回数
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout)
        logger.info(f"EGovAPIClient initialized: base_url={base_url}, timeout={timeout}s")
    
    def __del__(self):
        """クライアントのクリーンアップ"""
        try:
            self.client.close()
        except Exception:
            pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True
    )
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        HTTPリクエストの実行（リトライ機能付き）
        
        Args:
            method: HTTPメソッド
            endpoint: エンドポイント
            params: クエリパラメータ
            **kwargs: その他のhttpxパラメータ
        
        Returns:
            httpx.Response
        
        Raises:
            EGovAPIError: API呼び出しエラー
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Requesting: {method} {url} with params={params}")
            response = self.client.request(method, url, params=params, **kwargs)
            response.raise_for_status()
            return response
        
        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {url}")
            raise EGovAPIError(f"API request timeout: {e}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            raise EGovAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
        
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {url}")
            raise EGovAPIError(f"Connection error: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}")
            raise EGovAPIError(f"Unexpected error: {e}")
    
    def search_by_keyword(
        self,
        keyword: str,
        law_type: Optional[List[str]] = None,
        category_cd: Optional[List[str]] = None,
        asof: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        キーワード検索API
        
        法令本文をキーワードで検索します。
        
        Args:
            keyword: 検索キーワード（必須）
            law_type: 法令種別のリスト（例: ["Act", "Rule"]）
            category_cd: 事項別分類コードのリスト（例: ["011", "021"]）
            asof: 法令の時点（YYYY-MM-DD形式）
            **kwargs: その他のパラメータ
        
        Returns:
            検索結果のJSON
        
        Raises:
            EGovAPIError: API呼び出しエラー
        """
        params = {"keyword": keyword}
        
        if law_type:
            params["law_type"] = ",".join(law_type)
        
        if category_cd:
            params["category_cd"] = ",".join(category_cd)
        
        if asof:
            params["asof"] = asof
        
        # 追加パラメータ
        params.update(kwargs)
        
        try:
            response = self._request("GET", "/keyword", params=params)
            result = response.json()
            logger.info(f"Keyword search successful: keyword='{keyword}', results={len(result.get('items', []))}")
            return result
        
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise
    
    def get_law_data(
        self,
        law_id_or_num: str,
        response_format: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """
        法令本文取得API
        
        法令IDまたは法令番号を指定して法令全文を取得します。
        
        Args:
            law_id_or_num: 法令IDまたは法令番号
            response_format: レスポンス形式（"json" or "xml"、デフォルト: "json"）
            **kwargs: その他のパラメータ
        
        Returns:
            法令データのJSON
        
        Raises:
            EGovAPIError: API呼び出しエラー
        """
        params = {"format": response_format}
        params.update(kwargs)
        
        try:
            response = self._request(
                "GET",
                f"/law_data/{law_id_or_num}",
                params=params
            )
            
            if response_format == "json":
                result = response.json()
            else:
                result = {"xml": response.text}
            
            logger.info(f"Law data retrieved: {law_id_or_num}")
            return result
        
        except Exception as e:
            logger.error(f"Get law data failed for {law_id_or_num}: {e}")
            raise
    
    def get_laws(
        self,
        category_cd: Optional[List[str]] = None,
        law_type: Optional[List[str]] = None,
        asof: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        法令一覧取得API
        
        法令の一覧を取得します。
        
        Args:
            category_cd: 事項別分類コードのリスト
            law_type: 法令種別のリスト
            asof: 法令の時点（YYYY-MM-DD形式）
            **kwargs: その他のパラメータ
        
        Returns:
            法令一覧のJSON
        
        Raises:
            EGovAPIError: API呼び出しエラー
        """
        params = {}
        
        if category_cd:
            params["category_cd"] = ",".join(category_cd)
        
        if law_type:
            params["law_type"] = ",".join(law_type)
        
        if asof:
            params["asof"] = asof
        
        params.update(kwargs)
        
        try:
            response = self._request("GET", "/laws", params=params)
            result = response.json()
            logger.info(f"Laws list retrieved: {len(result.get('items', []))} laws")
            return result
        
        except Exception as e:
            logger.error(f"Get laws list failed: {e}")
            raise
    
    def get_law_revisions(
        self,
        law_id_or_num: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        法令履歴一覧取得API
        
        法令の改正履歴を取得します。
        
        Args:
            law_id_or_num: 法令IDまたは法令番号
            **kwargs: その他のパラメータ
        
        Returns:
            法令履歴のJSON
        
        Raises:
            EGovAPIError: API呼び出しエラー
        """
        params = kwargs
        
        try:
            response = self._request(
                "GET",
                f"/law_revisions/{law_id_or_num}",
                params=params
            )
            result = response.json()
            logger.info(f"Law revisions retrieved: {law_id_or_num}")
            return result
        
        except Exception as e:
            logger.error(f"Get law revisions failed for {law_id_or_num}: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        APIの疎通確認
        
        Returns:
            APIが正常に応答する場合True
        """
        try:
            # 簡単なクエリで疎通確認
            self.get_laws()
            logger.info("API health check: OK")
            return True
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False

