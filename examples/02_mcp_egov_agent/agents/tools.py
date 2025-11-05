"""
LangChain Tools定義

e-Gov APIとローカル検索のためのToolsを定義します。
"""
import logging
from typing import Dict, Any, Optional, List
from langchain_core.tools import Tool
from .egov_client import EGovAPIClient, EGovAPIError
from ..config import MCPEgovConfig

logger = logging.getLogger(__name__)


class EgovToolsFactory:
    """
    e-Gov API用のLangChain Toolsを生成するファクトリクラス
    """
    
    def __init__(
        self,
        config: MCPEgovConfig,
        local_retriever = None
    ):
        """
        Args:
            config: MCP e-Gov Agent設定
            local_retriever: ローカル検索用のRetriever（オプション）
        """
        self.config = config
        self.client = EGovAPIClient(
            base_url=config.api_base_url,
            timeout=config.api_timeout,
            max_retries=config.api_max_retries
        )
        self.local_retriever = local_retriever
    
    def _keyword_search_function(self, query: str) -> str:
        """
        キーワード検索のツール関数
        
        Args:
            query: 検索クエリ
        
        Returns:
            検索結果のテキスト表現
        """
        try:
            result = self.client.search_by_keyword(keyword=query)
            
            if not result.get("items"):
                return "検索結果が見つかりませんでした。"
            
            # 結果を整形
            laws = result.get("items", [])
            output_lines = [f"検索結果: {len(laws)}件の法令が見つかりました。\n"]
            
            for i, law in enumerate(laws[:5], 1):  # 上位5件のみ
                law_info = law.get("law_info", {})
                law_name = law_info.get("law_title", "不明")
                law_num = law_info.get("law_num", "")
                
                # 該当箇所の情報
                matched_provisions = law.get("matched_provisions", [])
                if matched_provisions:
                    provision_info = matched_provisions[0]
                    article = provision_info.get("article", {}).get("article_title", "")
                    text = provision_info.get("text", "")[:100]  # 最初の100文字
                    
                    output_lines.append(
                        f"[{i}] {law_name} ({law_num})\n"
                        f"    {article}\n"
                        f"    {text}...\n"
                    )
                else:
                    output_lines.append(f"[{i}] {law_name} ({law_num})\n")
            
            return "\n".join(output_lines)
        
        except EGovAPIError as e:
            logger.error(f"API error in keyword search: {e}")
            return f"API検索エラー: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in keyword search: {e}")
            return f"検索エラー: {str(e)}"
    
    def _get_law_function(self, law_id_or_num: str) -> str:
        """
        法令本文取得のツール関数
        
        Args:
            law_id_or_num: 法令IDまたは法令番号
        
        Returns:
            法令本文のテキスト表現
        """
        try:
            result = self.client.get_law_data(law_id_or_num, response_format="json")
            
            law_body = result.get("law_body", {})
            law_title = result.get("law_title", "不明")
            law_num = result.get("law_num", "")
            
            output_lines = [
                f"法令名: {law_title}",
                f"法令番号: {law_num}\n"
            ]
            
            # 本文の抽出（簡略版）
            main_provision = law_body.get("main_provision", {})
            if isinstance(main_provision, dict):
                articles = main_provision.get("article", [])
                if not isinstance(articles, list):
                    articles = [articles] if articles else []
                
                for article in articles[:10]:  # 最初の10条まで
                    article_title = article.get("article_title", "")
                    article_caption = article.get("article_caption", "")
                    
                    if article_title or article_caption:
                        output_lines.append(f"{article_title} {article_caption}")
            
            return "\n".join(output_lines)
        
        except EGovAPIError as e:
            logger.error(f"API error in get law: {e}")
            return f"法令取得エラー: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in get law: {e}")
            return f"取得エラー: {str(e)}"
    
    def _local_search_function(self, query: str) -> str:
        """
        ローカル検索のツール関数（フォールバック）
        
        Args:
            query: 検索クエリ
        
        Returns:
            検索結果のテキスト表現
        """
        if self.local_retriever is None:
            return "ローカル検索機能は利用できません。"
        
        try:
            from app.retrieval.base import Document
            
            # ローカルリトリーバーで検索
            documents = self.local_retriever.retrieve(
                query,
                top_k=self.config.retrieval_top_k
            )
            
            if not documents:
                return "ローカルデータで検索結果が見つかりませんでした。"
            
            # 結果を整形
            output_lines = [f"ローカル検索結果: {len(documents)}件\n"]
            
            for i, doc in enumerate(documents[:5], 1):
                meta = doc.metadata
                law_title = meta.get("law_title", "不明")
                article = meta.get("article", "")
                text = doc.page_content[:100]
                
                output_lines.append(
                    f"[{i}] {law_title} 第{article}条\n"
                    f"    {text}...\n"
                )
            
            return "\n".join(output_lines)
        
        except Exception as e:
            logger.error(f"Error in local search: {e}")
            return f"ローカル検索エラー: {str(e)}"
    
    def create_tools(self) -> List[Tool]:
        """
        LangChain Toolsのリストを生成
        
        Returns:
            Toolsのリスト
        """
        tools = []
        
        # Tool 1: キーワード検索
        keyword_search_tool = Tool(
            name="egov_keyword_search",
            func=self._keyword_search_function,
            description=(
                "e-Gov APIで法令本文をキーワード検索します。"
                "最新の法令データを取得できます。"
                "入力: 検索キーワード（文字列）"
            )
        )
        tools.append(keyword_search_tool)
        
        # Tool 2: 法令本文取得
        get_law_tool = Tool(
            name="egov_get_law",
            func=self._get_law_function,
            description=(
                "法令番号を指定してe-Gov APIから法令全文を取得します。"
                "入力: 法令番号（例: '平成二十八年個人情報保護委員会規則第六号'）"
            )
        )
        tools.append(get_law_tool)
        
        # Tool 3: ローカル検索（フォールバック）
        if self.local_retriever is not None:
            local_search_tool = Tool(
                name="local_search",
                func=self._local_search_function,
                description=(
                    "ローカルの法令データベースから検索します。"
                    "API障害時のフォールバックとして使用します。"
                    "入力: 検索クエリ（文字列）"
                )
            )
            tools.append(local_search_tool)
        
        logger.info(f"Created {len(tools)} tools")
        return tools
    
    def get_api_client(self) -> EGovAPIClient:
        """
        e-Gov APIクライアントを取得
        
        Returns:
            EGovAPIClient インスタンス
        """
        return self.client


def create_egov_tools(
    config: MCPEgovConfig,
    local_retriever = None
) -> List[Tool]:
    """
    e-Gov API用のToolsを作成する便利関数
    
    Args:
        config: MCP e-Gov Agent設定
        local_retriever: ローカル検索用のRetriever（オプション）
    
    Returns:
        Toolsのリスト
    """
    factory = EgovToolsFactory(config, local_retriever)
    return factory.create_tools()

