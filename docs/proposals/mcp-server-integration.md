# MCPサーバ統合提案

## 概要

本ドキュメントは、Model Context Protocol（MCP）サーバを実装し、法令データの取得方法を現在の静的データセットから動的API連携へ移行することを提案します。MCPサーバを介することで、e-Gov法令APIとの標準化された統合が可能になり、常に最新の法令データへのアクセスが実現します。

## Model Context Protocol（MCP）とは

MCPは、AIモデルとアプリケーション間の標準化された通信プロトコルです。

### 主要な特徴

- AIモデルと外部データソース間の標準化されたインターフェース
- リソース、ツール、プロンプトの統一的な提供方法
- セキュアなAPI統合
- クライアント非依存の設計
- 拡張可能なアーキテクチャ

### MCPの利点

1. 標準化: 統一されたプロトコルにより統合が容易
2. セキュリティ: アクセス制御とデータ保護の組み込み
3. 拡張性: 新しいデータソースの追加が簡単
4. 保守性: クリーンな分離により保守が容易
5. 再利用性: 他のプロジェクトでもMCPサーバを再利用可能

## 現在のアーキテクチャの課題

### 静的データセットの問題点

現在のシステムは以下の課題を抱えています:

1. データの鮮度
   - 法令改正への対応が遅れる
   - 手動でのデータ更新が必要
   - 最新の条文を反映できない

2. データ管理の負担
   - 大規模なデータセットのストレージコスト
   - 更新プロセスの複雑さ
   - バージョン管理の困難さ

3. 柔軟性の欠如
   - 必要なデータのみを取得できない
   - 動的なデータ拡張が困難
   - 複数データソースの統合が難しい

## MCPサーバによる解決策

### アーキテクチャ概要

```
┌─────────────────────────────────────────────────┐
│           Legal RAG Application                 │
│                                                 │
│  ┌─────────────────────────────────────┐       │
│  │      RAG Pipeline                   │       │
│  │  - Query Processing                 │       │
│  │  - Response Generation              │       │
│  └─────────────────────────────────────┘       │
│              ↓                                  │
│  ┌─────────────────────────────────────┐       │
│  │      MCP Client                     │       │
│  │  - Protocol Handler                 │       │
│  │  - Request Builder                  │       │
│  └─────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
                    ↓ MCP Protocol
┌─────────────────────────────────────────────────┐
│           Legal MCP Server                      │
│                                                 │
│  ┌─────────────────────────────────────┐       │
│  │      Resources                      │       │
│  │  - 法令一覧                          │       │
│  │  - 条文データ                        │       │
│  │  - 改正履歴                          │       │
│  └─────────────────────────────────────┘       │
│                                                 │
│  ┌─────────────────────────────────────┐       │
│  │      Tools                          │       │
│  │  - search_law: 法令検索             │       │
│  │  - get_article: 条文取得            │       │
│  │  - get_amendments: 改正履歴取得     │       │
│  │  - search_related: 関連法令検索     │       │
│  └─────────────────────────────────────┘       │
│                                                 │
│  ┌─────────────────────────────────────┐       │
│  │      API Connectors                 │       │
│  │  - e-Gov法令API                     │       │
│  │  - デジタル庁法令APIv2（将来）       │       │
│  └─────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
                    ↓ HTTPS
┌─────────────────────────────────────────────────┐
│         External APIs                           │
│                                                 │
│  - e-Gov法令検索API                             │
│  - デジタル庁法令API                             │
│  - その他政府API                                 │
└─────────────────────────────────────────────────┘
```

## e-Gov法令APIとの統合

### e-Gov法令APIの概要

デジタル庁が提供する法令データAPIには以下の種類があります:

1. 法令名一覧取得API
   - 法令の種類に基づいて法令一覧を取得
   - フィルタリング機能

2. 法令取得API
   - 法令番号を指定して全文を取得
   - XML/JSON形式での取得

3. 条文内容取得API
   - 法令番号と条文番号を指定して特定条文を取得
   - 細かい粒度でのデータ取得

### デジタル庁法令APIv2の進化

現在開発中の法令APIv2は以下の機能を提供予定:

- 改正前の過去条文履歴の取得
- キーワード検索機能
- 時点指定での法令取得
- より使いやすいJSON形式

## 実装設計

### MCPサーバ実装（Python）

```python
# legal_mcp_server.py
from typing import Any, Dict, List, Optional
from modelcontextprotocol import MCPServer, Tool, Resource
import httpx
import json
from datetime import datetime

class LegalMCPServer(MCPServer):
    """法令データ用MCPサーバ"""
    
    def __init__(self, egov_api_base: str = "https://elaws.e-gov.go.jp/api/1"):
        super().__init__(name="legal-mcp-server", version="1.0.0")
        self.api_base = egov_api_base
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # ツールの登録
        self.register_tools()
        
    def register_tools(self):
        """MCPツールの登録"""
        
        @self.tool(
            name="search_law",
            description="法令名または法令番号で法令を検索します"
        )
        async def search_law(
            query: str,
            law_type: Optional[str] = None,
            limit: int = 10
        ) -> Dict[str, Any]:
            """
            法令検索ツール
            
            Args:
                query: 検索クエリ（法令名または法令番号）
                law_type: 法令種別（法律、政令、省令等）
                limit: 取得件数
            
            Returns:
                検索結果の法令リスト
            """
            return await self._search_law_impl(query, law_type, limit)
        
        @self.tool(
            name="get_law_content",
            description="法令番号を指定して法令全文を取得します"
        )
        async def get_law_content(
            law_number: str,
            format: str = "json"
        ) -> Dict[str, Any]:
            """
            法令全文取得ツール
            
            Args:
                law_number: 法令番号
                format: 取得形式（json/xml）
            
            Returns:
                法令の全文データ
            """
            return await self._get_law_content_impl(law_number, format)
        
        @self.tool(
            name="get_article",
            description="特定の条文を取得します"
        )
        async def get_article(
            law_number: str,
            article_number: str,
            paragraph: Optional[str] = None,
            item: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            条文取得ツール
            
            Args:
                law_number: 法令番号
                article_number: 条文番号
                paragraph: 項番号（オプション）
                item: 号番号（オプション）
            
            Returns:
                条文データ
            """
            return await self._get_article_impl(
                law_number, article_number, paragraph, item
            )
        
        @self.tool(
            name="search_related_laws",
            description="関連法令を検索します"
        )
        async def search_related_laws(
            law_number: str,
            max_results: int = 5
        ) -> List[Dict[str, Any]]:
            """
            関連法令検索ツール
            
            Args:
                law_number: 基準となる法令番号
                max_results: 最大取得件数
            
            Returns:
                関連法令のリスト
            """
            return await self._search_related_laws_impl(law_number, max_results)
        
        @self.tool(
            name="get_amendments",
            description="法令の改正履歴を取得します"
        )
        async def get_amendments(
            law_number: str
        ) -> List[Dict[str, Any]]:
            """
            改正履歴取得ツール
            
            Args:
                law_number: 法令番号
            
            Returns:
                改正履歴のリスト
            """
            return await self._get_amendments_impl(law_number)
    
    async def _search_law_impl(
        self,
        query: str,
        law_type: Optional[str],
        limit: int
    ) -> Dict[str, Any]:
        """法令検索の実装"""
        url = f"{self.api_base}/lawlists/1"
        
        response = await self.client.get(url)
        response.raise_for_status()
        
        # XMLまたはJSONのパース
        data = self._parse_response(response)
        
        # フィルタリング
        laws = self._filter_laws(data, query, law_type, limit)
        
        return {
            "laws": laws,
            "total": len(laws),
            "query": query
        }
    
    async def _get_law_content_impl(
        self,
        law_number: str,
        format: str
    ) -> Dict[str, Any]:
        """法令全文取得の実装"""
        url = f"{self.api_base}/lawdata/{law_number}"
        
        response = await self.client.get(url)
        response.raise_for_status()
        
        if format == "json":
            return self._xml_to_json(response.text)
        else:
            return {"xml": response.text}
    
    async def _get_article_impl(
        self,
        law_number: str,
        article_number: str,
        paragraph: Optional[str],
        item: Optional[str]
    ) -> Dict[str, Any]:
        """条文取得の実装"""
        # e-Gov条文内容取得APIの呼び出し
        url = f"{self.api_base}/articles/{law_number}/{article_number}"
        
        params = {}
        if paragraph:
            params["paragraph"] = paragraph
        if item:
            params["item"] = item
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        return self._parse_article(response)
    
    async def _search_related_laws_impl(
        self,
        law_number: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """関連法令検索の実装"""
        # 法令全文を取得
        law_content = await self._get_law_content_impl(law_number, "json")
        
        # 本文中から引用されている法令を抽出
        referenced_laws = self._extract_referenced_laws(law_content)
        
        # 関連法令の情報を取得
        related = []
        for ref_law_number in referenced_laws[:max_results]:
            try:
                law_info = await self._get_law_info(ref_law_number)
                related.append(law_info)
            except Exception as e:
                continue
        
        return related
    
    async def _get_amendments_impl(
        self,
        law_number: str
    ) -> List[Dict[str, Any]]:
        """改正履歴取得の実装"""
        # 将来的にデジタル庁法令APIv2を使用
        # 現時点では法令データから改正情報を抽出
        law_content = await self._get_law_content_impl(law_number, "json")
        
        amendments = self._extract_amendments(law_content)
        
        return amendments
    
    def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        """APIレスポンスのパース"""
        content_type = response.headers.get("content-type", "")
        
        if "json" in content_type:
            return response.json()
        elif "xml" in content_type:
            return self._xml_to_json(response.text)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    
    def _xml_to_json(self, xml_text: str) -> Dict[str, Any]:
        """XMLをJSONに変換"""
        import xmltodict
        return xmltodict.parse(xml_text)
    
    def _filter_laws(
        self,
        data: Dict[str, Any],
        query: str,
        law_type: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """法令データのフィルタリング"""
        # 実装: クエリと法令種別でフィルタリング
        laws = data.get("laws", [])
        filtered = []
        
        for law in laws:
            if query.lower() in law.get("name", "").lower():
                if law_type is None or law.get("type") == law_type:
                    filtered.append(law)
                    if len(filtered) >= limit:
                        break
        
        return filtered
    
    def _extract_referenced_laws(self, law_content: Dict[str, Any]) -> List[str]:
        """引用法令の抽出"""
        # 実装: 本文から他の法令への参照を抽出
        # 例: "民法第○条", "行政手続法"などのパターンマッチング
        import re
        
        text = json.dumps(law_content)
        patterns = [
            r'([^法]+法)(?:第?\d+条)?',
            r'(政令第\d+号)',
            r'(省令第\d+号)'
        ]
        
        referenced = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            referenced.update(matches)
        
        return list(referenced)
    
    def _extract_amendments(self, law_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """改正情報の抽出"""
        # 実装: 法令データから改正履歴を抽出
        amendments = []
        
        # 附則から改正情報を抽出
        if "supplementary_provisions" in law_content:
            for provision in law_content["supplementary_provisions"]:
                if "施行" in provision.get("text", ""):
                    amendments.append({
                        "date": self._extract_date(provision["text"]),
                        "description": provision["text"]
                    })
        
        return amendments
    
    def _extract_date(self, text: str) -> Optional[str]:
        """テキストから日付を抽出"""
        import re
        
        # 例: "令和5年4月1日"
        pattern = r'(令和|平成)\d+年\d+月\d+日'
        match = re.search(pattern, text)
        
        return match.group(0) if match else None


# サーバの起動
if __name__ == "__main__":
    server = LegalMCPServer()
    server.run(transport="stdio")  # 標準入出力経由で通信
```

### MCPクライアント実装

```python
# app/mcp/legal_client.py
from typing import Any, Dict, List, Optional
import httpx
from modelcontextprotocol import MCPClient

class LegalMCPClient:
    """法令MCP���ーバ用クライアント"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.client = MCPClient(server_url)
        
    async def search_law(
        self,
        query: str,
        law_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """法令検索"""
        return await self.client.call_tool(
            "search_law",
            query=query,
            law_type=law_type,
            limit=limit
        )
    
    async def get_law_content(
        self,
        law_number: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """法令全文取得"""
        return await self.client.call_tool(
            "get_law_content",
            law_number=law_number,
            format=format
        )
    
    async def get_article(
        self,
        law_number: str,
        article_number: str,
        paragraph: Optional[str] = None,
        item: Optional[str] = None
    ) -> Dict[str, Any]:
        """条文取得"""
        return await self.client.call_tool(
            "get_article",
            law_number=law_number,
            article_number=article_number,
            paragraph=paragraph,
            item=item
        )
    
    async def search_related_laws(
        self,
        law_number: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """関連法令検索"""
        return await self.client.call_tool(
            "search_related_laws",
            law_number=law_number,
            max_results=max_results
        )
    
    async def get_amendments(
        self,
        law_number: str
    ) -> List[Dict[str, Any]]:
        """改正履歴取得"""
        return await self.client.call_tool(
            "get_amendments",
            law_number=law_number
        )


# RAGパイプラインへの統合
class MCPEnhancedRetriever(BaseRetriever):
    """MCP統合レトリーバー"""
    
    def __init__(
        self,
        mcp_client: LegalMCPClient,
        local_retriever: Optional[BaseRetriever] = None,
        use_hybrid: bool = True
    ):
        self.mcp_client = mcp_client
        self.local_retriever = local_retriever
        self.use_hybrid = use_hybrid
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Document]:
        """ハイブリッド検索"""
        documents = []
        
        # ローカル検索（高速）
        if self.local_retriever and self.use_hybrid:
            local_docs = await self.local_retriever.retrieve(query, top_k)
            documents.extend(local_docs)
        
        # MCP経由のリアルタイム検索
        try:
            # 法令検索
            search_result = await self.mcp_client.search_law(query, limit=top_k)
            
            # 検索結果から詳細を取得
            for law in search_result.get("laws", [])[:top_k]:
                law_content = await self.mcp_client.get_law_content(
                    law["law_number"]
                )
                
                doc = Document(
                    page_content=self._extract_relevant_content(law_content, query),
                    metadata={
                        "law_number": law["law_number"],
                        "law_title": law["name"],
                        "source": "mcp",
                        "updated_at": law.get("updated_at")
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            # MCPサーバ接続失敗時はローカルのみ使用
            print(f"MCP search failed: {e}")
        
        # 重複除去とスコアリング
        documents = self._deduplicate_and_score(documents)
        
        return documents[:top_k]
    
    def _extract_relevant_content(
        self,
        law_content: Dict[str, Any],
        query: str
    ) -> str:
        """クエリに関連する条文を抽出"""
        # 実装: 全文から関連部分を抽出
        # 簡易版では全文を返す
        return json.dumps(law_content, ensure_ascii=False)
    
    def _deduplicate_and_score(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """重複除去とスコアリング"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            key = (
                doc.metadata.get("law_number"),
                doc.metadata.get("article")
            )
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return unique_docs
```

## 実装計画

### フェーズ1: MCPサーバ基本実装（2週間）

1. MCPサーバのセットアップ
   - Python MCP SDK導入
   - 基本的なサーバ構造の実装
   - e-Gov API連携の基礎実装

2. 基本ツールの実装
   - search_law
   - get_law_content
   - get_article

3. テストとデバッグ
   - 単体テスト
   - e-Gov APIとの接続テスト

### フェーズ2: MCPクライアント統合（1週間）

1. クライアントライブラリ実装
2. RAGパイプラインへの統合
3. ハイブリッド検索の実装

### フェーズ3: 高度な機能追加（2週間）

1. 関連法令検索
2. 改正履歴取得
3. キャッシング機能
4. エラーハンドリングの強化

### フェーズ4: 最適化と本番化（1週間）

1. パフォーマンスチューニング
2. モニタリング機能
3. ドキュメント整備

## 期待される効果

### データ鮮度の向上

- 常に最新の法令データへアクセス
- 改正への即時対応
- 手動更新作業の削減

### システムの柔軟性

- 動的なデータ取得
- 必要なデータのみを取得（効率化）
- 複数データソースの容易な統合

### 保守性の向上

- クリーンなアーキテクチャ
- データ取得ロジックの分離
- テストの容易化

### 拡張性

- 新しいAPIの追加が容易
- 他のシステムでのMCPサーバ再利用
- 段階的な機能追加が可能

## 技術的考慮事項

### パフォーマンス

- APIレスポンスタイムの考慮
- キャッシング戦略の実装
- ローカルデータとのハイブリッド運用

### セキュリティ

- API認証の適切な実装
- データの暗号化
- アクセスログの記録

### 可用性

- APIサーバダウン時のフォールバック
- ローカルキャッシュの活用
- エラーハンドリング

### コスト

- API呼び出し回数の最適化
- キャッシング活用によるコスト削減
- ローカルデータとの使い分け

## 参考資料

- Model Context Protocol公式ドキュメント: https://modelcontextprotocol.io/
- e-Gov法令API: https://elaws.e-gov.go.jp/apitop/
- デジタル庁法令API: https://www.digital.go.jp/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCPサーバ事例集: https://glama.ai/mcp/servers
