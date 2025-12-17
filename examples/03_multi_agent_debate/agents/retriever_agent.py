"""
RetrieverAgent - 法令検索役

質問に関連する法令条文を検索・抽出し、
CitationRegistryに登録する。
検索意図CoTを生成して次のエージェントに共有。
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import re

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent
from .citation import CitationRegistry, Citation

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    法令検索エージェント
    
    責務:
    - 質問を分析し、検索クエリを最適化
    - 関連法令条文を検索
    - 検索結果をCitationRegistryに登録
    - 検索意図CoTを生成して共有
    """
    
    def __init__(self, llm, config, retriever, citation_registry: CitationRegistry):
        """
        Args:
            llm: LLMインスタンス
            config: 設定オブジェクト
            retriever: 検索用Retriever
            citation_registry: 引用レジストリ
        """
        super().__init__(llm, config)
        self.retriever = retriever
        self.citation_registry = citation_registry
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        法令検索を実行
        
        Args:
            input_data: {
                "query": str,  # 質問文
                "choices": List[str],  # 選択肢（オプション）
            }
        
        Returns:
            {
                "citation_ids": List[str],  # 登録した引用ID
                "documents": List[Document],  # 検索した文書
                "search_queries": List[str],  # 使用した検索クエリ
                "retrieval_cot": str,  # 検索意図CoT
                "metadata": Dict
            }
        """
        query = input_data.get("query", "")
        choices = input_data.get("choices", [])
        
        if not query:
            return {
                "citation_ids": [],
                "documents": [],
                "search_queries": [],
                "retrieval_cot": "",
                "extracted_keywords": {},
                "metadata": {"error": "Empty query"}
            }
        
        logger.info(f"RetrieverAgent executing for query: {query[:100]}...")
        
        # 検索意図を分析し、クエリを生成
        analysis = self._analyze_and_expand_query(query, choices)
        search_queries = analysis.get("queries", [query])
        retrieval_cot = analysis.get("retrieval_cot", "")
        extracted_keywords = analysis.get("extracted_keywords", {})
        
        # 検索実行
        all_documents = []
        seen_texts = set()
        
        for search_query in search_queries:
            try:
                docs = self.retriever.retrieve(
                    search_query, 
                    top_k=self.config.retrieval_top_k
                )
                
                # 重複除去
                for doc in docs:
                    text_hash = hash(doc.page_content[:100])
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        all_documents.append(doc)
                        
            except Exception as e:
                logger.error(f"Search failed for query '{search_query}': {e}")
        
        # CitationRegistryに登録
        citation_ids = []
        for doc in all_documents:
            cid = self.citation_registry.register_from_document(doc)
            citation_ids.append(cid)
        
        # 推論ステップを記録
        self.citation_registry.add_reasoning_step(
            agent="RetrieverAgent",
            action="search",
            claim=f"{len(all_documents)}件の関連法令条文を検索",
            supporting_citations=citation_ids[:5],
            confidence=0.0,
            metadata={
                "search_queries": search_queries,
                "total_documents": len(all_documents),
                "retrieval_cot": retrieval_cot
            }
        )
        
        logger.info(f"Retrieved {len(all_documents)} documents, registered {len(citation_ids)} citations")
        
        return {
            "citation_ids": citation_ids,
            "documents": all_documents,
            "search_queries": search_queries,
            "retrieval_cot": retrieval_cot,
            "extracted_keywords": extracted_keywords,
            "metadata": {
                "total_documents": len(all_documents),
                "unique_laws": self._count_unique_laws(all_documents)
            }
        }
    
    def _analyze_and_expand_query(self, query: str, choices: List[str]) -> Dict[str, Any]:
        """
        検索意図を分析し、クエリを分解・拡張
        法令名・条文番号を明示的に抽出して検索精度を向上
        """
        # Step 1: 質問と選択肢から法令キーワードを抽出
        all_text = query + " " + " ".join(choices[:4])
        extracted = self._extract_structured_keywords(all_text)
        
        queries = [query]
        
        # Step 2: 抽出した法令名で検索クエリを追加
        if extracted["law_names"]:
            law_query = " ".join(extracted["law_names"][:3])
            queries.append(law_query)
        
        # Step 3: 条文番号があれば、法令名+条文番号で検索
        if extracted["law_names"] and extracted["articles"]:
            for law in extracted["law_names"][:2]:
                for article in extracted["articles"][:2]:
                    queries.append(f"{law} {article}")
        
        # Step 4: LLMで追加のキーワード拡張
        expanded = self._expand_with_llm_simple(query, extracted)
        if expanded:
            queries.append(expanded)
        
        # Step 5: 選択肢から追加キーワード
        for choice in choices[:4]:
            keywords = self._extract_legal_keywords(choice)
            if keywords and keywords not in queries:
                queries.append(keywords)
        
        # 検索意図CoTを生成
        retrieval_cot = self._generate_retrieval_cot(query, extracted)
        
        # 重複除去して最大7クエリ
        unique_queries = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return {
            "queries": unique_queries[:7],
            "retrieval_cot": retrieval_cot,
            "extracted_keywords": extracted
        }
    
    def _extract_structured_keywords(self, text: str) -> Dict[str, List[str]]:
        """テキストから法令名・条文番号・その他キーワードを構造化抽出"""
        result = {
            "law_names": [],
            "articles": [],
            "numbers": [],
            "terms": []
        }
        
        # 法令名パターン（より包括的）
        law_patterns = [
            r'([\u4e00-\u9fff]{2,}法)',
            r'([\u4e00-\u9fff]{2,}令)',
            r'([\u4e00-\u9fff]{2,}規則)',
            r'([\u4e00-\u9fff]{2,}条例)',
        ]
        for pattern in law_patterns:
            matches = re.findall(pattern, text)
            result["law_names"].extend(matches)
        
        # 条文番号パターン
        article_patterns = [
            r'第[\d一二三四五六七八九十百千]+条(?:の[\d一二三四五六七八九十]+)?',
            r'第[\d一二三四五六七八九十百]+項',
            r'第[\d一二三四五六七八九十百]+号',
        ]
        for pattern in article_patterns:
            matches = re.findall(pattern, text)
            result["articles"].extend(matches)
        
        # 数値（期間、金額など）
        number_patterns = [
            r'[\d一二三四五六七八九十百千万億]+(?:年|月|日|円|万円|億円|%)',
            r'[\d]+(?:年|月|日)',
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            result["numbers"].extend(matches)
        
        # 重複除去
        result["law_names"] = list(dict.fromkeys(result["law_names"]))
        result["articles"] = list(dict.fromkeys(result["articles"]))
        result["numbers"] = list(dict.fromkeys(result["numbers"]))
        
        return result
    
    def _expand_with_llm_simple(self, query: str, extracted: Dict) -> Optional[str]:
        """LLMで簡潔にキーワード拡張"""
        laws = ", ".join(extracted["law_names"][:3]) if extracted["law_names"] else "不明"
        articles = ", ".join(extracted["articles"][:3]) if extracted["articles"] else "不明"
        
        prompt = f"""質問: {query[:200]}
抽出済み法令: {laws}
抽出済み条文: {articles}

この質問に関連する追加の検索キーワードを3-5語で出力してください（スペース区切り）:"""
        
        response = self._safe_llm_invoke(prompt)
        if response:
            return response.strip()[:150]
        return None
    
    def _generate_retrieval_cot(self, query: str, extracted: Dict) -> str:
        """検索意図CoTを生成"""
        laws = ", ".join(extracted["law_names"][:2]) if extracted["law_names"] else "関連法令"
        articles = ", ".join(extracted["articles"][:2]) if extracted["articles"] else "関連条文"
        
        cot = f"この質問は{laws}の{articles}に関する問題。"
        
        if extracted["numbers"]:
            cot += f"数値条件（{', '.join(extracted['numbers'][:2])}）の確認が必要。"
        
        return cot[:300]
    
    def _extract_legal_keywords(self, text: str) -> Optional[str]:
        """テキストから法律関連のキーワードを抽出"""
        keywords = []
        
        # 法令名のパターン
        law_patterns = [
            r'([\u4e00-\u9fff]+法)',
            r'([\u4e00-\u9fff]+令)',
            r'([\u4e00-\u9fff]+規則)',
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        # 条文番号のパターン
        article_pattern = r'第[\d一二三四五六七八九十百]+条'
        articles = re.findall(article_pattern, text)
        keywords.extend(articles)
        
        if keywords:
            return " ".join(keywords[:5])
        return None
    
    def _count_unique_laws(self, documents: List[Any]) -> int:
        """ユニークな法令数をカウント"""
        law_names = set()
        for doc in documents:
            law_name = doc.metadata.get("law_title", "")
            if law_name:
                law_names.add(law_name)
        return len(law_names)
