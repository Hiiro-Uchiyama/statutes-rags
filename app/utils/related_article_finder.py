"""
関連条文の自動追加検索

「政令で定める」などの参照関係を追跡し、関連条文を自動検索する
"""
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArticleReference:
    """条文参照情報"""
    source_text: str       # 参照元テキスト
    reference_type: str    # 参照タイプ（政令、施行令、他条など）
    target_law: str        # 参照先法令
    target_article: str    # 参照先条文
    context: str           # 文脈


class RelatedArticleFinder:
    """関連条文を検索"""
    
    # 参照パターン
    REFERENCE_PATTERNS = [
        # 政令・施行令への参照
        (r'政令で定める', 'decree', '施行令'),
        (r'内閣府令で定める', 'cabinet_order', '施行規則'),
        (r'省令で定める', 'ministry_order', '施行規則'),
        (r'施行令第([一二三四五六七八九十百\d]+)条', 'decree_article', '施行令'),
        (r'施行規則第([一二三四五六七八九十百\d]+)条', 'rule_article', '施行規則'),
        
        # 他条への参照
        (r'第([一二三四五六七八九十百\d]+)条(の規定|に規定する|の[一二三四五六七八九十\d]+)', 'same_law_article', None),
        (r'前条', 'prev_article', None),
        (r'次条', 'next_article', None),
        
        # 他法令への参照
        (r'([一-龥ぁ-んァ-ン]+法)第([一二三四五六七八九十百\d]+)条', 'other_law', None),
    ]
    
    def __init__(self, retriever=None):
        """
        Args:
            retriever: 検索に使用するRetriever
        """
        self.retriever = retriever
    
    def find_references(self, text: str) -> List[ArticleReference]:
        """テキスト内の条文参照を抽出"""
        references = []
        
        for pattern, ref_type, target_law in self.REFERENCE_PATTERNS:
            for match in re.finditer(pattern, text):
                # 文脈を抽出
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # 参照先条文を特定
                if ref_type in ['decree_article', 'rule_article', 'same_law_article']:
                    target_article = f"第{match.group(1)}条" if match.groups() else ""
                elif ref_type == 'other_law':
                    target_law = match.group(1)
                    target_article = f"第{match.group(2)}条"
                else:
                    target_article = ""
                
                references.append(ArticleReference(
                    source_text=match.group(0),
                    reference_type=ref_type,
                    target_law=target_law or "",
                    target_article=target_article,
                    context=context
                ))
        
        return references
    
    def search_related_articles(
        self,
        base_text: str,
        base_law_title: str = "",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        関連条文を検索
        
        Args:
            base_text: 基となるテキスト
            base_law_title: 基となる法令名
            top_k: 各参照につき取得する件数
            
        Returns:
            関連条文リスト
        """
        if not self.retriever:
            logger.warning("Retriever not set. Cannot search related articles.")
            return []
        
        references = self.find_references(base_text)
        related_articles = []
        seen_queries = set()
        
        for ref in references:
            # 検索クエリを構築
            query = self._build_search_query(ref, base_law_title)
            
            if query and query not in seen_queries:
                seen_queries.add(query)
                
                try:
                    # 検索実行
                    results = self.retriever.retrieve(query, top_k=top_k)
                    
                    for doc in results:
                        if hasattr(doc, 'page_content'):
                            content = doc.page_content
                            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        elif isinstance(doc, dict):
                            content = doc.get('text', doc.get('content', ''))
                            metadata = doc.get('metadata', {})
                        else:
                            content = str(doc)
                            metadata = {}
                        
                        related_articles.append({
                            'content': content,
                            'metadata': metadata,
                            'reference': ref,
                            'query': query
                        })
                except Exception as e:
                    logger.warning(f"Failed to search related articles: {e}")
        
        return related_articles
    
    def _build_search_query(
        self,
        ref: ArticleReference,
        base_law_title: str
    ) -> Optional[str]:
        """参照から検索クエリを構築"""
        if ref.reference_type == 'decree':
            # 政令で定める → 施行令を検索
            if base_law_title:
                # 法令名から施行令名を推測
                decree_name = base_law_title.replace('法', '法施行令')
                return f"{decree_name}"
            return None
        
        elif ref.reference_type == 'cabinet_order':
            if base_law_title:
                rule_name = base_law_title.replace('法', '法施行規則')
                return f"{rule_name}"
            return None
        
        elif ref.reference_type in ['decree_article', 'rule_article']:
            target_law = ref.target_law
            target_article = ref.target_article
            return f"{target_law} {target_article}"
        
        elif ref.reference_type == 'same_law_article':
            return f"{base_law_title} {ref.target_article}"
        
        elif ref.reference_type == 'other_law':
            return f"{ref.target_law} {ref.target_article}"
        
        return None
    
    def expand_context_with_related(
        self,
        original_context: str,
        base_law_title: str = "",
        max_related: int = 3
    ) -> str:
        """
        コンテキストを関連条文で拡張
        
        Args:
            original_context: 元のコンテキスト
            base_law_title: 基となる法令名
            max_related: 追加する関連条文の最大数
            
        Returns:
            拡張されたコンテキスト
        """
        related = self.search_related_articles(
            original_context,
            base_law_title,
            top_k=max_related
        )
        
        if not related:
            return original_context
        
        # 関連条文を追加
        expanded = original_context + "\n\n【関連条文】\n"
        
        seen_contents = set()
        added = 0
        
        for r in related:
            content = r['content']
            # 重複チェック
            content_key = content[:100]
            if content_key not in seen_contents and content_key not in original_context[:100]:
                seen_contents.add(content_key)
                ref_info = r['reference']
                expanded += f"\n--- {ref_info.source_text} より ---\n"
                expanded += content[:500] + "\n"
                added += 1
                
                if added >= max_related:
                    break
        
        return expanded


class ReferenceTracker:
    """条文間の参照関係を追跡"""
    
    def __init__(self):
        self.reference_graph: Dict[str, Set[str]] = {}
    
    def add_reference(self, from_article: str, to_article: str):
        """参照関係を追加"""
        if from_article not in self.reference_graph:
            self.reference_graph[from_article] = set()
        self.reference_graph[from_article].add(to_article)
    
    def get_all_related(
        self,
        article: str,
        max_depth: int = 2
    ) -> Set[str]:
        """指定条文に関連する全ての条文を取得（再帰的）"""
        related = set()
        self._collect_related(article, related, max_depth, 0)
        return related
    
    def _collect_related(
        self,
        article: str,
        related: Set[str],
        max_depth: int,
        current_depth: int
    ):
        if current_depth >= max_depth:
            return
        
        if article in self.reference_graph:
            for ref in self.reference_graph[article]:
                if ref not in related:
                    related.add(ref)
                    self._collect_related(ref, related, max_depth, current_depth + 1)

