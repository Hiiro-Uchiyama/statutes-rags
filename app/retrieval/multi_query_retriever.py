"""
複数クエリ検索モジュール

複数の条文を参照する問題に対応するため、
問題文+選択肢から条文参照を抽出し、
各条文に対して個別検索を実行して結果を統合する。
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.query_processor import QueryProcessor


@dataclass
class ArticleReference:
    """条文参照"""
    law_name: Optional[str]  # 法令名（判明している場合）
    article_num: str         # 条番号（「27」など）
    article_suffix: str = "" # 「の2」など
    
    @property
    def full_article(self) -> str:
        """完全な条文番号（例：第二十七条の二）"""
        from app.utils.number_normalizer import arabic_to_kanji_number
        
        kanji_num = arabic_to_kanji_number(int(self.article_num))
        result = f"第{kanji_num}条"
        
        if self.article_suffix:
            suffix_num = arabic_to_kanji_number(int(self.article_suffix))
            result += f"の{suffix_num}"
        
        return result


class MultiQueryRetriever:
    """
    複数クエリ検索
    
    問題文と選択肢から条文参照を抽出し、
    各条文に対して個別検索を実行して結果を統合する。
    """
    
    def __init__(
        self,
        base_retriever,
        query_processor: Optional[QueryProcessor] = None,
        per_article_top_k: int = 5,
        max_total_results: int = 20
    ):
        """
        Args:
            base_retriever: ベースの検索器（VectorRetriever, HybridRetrieverなど）
            query_processor: クエリプロセッサ
            per_article_top_k: 各条文検索でのtop_k
            max_total_results: 統合後の最大結果数
        """
        self.base_retriever = base_retriever
        self.query_processor = query_processor or QueryProcessor()
        self.per_article_top_k = per_article_top_k
        self.max_total_results = max_total_results
        
        # 法令名パターン
        self.law_patterns = [
            (r'金融商品取引法(?:施行令)?', '金融商品取引法'),
            (r'医薬品[、\s]*医療機器等[^\s]*法(?:施行規則)?', '医薬品、医療機器等の品質、有効性及び安全性の確保等に関する法律'),
            (r'借地借家法', '借地借家法'),
        ]
    
    def extract_article_references(
        self, 
        question: str, 
        choices: str = ""
    ) -> List[ArticleReference]:
        """
        問題文と選択肢から条文参照を抽出
        
        Args:
            question: 問題文
            choices: 選択肢テキスト
        
        Returns:
            条文参照のリスト
        """
        full_text = question + " " + choices
        references = []
        seen = set()  # 重複排除用
        
        # 法令名を抽出
        law_name = None
        for pattern, name in self.law_patterns:
            if re.search(pattern, full_text):
                law_name = name
                break
        
        # 条文参照を抽出（第X条、第X条のY）
        pattern = r'第(\d+)条(?:の(\d+))?'
        for match in re.finditer(pattern, full_text):
            article_num = match.group(1)
            suffix = match.group(2) or ""
            
            key = (article_num, suffix)
            if key not in seen:
                seen.add(key)
                references.append(ArticleReference(
                    law_name=law_name,
                    article_num=article_num,
                    article_suffix=suffix
                ))
        
        return references
    
    def retrieve(
        self, 
        question: str, 
        choices: str = "",
        top_k: int = None
    ) -> List[Any]:
        """
        複数クエリ検索を実行
        
        Args:
            question: 問題文
            choices: 選択肢テキスト
            top_k: 最終結果数（省略時はmax_total_results）
        
        Returns:
            検索結果のリスト（重複排除・スコア順）
        """
        top_k = top_k or self.max_total_results
        
        # 条文参照を抽出
        refs = self.extract_article_references(question, choices)
        
        # メインクエリ（問題文+選択肢、正規化済み）
        full_query = question + " " + choices
        proc = self.query_processor.process(full_query)
        main_query = proc["normalized"]
        
        all_results = []
        seen_contents = set()  # 重複排除用
        
        # 1. メインクエリで検索
        main_results = self.base_retriever.retrieve(main_query, top_k=self.per_article_top_k * 2)
        for r in main_results:
            content_hash = hash(r.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_results.append(r)
        
        # 2. 各条文参照で個別検索
        for ref in refs[:5]:  # 最大5条文まで
            article_query = ref.full_article
            if ref.law_name:
                article_query = f"{ref.law_name} {article_query}"
            
            article_results = self.base_retriever.retrieve(
                article_query, 
                top_k=self.per_article_top_k
            )
            
            for r in article_results:
                content_hash = hash(r.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    # スコアを少し下げて追加（メインクエリ結果を優先）
                    r.score = r.score * 0.95
                    all_results.append(r)
        
        # 3. スコア順にソートして上位を返す
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k]
    
    def retrieve_with_details(
        self, 
        question: str, 
        choices: str = "",
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        詳細情報付きで検索を実行
        
        Returns:
            {
                "results": 検索結果リスト,
                "article_refs": 抽出された条文参照,
                "query_count": 実行されたクエリ数
            }
        """
        refs = self.extract_article_references(question, choices)
        results = self.retrieve(question, choices, top_k)
        
        return {
            "results": results,
            "article_refs": [ref.full_article for ref in refs],
            "query_count": 1 + len(refs[:5])
        }


# テスト
if __name__ == "__main__":
    # テスト用の簡単な確認
    retriever = MultiQueryRetriever(base_retriever=None)
    
    question = "金融商品取引法第27条の2の2の規定により、公開買付けによらなければならないものとして、正しいものを教えてください。"
    choices = """a 第14条の3に基づく報告
b 第27条の22に基づく届出
c その他"""
    
    refs = retriever.extract_article_references(question, choices)
    
    print("=== 条文参照抽出テスト ===")
    print(f"問題: {question[:50]}...")
    print(f"\n抽出された条文参照:")
    for ref in refs:
        print(f"  {ref.full_article} (法令: {ref.law_name})")

