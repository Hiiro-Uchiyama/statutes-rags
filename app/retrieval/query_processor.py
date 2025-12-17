"""
検索クエリの前処理・拡張モジュール

機能:
1. 数字正規化（アラビア数字→漢数字）
2. 条文参照の抽出・展開
3. Query Expansion（キーワード拡張）
"""

import re
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.number_normalizer import (
    normalize_article_numbers,
    get_both_notations,
    extract_article_references
)


class QueryProcessor:
    """検索クエリの前処理・拡張を行うクラス"""
    
    def __init__(
        self,
        normalize_numbers: bool = True,
        expand_articles: bool = True,
        add_keywords: bool = True
    ):
        """
        Args:
            normalize_numbers: 条文番号を漢数字に正規化
            expand_articles: 条文参照を両表記で展開
            add_keywords: キーワードを追加
        """
        self.normalize_numbers = normalize_numbers
        self.expand_articles = expand_articles
        self.add_keywords = add_keywords
        
        # 法令名の略称→正式名称マッピング
        self.law_aliases = {
            "金商法": "金融商品取引法",
            "民法": "民法",
            "会社法": "会社法",
            "商法": "商法",
            "金融商品取引法施行令": "金融商品取引法施行令",
            "金商法施行令": "金融商品取引法施行令",
        }
        
        # 法令関連キーワード
        self.legal_keywords = {
            "損害賠償": ["賠償", "責任", "損害"],
            "虚偽記載": ["虚偽", "記載", "偽り"],
            "届出": ["届出", "届出書", "提出"],
            "開示": ["開示", "公表", "報告"],
            "有価証券": ["有価証券", "株券", "社債"],
        }
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        クエリを処理して拡張版を返す
        
        Args:
            query: 元のクエリ
        
        Returns:
            {
                "original": 元のクエリ,
                "normalized": 正規化されたクエリ,
                "expanded": 拡張キーワードを含むクエリ,
                "article_refs": 抽出された条文参照,
                "search_terms": 検索に使用する用語リスト
            }
        """
        result = {
            "original": query,
            "normalized": query,
            "expanded": query,
            "article_refs": [],
            "search_terms": []
        }
        
        # 1. 数字正規化
        if self.normalize_numbers:
            result["normalized"] = normalize_article_numbers(query, to_kanji=True)
        
        # 2. 条文参照の抽出
        result["article_refs"] = extract_article_references(query)
        
        # 3. 法令名の正規化
        normalized = result["normalized"]
        for alias, full_name in self.law_aliases.items():
            if alias in normalized and full_name not in normalized:
                normalized = normalized.replace(alias, full_name)
        result["normalized"] = normalized
        
        # 4. 検索用語の構築
        search_terms = [result["normalized"]]
        
        # 条文参照を両表記で追加
        if self.expand_articles:
            for ref in result["article_refs"]:
                kanji_ref, arabic_ref = get_both_notations(ref)
                if kanji_ref not in search_terms:
                    search_terms.append(kanji_ref)
        
        # 関連キーワードを追加
        if self.add_keywords:
            for keyword, related in self.legal_keywords.items():
                if keyword in query:
                    for rel in related:
                        if rel not in query:
                            search_terms.append(rel)
        
        result["search_terms"] = search_terms
        
        # 5. 拡張クエリの構築
        expanded_parts = [result["normalized"]]
        # 条文参照を明示的に追加
        for ref in result["article_refs"]:
            expanded_parts.append(ref)
        result["expanded"] = " ".join(expanded_parts)
        
        return result
    
    def get_search_query(self, query: str) -> str:
        """
        検索に最適化されたクエリを返す
        
        Args:
            query: 元のクエリ
        
        Returns:
            検索用クエリ文字列
        """
        processed = self.process(query)
        return processed["normalized"]
    
    def get_multi_queries(self, query: str) -> List[str]:
        """
        複数の検索クエリを生成（多角的検索用）
        
        Args:
            query: 元のクエリ
        
        Returns:
            検索クエリのリスト
        """
        processed = self.process(query)
        
        queries = [processed["normalized"]]
        
        # 条文参照がある場合、条文に特化したクエリを追加
        for ref in processed["article_refs"]:
            # 法令名を抽出
            law_names = []
            for alias, full_name in self.law_aliases.items():
                if alias in query or full_name in query:
                    law_names.append(full_name)
            
            if law_names:
                for law in law_names:
                    queries.append(f"{law} {ref}")
            else:
                queries.append(ref)
        
        return list(dict.fromkeys(queries))  # 重複除去しつつ順序保持


class ContextAwareQueryProcessor(QueryProcessor):
    """
    コンテキスト（正解条文）を意識したクエリ処理
    
    4択QAのコンテキストと同様の情報を検索で取得することを目指す
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def extract_target_context(self, question: str, choices: str = "") -> Dict[str, Any]:
        """
        問題文と選択肢から、必要なコンテキスト情報を推定
        
        Args:
            question: 問題文
            choices: 選択肢テキスト
        
        Returns:
            {
                "law_name": 推定される法令名,
                "article_refs": 条文参照リスト,
                "key_terms": キーワードリスト,
                "target_query": 最適化された検索クエリ
            }
        """
        result = {
            "law_name": "",
            "article_refs": [],
            "key_terms": [],
            "target_query": ""
        }
        
        full_text = question + " " + choices
        
        # 法令名を抽出
        law_patterns = [
            r'(金融商品取引法(?:施行令)?)',
            r'(民法)',
            r'(会社法)',
            r'(商法)',
            r'([ァ-ヴー一-龯]+法(?:施行令|施行規則)?)',
        ]
        for pattern in law_patterns:
            match = re.search(pattern, full_text)
            if match:
                result["law_name"] = match.group(1)
                break
        
        # 条文参照を抽出
        result["article_refs"] = extract_article_references(full_text)
        
        # キーワードを抽出
        key_patterns = [
            r'(損害賠償|賠償責任)',
            r'(虚偽記載|虚偽の記載)',
            r'(届出書|報告書)',
            r'(開示|公表)',
            r'(有価証券|株券)',
            r'(取締役|役員)',
        ]
        for pattern in key_patterns:
            matches = re.findall(pattern, full_text)
            result["key_terms"].extend(matches)
        
        # 最適化クエリを構築
        query_parts = []
        if result["law_name"]:
            query_parts.append(result["law_name"])
        for ref in result["article_refs"][:2]:  # 最初の2つ
            query_parts.append(ref)
        for term in result["key_terms"][:3]:  # 最初の3つ
            query_parts.append(term)
        
        result["target_query"] = " ".join(query_parts)
        
        return result


# テスト
if __name__ == "__main__":
    processor = QueryProcessor()
    
    test_queries = [
        "金融商品取引法第21条により、損害賠償責任を負う可能性のある者として、正しいものを教えてください。",
        "第24条の規定に基づく有価証券報告書の提出義務について",
        "第27条の38の規定について、正しいものを教えてください。",
    ]
    
    print("=== Query Processing Test ===\n")
    for query in test_queries:
        result = processor.process(query)
        print(f"Original: {query[:60]}...")
        print(f"Normalized: {result['normalized'][:60]}...")
        print(f"Article refs: {result['article_refs']}")
        print(f"Search terms: {result['search_terms'][:3]}...")
        print()
    
    print("=== Context-Aware Test ===\n")
    ca_processor = ContextAwareQueryProcessor()
    
    question = "金融商品取引法第21条により、損害賠償責任を負う可能性のある者として、正しいものを教えてください。"
    choices = """a 当該有価証券届出書を提出した会社のその提出の時における役員
b 当該有価証券の売出しをした者
c 以上全て
d 以上のいずれでもない"""
    
    context_info = ca_processor.extract_target_context(question, choices)
    print(f"Law name: {context_info['law_name']}")
    print(f"Article refs: {context_info['article_refs']}")
    print(f"Key terms: {context_info['key_terms']}")
    print(f"Target query: {context_info['target_query']}")

