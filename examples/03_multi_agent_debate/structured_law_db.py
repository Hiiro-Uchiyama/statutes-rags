#!/usr/bin/env python3
"""
構造化法令DB

法令の条文から数値情報、参照関係を抽出して構造化保存
"""
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class NumberInfo:
    """条文内の数値情報"""
    periods: List[str] = field(default_factory=list)    # 期間: ["六十日以内", "三十日以内"]
    ratios: List[str] = field(default_factory=list)     # 割合: ["三分の二以上", "過半数"]
    amounts: List[str] = field(default_factory=list)    # 金額: ["一億円以上"]
    counts: List[str] = field(default_factory=list)     # 人数等: ["三人以上"]


@dataclass
class Reference:
    """参照関係"""
    ref_type: str              # 施行令/内閣府令/他条文
    target: str                # 参照先
    context: str = ""          # 参照文脈


@dataclass
class StructuredArticle:
    """構造化された条文"""
    article_id: str            # 第二十七条の二十三の三
    text: str                  # 条文本文
    numbers: NumberInfo = field(default_factory=NumberInfo)
    references: List[Reference] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class StructuredLawDB:
    """構造化法令DB"""
    
    # 数値抽出パターン
    PERIOD_PATTERNS = [
        r'[一二三四五六七八九十百]+[日週月年]以?内?',
        r'\d+[日週月年]以?内?',
        r'[一二三四五六七八九十百]+箇?月',
    ]
    
    RATIO_PATTERNS = [
        r'[一二三四五六七八九十]+分の[一二三四五六七八九十]+以?上?',
        r'\d+分の\d+以?上?',
        r'過半数',
        r'\d+[%％]',
        r'\d+パーセント',
    ]
    
    AMOUNT_PATTERNS = [
        r'[一二三四五六七八九十百千万億]+円',
        r'\d+[万億]?円',
    ]
    
    COUNT_PATTERNS = [
        r'[一二三四五六七八九十百]+人以?[上下]?',
        r'\d+人以?[上下]?',
        r'[一二三四五六七八九十百]+名',
    ]
    
    # 参照パターン
    REFERENCE_PATTERNS = [
        (r'政令で定める', '施行令'),
        (r'内閣府令で定める', '内閣府令'),
        (r'省令で定める', '省令'),
        (r'第[一二三四五六七八九十百]+条(?:の[一二三四五六七八九十]+)?(?:第[一二三四五六七八九十]+項)?', '条文'),
    ]
    
    def __init__(self):
        self.articles: Dict[str, StructuredArticle] = {}
        self._cache: Dict[str, Any] = {}
    
    def extract_numbers(self, text: str) -> NumberInfo:
        """テキストから数値情報を抽出"""
        info = NumberInfo()
        
        for pattern in self.PERIOD_PATTERNS:
            matches = re.findall(pattern, text)
            info.periods.extend(matches)
        
        for pattern in self.RATIO_PATTERNS:
            matches = re.findall(pattern, text)
            info.ratios.extend(matches)
        
        for pattern in self.AMOUNT_PATTERNS:
            matches = re.findall(pattern, text)
            info.amounts.extend(matches)
        
        for pattern in self.COUNT_PATTERNS:
            matches = re.findall(pattern, text)
            info.counts.extend(matches)
        
        # 重複除去
        info.periods = list(set(info.periods))
        info.ratios = list(set(info.ratios))
        info.amounts = list(set(info.amounts))
        info.counts = list(set(info.counts))
        
        return info
    
    def extract_references(self, text: str) -> List[Reference]:
        """テキストから参照関係を抽出"""
        references = []
        
        for pattern, ref_type in self.REFERENCE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if ref_type == '条文':
                    references.append(Reference(
                        ref_type='条文',
                        target=match,
                        context=self._get_context(text, match)
                    ))
                else:
                    references.append(Reference(
                        ref_type=ref_type,
                        target='',
                        context=self._get_context(text, match)
                    ))
        
        return references
    
    def _get_context(self, text: str, match: str, window: int = 30) -> str:
        """マッチ周辺のコンテキストを取得"""
        idx = text.find(match)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(match) + window)
        return text[start:end]
    
    def extract_keywords(self, text: str) -> List[str]:
        """重要キーワードを抽出"""
        keywords = []
        
        important_terms = [
            '届出', '届け出', '報告', '公告', '開示',
            '公開買付', 'TOB', '大量保有',
            '内部者取引', 'インサイダー',
            '有価証券報告書', '半期報告書', '四半期報告書',
            '目論見書', '訂正', '変更',
            '罰則', '違反', '課徴金',
        ]
        
        for term in important_terms:
            if term in text:
                keywords.append(term)
        
        return keywords
    
    def add_article(self, article_id: str, text: str) -> StructuredArticle:
        """条文を追加"""
        article = StructuredArticle(
            article_id=article_id,
            text=text,
            numbers=self.extract_numbers(text),
            references=self.extract_references(text),
            keywords=self.extract_keywords(text)
        )
        self.articles[article_id] = article
        return article
    
    def build_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """検索結果のドキュメントから構造化DBを構築"""
        for doc in documents:
            content = doc.get('content', '') or doc.get('page_content', '')
            metadata = doc.get('metadata', {})
            
            # 条文IDの抽出を試みる
            article_id = metadata.get('article_id', '')
            if not article_id:
                # テキストから条文番号を抽出
                match = re.search(r'第[一二三四五六七八九十百]+条(?:の[一二三四五六七八九十]+)?', content)
                if match:
                    article_id = match.group()
                else:
                    article_id = f"doc_{len(self.articles)}"
            
            self.add_article(article_id, content)
    
    def get_numbers_table(self, choices: List[str]) -> Dict[str, Any]:
        """選択肢と条文の数値対比表を生成"""
        table = {
            'choices': {},
            'articles': {},
            'comparison': []
        }
        
        # 選択肢から数値抽出
        for i, choice in enumerate(choices):
            choice_id = chr(97 + i)  # a, b, c, d
            numbers = self.extract_numbers(choice)
            table['choices'][choice_id] = {
                'periods': numbers.periods,
                'ratios': numbers.ratios,
                'amounts': numbers.amounts,
                'counts': numbers.counts
            }
        
        # 条文から数値抽出（集約）
        all_periods = []
        all_ratios = []
        all_amounts = []
        all_counts = []
        
        for article in self.articles.values():
            all_periods.extend(article.numbers.periods)
            all_ratios.extend(article.numbers.ratios)
            all_amounts.extend(article.numbers.amounts)
            all_counts.extend(article.numbers.counts)
        
        table['articles'] = {
            'periods': list(set(all_periods)),
            'ratios': list(set(all_ratios)),
            'amounts': list(set(all_amounts)),
            'counts': list(set(all_counts))
        }
        
        # 比較結果
        for choice_id, choice_nums in table['choices'].items():
            for num_type in ['periods', 'ratios', 'amounts', 'counts']:
                for choice_num in choice_nums.get(num_type, []):
                    article_nums = table['articles'].get(num_type, [])
                    match_status = '一致' if choice_num in article_nums else '不一致/未確認'
                    table['comparison'].append({
                        'choice': choice_id,
                        'type': num_type,
                        'choice_value': choice_num,
                        'match_status': match_status,
                        'article_values': article_nums
                    })
        
        return table
    
    def format_numbers_table(self, choices: List[str]) -> str:
        """数値対比表を文字列形式で出力"""
        table = self.get_numbers_table(choices)
        
        lines = ["【数値対比表】"]
        
        # 条文の数値
        articles = table['articles']
        if any([articles['periods'], articles['ratios'], articles['amounts'], articles['counts']]):
            lines.append("条文の数値:")
            if articles['periods']:
                lines.append(f"  期間: {', '.join(articles['periods'])}")
            if articles['ratios']:
                lines.append(f"  割合: {', '.join(articles['ratios'])}")
            if articles['amounts']:
                lines.append(f"  金額: {', '.join(articles['amounts'])}")
            if articles['counts']:
                lines.append(f"  人数: {', '.join(articles['counts'])}")
        
        # 選択肢の数値
        lines.append("")
        lines.append("選択肢の数値:")
        for choice_id, nums in table['choices'].items():
            choice_nums = []
            for num_type in ['periods', 'ratios', 'amounts', 'counts']:
                choice_nums.extend(nums.get(num_type, []))
            if choice_nums:
                lines.append(f"  {choice_id}: {', '.join(choice_nums)}")
        
        # 比較
        if table['comparison']:
            lines.append("")
            lines.append("照合結果:")
            for comp in table['comparison']:
                lines.append(f"  {comp['choice']}: {comp['choice_value']} → {comp['match_status']}")
        
        return '\n'.join(lines)
    
    def get_structured_context(self) -> str:
        """構造化されたコンテキストを文字列で出力"""
        lines = []
        
        for article_id, article in self.articles.items():
            lines.append(f"【{article_id}】")
            lines.append(article.text[:500])  # テキストは短縮
            
            if article.numbers.periods or article.numbers.ratios:
                nums = []
                if article.numbers.periods:
                    nums.append(f"期間: {', '.join(article.numbers.periods)}")
                if article.numbers.ratios:
                    nums.append(f"割合: {', '.join(article.numbers.ratios)}")
                lines.append(f"  数値: {'; '.join(nums)}")
            
            if article.references:
                refs = [f"{r.ref_type}" for r in article.references[:3]]
                lines.append(f"  参照: {', '.join(refs)}")
            
            lines.append("")
        
        return '\n'.join(lines)


# グローバルインスタンス
_db_instance: Optional[StructuredLawDB] = None

def get_db() -> StructuredLawDB:
    """シングルトンインスタンスを取得"""
    global _db_instance
    if _db_instance is None:
        _db_instance = StructuredLawDB()
    return _db_instance

def reset_db() -> None:
    """DBをリセット"""
    global _db_instance
    _db_instance = StructuredLawDB()
