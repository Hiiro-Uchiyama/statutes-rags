"""
数値抽出・比較の専用ロジック

法令文書内の期間、割合、金額などの数値を抽出し、構造化して比較する
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedNumeric:
    """抽出された数値情報"""
    raw_text: str          # 元のテキスト
    value: float           # 数値
    unit: str              # 単位（日、月、年、%など）
    context: str           # 前後の文脈
    category: str          # カテゴリ（期間、割合、金額など）
    source_article: str    # 出典条文


class NumericExtractor:
    """法令文書から数値を抽出"""
    
    # 漢数字変換テーブル
    KANJI_NUMS = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000
    }
    
    # 期間パターン
    PERIOD_PATTERNS = [
        # 漢数字 + 単位
        (r'([一二三四五六七八九十百]+)(日|週間?|月|年)(以内|以上|以下|を超え|未満|間)?', 'period'),
        # アラビア数字 + 単位
        (r'(\d+)(日|週間?|月|年)(以内|以上|以下|を超え|未満|間)?', 'period'),
    ]
    
    # 割合パターン
    RATIO_PATTERNS = [
        (r'([一二三四五六七八九十百]+)(パーセント|%|割|分)', 'ratio'),
        (r'(\d+\.?\d*)(パーセント|%|割|分)', 'ratio'),
        (r'(百分の[一二三四五六七八九十百]+)', 'ratio'),
        (r'([一二三四五六七八九十]+分の[一二三四五六七八九十]+)', 'ratio'),
    ]
    
    # 金額パターン
    AMOUNT_PATTERNS = [
        (r'([一二三四五六七八九十百千万億]+)(円|ドル|ユーロ)', 'amount'),
        (r'(\d+(?:,\d{3})*)(円|ドル|ユーロ)', 'amount'),
    ]
    
    # 条文番号パターン
    ARTICLE_PATTERNS = [
        (r'第([一二三四五六七八九十百]+)条(の[一二三四五六七八九十]+)?', 'article'),
        (r'第(\d+)条(の\d+)?', 'article'),
    ]
    
    def __init__(self):
        self.all_patterns = (
            self.PERIOD_PATTERNS + 
            self.RATIO_PATTERNS + 
            self.AMOUNT_PATTERNS +
            self.ARTICLE_PATTERNS
        )
    
    def kanji_to_number(self, kanji: str) -> float:
        """漢数字をアラビア数字に変換"""
        if not kanji:
            return 0
        
        # 単純な漢数字（一〜九）
        if len(kanji) == 1 and kanji in self.KANJI_NUMS:
            return self.KANJI_NUMS[kanji]
        
        result = 0
        current = 0
        
        for char in kanji:
            if char in self.KANJI_NUMS:
                val = self.KANJI_NUMS[char]
                if val >= 10:
                    if current == 0:
                        current = 1
                    result += current * val
                    current = 0
                else:
                    current = current * 10 + val if current > 0 else val
            else:
                # 不明な文字はスキップ
                pass
        
        result += current
        return float(result) if result > 0 else float(current)
    
    def extract_numerics(
        self,
        text: str,
        source_article: str = ""
    ) -> List[ExtractedNumeric]:
        """テキストから数値情報を抽出"""
        results = []
        
        for pattern, category in self.all_patterns:
            for match in re.finditer(pattern, text):
                raw_text = match.group(0)
                
                # 数値部分を抽出
                num_str = match.group(1)
                
                # 漢数字かアラビア数字か判定して変換
                if re.match(r'^[一二三四五六七八九十百千万億]+$', num_str):
                    value = self.kanji_to_number(num_str)
                elif re.match(r'^\d+\.?\d*$', num_str.replace(',', '')):
                    value = float(num_str.replace(',', ''))
                else:
                    value = 0
                
                # 単位を抽出
                unit = match.group(2) if len(match.groups()) > 1 else ""
                
                # 文脈を抽出（前後30文字）
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                
                results.append(ExtractedNumeric(
                    raw_text=raw_text,
                    value=value,
                    unit=unit,
                    context=context,
                    category=category,
                    source_article=source_article
                ))
        
        return results
    
    def extract_from_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, List[ExtractedNumeric]]:
        """複数ドキュメントから数値を抽出してカテゴリ別に整理"""
        by_category = {
            'period': [],
            'ratio': [],
            'amount': [],
            'article': []
        }
        
        for doc in documents:
            if hasattr(doc, 'page_content'):
                text = doc.page_content
                source = doc.metadata.get('article_num', '') if hasattr(doc, 'metadata') else ''
            elif isinstance(doc, dict):
                text = doc.get('text', doc.get('content', ''))
                source = doc.get('article_num', '')
            else:
                text = str(doc)
                source = ''
            
            numerics = self.extract_numerics(text, source)
            for num in numerics:
                by_category[num.category].append(num)
        
        return by_category


class NumericComparator:
    """数値の比較ロジック"""
    
    def __init__(self, extractor: Optional[NumericExtractor] = None):
        self.extractor = extractor or NumericExtractor()
    
    def compare_periods(
        self,
        choice_text: str,
        context_text: str
    ) -> Dict[str, Any]:
        """選択肢とコンテキストの期間を比較"""
        choice_periods = self.extractor.extract_numerics(choice_text)
        choice_periods = [p for p in choice_periods if p.category == 'period']
        
        context_periods = self.extractor.extract_numerics(context_text)
        context_periods = [p for p in context_periods if p.category == 'period']
        
        result = {
            'choice_periods': [(p.raw_text, p.value, p.unit) for p in choice_periods],
            'context_periods': [(p.raw_text, p.value, p.unit) for p in context_periods],
            'matches': [],
            'mismatches': []
        }
        
        # 比較
        for cp in choice_periods:
            matched = False
            for ctx_p in context_periods:
                if cp.unit == ctx_p.unit:
                    if cp.value == ctx_p.value:
                        result['matches'].append({
                            'choice': cp.raw_text,
                            'context': ctx_p.raw_text,
                            'match': True
                        })
                        matched = True
                    else:
                        result['mismatches'].append({
                            'choice': cp.raw_text,
                            'context': ctx_p.raw_text,
                            'choice_value': cp.value,
                            'context_value': ctx_p.value,
                            'difference': cp.value - ctx_p.value
                        })
            
            if not matched and not any(
                m['choice'] == cp.raw_text for m in result['mismatches']
            ):
                result['mismatches'].append({
                    'choice': cp.raw_text,
                    'context': None,
                    'note': 'コンテキストに該当する期間なし'
                })
        
        return result
    
    def generate_comparison_summary(
        self,
        choice_text: str,
        context_text: str
    ) -> str:
        """選択肢とコンテキストの数値比較サマリーを生成"""
        period_result = self.compare_periods(choice_text, context_text)
        
        lines = ["【数値比較結果】"]
        
        if period_result['matches']:
            lines.append("一致:")
            for m in period_result['matches']:
                lines.append(f"  - {m['choice']} = {m['context']}")
        
        if period_result['mismatches']:
            lines.append("不一致:")
            for m in period_result['mismatches']:
                if m.get('context'):
                    lines.append(
                        f"  - 選択肢: {m['choice']} vs 条文: {m['context']} "
                        f"(差: {m.get('difference', 'N/A')})"
                    )
                else:
                    lines.append(f"  - 選択肢: {m['choice']} ({m.get('note', '')})")
        
        if not period_result['matches'] and not period_result['mismatches']:
            lines.append("数値情報なし")
        
        return "\n".join(lines)

