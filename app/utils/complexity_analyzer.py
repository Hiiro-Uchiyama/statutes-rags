"""
問題複雑さ分析器

問題の複雑さを判定し、適切な処理パスを選択する
"""
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """複雑さレベル"""
    SIMPLE = "simple"       # 単純: 1条文、明確な回答
    MODERATE = "moderate"   # 中程度: 2-3条文、比較必要
    COMPLEX = "complex"     # 複雑: 4条文以上、数値比較、参照関係


@dataclass
class ComplexityAnalysis:
    """複雑さ分析結果"""
    level: ComplexityLevel
    score: float  # 0.0-1.0
    factors: Dict[str, Any]
    recommended_strategy: str


class ComplexityAnalyzer:
    """問題の複雑さを分析"""
    
    # 複雑さを示すキーワード
    COMPLEXITY_KEYWORDS = {
        'high': [
            '組み合わせ', '組合せ', 'すべて', '全て',
            '及び', '並びに', '又は', '若しくは',
            '政令で定める', '施行令', '施行規則',
            'ただし', '但し', 'この限りでない',
        ],
        'medium': [
            '以内', '以上', '以下', '超', '未満',
            '期間', '期限', '日', '月', '年',
            'パーセント', '%', '割合',
        ],
        'low': [
            '次のうち', '正しいもの', '誤っているもの',
        ]
    }
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        question: str,
        choices: List[str],
        context: str = ""
    ) -> ComplexityAnalysis:
        """
        問題の複雑さを分析
        
        Args:
            question: 問題文
            choices: 選択肢リスト
            context: コンテキスト（あれば）
            
        Returns:
            複雑さ分析結果
        """
        factors = {}
        score = 0.0
        
        full_text = question + " " + " ".join(choices)
        
        # 1. 条文参照数
        article_refs = self._count_article_references(full_text)
        factors['article_refs'] = article_refs
        if article_refs >= 4:
            score += 0.3
        elif article_refs >= 2:
            score += 0.15
        
        # 2. キーワード分析
        high_kw_count = sum(1 for kw in self.COMPLEXITY_KEYWORDS['high'] if kw in full_text)
        medium_kw_count = sum(1 for kw in self.COMPLEXITY_KEYWORDS['medium'] if kw in full_text)
        
        factors['high_complexity_keywords'] = high_kw_count
        factors['medium_complexity_keywords'] = medium_kw_count
        
        score += min(high_kw_count * 0.1, 0.3)
        score += min(medium_kw_count * 0.05, 0.15)
        
        # 3. 数値の存在
        has_numbers = self._has_numeric_comparison(full_text)
        factors['has_numeric_comparison'] = has_numbers
        if has_numbers:
            score += 0.15
        
        # 4. 選択肢の長さ
        avg_choice_len = sum(len(c) for c in choices) / max(len(choices), 1)
        factors['avg_choice_length'] = avg_choice_len
        if avg_choice_len > 80:
            score += 0.1
        
        # 5. 質問タイプ
        question_type = self._determine_question_type(question)
        factors['question_type'] = question_type
        if question_type in ['組み合わせ', '複数条文比較']:
            score += 0.2
        
        # レベル判定
        if score >= 0.5:
            level = ComplexityLevel.COMPLEX
            strategy = "full_multi_agent"
        elif score >= 0.25:
            level = ComplexityLevel.MODERATE
            strategy = "enhanced_rag"
        else:
            level = ComplexityLevel.SIMPLE
            strategy = "simple_rag"
        
        factors['final_score'] = score
        
        return ComplexityAnalysis(
            level=level,
            score=score,
            factors=factors,
            recommended_strategy=strategy
        )
    
    def _count_article_references(self, text: str) -> int:
        """条文参照数をカウント"""
        # 「第X条」のパターンをカウント
        pattern = r'第[一二三四五六七八九十百\d]+条'
        matches = re.findall(pattern, text)
        # 重複除去
        unique = set(matches)
        return len(unique)
    
    def _has_numeric_comparison(self, text: str) -> bool:
        """数値比較が必要かどうか"""
        # 期間や割合のパターン
        patterns = [
            r'[一二三四五六七八九十]+[日月年]',
            r'\d+[日月年]',
            r'[一二三四五六七八九十]+パーセント',
            r'\d+%',
            r'以内|以上|以下|超|未満',
        ]
        return any(re.search(p, text) for p in patterns)
    
    def _determine_question_type(self, question: str) -> str:
        """質問タイプを判定"""
        if '組み合わせ' in question or '組合せ' in question:
            return '組み合わせ'
        elif '誤っている' in question or '誤り' in question:
            return '誤り選択'
        elif '正しい' in question:
            return '正しい選択'
        elif 'すべて' in question or '全て' in question:
            return '複数選択'
        else:
            return '単純選択'


class AdaptiveProcessor:
    """複雑さに応じた適応的処理"""
    
    def __init__(self, analyzer: ComplexityAnalyzer = None):
        self.analyzer = analyzer or ComplexityAnalyzer()
    
    def get_processing_config(
        self,
        question: str,
        choices: List[str]
    ) -> Dict[str, Any]:
        """
        問題に応じた処理設定を取得
        
        Returns:
            処理設定（top_k, use_reranker, use_multi_agent, etc.）
        """
        analysis = self.analyzer.analyze(question, choices)
        
        if analysis.level == ComplexityLevel.COMPLEX:
            return {
                'strategy': 'full_multi_agent',
                'top_k': 30,
                'use_reranker': True,
                'use_numeric_comparison': True,
                'use_related_search': True,
                'max_iterations': 2,
                'complexity': analysis
            }
        elif analysis.level == ComplexityLevel.MODERATE:
            return {
                'strategy': 'enhanced_rag',
                'top_k': 20,
                'use_reranker': True,
                'use_numeric_comparison': True,
                'use_related_search': False,
                'max_iterations': 1,
                'complexity': analysis
            }
        else:
            return {
                'strategy': 'simple_rag',
                'top_k': 15,
                'use_reranker': False,
                'use_numeric_comparison': False,
                'use_related_search': False,
                'max_iterations': 0,
                'complexity': analysis
            }

