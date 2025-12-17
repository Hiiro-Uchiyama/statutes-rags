#!/usr/bin/env python3
"""
事後分析モジュール (Post-Analysis Module)

v4の判断を変えずに、事後的に以下を分析・評価:
1. Citation Analysis: 回答から引用された条文を抽出・評価
2. Confidence Scoring: 判断の確信度を算出
3. Error Pattern Detection: 失敗しやすいパターンを検出

新規性: v4の精度を維持しつつ、説明可能性・追跡可能性を向上
"""
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConfidenceLevel(Enum):
    """確信度レベル"""
    HIGH = "high"       # 0.8-1.0: 明確な根拠あり
    MEDIUM = "medium"   # 0.5-0.8: 部分的な根拠
    LOW = "low"         # 0.3-0.5: 根拠不明瞭
    VERY_LOW = "very_low"  # 0-0.3: 推測的


@dataclass
class Citation:
    """条文引用"""
    article_ref: str      # 「第二十四条」
    law_name: str = ""    # 「金融商品取引法」
    paragraph: str = ""   # 「第一項」
    content: str = ""     # 引用された内容
    verified: bool = False  # 検索結果に存在するか


@dataclass
class AnalysisResult:
    """分析結果"""
    question_id: int
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    error_patterns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning_quality: str = ""


class PostAnalyzer:
    """
    事後分析器
    
    v4の出力を分析し、以下を提供:
    - 引用条文の抽出・検証
    - 確信度スコアの算出
    - エラーパターンの検出
    """
    
    # 条文参照パターン
    ARTICLE_PATTERN = r'第[一二三四五六七八九十百千\d]+条(?:の[一二三四五六七八九十\d]+)?(?:第[一二三四五六七八九十\d]+項)?(?:第[一二三四五六七八九十\d]+号)?'
    
    # エラーパターン（失敗しやすい特徴）
    ERROR_PATTERNS = {
        'multiple_articles': r'政令で定める|施行令|内閣府令',
        'numerical_comparison': r'\d+[月日年]|[一二三四五六七八九十百千]+[月日年]|パーセント|%',
        'combination': r'組み合わせ|組合せ',
        'negation': r'誤っている|正しくない|該当しない',
    }
    
    # 高リスクキーワード
    HIGH_RISK_KEYWORDS = [
        '政令で定める', '施行令', '内閣府令',
        '準用する', '適用しない', '除く',
        '以内', '以上', '未満', '超える'
    ]
    
    def __init__(self, context_store: Optional[Dict[str, str]] = None):
        """
        Args:
            context_store: 検索されたコンテキストのストア（引用検証用）
        """
        self.context_store = context_store or {}
    
    def analyze(
        self,
        question_id: int,
        question: str,
        choices: List[str],
        answer: str,
        llm_response: str,
        context: str = ""
    ) -> AnalysisResult:
        """
        v4の出力を事後分析
        
        Args:
            question_id: 問題ID
            question: 問題文
            choices: 選択肢リスト
            answer: v4の回答
            llm_response: LLMの生成テキスト
            context: 検索されたコンテキスト
        
        Returns:
            AnalysisResult: 分析結果
        """
        result = AnalysisResult(
            question_id=question_id,
            answer=answer
        )
        
        # 1. 引用抽出・検証
        result.citations = self._extract_citations(llm_response, context)
        
        # 2. 確信度算出
        result.confidence_score = self._calculate_confidence(
            question, choices, answer, llm_response, result.citations
        )
        result.confidence_level = self._score_to_level(result.confidence_score)
        
        # 3. エラーパターン検出
        result.error_patterns = self._detect_error_patterns(question, choices)
        
        # 4. 警告生成
        result.warnings = self._generate_warnings(
            result.citations, result.error_patterns, result.confidence_score
        )
        
        # 5. 推論品質評価
        result.reasoning_quality = self._evaluate_reasoning(
            llm_response, result.citations, result.confidence_score
        )
        
        return result
    
    def _extract_citations(
        self,
        llm_response: str,
        context: str
    ) -> List[Citation]:
        """LLM応答から条文引用を抽出"""
        citations = []
        
        # 条文参照を抽出
        matches = re.findall(self.ARTICLE_PATTERN, llm_response)
        
        for match in matches:
            citation = Citation(
                article_ref=match,
                verified=self._verify_citation(match, context)
            )
            
            # 法令名を推測
            citation.law_name = self._infer_law_name(llm_response, match)
            
            # 重複を避けて追加
            if not any(c.article_ref == match for c in citations):
                citations.append(citation)
        
        return citations
    
    def _verify_citation(self, article_ref: str, context: str) -> bool:
        """引用がコンテキストに存在するか検証"""
        if not context:
            return False
        
        # 正規化して比較
        normalized_ref = article_ref.replace(' ', '')
        normalized_context = context.replace(' ', '')
        
        return normalized_ref in normalized_context
    
    def _infer_law_name(self, response: str, article_ref: str) -> str:
        """応答から法令名を推測"""
        law_patterns = [
            ('金融商品取引法', r'金融商品取引法'),
            ('借地借家法', r'借地借家法'),
            ('医薬品医療機器等法', r'医薬品.{0,10}法|薬機法'),
            ('施行令', r'施行令'),
            ('施行規則', r'施行規則'),
        ]
        
        for law_name, pattern in law_patterns:
            if re.search(pattern, response):
                return law_name
        
        return ""
    
    def _calculate_confidence(
        self,
        question: str,
        choices: List[str],
        answer: str,
        llm_response: str,
        citations: List[Citation]
    ) -> float:
        """確信度スコアを算出（0.0-1.0）"""
        score = 0.5  # ベーススコア
        
        # 1. 引用の有無と検証状況
        if citations:
            verified_count = sum(1 for c in citations if c.verified)
            citation_ratio = verified_count / len(citations) if citations else 0
            score += 0.2 * citation_ratio
        else:
            score -= 0.1  # 引用なしはペナルティ
        
        # 2. 回答の明確さ
        if answer in ['a', 'b', 'c', 'd']:
            score += 0.1
        
        # 3. 応答の長さ（適切な説明があるか）
        response_length = len(llm_response)
        if 200 < response_length < 2000:
            score += 0.1
        elif response_length < 100:
            score -= 0.1  # 短すぎる応答はペナルティ
        
        # 4. 高リスクキーワードの存在
        risk_count = sum(1 for kw in self.HIGH_RISK_KEYWORDS if kw in question)
        score -= 0.05 * min(risk_count, 3)
        
        # 5. 質問タイプによる調整
        if '誤っている' in question:
            score -= 0.05  # 誤り選択は難しい
        if '組み合わせ' in question:
            score -= 0.1  # 組み合わせはさらに難しい
        
        return max(0.0, min(1.0, score))
    
    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """スコアを確信度レベルに変換"""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _detect_error_patterns(
        self,
        question: str,
        choices: List[str]
    ) -> List[str]:
        """エラーパターンを検出"""
        full_text = question + " " + " ".join(choices)
        detected = []
        
        for pattern_name, pattern in self.ERROR_PATTERNS.items():
            if re.search(pattern, full_text):
                detected.append(pattern_name)
        
        return detected
    
    def _generate_warnings(
        self,
        citations: List[Citation],
        error_patterns: List[str],
        confidence_score: float
    ) -> List[str]:
        """警告を生成"""
        warnings = []
        
        # 引用関連の警告
        if not citations:
            warnings.append("Warning: 条文引用が検出されませんでした")
        else:
            unverified = [c for c in citations if not c.verified]
            if unverified:
                warnings.append(f"Warning: {len(unverified)}件の引用がコンテキストで確認できません")
        
        # エラーパターン関連の警告
        if 'multiple_articles' in error_patterns:
            warnings.append("Caution: 複数条文参照を含む問題です（失敗リスク高）")
        
        if 'numerical_comparison' in error_patterns:
            warnings.append("Caution: 数値比較を含む問題です（精査が必要）")
        
        if 'combination' in error_patterns:
            warnings.append("Caution: 組み合わせ問題です（複合判断が必要）")
        
        # 確信度関連の警告
        if confidence_score < 0.3:
            warnings.append("Alert: 確信度が非常に低いです（要レビュー）")
        elif confidence_score < 0.5:
            warnings.append("Info: 確信度が低めです")
        
        return warnings
    
    def _evaluate_reasoning(
        self,
        llm_response: str,
        citations: List[Citation],
        confidence_score: float
    ) -> str:
        """推論品質を評価"""
        factors = []
        
        # 引用品質
        if citations:
            verified_ratio = sum(1 for c in citations if c.verified) / len(citations)
            if verified_ratio > 0.8:
                factors.append("引用品質:高")
            elif verified_ratio > 0.5:
                factors.append("引用品質:中")
            else:
                factors.append("引用品質:低")
        else:
            factors.append("引用品質:なし")
        
        # 説明の詳細さ
        if len(llm_response) > 500:
            factors.append("説明:詳細")
        elif len(llm_response) > 200:
            factors.append("説明:適度")
        else:
            factors.append("説明:簡潔")
        
        # 総合評価
        if confidence_score >= 0.7 and citations and len(llm_response) > 300:
            overall = "Good"
        elif confidence_score >= 0.5:
            overall = "Acceptable"
        else:
            overall = "Needs Review"
        
        return f"{overall} ({', '.join(factors)})"
    
    def generate_report(self, results: List[AnalysisResult]) -> str:
        """分析結果のレポートを生成"""
        total = len(results)
        
        # 統計
        high_conf = sum(1 for r in results if r.confidence_level == ConfidenceLevel.HIGH)
        medium_conf = sum(1 for r in results if r.confidence_level == ConfidenceLevel.MEDIUM)
        low_conf = sum(1 for r in results if r.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW])
        
        with_citations = sum(1 for r in results if r.citations)
        verified_citations = sum(1 for r in results if any(c.verified for c in r.citations))
        
        # エラーパターン統計
        pattern_counts = {}
        for r in results:
            for p in r.error_patterns:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1
        
        report = f"""# 事後分析レポート

## 概要
- 分析対象: {total}問
- 生成日時: (自動生成)

## 確信度分布

| レベル | 件数 | 割合 |
|--------|------|------|
| HIGH (0.8+) | {high_conf} | {high_conf/total*100:.1f}% |
| MEDIUM (0.5-0.8) | {medium_conf} | {medium_conf/total*100:.1f}% |
| LOW/VERY_LOW (<0.5) | {low_conf} | {low_conf/total*100:.1f}% |

## 引用分析

| 指標 | 値 |
|------|-----|
| 引用検出率 | {with_citations/total*100:.1f}% ({with_citations}/{total}) |
| 引用検証率 | {verified_citations/total*100:.1f}% ({verified_citations}/{total}) |

## エラーパターン分布

| パターン | 件数 | 説明 |
|----------|------|------|
"""
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            desc = {
                'multiple_articles': '複数条文参照（政令/施行令）',
                'numerical_comparison': '数値比較（期間/割合）',
                'combination': '組み合わせ問題',
                'negation': '否定形質問（誤り選択）'
            }.get(pattern, pattern)
            report += f"| {pattern} | {count} | {desc} |\n"
        
        report += """
## リスク評価

低確信度の問題（要レビュー）:
"""
        low_conf_results = [r for r in results if r.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]]
        for r in low_conf_results[:10]:
            report += f"- Q{r.question_id}: 確信度 {r.confidence_score:.2f}, パターン: {', '.join(r.error_patterns)}\n"
        
        if len(low_conf_results) > 10:
            report += f"- ...他 {len(low_conf_results) - 10}件\n"
        
        return report


class CitationTracker:
    """
    引用追跡器
    
    判断根拠の透明性を提供
    """
    
    def __init__(self):
        self.citations_log: Dict[int, List[Citation]] = {}
    
    def log_citations(self, question_id: int, citations: List[Citation]):
        """引用をログに記録"""
        self.citations_log[question_id] = citations
    
    def get_citation_summary(self) -> Dict[str, Any]:
        """引用の統計サマリを取得"""
        all_citations = []
        for cites in self.citations_log.values():
            all_citations.extend(cites)
        
        # 条文ごとの引用回数
        article_counts = {}
        for c in all_citations:
            article_counts[c.article_ref] = article_counts.get(c.article_ref, 0) + 1
        
        # 法令ごとの引用回数
        law_counts = {}
        for c in all_citations:
            if c.law_name:
                law_counts[c.law_name] = law_counts.get(c.law_name, 0) + 1
        
        return {
            'total_citations': len(all_citations),
            'unique_articles': len(article_counts),
            'top_articles': sorted(article_counts.items(), key=lambda x: -x[1])[:10],
            'by_law': law_counts,
            'verification_rate': sum(1 for c in all_citations if c.verified) / len(all_citations) if all_citations else 0
        }


# テスト用
if __name__ == "__main__":
    analyzer = PostAnalyzer()
    
    # サンプル分析
    result = analyzer.analyze(
        question_id=1,
        question="金融商品取引法第24条の規定について、誤っているものを選べ",
        choices=[
            "内国会社は三月以内に提出",
            "外国会社は六月以内に提出",
            "提出先は内閣総理大臣",
            "縦覧は不要"
        ],
        answer="d",
        llm_response="第24条の規定により、有価証券報告書は内閣総理大臣に提出が必要です。選択肢dの「縦覧は不要」は誤りです。",
        context="第二十四条 有価証券報告書を内閣総理大臣に提出しなければならない。"
    )
    
    print(f"確信度: {result.confidence_score:.2f} ({result.confidence_level.value})")
    print(f"引用: {[c.article_ref for c in result.citations]}")
    print(f"エラーパターン: {result.error_patterns}")
    print(f"警告: {result.warnings}")
    print(f"推論品質: {result.reasoning_quality}")



