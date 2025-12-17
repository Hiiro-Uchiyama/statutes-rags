"""
Proposed Metrics - 提案手法の評価指標

従来の正答率に加え、以下の指標を計測:
- 議論プロセス指標（正答転換率、誤答転換率、議論有効性）
- 法的根拠指標（条文引用率、引用正確性）
- エラー分類（検索失敗、解釈誤り、適用誤り、合意バイアス）
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """単一問題の評価結果"""
    question_id: str
    question: str
    correct_answer: str
    
    # 予測結果
    predicted_answer: str
    confidence: float
    is_correct: bool
    
    # 議論プロセス
    interpreter_answer: str = ""
    critic_alternative: str = ""
    answer_changed: bool = False
    criticism_severity: str = ""
    
    # 法的根拠
    citation_count: int = 0
    citation_rate: float = 0.0
    
    # メタデータ
    elapsed_time: float = 0.0
    error: Optional[str] = None
    audit_trail: Dict = field(default_factory=dict)


@dataclass 
class DebateProcessMetrics:
    """議論プロセス指標"""
    # 正答転換: 初回不正解 -> 最終正解
    correct_conversion_count: int = 0
    correct_conversion_rate: float = 0.0
    
    # 誤答転換: 初回正解 -> 最終不正解
    wrong_conversion_count: int = 0
    wrong_conversion_rate: float = 0.0
    
    # 議論有効性: 正答転換率 / 誤答転換率
    debate_effectiveness: float = 0.0
    
    # 回答変更率
    answer_change_rate: float = 0.0
    
    # 批判重大度分布
    severity_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class CitationMetrics:
    """法的根拠指標"""
    # 条文引用率
    avg_citation_count: float = 0.0
    avg_citation_rate: float = 0.0
    
    # 引用カバレッジ
    questions_with_citations: int = 0
    citation_coverage_rate: float = 0.0


@dataclass
class ErrorAnalysis:
    """エラー分析"""
    # エラータイプ別カウント
    search_failure: int = 0  # 検索失敗
    interpretation_error: int = 0  # 解釈誤り
    application_error: int = 0  # 適用誤り
    consensus_bias: int = 0  # 合意バイアス（グループシンキング）
    
    # エラー分布
    error_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 選択肢別エラー率
    choice_error_rates: Dict[str, float] = field(default_factory=dict)


def calculate_proposed_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    提案手法の評価指標を計算
    
    Args:
        results: 評価結果のリスト
    
    Returns:
        全ての評価指標を含む辞書
    """
    if not results:
        return {"error": "No results to evaluate"}
    
    total = len(results)
    
    # 基本指標
    correct_count = sum(1 for r in results if r.is_correct)
    accuracy = correct_count / total
    
    # 議論プロセス指標
    debate_metrics = _calculate_debate_process_metrics(results)
    
    # 法的根拠指標
    citation_metrics = _calculate_citation_metrics(results)
    
    # エラー分析
    error_analysis = _analyze_errors(results)
    
    # 確信度統計
    confidences = [r.confidence for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # 処理時間統計
    times = [r.elapsed_time for r in results if r.elapsed_time > 0]
    avg_time = sum(times) / len(times) if times else 0.0
    
    return {
        "basic_metrics": {
            "total_questions": total,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_time_per_question": avg_time
        },
        "debate_process_metrics": {
            "correct_conversion_count": debate_metrics.correct_conversion_count,
            "correct_conversion_rate": debate_metrics.correct_conversion_rate,
            "wrong_conversion_count": debate_metrics.wrong_conversion_count,
            "wrong_conversion_rate": debate_metrics.wrong_conversion_rate,
            "debate_effectiveness": debate_metrics.debate_effectiveness,
            "answer_change_rate": debate_metrics.answer_change_rate,
            "severity_distribution": debate_metrics.severity_distribution
        },
        "citation_metrics": {
            "avg_citation_count": citation_metrics.avg_citation_count,
            "avg_citation_rate": citation_metrics.avg_citation_rate,
            "citation_coverage_rate": citation_metrics.citation_coverage_rate
        },
        "error_analysis": {
            "search_failure": error_analysis.search_failure,
            "interpretation_error": error_analysis.interpretation_error,
            "application_error": error_analysis.application_error,
            "consensus_bias": error_analysis.consensus_bias,
            "error_distribution": error_analysis.error_distribution,
            "choice_error_rates": error_analysis.choice_error_rates
        }
    }


def _calculate_debate_process_metrics(results: List[EvaluationResult]) -> DebateProcessMetrics:
    """議論プロセス指標を計算"""
    metrics = DebateProcessMetrics()
    
    # 初回回答（interpreter_answer）と最終回答の比較
    interpreter_correct = 0
    interpreter_wrong = 0
    correct_conversions = 0
    wrong_conversions = 0
    answer_changes = 0
    severity_counts = Counter()
    
    for r in results:
        # 初回回答の正誤
        interpreter_was_correct = r.interpreter_answer == r.correct_answer
        final_is_correct = r.is_correct
        
        if interpreter_was_correct:
            interpreter_correct += 1
            if not final_is_correct:
                wrong_conversions += 1  # 正解 -> 不正解（悪化）
        else:
            interpreter_wrong += 1
            if final_is_correct:
                correct_conversions += 1  # 不正解 -> 正解（改善）
        
        # 回答変更
        if r.answer_changed:
            answer_changes += 1
        
        # 批判重大度
        if r.criticism_severity:
            severity_counts[r.criticism_severity] += 1
    
    # 率を計算
    metrics.correct_conversion_count = correct_conversions
    metrics.correct_conversion_rate = correct_conversions / interpreter_wrong if interpreter_wrong > 0 else 0.0
    
    metrics.wrong_conversion_count = wrong_conversions
    metrics.wrong_conversion_rate = wrong_conversions / interpreter_correct if interpreter_correct > 0 else 0.0
    
    # 議論有効性 = 正答転換率 / 誤答転換率
    if metrics.wrong_conversion_rate > 0:
        metrics.debate_effectiveness = metrics.correct_conversion_rate / metrics.wrong_conversion_rate
    else:
        metrics.debate_effectiveness = float('inf') if metrics.correct_conversion_rate > 0 else 1.0
    
    metrics.answer_change_rate = answer_changes / len(results) if results else 0.0
    metrics.severity_distribution = dict(severity_counts)
    
    return metrics


def _calculate_citation_metrics(results: List[EvaluationResult]) -> CitationMetrics:
    """法的根拠指標を計算"""
    metrics = CitationMetrics()
    
    citation_counts = [r.citation_count for r in results]
    citation_rates = [r.citation_rate for r in results]
    
    metrics.avg_citation_count = sum(citation_counts) / len(citation_counts) if citation_counts else 0.0
    metrics.avg_citation_rate = sum(citation_rates) / len(citation_rates) if citation_rates else 0.0
    
    # 引用カバレッジ
    questions_with_citations = sum(1 for r in results if r.citation_count > 0)
    metrics.questions_with_citations = questions_with_citations
    metrics.citation_coverage_rate = questions_with_citations / len(results) if results else 0.0
    
    return metrics


def _analyze_errors(results: List[EvaluationResult]) -> ErrorAnalysis:
    """エラー分析"""
    analysis = ErrorAnalysis()
    
    incorrect_results = [r for r in results if not r.is_correct]
    
    if not incorrect_results:
        return analysis
    
    for r in incorrect_results:
        error_type = _classify_error(r)
        
        if error_type == "search_failure":
            analysis.search_failure += 1
        elif error_type == "interpretation_error":
            analysis.interpretation_error += 1
        elif error_type == "application_error":
            analysis.application_error += 1
        elif error_type == "consensus_bias":
            analysis.consensus_bias += 1
    
    # エラー分布
    total_errors = len(incorrect_results)
    analysis.error_distribution = {
        "search_failure": analysis.search_failure / total_errors,
        "interpretation_error": analysis.interpretation_error / total_errors,
        "application_error": analysis.application_error / total_errors,
        "consensus_bias": analysis.consensus_bias / total_errors
    }
    
    # 選択肢別エラー率
    choice_errors = Counter()
    choice_totals = Counter()
    
    for r in results:
        choice_totals[r.correct_answer] += 1
        if not r.is_correct:
            choice_errors[r.correct_answer] += 1
    
    analysis.choice_error_rates = {
        choice: choice_errors[choice] / choice_totals[choice] if choice_totals[choice] > 0 else 0.0
        for choice in ["a", "b", "c", "d"]
    }
    
    return analysis


def _classify_error(result: EvaluationResult) -> str:
    """エラータイプを分類"""
    
    # 検索失敗: 引用が0件
    if result.citation_count == 0:
        return "search_failure"
    
    # 合意バイアス: 初回正解だったが最終的に不正解
    if result.interpreter_answer == result.correct_answer and result.answer_changed:
        return "consensus_bias"
    
    # 解釈誤り: 引用はあるが初回から不正解
    if result.interpreter_answer != result.correct_answer and not result.answer_changed:
        return "interpretation_error"
    
    # 適用誤り: その他（引用あり、解釈は正しいが適用を誤った）
    return "application_error"


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """評価指標をフォーマットしたレポートを生成"""
    
    lines = []
    lines.append("=" * 60)
    lines.append("PROPOSED METHOD EVALUATION REPORT")
    lines.append("=" * 60)
    
    # 基本指標
    basic = metrics.get("basic_metrics", {})
    lines.append("\n[Basic Metrics]")
    lines.append(f"  Total Questions: {basic.get('total_questions', 0)}")
    lines.append(f"  Correct Answers: {basic.get('correct_answers', 0)}")
    lines.append(f"  Accuracy: {basic.get('accuracy', 0):.2%}")
    lines.append(f"  Avg Confidence: {basic.get('avg_confidence', 0):.2f}")
    lines.append(f"  Avg Time/Question: {basic.get('avg_time_per_question', 0):.1f}s")
    
    # 議論プロセス指標
    debate = metrics.get("debate_process_metrics", {})
    lines.append("\n[Debate Process Metrics]")
    lines.append(f"  Correct Conversion Rate: {debate.get('correct_conversion_rate', 0):.2%}")
    lines.append(f"    (Wrong->Correct: {debate.get('correct_conversion_count', 0)} cases)")
    lines.append(f"  Wrong Conversion Rate: {debate.get('wrong_conversion_rate', 0):.2%}")
    lines.append(f"    (Correct->Wrong: {debate.get('wrong_conversion_count', 0)} cases)")
    lines.append(f"  Debate Effectiveness: {debate.get('debate_effectiveness', 0):.2f}")
    lines.append(f"  Answer Change Rate: {debate.get('answer_change_rate', 0):.2%}")
    lines.append(f"  Severity Distribution: {debate.get('severity_distribution', {})}")
    
    # 法的根拠指標
    citation = metrics.get("citation_metrics", {})
    lines.append("\n[Citation Metrics]")
    lines.append(f"  Avg Citation Count: {citation.get('avg_citation_count', 0):.1f}")
    lines.append(f"  Avg Citation Rate: {citation.get('avg_citation_rate', 0):.2%}")
    lines.append(f"  Citation Coverage: {citation.get('citation_coverage_rate', 0):.2%}")
    
    # エラー分析
    error = metrics.get("error_analysis", {})
    lines.append("\n[Error Analysis]")
    lines.append(f"  Search Failure: {error.get('search_failure', 0)}")
    lines.append(f"  Interpretation Error: {error.get('interpretation_error', 0)}")
    lines.append(f"  Application Error: {error.get('application_error', 0)}")
    lines.append(f"  Consensus Bias: {error.get('consensus_bias', 0)}")
    lines.append(f"  Choice Error Rates: {error.get('choice_error_rates', {})}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)

