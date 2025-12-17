"""
Citation Registry - 法的根拠の構造化共有

議論の各ステップで引用された法令条文を一元管理し、
追跡可能性と検証可能性を担保する。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CitationStatus(Enum):
    """引用の検証ステータス"""
    RETRIEVED = "retrieved"     # 検索で取得
    VERIFIED = "verified"       # 検証済み
    DISPUTED = "disputed"       # 議論中
    REJECTED = "rejected"       # 却下


@dataclass
class Citation:
    """法的根拠の構造化表現"""
    citation_id: str
    law_name: str
    article: str
    paragraph: Optional[str] = None
    item: Optional[str] = None
    text: str = ""
    relevance_score: float = 0.0
    status: CitationStatus = CitationStatus.RETRIEVED
    source: str = "local"  # "local" or "api"
    
    def to_reference_string(self) -> str:
        """参照文字列を生成（例: 民法第709条第1項）"""
        ref = f"{self.law_name}"
        if self.article:
            ref += f"第{self.article}条"
        if self.paragraph:
            ref += f"第{self.paragraph}項"
        if self.item:
            ref += f"第{self.item}号"
        return ref
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "citation_id": self.citation_id,
            "law_name": self.law_name,
            "article": self.article,
            "paragraph": self.paragraph,
            "item": self.item,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "relevance_score": self.relevance_score,
            "status": self.status.value,
            "source": self.source,
            "reference": self.to_reference_string()
        }


@dataclass
class ReasoningStep:
    """推論ステップの構造化表現"""
    step_id: int
    agent: str
    action: str  # search, interpret, critique, judge
    claim: str
    supporting_citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "action": self.action,
            "claim": self.claim,
            "supporting_citations": self.supporting_citations,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class CitationRegistry:
    """条文引用の中央管理"""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self.reasoning_chain: List[ReasoningStep] = []
        self._counter = 0
        self._step_counter = 0
    
    def register(self, 
                 law_name: str,
                 article: str,
                 text: str,
                 paragraph: Optional[str] = None,
                 item: Optional[str] = None,
                 relevance_score: float = 0.0,
                 source: str = "local") -> str:
        """
        条文を登録してIDを返す
        
        Args:
            law_name: 法令名
            article: 条番号
            text: 条文テキスト
            paragraph: 項番号
            item: 号番号
            relevance_score: 関連度スコア
            source: データソース
        
        Returns:
            citation_id
        """
        # 重複チェック
        for cid, citation in self.citations.items():
            if (citation.law_name == law_name and 
                citation.article == article and
                citation.paragraph == paragraph):
                return cid
        
        self._counter += 1
        citation_id = f"C{self._counter:03d}"
        
        citation = Citation(
            citation_id=citation_id,
            law_name=law_name,
            article=article,
            paragraph=paragraph,
            item=item,
            text=text,
            relevance_score=relevance_score,
            source=source
        )
        
        self.citations[citation_id] = citation
        logger.debug(f"Registered citation: {citation_id} - {citation.to_reference_string()}")
        
        return citation_id
    
    def register_from_document(self, doc: Any) -> str:
        """
        Documentオブジェクトから条文を登録
        
        Args:
            doc: Document オブジェクト
        
        Returns:
            citation_id
        """
        meta = doc.metadata
        return self.register(
            law_name=meta.get("law_title", "不明"),
            article=meta.get("article", ""),
            paragraph=meta.get("paragraph"),
            item=meta.get("item"),
            text=doc.page_content,
            relevance_score=getattr(doc, "score", 0.0),
            source=meta.get("source", "local")
        )
    
    def get(self, citation_id: str) -> Optional[Citation]:
        """IDから条文を取得"""
        return self.citations.get(citation_id)
    
    def update_status(self, citation_id: str, status: CitationStatus) -> bool:
        """引用のステータスを更新"""
        if citation_id in self.citations:
            self.citations[citation_id].status = status
            return True
        return False
    
    def add_reasoning_step(self,
                           agent: str,
                           action: str,
                           claim: str,
                           supporting_citations: List[str] = None,
                           confidence: float = 0.0,
                           metadata: Dict[str, Any] = None) -> int:
        """
        推論ステップを追加
        
        Args:
            agent: エージェント名
            action: アクション（search, interpret, critique, judge）
            claim: 主張
            supporting_citations: 根拠となる引用ID
            confidence: 確信度
            metadata: 追加メタデータ
        
        Returns:
            step_id
        """
        self._step_counter += 1
        step = ReasoningStep(
            step_id=self._step_counter,
            agent=agent,
            action=action,
            claim=claim,
            supporting_citations=supporting_citations or [],
            confidence=confidence,
            metadata=metadata or {}
        )
        self.reasoning_chain.append(step)
        return self._step_counter
    
    def get_citations_for_choice(self, choice: str) -> List[Citation]:
        """特定の選択肢を支持する引用を取得"""
        result = []
        for step in self.reasoning_chain:
            if choice.lower() in step.claim.lower():
                for cid in step.supporting_citations:
                    citation = self.get(cid)
                    if citation and citation not in result:
                        result.append(citation)
        return result
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """監査用サマリーを生成"""
        status_counts = {}
        for citation in self.citations.values():
            status = citation.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_citations": len(self.citations),
            "total_reasoning_steps": len(self.reasoning_chain),
            "status_distribution": status_counts,
            "citations": [c.to_dict() for c in self.citations.values()],
            "reasoning_chain": [s.to_dict() for s in self.reasoning_chain]
        }
    
    def get_citation_coverage(self) -> Dict[str, List[str]]:
        """選択肢ごとの引用カバレッジを計算"""
        coverage = {"a": [], "b": [], "c": [], "d": []}
        
        for step in self.reasoning_chain:
            claim_lower = step.claim.lower()
            for choice in coverage.keys():
                if f"選択肢{choice}" in claim_lower or f"{choice})" in claim_lower:
                    for cid in step.supporting_citations:
                        if cid not in coverage[choice]:
                            coverage[choice].append(cid)
        
        return coverage
    
    def calculate_citation_rate(self) -> float:
        """条文引用率を計算"""
        if not self.reasoning_chain:
            return 0.0
        
        steps_with_citations = sum(
            1 for step in self.reasoning_chain 
            if step.supporting_citations
        )
        return steps_with_citations / len(self.reasoning_chain)
    
    def clear(self):
        """レジストリをクリア"""
        self.citations.clear()
        self.reasoning_chain.clear()
        self._counter = 0
        self._step_counter = 0

