"""
Retrieval Agent

動的に検索戦略を選択し、高品質な検索結果を提供するエージェント。
"""
import re
from typing import Dict, Any, List
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class RetrievalAgent(BaseAgent):
    """
    検索エージェント
    
    責務:
    - 検索戦略の動的選択
    - 検索実行
    - 検索結果の品質評価
    """
    
    def __init__(self, llm, config, retrievers: Dict[str, Any]):
        """
        Args:
            llm: LLMインスタンス
            config: 設定オブジェクト
            retrievers: {"vector": ..., "bm25": ..., "hybrid": ...}
        """
        super().__init__(llm, config)
        self.retrievers = retrievers
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        検索を実行
        
        Args:
            input_data: {
                "query": str,
                "query_type": str,
                "complexity": str (optional)
            }
        
        Returns:
            {
                "documents": List[Document],
                "strategy": str,
                "quality": Dict[str, Any]
            }
        """
        query = input_data.get("query", "")
        query_type = input_data.get("query_type", "interpretation")
        
        # 戦略選択
        strategy = self.select_strategy(query, query_type)
        
        # 検索実行
        top_k = getattr(self.config, "retrieval_top_k", 10)
        documents = self.retrieve(query, strategy, top_k)
        
        # 品質評価
        quality = self.evaluate_quality(query, documents)
        
        return {
            "documents": documents,
            "strategy": strategy,
            "quality": quality
        }
    
    def select_strategy(self, query: str, query_type: str) -> str:
        """
        クエリに応じた検索戦略を選択
        
        Args:
            query: 質問文
            query_type: クエリタイプ
        
        Returns:
            "vector" | "bm25" | "hybrid"
        """
        # ルールベース判定
        
        # ルール1: 具体的な条文番号を含む場合はBM25
        if re.search(r'第\d+条', query) or re.search(r'第\d+項', query):
            self.logger.info(f"Strategy: bm25 (contains article number)")
            return "bm25"
        
        # ルール2: 法令名が明示されている場合はBM25
        if re.search(r'[^。、]+法第', query):
            self.logger.info(f"Strategy: bm25 (contains law name)")
            return "bm25"
        
        # ルール3: 抽象的な概念の場合はvector
        if query_type == "interpretation":
            abstract_keywords = ['意味', '解釈', '趣旨', '目的', '理由']
            if any(kw in query for kw in abstract_keywords):
                self.logger.info(f"Strategy: vector (interpretation query)")
                return "vector"
        
        # デフォルトはhybrid
        self.logger.info(f"Strategy: hybrid (default)")
        return "hybrid"
    
    def retrieve(self, query: str, strategy: str, top_k: int = 10) -> List[Any]:
        """
        検索を実行
        
        Args:
            query: 質問文
            strategy: 検索戦略
            top_k: 取得する文書数
        
        Returns:
            Document のリスト
        """
        if strategy not in self.retrievers:
            self.logger.warning(f"Strategy {strategy} not found, using hybrid")
            strategy = "hybrid"
        
        retriever = self.retrievers[strategy]
        
        try:
            documents = retriever.retrieve(query, top_k=top_k)
            self.logger.info(f"Retrieved {len(documents)} documents using {strategy}")
            return documents
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}", exc_info=True)
            return []
    
    def evaluate_quality(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        """
        検索結果の品質を評価
        
        Args:
            query: 質問文
            documents: 検索結果
        
        Returns:
            {
                "score": float (0-1),
                "is_sufficient": bool,
                "feedback": str
            }
        """
        if not documents:
            return {
                "score": 0.0,
                "is_sufficient": False,
                "feedback": "No documents retrieved"
            }
        
        # スコアベースの評価（scoreがない場合は0.5をデフォルトとする）
        scores = [getattr(doc, 'score', 0.5) for doc in documents]
        
        # スコアを0-1の範囲に正規化
        # BM25などのスコアは大きな値になる可能性があるため、正規化が必要
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score > min_score:
                # Min-Max正規化
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                # すべて同じスコアの場合は0.5とする
                normalized_scores = [0.5] * len(scores)
            
            avg_score = sum(normalized_scores) / len(normalized_scores)
        else:
            avg_score = 0.0
        
        # 浮動小数点の精度問題を避けるため、スコアを丸める
        avg_score = round(avg_score, 6)
        
        # 閾値判定
        threshold = 0.5
        is_sufficient = avg_score >= threshold and len(documents) >= 3
        
        feedback = "Good quality results" if is_sufficient else "Low relevance scores"
        
        self.logger.info(f"Quality: score={avg_score:.3f}, sufficient={is_sufficient}")
        
        return {
            "score": float(avg_score),
            "is_sufficient": is_sufficient,
            "feedback": feedback
        }

