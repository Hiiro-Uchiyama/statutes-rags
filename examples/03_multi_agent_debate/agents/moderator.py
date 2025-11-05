"""
Moderator Agent

議論を調整し、合意判定と最終回答を生成するエージェント。
"""
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.shared.base_agent import BaseAgent


class ModeratorAgent(BaseAgent):
    """
    モデレーターエージェント
    
    責務:
    - 両者の議論を評価
    - 合意判定（類似度計算）
    - 最終回答の生成
    - 議論の強制終了判断
    """
    
    def __init__(self, llm, config):
        """
        Args:
            llm: LLMインスタンス
            config: 設定オブジェクト
        """
        super().__init__(llm, config)
        
        # 埋め込みモデルの初期化（合意判定用）
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """埋め込みモデルを初期化"""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        try:
            model_name = self.config.embedding_model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.logger.info(f"Embedding model initialized: {model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize embedding model: {e}")
            self.embeddings = None
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        議論を評価し、必要に応じて最終回答を生成
        
        Args:
            input_data: {
                "query": str,
                "documents": List[Document],
                "debater_a_position": Dict,
                "debater_b_position": Dict,
                "round": int
            }
        
        Returns:
            {
                "agreement_score": float,
                "should_continue": bool,
                "final_answer": str (終了時のみ),
                "moderator_comment": str
            }
        """
        query = input_data.get("query", "")
        documents = input_data.get("documents", [])
        position_a = input_data.get("debater_a_position", {})
        position_b = input_data.get("debater_b_position", {})
        round_num = input_data.get("round", 1)
        max_rounds = input_data.get("max_rounds", 3)
        
        # 合意スコアを計算
        agreement_score = self._calculate_agreement(
            position_a.get("position", ""),
            position_b.get("position", "")
        )
        
        # 継続判定
        should_continue = self._should_continue_debate(
            agreement_score,
            round_num,
            max_rounds
        )
        
        result = {
            "agreement_score": agreement_score,
            "should_continue": should_continue,
        }
        
        # 議論終了時は最終回答を生成
        if not should_continue:
            final_answer = self._generate_final_answer(
                query,
                documents,
                position_a,
                position_b,
                agreement_score
            )
            result["final_answer"] = final_answer
        else:
            result["moderator_comment"] = self._generate_moderator_comment(
                position_a,
                position_b,
                agreement_score,
                round_num
            )
        
        return result
    
    def _calculate_agreement(
        self,
        position_a: str,
        position_b: str
    ) -> float:
        """
        両者の主張の類似度を計算（合意スコア）
        
        Args:
            position_a: Debater Aの主張
            position_b: Debater Bの主張
        
        Returns:
            合意スコア（0.0〜1.0）
        """
        if not position_a or not position_b:
            return 0.0
        
        # 埋め込みモデルが利用可能な場合は類似度計算
        if self.embeddings:
            try:
                # 埋め込みベクトルを取得
                embed_a = self.embeddings.embed_query(position_a)
                embed_b = self.embeddings.embed_query(position_b)
                
                # コサイン類似度（正規化済みなので内積で計算可能）
                similarity = np.dot(embed_a, embed_b)
                
                # 0〜1の範囲に正規化
                return float((similarity + 1.0) / 2.0)
            except Exception as e:
                self.logger.warning(f"Failed to calculate embedding similarity: {e}")
        
        # フォールバック: 単純なキーワード一致率
        words_a = set(position_a.split())
        words_b = set(position_b.split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _should_continue_debate(
        self,
        agreement_score: float,
        current_round: int,
        max_rounds: int
    ) -> bool:
        """
        議論を継続すべきか判定
        
        Args:
            agreement_score: 合意スコア
            current_round: 現在のラウンド
            max_rounds: 最大ラウンド数
        
        Returns:
            継続する場合True
        """
        # 最大ラウンドに達した場合は終了
        if current_round >= max_rounds:
            return False
        
        # 合意閾値を超えた場合は終了
        threshold = self.config.agreement_threshold
        if agreement_score >= threshold:
            return False
        
        # それ以外は継続
        return True
    
    def _generate_moderator_comment(
        self,
        position_a: Dict[str, Any],
        position_b: Dict[str, Any],
        agreement_score: float,
        round_num: int
    ) -> str:
        """
        議論継続時のモデレーターコメントを生成
        
        Args:
            position_a: Debater Aの主張
            position_b: Debater Bの主張
            agreement_score: 合意スコア
            round_num: 現在のラウンド
        
        Returns:
            モデレーターコメント
        """
        prompt = f"""あなたは法律議論のモデレーターです。
2人の議論者の主張を聞き、簡潔なコメントを提供してください。

【Debater Aの主張】
{position_a.get("position", "")}

【Debater Bの主張】
{position_b.get("position", "")}

現在ラウンド{round_num}、合意スコア: {agreement_score:.2f}

両者の主張の要点と、まだ議論が必要な点を簡潔に指摘してください。
（2-3文程度）
"""
        
        response = self._safe_llm_invoke(prompt)
        return response if response else "議論を継続してください。"
    
    def _generate_final_answer(
        self,
        query: str,
        documents: List[Any],
        position_a: Dict[str, Any],
        position_b: Dict[str, Any],
        agreement_score: float
    ) -> str:
        """
        最終回答を生成
        
        Args:
            query: 元の質問
            documents: 検索された文書
            position_a: Debater Aの主張
            position_b: Debater Bの主張
            agreement_score: 合意スコア
        
        Returns:
            最終回答
        """
        context = self._format_documents(documents)
        
        prompt = f"""あなたは法律議論のモデレーターです。
2人の議論者の主張を聞き、最終的な回答を統合してください。

【元の質問】
{query}

【関連法令】
{context}

【Debater A（肯定的解釈）の主張】
{position_a.get("position", "")}

推論:
{position_a.get("reasoning", "")}

【Debater B（批判的解釈）の主張】
{position_b.get("position", "")}

推論:
{position_b.get("reasoning", "")}

【合意スコア】
{agreement_score:.2f}

両者の主張を総合的に考慮し、バランスの取れた最終回答を作成してください。
両者の主張で一致している点、異なる点を明確にし、法的根拠に基づいた結論を示してください。

回答は以下の構成で：
1. 結論（簡潔に）
2. 根拠（両者の主張を踏まえて）
3. 注意点（もしあれば）
"""
        
        response = self._safe_llm_invoke(prompt)
        return response if response else "最終回答の生成に失敗しました。"

