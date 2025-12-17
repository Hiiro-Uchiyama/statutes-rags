"""
ChoiceVerifierAgent - 選択肢ごとの条文検証エージェント

各選択肢を個別に条文と照合し、一致/不一致の判定と根拠を出力する。
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)


@dataclass
class ChoiceVerification:
    """選択肢の検証結果"""
    choice_id: str  # a, b, c, d
    choice_text: str
    matched_article: str  # 対応する条文
    judgment: str  # "一致", "不一致", "不明"
    key_values: Dict[str, str]  # 抽出した数値・条件
    reasoning: str  # 判定理由
    confidence: float  # 確信度 0-1


class ChoiceVerifierAgent:
    """選択肢ごとに条文と照合するエージェント"""
    
    def __init__(
        self,
        llm_model: str = "qwen3:8b",
        timeout: int = 120,
        num_ctx: int = 16000
    ):
        self.llm = Ollama(model=llm_model, timeout=timeout, num_ctx=num_ctx)
        self.llm_model = llm_model
    
    def verify_choice(
        self,
        choice_id: str,
        choice_text: str,
        context: str,
        question_type: str  # "correct" or "incorrect"
    ) -> ChoiceVerification:
        """単一の選択肢を検証"""
        
        prompt = f"""あなたは法令の専門家です。以下の選択肢が法令条文と一致するか厳密に検証してください。

【検証対象の選択肢】
{choice_id}. {choice_text}

【参考法令条文】
{context}

【検証手順】
1. 選択肢に含まれる具体的な数値・期間・条件を抽出
2. 条文中の対応する数値・期間・条件を特定
3. 両者を比較して一致/不一致を判定

【出力形式】
対応条文: [該当する条文名と条項]
選択肢の主張: [選択肢が主張している内容]
条文の規定: [条文が規定している内容]
抽出した数値:
  - 選択肢: [数値/期間/条件]
  - 条文: [数値/期間/条件]
判定: [一致/不一致/不明]
判定理由: [なぜそう判断したか]
確信度: [0.0-1.0]

/no_think"""

        try:
            response = self.llm.invoke(prompt)
            return self._parse_verification(choice_id, choice_text, response)
        except Exception as e:
            logger.error(f"Choice verification failed for {choice_id}: {e}")
            return ChoiceVerification(
                choice_id=choice_id,
                choice_text=choice_text,
                matched_article="不明",
                judgment="不明",
                key_values={},
                reasoning=f"検証エラー: {str(e)}",
                confidence=0.0
            )
    
    def _parse_verification(
        self,
        choice_id: str,
        choice_text: str,
        response: str
    ) -> ChoiceVerification:
        """LLMの応答をパース"""
        lines = response.strip().split("\n")
        
        matched_article = ""
        judgment = "不明"
        key_values = {}
        reasoning = ""
        confidence = 0.5
        
        for line in lines:
            line = line.strip()
            if line.startswith("対応条文:"):
                matched_article = line.replace("対応条文:", "").strip()
            elif line.startswith("判定:"):
                j = line.replace("判定:", "").strip()
                if "一致" in j and "不一致" not in j:
                    judgment = "一致"
                elif "不一致" in j:
                    judgment = "不一致"
                else:
                    judgment = "不明"
            elif line.startswith("判定理由:"):
                reasoning = line.replace("判定理由:", "").strip()
            elif line.startswith("確信度:"):
                try:
                    conf_str = line.replace("確信度:", "").strip()
                    confidence = float(conf_str.replace("[", "").replace("]", ""))
                except:
                    confidence = 0.5
            elif "選択肢:" in line:
                key_values["選択肢"] = line.split("選択肢:")[-1].strip()
            elif "条文:" in line and "対応条文" not in line:
                key_values["条文"] = line.split("条文:")[-1].strip()
        
        return ChoiceVerification(
            choice_id=choice_id,
            choice_text=choice_text,
            matched_article=matched_article,
            judgment=judgment,
            key_values=key_values,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def verify_all_choices(
        self,
        question: str,
        choices: List[str],
        context: str
    ) -> Dict[str, Any]:
        """全選択肢を検証"""
        
        # 質問タイプ判定
        if "誤っている" in question or "誤り" in question:
            question_type = "incorrect"
        elif "正しい" in question:
            question_type = "correct"
        else:
            question_type = "unknown"
        
        verifications = []
        choice_ids = ['a', 'b', 'c', 'd']
        
        for i, choice_text in enumerate(choices):
            if i >= len(choice_ids):
                break
            
            choice_id = choice_ids[i]
            logger.info(f"Verifying choice {choice_id}...")
            
            verification = self.verify_choice(
                choice_id=choice_id,
                choice_text=choice_text,
                context=context,
                question_type=question_type
            )
            verifications.append(verification)
        
        return {
            "question_type": question_type,
            "verifications": verifications,
            "summary": self._create_summary(verifications, question_type)
        }
    
    def _create_summary(
        self,
        verifications: List[ChoiceVerification],
        question_type: str
    ) -> Dict[str, Any]:
        """検証結果のサマリを作成"""
        
        matched = [v for v in verifications if v.judgment == "一致"]
        unmatched = [v for v in verifications if v.judgment == "不一致"]
        unknown = [v for v in verifications if v.judgment == "不明"]
        
        # 暫定回答の決定
        if question_type == "incorrect":
            # 「誤っているもの」を選ぶ場合、不一致を探す
            if unmatched:
                # 確信度が最も高い不一致を選択
                best = max(unmatched, key=lambda v: v.confidence)
                provisional_answer = best.choice_id
                answer_confidence = best.confidence
            else:
                provisional_answer = "不明"
                answer_confidence = 0.0
        elif question_type == "correct":
            # 「正しいもの」を選ぶ場合、一致を探す
            if matched:
                best = max(matched, key=lambda v: v.confidence)
                provisional_answer = best.choice_id
                answer_confidence = best.confidence
            else:
                provisional_answer = "不明"
                answer_confidence = 0.0
        else:
            # 不明な場合、一致度が最も高いものを選択
            if matched:
                best = max(matched, key=lambda v: v.confidence)
                provisional_answer = best.choice_id
                answer_confidence = best.confidence
            else:
                provisional_answer = "不明"
                answer_confidence = 0.0
        
        return {
            "matched_choices": [v.choice_id for v in matched],
            "unmatched_choices": [v.choice_id for v in unmatched],
            "unknown_choices": [v.choice_id for v in unknown],
            "provisional_answer": provisional_answer,
            "answer_confidence": answer_confidence,
            "needs_recheck": answer_confidence < 0.7 or len(unknown) > 0
        }

