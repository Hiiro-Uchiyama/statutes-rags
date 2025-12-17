"""
IntegratorAgent - 検証結果統合エージェント

ChoiceVerifierAgentの結果を統合し、法令CoTを生成する。
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """統合結果"""
    provisional_answer: str
    legal_cot: str  # 法令Chain-of-Thought
    confidence: float
    uncertain_points: List[str]  # 不確実な点
    recheck_queries: List[str]  # 再確認用クエリ


class IntegratorAgent:
    """検証結果を統合するエージェント"""
    
    def __init__(
        self,
        llm_model: str = "qwen3:8b",
        timeout: int = 120,
        num_ctx: int = 16000
    ):
        self.llm = Ollama(model=llm_model, timeout=timeout, num_ctx=num_ctx)
        self.llm_model = llm_model
    
    def integrate(
        self,
        question: str,
        choices: List[str],
        verification_result: Dict[str, Any]
    ) -> IntegrationResult:
        """検証結果を統合"""
        
        # 検証結果をフォーマット
        verifications = verification_result.get("verifications", [])
        question_type = verification_result.get("question_type", "unknown")
        summary = verification_result.get("summary", {})
        
        verification_text = self._format_verifications(verifications)
        
        type_instruction = ""
        if question_type == "incorrect":
            type_instruction = "この問題は「誤っているもの」を選ぶ問題です。条文と不一致の選択肢を探してください。"
        elif question_type == "correct":
            type_instruction = "この問題は「正しいもの」を選ぶ問題です。条文と一致する選択肢を探してください。"
        
        prompt = f"""あなたは法令の専門家です。以下の選択肢検証結果を統合し、法令CoT（推論チェーン）を生成してください。

【質問】
{question}

{type_instruction}

【選択肢】
{self._format_choices(choices)}

【各選択肢の検証結果】
{verification_text}

【タスク】
1. 各選択肢の検証結果を整理
2. 法令CoT（推論チェーン）を作成
3. 暫定回答を決定
4. 不確実な点があれば指摘

【出力形式】
法令CoT:
[検証結果に基づく推論の流れ。各選択肢の判定根拠を含める]

暫定回答: [a/b/c/d]
確信度: [0.0-1.0]

不確実な点:
- [あれば記載、なければ「なし」]

再確認が必要な条文:
- [あれば記載、なければ「なし」]

/no_think"""

        try:
            response = self.llm.invoke(prompt)
            return self._parse_integration(response)
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return IntegrationResult(
                provisional_answer=summary.get("provisional_answer", "a"),
                legal_cot=f"統合エラー: {str(e)}",
                confidence=0.0,
                uncertain_points=[],
                recheck_queries=[]
            )
    
    def _format_verifications(self, verifications: List) -> str:
        """検証結果をフォーマット"""
        parts = []
        for v in verifications:
            key_vals = v.key_values if hasattr(v, 'key_values') else {}
            parts.append(f"""
選択肢{v.choice_id}: {v.choice_text[:50]}...
  - 対応条文: {v.matched_article}
  - 判定: {v.judgment}
  - 抽出値: {key_vals}
  - 理由: {v.reasoning}
  - 確信度: {v.confidence}
""")
        return "\n".join(parts)
    
    def _format_choices(self, choices: List[str]) -> str:
        """選択肢をフォーマット"""
        choice_ids = ['a', 'b', 'c', 'd']
        return "\n".join([f"{choice_ids[i]}. {c}" for i, c in enumerate(choices) if i < len(choice_ids)])
    
    def _parse_integration(self, response: str) -> IntegrationResult:
        """LLMの応答をパース"""
        lines = response.strip().split("\n")
        
        legal_cot = ""
        provisional_answer = "a"
        confidence = 0.5
        uncertain_points = []
        recheck_queries = []
        
        in_cot = False
        in_uncertain = False
        in_recheck = False
        cot_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith("法令CoT:"):
                in_cot = True
                in_uncertain = False
                in_recheck = False
                continue
            elif line_stripped.startswith("暫定回答:"):
                in_cot = False
                ans = line_stripped.replace("暫定回答:", "").strip().lower()
                for c in ['a', 'b', 'c', 'd']:
                    if c in ans:
                        provisional_answer = c
                        break
            elif line_stripped.startswith("確信度:"):
                try:
                    conf_str = line_stripped.replace("確信度:", "").strip()
                    confidence = float(conf_str.replace("[", "").replace("]", ""))
                except:
                    confidence = 0.5
            elif line_stripped.startswith("不確実な点:"):
                in_cot = False
                in_uncertain = True
                in_recheck = False
            elif line_stripped.startswith("再確認が必要な条文:"):
                in_cot = False
                in_uncertain = False
                in_recheck = True
            elif in_cot:
                cot_lines.append(line)
            elif in_uncertain and line_stripped.startswith("-"):
                point = line_stripped[1:].strip()
                if point and point != "なし":
                    uncertain_points.append(point)
            elif in_recheck and line_stripped.startswith("-"):
                query = line_stripped[1:].strip()
                if query and query != "なし":
                    recheck_queries.append(query)
        
        legal_cot = "\n".join(cot_lines).strip()
        
        return IntegrationResult(
            provisional_answer=provisional_answer,
            legal_cot=legal_cot,
            confidence=confidence,
            uncertain_points=uncertain_points,
            recheck_queries=recheck_queries
        )

