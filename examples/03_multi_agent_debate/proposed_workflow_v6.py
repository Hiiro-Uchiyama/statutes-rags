#!/usr/bin/env python3
"""
提案手法 v6: CLMR (Citation-grounded Legal Multi-agent Reasoning)

段階的実装:
- v6.1: Citation-Grounded推論（引用必須）
- v6.2: + 階層的参照追跡
- v6.3: + 対照的選択肢検証

新規性:
1. 法令QA特化のマルチエージェントフレームワーク
2. Citation-Grounded推論（全判断に条文引用を必須化）
3. 階層的参照追跡（法律→施行令→規則）
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langchain_community.llms import Ollama

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

from app.utils.number_normalizer import normalize_article_numbers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """ワークフロー設定"""
    llm_model: str = "qwen3:8b"
    timeout: int = 180
    num_ctx: int = 16000
    top_k: int = 30
    # v6機能フラグ
    use_citation_grounded: bool = True      # v6.1
    use_reference_tracking: bool = False    # v6.2
    use_contrastive_verify: bool = False    # v6.3


class ProposedWorkflowV6:
    """
    CLMR: Citation-grounded Legal Multi-agent Reasoning
    
    v6.1: Citation-Grounded推論
    - 全ての判断に条文引用を必須化
    - 引用なしの判断は信頼性低として扱う
    """
    
    def __init__(
        self,
        retriever,
        config: WorkflowConfig = None
    ):
        self.retriever = retriever
        self.config = config or WorkflowConfig()
        
        # LLM
        self.llm = Ollama(
            model=self.config.llm_model,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx
        )
    
    def query(
        self,
        question: str,
        choices: List[str]
    ) -> Dict[str, Any]:
        """質問に回答"""
        # クエリ正規化
        full_query = normalize_article_numbers(question + " " + " ".join(choices))
        
        # Step 1: 検索
        docs = self.retriever.retrieve(full_query, top_k=self.config.top_k)
        context = self._build_context(docs)
        
        # Step 2: 質問タイプ判定
        question_type = self._detect_question_type(question)
        
        # Step 3: Citation-Grounded分析（RetrieverAgent）
        analysis = self._citation_grounded_analysis(
            question, choices, context, question_type
        )
        
        # Step 4: 最終判断（IntegratorAgent）
        answer, citations = self._integrate_with_citations(
            question, choices, context, analysis, question_type
        )
        
        return {
            'answer': answer,
            'method': 'v6_clmr',
            'citations': citations,
            'question_type': question_type
        }
    
    def _build_context(self, docs: List[Any]) -> str:
        """ドキュメントからコンテキストを構築"""
        texts = []
        for doc in docs[:20]:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif isinstance(doc, dict):
                texts.append(doc.get('text', doc.get('content', str(doc))))
            else:
                texts.append(str(doc))
        return "\n\n".join(texts)
    
    def _detect_question_type(self, question: str) -> str:
        """質問タイプを検出"""
        if '誤っている' in question or '誤り' in question:
            return '誤り選択'
        elif '正しい' in question:
            return '正しい選択'
        elif '組み合わせ' in question or '組合せ' in question:
            return '組み合わせ'
        else:
            return '単純選択'
    
    def _citation_grounded_analysis(
        self,
        question: str,
        choices: List[str],
        context: str,
        question_type: str
    ) -> Dict[str, Any]:
        """
        Citation-Grounded分析
        
        各選択肢の判断に必ず条文引用を要求
        """
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        
        prompt = f"""あなたは法令QAの分析エージェントです。
各選択肢について、**必ず根拠となる条文を引用して**判断してください。

【検索された条文】
{context[:5000]}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【分析ルール】
1. 各選択肢について、関連する条文を**具体的に引用**してください
2. 引用形式: 「第X条第Y項: 〜」
3. 条文を引用できない場合は「引用なし（根拠不明）」と記載
4. 引用に基づいて一致/不一致を判断

【各選択肢の分析】（以下の形式で記載）

選択肢a:
  引用条文: [具体的な条文を引用]
  判断: [一致/不一致/不明]
  理由: [引用条文に基づく理由]

選択肢b:
  引用条文: [具体的な条文を引用]
  判断: [一致/不一致/不明]
  理由: [引用条文に基づく理由]

選択肢c:
  引用条文: [具体的な条文を引用]
  判断: [一致/不一致/不明]
  理由: [引用条文に基づく理由]

選択肢d:
  引用条文: [具体的な条文を引用]
  判断: [一致/不一致/不明]
  理由: [引用条文に基づく理由]

【暫定回答】: [a/b/c/d]

分析:"""
        
        response = self.llm.invoke(prompt)
        
        # 引用を抽出
        citations = self._extract_citations(response)
        
        return {
            'analysis': response,
            'question_type': question_type,
            'citations': citations
        }
    
    def _extract_citations(self, text: str) -> List[str]:
        """分析結果から条文引用を抽出"""
        # 「第X条」パターンを抽出
        pattern = r'第[一二三四五六七八九十百\d]+条(第[一二三四五六七八九十\d]+項)?(第[一二三四五六七八九十\d]+号)?'
        citations = re.findall(pattern, text)
        
        # 完全な引用を再構築
        full_citations = []
        for match in re.finditer(pattern, text):
            full_citations.append(match.group(0))
        
        return list(set(full_citations))
    
    def _integrate_with_citations(
        self,
        question: str,
        choices: List[str],
        context: str,
        analysis: Dict[str, Any],
        question_type: str
    ) -> Tuple[str, List[str]]:
        """
        Citation-Groundedな最終判断
        
        引用の妥当性を確認し、最終回答を決定
        """
        choices_text = "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(choices)])
        citations = analysis.get('citations', [])
        
        prompt = f"""あなたは法令QAの最終判断エージェントです。

【分析結果】
{analysis.get('analysis', '')[:3000]}

【引用された条文】
{', '.join(citations) if citations else '引用なし'}

【問題】
{question}

【選択肢】
{choices_text}

【質問タイプ】: {question_type}

【最終判断ルール】
1. 分析で引用された条文の妥当性を確認
2. 質問タイプ「{question_type}」に応じた回答を選択
3. 引用根拠が明確な判断を優先

【最終判断】
引用根拠に基づき、最も適切な選択肢を選んでください。
回答は a, b, c, d のいずれか1文字のみで最後に答えてください。

最終回答:"""
        
        response = self.llm.invoke(prompt)
        answer = self._extract_answer(response)
        
        return answer, citations
    
    def _extract_answer(self, response: str) -> str:
        """LLM応答から回答を抽出"""
        response = response.strip().lower()
        
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().lower()
            match = re.search(r'(?:回答|答え|answer)[:\s]*([abcd])', line)
            if match:
                return match.group(1)
            if line in ['a', 'b', 'c', 'd']:
                return line
            match = re.search(r'^[\(\[]?([abcd])[\)\]\.。]?$', line)
            if match:
                return match.group(1)
        
        matches = re.findall(r'\b([abcd])\b', response)
        if matches:
            return matches[-1]
        
        return 'a'


# ============================================================
# v6.2: 階層的参照追跡（後で追加）
# ============================================================

class ReferenceTracker:
    """
    階層的参照追跡
    
    「政令で定める」等のパターンを検出し、
    関連する施行令・規則を追加検索
    """
    
    # 参照パターン
    REFERENCE_PATTERNS = [
        (r'政令で定める', '施行令'),
        (r'内閣府令で定める', '施行規則'),
        (r'省令で定める', '施行規則'),
    ]
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def detect_references(self, text: str) -> List[Dict[str, str]]:
        """参照パターンを検出"""
        references = []
        for pattern, target_type in self.REFERENCE_PATTERNS:
            if re.search(pattern, text):
                references.append({
                    'pattern': pattern,
                    'target_type': target_type
                })
        return references
    
    def expand_context(
        self,
        original_context: str,
        question: str,
        law_name: str = ""
    ) -> str:
        """参照先を追加検索してコンテキストを拡張"""
        references = self.detect_references(question + original_context)
        
        if not references:
            return original_context
        
        # 参照先を検索
        additional_context = []
        for ref in references:
            target = ref['target_type']
            if law_name:
                query = f"{law_name}{target}"
            else:
                query = target
            
            # 追加検索（少数のみ）
            docs = self.retriever.retrieve(query, top_k=5)
            for doc in docs[:3]:
                if hasattr(doc, 'page_content'):
                    additional_context.append(doc.page_content)
        
        if additional_context:
            return original_context + "\n\n【参照先条文】\n" + "\n\n".join(additional_context[:3])
        
        return original_context


# ============================================================
# v6.3: 対照的選択肢検証（後で追加）
# ============================================================

class ContrastiveVerifier:
    """
    対照的選択肢検証
    
    選択肢間の差分を抽出し、差分に焦点を当てた検証を行う
    """
    
    def extract_choice_diffs(self, choices: List[str]) -> List[Dict[str, Any]]:
        """選択肢間の差分を抽出"""
        diffs = []
        
        for i, choice_a in enumerate(choices):
            for j, choice_b in enumerate(choices):
                if i >= j:
                    continue
                
                # 単語レベルでの差分
                words_a = set(choice_a.split())
                words_b = set(choice_b.split())
                
                only_a = words_a - words_b
                only_b = words_b - words_a
                
                if only_a or only_b:
                    diffs.append({
                        'choice_a': chr(97 + i),
                        'choice_b': chr(97 + j),
                        'diff_a': list(only_a)[:5],
                        'diff_b': list(only_b)[:5]
                    })
        
        return diffs
    
    def generate_focused_queries(
        self,
        question: str,
        diffs: List[Dict[str, Any]]
    ) -> List[str]:
        """差分に基づいた検索クエリを生成"""
        queries = [question]
        
        for diff in diffs[:3]:
            # 差分キーワードを含むクエリ
            keywords = diff['diff_a'] + diff['diff_b']
            if keywords:
                query = question + " " + " ".join(keywords[:3])
                queries.append(query)
        
        return queries

