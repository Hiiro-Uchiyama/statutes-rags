"""
Legal Case Generator Pipeline

LangGraphを使用した事例生成パイプライン。
"""
import os
import logging
import argparse
import json
import re
from typing import Dict, Any, List, TypedDict, Optional
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# 数字で始まるモジュール名のため、importlibを使用
import importlib
config_module = importlib.import_module('examples.04_legal_case_generator.config')
agents_module = importlib.import_module('examples.04_legal_case_generator.agents')

LegalCaseConfig = config_module.LegalCaseConfig
load_config = config_module.load_config
ScenarioGeneratorAgent = agents_module.ScenarioGeneratorAgent
LegalCheckerAgent = agents_module.LegalCheckerAgent
RefinerAgent = agents_module.RefinerAgent
MCQParserAgent = agents_module.MCQParserAgent
MCQCaseGeneratorAgent = agents_module.MCQCaseGeneratorAgent
MCQConsistencyCheckerAgent = agents_module.MCQConsistencyCheckerAgent
MCQRefinerAgent = agents_module.MCQRefinerAgent

logger = logging.getLogger(__name__)


class LegalCaseState(TypedDict):
    """LangGraphの状態定義"""
    # 入力情報
    law_number: str
    law_title: str
    article: str
    article_content: str
    case_type: str  # "applicable" | "non_applicable" | "boundary"
    
    # 生成結果
    scenario: str
    legal_analysis: str
    educational_point: str
    
    # 検証結果
    is_valid: bool
    validation_score: float
    feedback: List[str]
    
    # メタデータ
    iteration: int
    max_iterations: int
    agents_used: List[str]
    
    # エラーハンドリング
    error: Optional[str]


class MCQCaseState(TypedDict):
    """4択問題用事例生成の状態定義"""
    question_id: str
    question: str
    context: str
    choices: Dict[str, str]
    correct_choice: str
    parser_result: Dict[str, Any]

    scenario: str
    character_count: int

    is_valid: bool
    validation_score: float
    feedback: List[str]

    iteration: int
    max_iterations: int
    agents_used: List[str]

    error: Optional[str]


class LegalCaseGenerator:
    """法令適用事例生成システム"""
    
    def __init__(self, config: LegalCaseConfig):
        """
        Args:
            config: Legal Case Generator設定
        """
        self.config = config
        
        # LLMの初期化
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = Ollama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=ollama_host,
            timeout=config.llm_timeout
        )
        
        # エージェントの初期化
        self.scenario_gen = ScenarioGeneratorAgent(self.llm, config)
        self.legal_checker = LegalCheckerAgent(self.llm, config)
        self.refiner = RefinerAgent(self.llm, config)
        
        # グラフの構築
        self.graph = self._build_graph()
        
        logger.info("LegalCaseGenerator initialized")
    
    def _build_graph(self) -> Any:
        """LangGraphワークフローを構築"""
        workflow = StateGraph(LegalCaseState)
        
        # ノードの追加
        workflow.add_node("generate_scenario", self._generate_scenario_node)
        workflow.add_node("check_legal", self._check_legal_node)
        workflow.add_node("refine", self._refine_node)
        
        # エントリーポイント
        workflow.set_entry_point("generate_scenario")
        
        # シナリオ生成 → 法的検証
        workflow.add_edge("generate_scenario", "check_legal")
        
        # 法的検証 → 条件分岐
        workflow.add_conditional_edges(
            "check_legal",
            self._should_refine,
            {
                "refine": "refine",
                "end": END
            }
        )
        
        # 洗練 → 再検証または終了
        workflow.add_conditional_edges(
            "refine",
            self._should_continue,
            {
                "check": "check_legal",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _generate_scenario_node(self, state: LegalCaseState) -> LegalCaseState:
        """シナリオ生成ノード"""
        logger.info(f"Generating {state['case_type']} scenario")
        
        result = self.scenario_gen.execute({
            "article_content": state["article_content"],
            "case_type": state["case_type"],
            "law_title": state["law_title"],
            "article": state["article"]
        })
        
        state["scenario"] = result.get("scenario", "")
        state["legal_analysis"] = result.get("legal_analysis", "")
        state["educational_point"] = result.get("educational_point", "")
        state["agents_used"].append("scenario_generator")
        
        logger.info(f"Scenario generated: {len(state['scenario'])} chars")
        
        return state
    
    def _check_legal_node(self, state: LegalCaseState) -> LegalCaseState:
        """法的整合性検証ノード"""
        logger.info("Checking legal consistency")
        
        result = self.legal_checker.execute({
            "article_content": state["article_content"],
            "scenario": state["scenario"],
            "legal_analysis": state["legal_analysis"],
            "case_type": state["case_type"],
            "law_title": state["law_title"],
            "article": state["article"]
        })
        
        state["is_valid"] = result.get("is_valid", False)
        state["validation_score"] = result.get("validation_score", 0.0)
        state["feedback"] = result.get("feedback", [])
        state["agents_used"].append("legal_checker")
        
        logger.info(
            f"Validation: valid={state['is_valid']}, "
            f"score={state['validation_score']:.3f}"
        )
        
        return state
    
    def _refine_node(self, state: LegalCaseState) -> LegalCaseState:
        """事例洗練ノード"""
        logger.info(f"Refining scenario (iteration {state['iteration'] + 1})")
        
        result = self.refiner.execute({
            "scenario": state["scenario"],
            "legal_analysis": state["legal_analysis"],
            "educational_point": state["educational_point"],
            "feedback": state["feedback"],
            "article_content": state["article_content"],
            "case_type": state["case_type"],
            "law_title": state["law_title"],
            "article": state["article"]
        })
        
        state["scenario"] = result.get("scenario", state["scenario"])
        state["legal_analysis"] = result.get("legal_analysis", state["legal_analysis"])
        state["educational_point"] = result.get("educational_point", state["educational_point"])
        state["agents_used"].append("refiner")
        state["iteration"] += 1
        
        logger.info("Refinement completed")
        
        return state
    
    def _should_refine(self, state: LegalCaseState) -> str:
        """洗練が必要か判定"""
        # 既に有効な場合は終了
        if state["is_valid"]:
            logger.info("Validation passed, ending workflow")
            return "end"
        
        # 最大反復回数に達した場合は終了
        if state["iteration"] >= state["max_iterations"]:
            logger.info("Max iterations reached, ending workflow")
            return "end"
        
        # 洗練が必要
        logger.info("Validation failed, refinement needed")
        return "refine"
    
    def _should_continue(self, state: LegalCaseState) -> str:
        """反復を継続するか判定"""
        # 最大反復回数に達した場合は終了
        if state["iteration"] >= state["max_iterations"]:
            logger.info("Max iterations reached after refinement")
            return "end"
        
        # 再検証
        logger.info("Re-checking after refinement")
        return "check"
    
    def generate_case(
        self,
        law_number: str,
        law_title: str,
        article: str,
        article_content: str,
        case_type: str
    ) -> Dict[str, Any]:
        """
        単一の事例を生成
        
        Args:
            law_number: 法令番号
            law_title: 法令名
            article: 条文番号
            article_content: 条文内容
            case_type: 事例タイプ
        
        Returns:
            生成された事例
        """
        logger.info(f"Generating {case_type} case for {law_title} Article {article}")
        
        # 初期状態
        initial_state = {
            "law_number": law_number,
            "law_title": law_title,
            "article": article,
            "article_content": article_content,
            "case_type": case_type,
            "scenario": "",
            "legal_analysis": "",
            "educational_point": "",
            "is_valid": False,
            "validation_score": 0.0,
            "feedback": [],
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "agents_used": [],
            "error": None
        }
        
        try:
            # グラフ実行
            result = self.graph.invoke(initial_state)
            
            return {
                "law_number": result["law_number"],
                "law_title": result["law_title"],
                "article": result["article"],
                "case_type": result["case_type"],
                "scenario": result["scenario"],
                "legal_analysis": result["legal_analysis"],
                "educational_point": result["educational_point"],
                "is_valid": result["is_valid"],
                "validation_score": result["validation_score"],
                "iterations": result["iteration"],
                "agents_used": result["agents_used"]
            }
        except Exception as e:
            logger.error(f"Error generating case: {e}", exc_info=True)
            return {
                "law_number": law_number,
                "law_title": law_title,
                "article": article,
                "case_type": case_type,
                "scenario": "",
                "legal_analysis": "",
                "educational_point": "",
                "is_valid": False,
                "validation_score": 0.0,
                "iterations": 0,
                "agents_used": [],
                "error": str(e)
            }
    
    def generate_cases(
        self,
        law_number: str,
        law_title: str,
        article: str,
        article_content: str
    ) -> Dict[str, Any]:
        """
        全種類の事例を生成
        
        Args:
            law_number: 法令番号
            law_title: 法令名
            article: 条文番号
            article_content: 条文内容
        
        Returns:
            {
                "law_info": {...},
                "cases": [...]
            }
        """
        cases = []
        
        # 生成する事例タイプ
        case_types = []
        if self.config.generate_applicable:
            case_types.append("applicable")
        if self.config.generate_non_applicable:
            case_types.append("non_applicable")
        if self.config.generate_boundary:
            case_types.append("boundary")
        
        # 各タイプの事例を生成
        for case_type in case_types:
            case = self.generate_case(
                law_number=law_number,
                law_title=law_title,
                article=article,
                article_content=article_content,
                case_type=case_type
            )
            cases.append(case)
        
        return {
            "law_info": {
                "law_number": law_number,
                "law_title": law_title,
                "article": article,
                "article_content": article_content
            },
            "cases": cases
        }


class MCQCaseGenerator:
    """4択問題の正解選択肢に沿った事例を生成するパイプライン"""

    def __init__(self, config: LegalCaseConfig):
        self.config = config

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = Ollama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=ollama_host,
            timeout=config.llm_timeout
        )

        self.parser = MCQParserAgent(self.llm, config)
        self.case_generator = MCQCaseGeneratorAgent(self.llm, config)
        self.checker = MCQConsistencyCheckerAgent(self.llm, config)
        self.refiner = MCQRefinerAgent(self.llm, config)

        self.graph = self._build_graph()

    @staticmethod
    def _normalize_scenario_text(text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r'[（(]\s*\d+\s*文字\s*[)）]$', '', cleaned)
        cleaned = cleaned.rstrip()
        return cleaned

    def _build_graph(self):
        workflow = StateGraph(MCQCaseState)

        workflow.add_node("parse_mcq", self._parse_node)
        workflow.add_node("generate_case", self._generate_case_node)
        workflow.add_node("check_case", self._check_case_node)
        workflow.add_node("refine_case", self._refine_case_node)

        workflow.set_entry_point("parse_mcq")
        workflow.add_edge("parse_mcq", "generate_case")
        workflow.add_edge("generate_case", "check_case")

        workflow.add_conditional_edges(
            "check_case",
            self._should_refine,
            {
                "refine": "refine_case",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "refine_case",
            self._should_continue,
            {
                "check": "check_case",
                "end": END
            }
        )

        return workflow.compile()

    def _parse_node(self, state: MCQCaseState) -> MCQCaseState:
        self._ensure_feedback(state)
        result = self.parser.execute({
            "question": state["question"],
            "context": state["context"],
            "choices": state["choices"],
            "correct_choice": state["correct_choice"]
        })

        state["parser_result"] = result
        state["agents_used"].append("mcq_parser")
        return state

    def _generate_case_node(self, state: MCQCaseState) -> MCQCaseState:
        self._ensure_feedback(state, reset=True)
        result = self.case_generator.execute({
            "question": state["question"],
            "context": state["context"],
            "choices": state["choices"],
            "correct_choice": state["correct_choice"],
            "parser_result": state["parser_result"],
            "target_length": self.config.mcq_target_length,
            "min_length": self.config.mcq_min_length,
            "max_length": self.config.mcq_max_length
        })

        scenario = result.get("scenario", "")
        scenario = self._normalize_scenario_text(scenario)
        state["scenario"] = scenario
        state["character_count"] = len(scenario)
        state["agents_used"].append("mcq_case_generator")
        return state

    def _check_case_node(self, state: MCQCaseState) -> MCQCaseState:
        self._ensure_feedback(state, reset=True)

        checker_result = self.checker.execute({
            "question": state["question"],
            "context": state["context"],
            "choices": state["choices"],
            "correct_choice": state["correct_choice"],
            "scenario": state["scenario"],
            "parser_result": state["parser_result"]
        })

        state["agents_used"].append("mcq_checker")

        feedback: List[str] = checker_result.get("feedback", [])
        if not isinstance(feedback, list):
            feedback = [str(feedback)]
        state["feedback"] = [item for item in feedback if item]

        state["validation_score"] = float(checker_result.get("score", 0.0))
        is_valid = bool(checker_result.get("is_valid", False))

        char_count = len(state["scenario"])
        state["character_count"] = char_count

        if char_count < self.config.mcq_min_length or char_count > self.config.mcq_max_length:
            state["feedback"].append(
                f"文字数が指定範囲外です（{char_count}文字、許容範囲 {self.config.mcq_min_length}〜{self.config.mcq_max_length}）。"
            )
            is_valid = False

        prohibited_terms = ["選択肢", "正解", "回答"]
        violating_terms = [term for term in prohibited_terms if term in state["scenario"]]
        if violating_terms:
            joined_terms = "、".join(sorted(set(violating_terms)))
            state["feedback"].append(
                f"シナリオに問題形式を示す語（{joined_terms}）が含まれています。これらの語を削除し、事例の描写に集中してください。"
            )
            is_valid = False

        parser_result = state.get("parser_result", {}) or {}
        correct_choice_text = ""
        if isinstance(parser_result, dict):
            correct_choice_text = parser_result.get("correct_choice_text", "")
        if not correct_choice_text:
            correct_choice_text = state["choices"].get(state["correct_choice"], "")
        if correct_choice_text and correct_choice_text in state["scenario"]:
            state["feedback"].append(
                "参照用の結論文面を事例内で逐語的に引用しないでください。事実描写と分析で結論を示してください。"
            )
            is_valid = False

        if re.search(r"第[0-9一二三四五六七八九十百千万]+条", state["scenario"]):
            state["feedback"].append(
                "条文番号（例:「第〇条」）を直接記載しないでください。制度趣旨や要件を言い換えて説明してください。"
            )
            is_valid = False
        if re.search(r"第[0-9一二三四五六七八九十百千万]+項", state["scenario"]):
            state["feedback"].append(
                "条文の項番号（例:「第〇項」）を直接記載しないでください。必要な要件は言い換えで示してください。"
            )
            is_valid = False
        if re.search(r"第[0-9一二三四五六七八九十百千万]+号", state["scenario"]):
            state["feedback"].append(
                "条文の号番号（例:「第〇号」）を直接記載しないでください。内容を説明的に表現してください。"
            )
            is_valid = False

        if state["validation_score"] < self.config.mcq_validation_threshold:
            state["feedback"].append(
                f"整合性スコアが閾値を下回っています（{state['validation_score']:.2f} < {self.config.mcq_validation_threshold:.2f}）。"
            )
            is_valid = False

        state["is_valid"] = is_valid
        return state

    def _refine_case_node(self, state: MCQCaseState) -> MCQCaseState:
        current_length = len(state["scenario"])
        result = self.refiner.execute({
            "scenario": state["scenario"],
            "feedback": state["feedback"],
            "question": state["question"],
            "choices": state["choices"],
            "correct_choice": state["correct_choice"],
            "parser_result": state["parser_result"],
            "target_length": self.config.mcq_target_length,
            "min_length": self.config.mcq_min_length,
            "max_length": self.config.mcq_max_length,
            "current_length": current_length
        })

        scenario = result.get("scenario", state["scenario"])
        scenario = self._normalize_scenario_text(scenario)
        state["scenario"] = scenario
        state["character_count"] = len(scenario)
        state["agents_used"].append("mcq_refiner")
        state["iteration"] += 1
        return state

    def _should_refine(self, state: MCQCaseState) -> str:
        if state["is_valid"]:
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "refine"

    def _should_continue(self, state: MCQCaseState) -> str:
        return "check"

    def generate_case_from_sample(
        self,
        sample: Dict[str, Any],
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """サンプルデータから単一の事例を生成"""
        initial_state: MCQCaseState = {
            "question_id": sample.get("question_id", ""),
            "question": sample.get("question", ""),
            "context": sample.get("context", ""),
            "choices": sample.get("choices", {}),
            "correct_choice": sample.get("correct_choice", ""),
            "parser_result": {},
            "scenario": "",
            "character_count": 0,
            "is_valid": False,
            "validation_score": 0.0,
            "feedback": [],
            "iteration": 0,
            "max_iterations": max_iterations or self.config.mcq_max_iterations,
            "agents_used": [],
            "error": None
        }

        try:
            result = self.graph.invoke(initial_state)
            return {
                "question_id": result["question_id"],
                "question": result["question"],
                "context": result["context"],
                "choices": result["choices"],
                "correct_choice": result["correct_choice"],
                "correct_choice_text": result["parser_result"].get("correct_choice_text", ""),
                "parser_result": result["parser_result"],
                "scenario": result["scenario"],
                "character_count": result["character_count"],
                "is_valid": result["is_valid"],
                "validation_score": result["validation_score"],
                "feedback": result["feedback"],
                "iterations": result["iteration"],
                "agents_used": result["agents_used"],
                "error": result["error"]
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error generating MCQ case: %s", exc, exc_info=True)
            return {
                "question_id": initial_state["question_id"],
                "question": initial_state["question"],
                "context": initial_state["context"],
                "choices": initial_state["choices"],
                "correct_choice": initial_state["correct_choice"],
                "correct_choice_text": "",
                "parser_result": {},
                "scenario": "",
                "character_count": 0,
                "is_valid": False,
                "validation_score": 0.0,
                "feedback": [f"エラーが発生しました: {exc}"],
                "iterations": initial_state["iteration"],
                "agents_used": initial_state["agents_used"],
                "error": str(exc)
            }

    @staticmethod
    def _ensure_feedback(state: MCQCaseState, reset: bool = False) -> None:
        """フィードバックリストを初期化"""
        if "feedback" not in state or reset:
            state["feedback"] = []


def parse_choice_string(choice_text: str) -> Dict[str, str]:
    """選択肢文字列を辞書に変換"""
    import re

    choices: Dict[str, str] = {}
    for raw_line in choice_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r'^([a-zA-Zａ-ｚ])[\s\.\)：:\)]*(.*)$', line)
        if not match:
            continue
        label = match.group(1)
        label = label.lower()
        if ord(label[0]) > 255:
            # 全角英字対応
            label = chr(ord(label[0]) - 0xFEE0)
        content = match.group(2).strip()
        if content.startswith((":", "：")):
            content = content[1:].strip()
        choices[label] = content
    return choices


def normalize_mcq_sample(raw_sample: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """datasets/lawqa_jp形式のサンプルを正規化"""
    choice_field = raw_sample.get("選択肢") or raw_sample.get("choices")
    if isinstance(choice_field, str):
        choices = parse_choice_string(choice_field)
    elif isinstance(choice_field, dict):
        choices = {str(k).lower(): str(v) for k, v in choice_field.items()}
    else:
        choices = {}

    correct_choice = (raw_sample.get("output") or raw_sample.get("answer") or "").strip().lower()

    return {
        "question_id": raw_sample.get("ファイル名") or raw_sample.get("id") or f"sample_{index}",
        "question": raw_sample.get("問題文") or raw_sample.get("question") or "",
        "context": raw_sample.get("コンテキスト") or raw_sample.get("context") or "",
        "choices": choices,
        "correct_choice": correct_choice,
        "references": raw_sample.get("references", []),
        "metadata": {
            "index": index,
            "source": raw_sample.get("ファイル名") or raw_sample.get("id"),
            "指示": raw_sample.get("指示")
        }
    }


def run_mcq_generation(args: argparse.Namespace) -> None:
    """4択問題用事例生成の実行"""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset.get("samples") if isinstance(dataset, dict) else dataset
    if not isinstance(samples, list):
        raise ValueError("Dataset format is invalid. 'samples' 配列を含むJSONを指定してください。")

    start_index = max(args.index, 0)
    count = args.count if args.count is not None else 1
    end_index = min(start_index + count, len(samples))

    selected_samples = [
        normalize_mcq_sample(samples[i], index=i)
        for i in range(start_index, end_index)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = load_config()
    generator = MCQCaseGenerator(config)

    results = []
    for sample in selected_samples:
        result = generator.generate_case_from_sample(sample)
        result["metadata"] = sample.get("metadata", {})
        results.append(result)

    output_data = {
        "dataset": str(dataset_path),
        "start_index": start_index,
        "count": len(results),
        "results": results
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")
    else:
        print(json.dumps(output_data, ensure_ascii=False, indent=2))


def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description="法令事例・4択問題事例生成パイプライン")
    subparsers = parser.add_subparsers(dest="command", required=True)

    law_parser = subparsers.add_parser("law", help="条文ベースの事例を生成")
    law_parser.add_argument("--law-number", required=True, help="法令番号")
    law_parser.add_argument("--law-title", required=True, help="法令名")
    law_parser.add_argument("--article", required=True, help="条文番号")
    law_parser.add_argument("--article-content", required=True, help="条文内容")
    law_parser.add_argument("--output", help="出力ファイルパス（JSON）")

    mcq_parser = subparsers.add_parser("mcq", help="4択問題から具体的な事例を生成")
    mcq_parser.add_argument("--dataset", required=True, help="4択問題データセット（selection.json 等）のパス")
    mcq_parser.add_argument("--index", type=int, default=0, help="生成を開始するインデックス（0始まり）")
    mcq_parser.add_argument("--count", type=int, default=1, help="生成する問題数")
    mcq_parser.add_argument("--output", help="出力ファイルパス（JSON）。指定しない場合は標準出力に表示")

    args = parser.parse_args()

    if args.command == "law":
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        config = load_config()
        generator = LegalCaseGenerator(config)
        result = generator.generate_cases(
            law_number=args.law_number,
            law_title=args.law_title,
            article=args.article,
            article_content=args.article_content
        )

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "mcq":
        run_mcq_generation(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

