"""
Legal Case Generator Pipeline

LangGraphを使用した事例生成パイプライン。
"""
import os
import logging
import argparse
import json
from typing import Dict, Any, List, TypedDict, Optional
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

from examples.04_legal_case_generator.config import LegalCaseConfig, load_config
from examples.04_legal_case_generator.agents import (
    ScenarioGeneratorAgent,
    LegalCheckerAgent,
    RefinerAgent,
)

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


def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description="法令適用事例生成")
    parser.add_argument("--law-number", required=True, help="法令番号")
    parser.add_argument("--law-title", required=True, help="法令名")
    parser.add_argument("--article", required=True, help="条文番号")
    parser.add_argument("--article-content", required=True, help="条文内容")
    parser.add_argument("--output", help="出力ファイルパス（JSON）")
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 設定のロード
    config = load_config()
    
    # ジェネレータの初期化
    generator = LegalCaseGenerator(config)
    
    # 事例生成
    result = generator.generate_cases(
        law_number=args.law_number,
        law_title=args.law_title,
        article=args.article,
        article_content=args.article_content
    )
    
    # 結果の出力
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

