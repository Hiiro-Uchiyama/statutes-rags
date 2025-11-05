"""
Legal Case Generator 評価スクリプト

生成された事例の品質を評価します。
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 数字で始まるモジュール名のため、importlibを使用
import importlib
pipeline_module = importlib.import_module('examples.04_legal_case_generator.pipeline')
config_module = importlib.import_module('examples.04_legal_case_generator.config')

LegalCaseGenerator = pipeline_module.LegalCaseGenerator
load_config = config_module.load_config

logger = logging.getLogger(__name__)


class LegalCaseEvaluator:
    """事例生成の評価クラス"""
    
    def __init__(self, generator: LegalCaseGenerator):
        """
        Args:
            generator: LegalCaseGenerator インスタンス
        """
        self.generator = generator
    
    def evaluate_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        テストケースで評価
        
        Args:
            test_cases: テストケースのリスト
        
        Returns:
            評価結果
        """
        results = []
        total_time = 0
        success_count = 0
        total_iterations = 0
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating test case {i}/{len(test_cases)}")
            
            start_time = time.time()
            
            # 事例生成
            result = self.generator.generate_cases(
                law_number=test_case["law_number"],
                law_title=test_case["law_title"],
                article=test_case["article"],
                article_content=test_case["article_content"]
            )
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # 成功判定
            cases = result.get("cases", [])
            for case in cases:
                if not case.get("error"):
                    success_count += 1
                    total_iterations += case.get("iterations", 0)
            
            results.append({
                "test_case": test_case,
                "result": result,
                "elapsed_time": elapsed_time
            })
        
        # 集計
        total_cases = sum(len(r["result"].get("cases", [])) for r in results)
        avg_time = total_time / total_cases if total_cases > 0 else 0
        success_rate = success_count / total_cases if total_cases > 0 else 0
        avg_iterations = total_iterations / success_count if success_count > 0 else 0
        
        return {
            "summary": {
                "total_test_cases": len(test_cases),
                "total_cases_generated": total_cases,
                "success_count": success_count,
                "success_rate": success_rate,
                "total_time": total_time,
                "average_time_per_case": avg_time,
                "average_iterations": avg_iterations
            },
            "results": results
        }
    
    def generate_evaluation_template(
        self,
        cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        人手評価用のテンプレートを生成
        
        Args:
            cases: 生成された事例のリスト
        
        Returns:
            人手評価用テンプレート
        """
        template = []
        
        for case in cases:
            template.append({
                "law_info": {
                    "law_title": case.get("law_title", ""),
                    "article": case.get("article", ""),
                },
                "case_type": case.get("case_type", ""),
                "scenario": case.get("scenario", ""),
                "legal_analysis": case.get("legal_analysis", ""),
                "educational_point": case.get("educational_point", ""),
                "human_evaluation": {
                    "legal_validity": 0,  # 1-5点
                    "concreteness": 0,    # 1-5点
                    "educational_value": 0,  # 1-5点
                    "comments": ""
                }
            })
        
        return template


def create_sample_test_cases() -> List[Dict[str, Any]]:
    """サンプルのテストケースを作成"""
    return [
        {
            "law_number": "平成十七年法律第八十七号",
            "law_title": "会社法",
            "article": "26",
            "article_content": "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。"
        },
        {
            "law_number": "明治二十九年法律第八十九号",
            "law_title": "民法",
            "article": "96",
            "article_content": "詐欺又は強迫による意思表示は、取り消すことができる。"
        },
        {
            "law_number": "平成十五年法律第五十七号",
            "law_title": "個人情報の保護に関する法律",
            "article": "27",
            "article_content": "個人情報取扱事業者は、次に掲げる場合を除くほか、あらかじめ本人の同意を得ないで、個人データを第三者に提供してはならない。"
        }
    ]


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="事例生成の評価")
    parser.add_argument(
        "--test-cases",
        help="テストケースのJSONファイル"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="評価結果の出力ファイル"
    )
    parser.add_argument(
        "--generate-template",
        help="人手評価用テンプレートの出力ファイル"
    )
    
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
    
    # 評価器の初期化
    evaluator = LegalCaseEvaluator(generator)
    
    # テストケースの読み込み
    if args.test_cases:
        with open(args.test_cases, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    else:
        logger.info("Using sample test cases")
        test_cases = create_sample_test_cases()
    
    # 評価実行
    logger.info(f"Evaluating {len(test_cases)} test cases")
    evaluation_results = evaluator.evaluate_test_cases(test_cases)
    
    # 結果の出力
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output}")
    
    # サマリーの表示
    summary = evaluation_results["summary"]
    print("\n=== 評価サマリー ===")
    print(f"テストケース数: {summary['total_test_cases']}")
    print(f"生成事例数: {summary['total_cases_generated']}")
    print(f"成功数: {summary['success_count']}")
    print(f"成功率: {summary['success_rate']:.1%}")
    print(f"総実行時間: {summary['total_time']:.1f}秒")
    print(f"平均生成時間: {summary['average_time_per_case']:.1f}秒/事例")
    print(f"平均反復回数: {summary['average_iterations']:.2f}")
    
    # 人手評価用テンプレート生成
    if args.generate_template:
        all_cases = []
        for result in evaluation_results["results"]:
            all_cases.extend(result["result"].get("cases", []))
        
        template = evaluator.generate_evaluation_template(all_cases)
        
        with open(args.generate_template, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation template saved to {args.generate_template}")


if __name__ == "__main__":
    main()

