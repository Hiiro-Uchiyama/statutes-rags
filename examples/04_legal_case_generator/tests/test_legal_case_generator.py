"""
簡易テストスクリプト

LLMを使わずに基本的な動作を確認します。
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import importlib

# 動的インポート（ディレクトリ名に数字が含まれるため）
config_module = importlib.import_module("examples.04_legal_case_generator.config")
load_config = config_module.load_config


def test_config():
    """設定のロードテスト"""
    print("Testing config loading...")
    config = load_config()
    
    # 注: max_cases_per_law は現在コメントアウトされています
    assert config.llm_model == "qwen3:8b"
    assert config.llm_temperature == 0.3
    assert config.max_iterations == 2
    assert config.mcq_target_length == 500
    assert config.mcq_min_length == 460
    assert config.mcq_max_length == 540
    assert config.mcq_max_iterations == 6
    
    print(f"  llm_model: {config.llm_model}")
    print(f"  llm_temperature: {config.llm_temperature}")
    print(f"  max_iterations: {config.max_iterations}")
    print(f"  mcq_target_length: {config.mcq_target_length}")
    print("  Config loading: OK")


def test_agents_import():
    """エージェントのインポートテスト"""
    print("\nTesting agents import...")
    
    agents_module = importlib.import_module("examples.04_legal_case_generator.agents")
    ScenarioGeneratorAgent = agents_module.ScenarioGeneratorAgent
    LegalCheckerAgent = agents_module.LegalCheckerAgent
    RefinerAgent = agents_module.RefinerAgent
    MCQParserAgent = agents_module.MCQParserAgent
    MCQCaseGeneratorAgent = agents_module.MCQCaseGeneratorAgent
    MCQConsistencyCheckerAgent = agents_module.MCQConsistencyCheckerAgent
    MCQRefinerAgent = agents_module.MCQRefinerAgent
    
    print(f"  ScenarioGeneratorAgent: {ScenarioGeneratorAgent.__name__}")
    print(f"  LegalCheckerAgent: {LegalCheckerAgent.__name__}")
    print(f"  RefinerAgent: {RefinerAgent.__name__}")
    print(f"  MCQParserAgent: {MCQParserAgent.__name__}")
    print(f"  MCQCaseGeneratorAgent: {MCQCaseGeneratorAgent.__name__}")
    print(f"  MCQConsistencyCheckerAgent: {MCQConsistencyCheckerAgent.__name__}")
    print(f"  MCQRefinerAgent: {MCQRefinerAgent.__name__}")
    print("  Agents import: OK")


def test_pipeline_import():
    """パイプラインのインポートテスト"""
    print("\nTesting pipeline import...")
    
    pipeline_module = importlib.import_module("examples.04_legal_case_generator.pipeline")
    LegalCaseGenerator = pipeline_module.LegalCaseGenerator
    MCQCaseGenerator = pipeline_module.MCQCaseGenerator
    
    print(f"  LegalCaseGenerator: {LegalCaseGenerator.__name__}")
    print(f"  MCQCaseGenerator: {MCQCaseGenerator.__name__}")
    print("  Pipeline import: OK")


def test_evaluate_import():
    """評価スクリプトのインポートテスト"""
    print("\nTesting evaluate import...")
    
    evaluate_module = importlib.import_module("examples.04_legal_case_generator.evaluate")
    LegalCaseEvaluator = evaluate_module.LegalCaseEvaluator
    create_sample_test_cases = evaluate_module.create_sample_test_cases
    
    test_cases = create_sample_test_cases()
    
    print(f"  LegalCaseEvaluator: {LegalCaseEvaluator.__name__}")
    print(f"  Sample test cases: {len(test_cases)} cases")
    
    for i, tc in enumerate(test_cases, 1):
        print(f"    {i}. {tc['law_title']} 第{tc['article']}条")
    
    print("  Evaluate import: OK")


def main():
    """メイン関数"""
    print("=== Legal Case Generator - Simple Test ===\n")
    
    try:
        test_config()
        test_agents_import()
        test_pipeline_import()
        test_evaluate_import()
        
        print("\n=== All tests passed! ===")
        return 0
    except Exception as e:
        print(f"\n!!! Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

