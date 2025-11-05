"""
pytest設定とフィクスチャ（04_legal_case_generator用）
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_llm():
    """モックLLMを提供"""
    llm = Mock()
    llm.invoke = Mock(return_value="モック応答")
    return llm


@pytest.fixture
def legal_case_config():
    """Legal Case Generator設定を提供"""
    # 数字で始まるモジュール名のため、importlibを使用
    import importlib
    config_module = importlib.import_module('examples.04_legal_case_generator.config')
    LegalCaseConfig = config_module.LegalCaseConfig
    
    # 注: max_cases_per_law は現在コメントアウトされています
    return LegalCaseConfig(
        min_length=100,
        max_length=500,
        max_iterations=2,
        generate_applicable=True,
        generate_non_applicable=True,
        generate_boundary=True,
        llm_model="qwen3:8b",
        llm_temperature=0.3
    )


@pytest.fixture
def sample_law_info():
    """サンプル法令情報を提供"""
    return {
        "law_number": "平成十七年法律第八十七号",
        "law_title": "会社法",
        "article": "26",
        "article_content": "株式会社は、株主名簿を作成し、これに株主の氏名又は名称及び住所、各株主の有する株式の種類及び数並びに株式を取得した日を記載し、又は記録しなければならない。"
    }

