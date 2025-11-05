"""
Multi-Agent Debate クイックテスト

依存関係と基本的な動作を確認するための簡易スクリプト。
実際のLLMやインデックスなしでモックを使用して動作確認します。
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Multi-Agent Debate - Quick Test")
print("=" * 60)

# 1. インポートテスト
print("\n[1/5] Testing imports...")
try:
    # 数字で始まるモジュール名を避けるため、sys.pathを使用
    debate_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(debate_dir))
    
    from config import MultiAgentDebateConfig, load_config
    from agents.debater import DebaterAgent
    from agents.moderator import ModeratorAgent
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# 2. 設定テスト
print("\n[2/5] Testing configuration...")
try:
    config = MultiAgentDebateConfig()
    assert config.max_debate_rounds == 3
    assert config.agreement_threshold == 0.8
    print("  ✓ Configuration OK")
    print(f"    - Max rounds: {config.max_debate_rounds}")
    print(f"    - Agreement threshold: {config.agreement_threshold}")
    print(f"    - LLM model: {config.llm_model}")
except Exception as e:
    print(f"  ✗ Configuration test failed: {e}")
    sys.exit(1)

# 3. Debaterエージェントテスト
print("\n[3/5] Testing Debater Agent...")
try:
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value="""
主張：
これはテスト主張です。

推論：
テストのための推論です。

引用条文：
テスト法 第1条
""")
    
    debater_a = DebaterAgent(mock_llm, config, stance="affirmative")
    debater_b = DebaterAgent(mock_llm, config, stance="critical")
    
    assert debater_a.stance == "affirmative"
    assert debater_b.stance == "critical"
    
    print("  ✓ Debater Agent initialization OK")
    print(f"    - Debater A stance: {debater_a.stance}")
    print(f"    - Debater B stance: {debater_b.stance}")
except Exception as e:
    print(f"  ✗ Debater Agent test failed: {e}")
    sys.exit(1)

# 4. Moderatorエージェントテスト
print("\n[4/5] Testing Moderator Agent...")
try:
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value="モデレーターコメント")
    
    with patch.object(ModeratorAgent, '_init_embedding_model'):
        moderator = ModeratorAgent(mock_llm, config)
        moderator.embeddings = None  # 埋め込みモデルなしでテスト
        
        # 合意判定テスト
        agreement = moderator._calculate_agreement("同じ内容", "同じ内容")
        assert 0.0 <= agreement <= 1.0
        
        # 継続判定テスト
        should_continue = moderator._should_continue_debate(
            agreement_score=0.5,
            current_round=1,
            max_rounds=3
        )
        assert should_continue is True
        
        print("  ✓ Moderator Agent OK")
        print(f"    - Agreement calculation works")
        print(f"    - Continue decision works")
except Exception as e:
    print(f"  ✗ Moderator Agent test failed: {e}")
    sys.exit(1)

# 5. Debater実行テスト（モック使用）
print("\n[5/5] Testing Debater execution...")
try:
    mock_document = Mock()
    mock_document.page_content = "テスト法令内容"
    mock_document.metadata = {
        "law_title": "テスト法",
        "article": "1",
        "paragraph": "",
        "item": ""
    }
    
    input_data = {
        "query": "テスト質問",
        "documents": [mock_document],
        "round": 1
    }
    
    result = debater_a.execute(input_data)
    
    assert "position" in result
    assert "reasoning" in result
    assert "citations" in result
    
    print("  ✓ Debater execution OK")
    print(f"    - Position generated: {len(result['position'])} chars")
    print(f"    - Reasoning generated: {len(result['reasoning'])} chars")
except Exception as e:
    print(f"  ✗ Debater execution test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Set up Ollama and download qwen3:8b model")
print("2. Prepare datasets and build FAISS index")
print("3. Run actual evaluation: python evaluate.py --limit 5")
print("\nSee README.md for detailed instructions.")

