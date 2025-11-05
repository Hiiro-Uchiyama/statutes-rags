"""
Multi-Agent Debate のテスト
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 03_multi_agent_debateディレクトリをパスに追加
debate_dir = Path(__file__).parent.parent
sys.path.insert(0, str(debate_dir))

from config import MultiAgentDebateConfig, load_config
from agents.debater import DebaterAgent
from agents.moderator import ModeratorAgent


class TestMultiAgentDebateConfig:
    """設定のテスト"""
    
    def test_default_config(self):
        """デフォルト設定の確認"""
        config = MultiAgentDebateConfig()
        
        assert config.max_debate_rounds == 3
        assert config.agreement_threshold == 0.8
        assert config.llm_model == "qwen3:8b"
        assert config.llm_temperature == 0.1
        assert config.retrieval_top_k == 10
    
    def test_custom_config(self):
        """カスタム設定の確認"""
        config = MultiAgentDebateConfig(
            max_debate_rounds=5,
            agreement_threshold=0.9,
            llm_temperature=0.0
        )
        
        assert config.max_debate_rounds == 5
        assert config.agreement_threshold == 0.9
        assert config.llm_temperature == 0.0
    
    def test_load_config(self):
        """設定ロード機能のテスト"""
        config = load_config()
        
        assert isinstance(config, MultiAgentDebateConfig)
        assert config.max_debate_rounds >= 1


class TestDebaterAgent:
    """Debater Agent のテスト"""
    
    @pytest.fixture
    def mock_llm(self):
        """モックLLM"""
        llm = Mock()
        llm.invoke = Mock(return_value="テスト応答")
        return llm
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return MultiAgentDebateConfig()
    
    @pytest.fixture
    def mock_document(self):
        """モック文書"""
        doc = Mock()
        doc.page_content = "テスト法令内容"
        doc.metadata = {
            "law_title": "テスト法",
            "article": "1",
            "paragraph": "",
            "item": ""
        }
        return doc
    
    def test_debater_initialization_affirmative(self, mock_llm, config):
        """肯定的Debaterの初期化"""
        debater = DebaterAgent(mock_llm, config, stance="affirmative")
        
        assert debater.stance == "affirmative"
        assert debater.llm == mock_llm
        assert debater.config == config
    
    def test_debater_initialization_critical(self, mock_llm, config):
        """批判的Debaterの初期化"""
        debater = DebaterAgent(mock_llm, config, stance="critical")
        
        assert debater.stance == "critical"
    
    def test_execute_initial_position(self, mock_llm, config, mock_document):
        """初回ラウンドの主張生成"""
        debater = DebaterAgent(mock_llm, config, stance="affirmative")
        
        # モックLLMの応答を設定
        mock_llm.invoke.return_value = """
主張：
法令が適用されます。

推論：
条文に該当するため。

引用条文：
テスト法 第1条
"""
        
        input_data = {
            "query": "テスト質問",
            "documents": [mock_document],
            "round": 1
        }
        
        result = debater.execute(input_data)
        
        assert "position" in result
        assert "reasoning" in result
        assert "citations" in result
        assert mock_llm.invoke.called
    
    def test_execute_rebuttal(self, mock_llm, config, mock_document):
        """反論の生成"""
        debater = DebaterAgent(mock_llm, config, stance="critical")
        
        mock_llm.invoke.return_value = """
主張：
相手の主張には問題があります。

推論：
例外規定があります。

引用条文：
テスト法 第1条
"""
        
        input_data = {
            "query": "テスト質問",
            "documents": [mock_document],
            "opponent_position": "法令が適用される",
            "round": 2
        }
        
        result = debater.execute(input_data)
        
        assert "position" in result
        assert mock_llm.invoke.called
    
    def test_parse_position_response(self, mock_llm, config, mock_document):
        """応答のパース"""
        debater = DebaterAgent(mock_llm, config)
        
        response = """
主張：
これは主張です。

推論：
これは推論です。

引用条文：
テスト法 第1条
"""
        
        result = debater._parse_position_response(response, [mock_document])
        
        assert "これは主張です" in result["position"]
        assert "これは推論です" in result["reasoning"]
        assert isinstance(result["citations"], list)


class TestModeratorAgent:
    """Moderator Agent のテスト"""
    
    @pytest.fixture
    def mock_llm(self):
        """モックLLM"""
        llm = Mock()
        llm.invoke = Mock(return_value="モデレーターのコメント")
        return llm
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return MultiAgentDebateConfig()
    
    @pytest.fixture
    def mock_document(self):
        """モック文書"""
        doc = Mock()
        doc.page_content = "テスト法令内容"
        doc.metadata = {
            "law_title": "テスト法",
            "article": "1",
        }
        return doc
    
    def test_moderator_initialization(self, mock_llm, config):
        """Moderatorの初期化"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            
            assert moderator.llm == mock_llm
            assert moderator.config == config
    
    def test_calculate_agreement_similar(self, mock_llm, config):
        """類似した主張の合意スコア計算"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            moderator.embeddings = None  # 埋め込みモデルなし（フォールバック）
            
            # 同じ文字列
            score = moderator._calculate_agreement("同じ内容", "同じ内容")
            
            assert 0.0 <= score <= 1.0
            assert score > 0.5  # ある程度高いスコア
    
    def test_calculate_agreement_different(self, mock_llm, config):
        """異なる主張の合意スコア計算"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            moderator.embeddings = None
            
            # 全く異なる内容
            score = moderator._calculate_agreement(
                "完全に異なる内容A",
                "全く別の話題B"
            )
            
            assert 0.0 <= score <= 1.0
    
    def test_should_continue_debate_max_rounds(self, mock_llm, config):
        """最大ラウンド到達時の継続判定"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            
            # 最大ラウンドに達した
            should_continue = moderator._should_continue_debate(
                agreement_score=0.5,
                current_round=3,
                max_rounds=3
            )
            
            assert should_continue is False
    
    def test_should_continue_debate_high_agreement(self, mock_llm, config):
        """高い合意スコア時の継続判定"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            
            # 高い合意スコア
            should_continue = moderator._should_continue_debate(
                agreement_score=0.9,
                current_round=1,
                max_rounds=3
            )
            
            assert should_continue is False
    
    def test_should_continue_debate_continue(self, mock_llm, config):
        """継続すべき場合の判定"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            
            # まだ継続すべき
            should_continue = moderator._should_continue_debate(
                agreement_score=0.5,
                current_round=1,
                max_rounds=3
            )
            
            assert should_continue is True
    
    def test_execute(self, mock_llm, config, mock_document):
        """Moderatorの実行"""
        with patch.object(ModeratorAgent, '_init_embedding_model'):
            moderator = ModeratorAgent(mock_llm, config)
            moderator.embeddings = None
            
            input_data = {
                "query": "テスト質問",
                "documents": [mock_document],
                "debater_a_position": {
                    "position": "主張A",
                    "reasoning": "推論A"
                },
                "debater_b_position": {
                    "position": "主張B",
                    "reasoning": "推論B"
                },
                "round": 1,
                "max_rounds": 3
            }
            
            result = moderator.execute(input_data)
            
            assert "agreement_score" in result
            assert "should_continue" in result
            assert isinstance(result["agreement_score"], float)
            assert isinstance(result["should_continue"], bool)


class TestDebateWorkflowIntegration:
    """ワークフロー統合テスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return MultiAgentDebateConfig(
            max_debate_rounds=2,
            retrieval_top_k=3
        )
    
    def test_workflow_initialization(self, config):
        """ワークフローの初期化（モック使用）"""
        # ワークフローの初期化には実際のコンポーネントが必要なため、
        # ここでは設定が正しく渡されることのみテスト
        assert config.max_debate_rounds == 2
        assert config.retrieval_top_k == 3


# 実行
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

