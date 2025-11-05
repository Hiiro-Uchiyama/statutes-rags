"""
Agentic RAG テスト
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# プロジェクトルートとexamplesディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
examples_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(examples_dir))

from config import AgenticRAGConfig
from agents import (
    ManagerAgent,
    RetrievalAgent,
    ReasoningAgent,
    ValidationAgent
)


class TestManagerAgent:
    """ManagerAgentのテスト"""
    
    def test_classify_complexity_simple(self, mock_llm, agentic_rag_config):
        """単純な質問の複雑度判定"""
        agent = ManagerAgent(mock_llm, agentic_rag_config)
        
        query = "民法第1条の内容は何ですか？"
        complexity = agent.classify_complexity(query)
        
        assert complexity in ["simple", "medium", "complex"]
        # ルールベースで simple と判定されるはず
        assert complexity == "simple"
    
    def test_classify_complexity_complex(self, mock_llm, agentic_rag_config):
        """複雑な質問の複雑度判定"""
        agent = ManagerAgent(mock_llm, agentic_rag_config)
        
        query = "個人情報保護法と民法の関係について説明してください"
        complexity = agent.classify_complexity(query)
        
        # 複数の法令が言及されているので complex
        assert complexity == "complex"
    
    def test_classify_query_type_lookup(self, mock_llm, agentic_rag_config):
        """lookupタイプの判定"""
        agent = ManagerAgent(mock_llm, agentic_rag_config)
        
        query = "会社法第26条の内容は何ですか？"
        query_type = agent.classify_query_type(query)
        
        assert query_type == "lookup"
    
    def test_classify_query_type_application(self, mock_llm, agentic_rag_config):
        """applicationタイプの判定"""
        agent = ManagerAgent(mock_llm, agentic_rag_config)
        
        query = "この場合、どの法令が適用されますか？"
        query_type = agent.classify_query_type(query)
        
        assert query_type == "application"
    
    def test_execute(self, mock_llm, agentic_rag_config):
        """executeメソッドのテスト"""
        agent = ManagerAgent(mock_llm, agentic_rag_config)
        
        result = agent.execute({"query": "民法第1条とは？"})
        
        assert "complexity" in result
        assert "query_type" in result
        assert result["complexity"] in ["simple", "medium", "complex"]
        assert result["query_type"] in ["lookup", "interpretation", "application"]


class TestRetrievalAgent:
    """RetrievalAgentのテスト"""
    
    def test_select_strategy_bm25_with_article(self, mock_llm, agentic_rag_config):
        """条文番号を含む場合のBM25選択"""
        mock_retrievers = {
            "vector": Mock(),
            "bm25": Mock(),
            "hybrid": Mock()
        }
        
        agent = RetrievalAgent(mock_llm, agentic_rag_config, mock_retrievers)
        
        query = "民法第1条について"
        strategy = agent.select_strategy(query, "lookup")
        
        assert strategy == "bm25"
    
    def test_select_strategy_vector_interpretation(self, mock_llm, agentic_rag_config):
        """解釈質問の場合のvector選択"""
        mock_retrievers = {
            "vector": Mock(),
            "bm25": Mock(),
            "hybrid": Mock()
        }
        
        agent = RetrievalAgent(mock_llm, agentic_rag_config, mock_retrievers)
        
        query = "法の趣旨について説明してください"
        strategy = agent.select_strategy(query, "interpretation")
        
        assert strategy == "vector"
    
    def test_evaluate_quality_good(self, mock_llm, agentic_rag_config, mock_documents):
        """品質評価（良好）"""
        mock_retrievers = {}
        agent = RetrievalAgent(mock_llm, agentic_rag_config, mock_retrievers)
        
        quality = agent.evaluate_quality("テスト質問", mock_documents)
        
        assert "score" in quality
        assert "is_sufficient" in quality
        assert quality["is_sufficient"] is True
    
    def test_evaluate_quality_no_documents(self, mock_llm, agentic_rag_config):
        """品質評価（文書なし）"""
        mock_retrievers = {}
        agent = RetrievalAgent(mock_llm, agentic_rag_config, mock_retrievers)
        
        quality = agent.evaluate_quality("テスト質問", [])
        
        assert quality["score"] == 0.0
        assert quality["is_sufficient"] is False


class TestReasoningAgent:
    """ReasoningAgentのテスト"""
    
    def test_simple_reasoning(self, mock_llm, agentic_rag_config, mock_documents):
        """簡易推論のテスト"""
        mock_llm.invoke = Mock(return_value="第1条は基本原則を定めています。")
        
        agent = ReasoningAgent(mock_llm, agentic_rag_config)
        
        result = agent.simple_reasoning("第1条について", mock_documents)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_analyze_legal_structure(self, mock_llm, agentic_rag_config, mock_documents):
        """法的構造分析のテスト"""
        mock_response = """{
            "main_provisions": ["第1条"],
            "related_provisions": ["第2条"],
            "exceptions": [],
            "application_order": ["第1条", "第2条"]
        }"""
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = ReasoningAgent(mock_llm, agentic_rag_config)
        
        result = agent.analyze_legal_structure("テスト質問", mock_documents)
        
        assert "main_provisions" in result
        assert "related_provisions" in result
        assert isinstance(result["main_provisions"], list)
    
    def test_execute_simple_complexity(self, mock_llm, agentic_rag_config, mock_documents):
        """簡易推論の実行"""
        mock_llm.invoke = Mock(return_value="回答です")
        
        agent = ReasoningAgent(mock_llm, agentic_rag_config)
        
        result = agent.execute({
            "query": "テスト",
            "documents": mock_documents,
            "complexity": "simple"
        })
        
        assert "reasoning" in result
        assert "legal_structure" in result


class TestValidationAgent:
    """ValidationAgentのテスト"""
    
    def test_verify_citations_valid(self, mock_llm, agentic_rag_config, mock_documents):
        """引用検証（正確）"""
        mock_response = """{
            "is_accurate": true,
            "confidence": 0.9,
            "issues": []
        }"""
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = ValidationAgent(mock_llm, agentic_rag_config)
        
        result = agent.verify_citations("第1条によれば...", mock_documents)
        
        assert result["is_accurate"] is True
        assert result["confidence"] > 0.5
    
    def test_detect_hallucination_none(self, mock_llm, agentic_rag_config, mock_documents):
        """ハルシネーション検出（なし）"""
        mock_response = """{
            "has_hallucination": false,
            "hallucinated_parts": [],
            "suggestions": []
        }"""
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = ValidationAgent(mock_llm, agentic_rag_config)
        
        result = agent.detect_hallucination("正確な回答", mock_documents)
        
        assert result["has_hallucination"] is False
    
    def test_execute(self, mock_llm, agentic_rag_config, mock_documents):
        """executeメソッドのテスト"""
        # 引用検証をモック
        mock_citation_response = """{
            "is_accurate": true,
            "confidence": 0.9,
            "issues": []
        }"""
        # ハルシネーション検出をモック
        mock_hallucination_response = """{
            "has_hallucination": false,
            "hallucinated_parts": [],
            "suggestions": []
        }"""
        
        mock_llm.invoke = Mock(side_effect=[mock_citation_response, mock_hallucination_response])
        
        agent = ValidationAgent(mock_llm, agentic_rag_config)
        
        result = agent.execute({
            "query": "テスト",
            "answer": "回答",
            "documents": mock_documents
        })
        
        assert "is_valid" in result
        assert "confidence" in result
        assert result["is_valid"] is True


class TestAgenticRAGConfig:
    """AgenticRAGConfigのテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = AgenticRAGConfig()
        
        assert config.max_iterations > 0
        assert 0.0 <= config.confidence_threshold <= 1.0
        assert config.llm_model is not None
        assert config.llm_temperature >= 0.0
    
    def test_custom_config(self):
        """カスタム設定のテスト"""
        config = AgenticRAGConfig(
            max_iterations=5,
            confidence_threshold=0.9,
            enable_reasoning=False
        )
        
        assert config.max_iterations == 5
        assert config.confidence_threshold == 0.9
        assert config.enable_reasoning is False


# 統合テストは実際のLLMとRetrieverが必要なため、スキップ可能なマーカーを付与
@pytest.mark.integration
class TestAgenticRAGPipeline:
    """AgenticRAGPipelineの統合テスト（要実環境）"""
    
    @pytest.mark.skip(reason="Requires actual LLM and retrievers")
    def test_pipeline_query(self):
        """パイプラインのクエリ実行（スキップ）"""
        # 実環境でのテストが必要
        pass

