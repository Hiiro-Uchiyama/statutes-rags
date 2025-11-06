#!/usr/bin/env python3
"""
Quick test script for Agentic RAG
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Agentic RAGのディレクトリもパスに追加
agentic_rag_dir = Path(__file__).parent.parent
sys.path.insert(0, str(agentic_rag_dir))

# 環境変数のロード
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from config import load_config
from pipeline import AgenticRAGPipeline

def main():
    print("="*60)
    print("Agentic RAG - Quick Test")
    print("="*60)
    
    # 設定のロード
    print("\n1. Loading configuration...")
    config = load_config()
    print(f"   LLM Model: {config.llm_model}")
    print(f"   Max Iterations: {config.max_iterations}")
    print(f"   Reasoning Enabled: {config.enable_reasoning}")
    print(f"   Validation Enabled: {config.enable_validation}")
    
    # パイプラインの初期化
    print("\n2. Initializing pipeline...")
    try:
        pipeline = AgenticRAGPipeline(config)
        print("   ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize pipeline: {e}")
        return 1
    
    # テストクエリ
    test_queries = [
        "民法第1条について教えてください",
        "会社法第26条の規定について説明してください",
    ]
    
    print("\n3. Running test queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        print("   " + "-"*50)
        
        try:
            result = pipeline.query(query)
            
            print(f"   Complexity: {result['metadata']['complexity']}")
            print(f"   Confidence: {result['metadata']['confidence']:.2f}")
            print(f"   Iterations: {result['metadata']['iterations']}")
            print(f"   Agents Used: {', '.join(result['metadata']['agents_used'])}")
            print(f"   Answer Preview: {result['answer'][:100]}...")
            
            if result['citations']:
                print(f"   Citations: {len(result['citations'])} documents")
            
            print("   ✓ Query executed successfully")
            
        except Exception as e:
            print(f"   ✗ Query failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
