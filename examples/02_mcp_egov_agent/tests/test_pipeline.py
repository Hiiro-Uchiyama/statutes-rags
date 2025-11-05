#!/usr/bin/env python3
"""
簡単なパイプラインテスト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 60)
print("MCP e-Gov Agent パイプラインテスト")
print("=" * 60)
print()

# モジュールのインポート
try:
    import importlib
    mcp_module = importlib.import_module('examples.02_mcp_egov_agent')
    load_config = mcp_module.load_config
    EGovAPIClient = mcp_module.EGovAPIClient
    
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    print("[OK] モジュールのインポートに成功しました")
except Exception as e:
    print(f"[FAIL] モジュールのインポートに失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 設定のロード
try:
    config = load_config(validate=False)
    print(f"[OK] 設定のロード成功")
    print(f"  - API URL: {config.api_base_url}")
    print(f"  - ベクトルストアパス: {config.vector_store_path}")
except Exception as e:
    print(f"[FAIL] 設定のロードに失敗: {e}")
    sys.exit(1)

print()

# API接続テスト
try:
    client = EGovAPIClient()
    if client.health_check():
        print("[OK] e-Gov API接続確認成功")
    else:
        print("[FAIL] e-Gov API接続失敗")
except Exception as e:
    print(f"[FAIL] API接続テストに失敗: {e}")

print()

# Retrieverの初期化
try:
    vector_store_path = config.get_absolute_path(config.vector_store_path)
    
    if not vector_store_path.exists():
        print(f"[WARN] ベクトルストアが見つかりません: {vector_store_path}")
        print("  スキップします（インデックスをビルドしてください）")
        sys.exit(0)
    
    vector_retriever = VectorRetriever(
        embedding_model="intfloat/multilingual-e5-large",
        index_path=str(vector_store_path / "vector")
    )
    
    bm25_retriever = BM25Retriever(
        index_path=str(vector_store_path / "bm25")
    )
    
    retriever = HybridRetriever(vector_retriever, bm25_retriever)
    print("[OK] Retriever初期化成功")
    
    # 簡単な検索テスト
    test_query = "個人情報保護について"
    docs = retriever.retrieve(test_query, top_k=3)
    print(f"[OK] ローカル検索テスト成功 ({len(docs)}件の結果)")
    
except FileNotFoundError as e:
    print(f"[WARN] ファイルが見つかりません: {e}")
    print("  インデックスをビルドしてください: python scripts/build_index.py")
    sys.exit(0)
except Exception as e:
    print(f"[FAIL] Retriever初期化に失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("パイプラインテスト完了")
print("=" * 60)
print()
print("次のステップ:")
print("  - 完全な評価を実行: python examples/02_mcp_egov_agent/evaluate.py --limit 5")
print()
