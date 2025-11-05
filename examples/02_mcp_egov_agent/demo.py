#!/usr/bin/env python3
"""
MCP e-Gov Agent デモスクリプト

セットアップ後にすぐに試せるデモ。
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 60)
print("MCP e-Gov Agent - デモ")
print("=" * 60)
print()

# 必要な依存関係のチェック
print("依存関係をチェック中...")
missing_deps = []

try:
    import httpx
    print("  httpx: OK")
except ImportError:
    print("  httpx: 不足")
    missing_deps.append("httpx")

try:
    import tenacity
    print("  tenacity: OK")
except ImportError:
    print("  tenacity: 不足")
    missing_deps.append("tenacity")

try:
    import pydantic
    print("  pydantic: OK")
except ImportError:
    print("  pydantic: 不足")
    missing_deps.append("pydantic")

if missing_deps:
    print()
    print("エラー: 以下の依存関係が不足しています:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print()
    print("セットアップを実行してください:")
    print("  cd /path/to/statutes-rags")
    print("  ./setup/setup_uv_env.sh")
    sys.exit(1)

print()
print("=" * 60)
print("デモ1: e-Gov APIへの接続テスト")
print("=" * 60)
print()

import importlib
egov_module = importlib.import_module('examples.02_mcp_egov_agent.egov_client')
EGovAPIClient = egov_module.EGovAPIClient
EGovAPIError = egov_module.EGovAPIError

# APIクライアントの作成
client = EGovAPIClient()

print("APIクライアントを初期化しました")
print()

# 疎通確認
print("1. API疎通確認...")
try:
    if client.health_check():
        print("   結果: OK - APIに正常に接続できました")
    else:
        print("   結果: NG - API接続に失敗しました")
        print()
        print("注: ネットワーク接続を確認してください")
except Exception as e:
    print(f"   エラー: {e}")
    print()
    print("注: インターネット接続が必要です")

print()

# キーワード検索のテスト
print("2. キーワード検索テスト...")
keyword = "個人情報保護"
print(f"   キーワード: '{keyword}'")

try:
    result = client.search_by_keyword(keyword)
    laws = result.get("laws", [])
    print(f"   結果: {len(laws)}件の法令が見つかりました")
    
    if laws and len(laws) > 0:
        # 最初の1件を表示
        first_law = laws[0]
        law_info = first_law.get("law_info", {})
        law_title = law_info.get("law_title", "不明")
        print(f"   最初の法令: {law_title}")
except EGovAPIError as e:
    print(f"   エラー: {e}")
except Exception as e:
    print(f"   予期しないエラー: {e}")

print()
print("=" * 60)
print("デモ2: 設定の確認")
print("=" * 60)
print()

config_module = importlib.import_module('examples.02_mcp_egov_agent.config')
load_config = config_module.load_config

try:
    config = load_config(validate=False)
    print("設定を読み込みました:")
    print(f"  API URL: {config.api_base_url}")
    print(f"  API タイムアウト: {config.api_timeout}秒")
    print(f"  API優先モード: {config.prefer_api}")
    print(f"  ローカルフォールバック: {config.fallback_to_local}")
    print(f"  最近の法令判定日数: {config.recent_law_threshold_days}日")
except Exception as e:
    print(f"エラー: {e}")

print()
print("=" * 60)
print("デモ3: データパスの確認")
print("=" * 60)
print()

try:
    vector_store_path = config.get_absolute_path(config.vector_store_path)
    data_path = config.get_absolute_path(config.data_path)
    
    print("データパス:")
    print(f"  ベクトルストア: {vector_store_path}")
    print(f"    存在: {'Yes' if vector_store_path.exists() else 'No'}")
    print(f"  法令データ: {data_path}")
    print(f"    存在: {'Yes' if data_path.exists() else 'No'}")
    
    if not vector_store_path.exists() or not data_path.exists():
        print()
        print("注: ベクトルストアまたは法令データが見つかりません")
        print("     インデックスをビルドしてください:")
        print("     python scripts/build_index.py")
except Exception as e:
    print(f"エラー: {e}")

print()
print("=" * 60)
print("セットアップ完了")
print("=" * 60)
print()
print("次のステップ:")
print()
print("1. データが不足している場合:")
print("   python scripts/build_index.py")
print()
print("2. パイプラインを試す:")
print("   python -c \"")
print("from examples.02_mcp_egov_agent import MCPEgovPipeline, load_config")
print("config = load_config(validate=False)")
print("# retriever が必要です（build_index後）")
print("   \"")
print()
print("3. 評価を実行:")
print("   python examples/02_mcp_egov_agent/evaluate.py \\")
print("     --dataset datasets/lawqa_jp/data/selection.json \\")
print("     --mode api_preferred \\")
print("     --limit 10")
print()
print("4. API接続テスト:")
print("   python examples/02_mcp_egov_agent/tests/test_api_simple.py")
print()

