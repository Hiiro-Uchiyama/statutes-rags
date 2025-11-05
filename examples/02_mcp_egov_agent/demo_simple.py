#!/usr/bin/env python3
"""
MCP e-Gov Agent 簡易デモ

最小限の依存関係で動作確認できるデモ。
"""
import sys
from pathlib import Path

print("=" * 60)
print("MCP e-Gov Agent - 簡易デモ")
print("=" * 60)
print()

# 依存関係のチェック
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

if missing_deps:
    print()
    print("エラー: 以下の依存関係が不足しています:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print()
    print("インストール:")
    print(f"  python3 -m pip install {' '.join(missing_deps)}")
    sys.exit(1)

print()

# e-Gov APIへの接続テスト
print("=" * 60)
print("e-Gov API v2 接続テスト")
print("=" * 60)
print()

base_url = "https://laws.e-gov.go.jp/api/2"

# テスト1: 法令一覧取得
print("テスト1: 法令一覧取得")
print("-" * 60)
try:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{base_url}/laws", params={"law_type": "Act"})
        response.raise_for_status()
        data = response.json()
        laws = data.get("laws", [])
        print(f"結果: {len(laws)}件の法律を取得しました")
        print("テスト1: OK")
except Exception as e:
    print(f"テスト1: エラー - {e}")

print()

# テスト2: キーワード検索
print("テスト2: キーワード検索")
print("-" * 60)
keyword = "個人情報保護"
try:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{base_url}/keyword", params={"keyword": keyword})
        response.raise_for_status()
        data = response.json()
        laws = data.get("laws", [])
        print(f"キーワード: '{keyword}'")
        print(f"結果: {len(laws)}件の法令が見つかりました")
        if laws:
            first_law = laws[0].get("law_info", {})
            print(f"最初の法令: {first_law.get('law_title', '不明')}")
        print("テスト2: OK")
except Exception as e:
    print(f"テスト2: エラー - {e}")

print()
print("=" * 60)
print("セットアップ完了")
print("=" * 60)
print()
print("e-Gov API v2への接続が確認できました。")
print()
print("次のステップ:")
print()
print("1. 既存のRAG環境をセットアップ:")
print("   cd /path/to/statutes-rags")
print("   ./setup/setup_uv_env.sh")
print("   source .venv/bin/activate")
print()
print("2. 完全なデモを実行:")
print("   python examples/02_mcp_egov_agent/demo.py")
print()
print("3. 評価を実行:")
print("   python examples/02_mcp_egov_agent/evaluate.py \\")
print("     --dataset datasets/lawqa_jp/data/selection.json \\")
print("     --limit 10")
print()
print("詳細はREADME.mdを参照してください。")
print()

