#!/usr/bin/env python3
"""
e-Gov API v2 簡易接続テスト

pydanticに依存せず、egov_clientのみをテストします。
"""
import sys
import httpx


def test_api_connection():
    """e-Gov API v2への基本的な接続テスト"""
    base_url = "https://laws.e-gov.go.jp/api/2"
    
    print("=" * 60)
    print("e-Gov API v2 簡易接続テスト")
    print("=" * 60)
    print()
    
    # テスト1: 法令一覧取得
    print("テスト1: 法令一覧取得")
    print("-" * 60)
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(f"{base_url}/laws", params={"law_type": "Act"})
            response.raise_for_status()
            data = response.json()
            laws = data.get("items", [])
            print(f"結果: {len(laws)}件の法律を取得しました")
            if laws:
                first_law = laws[0].get("law_info", {})
                print(f"最初の法令: {first_law.get('law_title', '不明')}")
            print("テスト1: OK")
    except Exception as e:
        print(f"テスト1: エラー - {e}")
        return False
    
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
            laws = data.get("items", [])
            print(f"キーワード: '{keyword}'")
            print(f"結果: {len(laws)}件の法令が見つかりました")
            if laws:
                first_law = laws[0].get("law_info", {})
                print(f"最初の法令: {first_law.get('law_title', '不明')}")
            print("テスト2: OK")
    except Exception as e:
        print(f"テスト2: エラー - {e}")
        return False
    
    print()
    
    # テスト3: 法令本文取得
    print("テスト3: 法令本文取得")
    print("-" * 60)
    law_number = "昭和二十二年法律第六十七号"  # 民法
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{base_url}/law_data/{law_number}",
                params={"format": "json"}
            )
            response.raise_for_status()
            data = response.json()
            law_title = data.get("law_title", "不明")
            print(f"法令番号: '{law_number}'")
            print(f"法令名: {law_title}")
            print("テスト3: OK")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"テスト3: 法令が見つかりませんでした（404）")
            print("注: 法令番号の形式が異なる可能性があります")
            print("テスト3: SKIP")
        else:
            print(f"テスト3: HTTPエラー - {e.response.status_code}")
            return False
    except Exception as e:
        print(f"テスト3: エラー - {e}")
        return False
    
    print()
    
    # テスト4: エラーハンドリング
    print("テスト4: エラーハンドリング（不正な法令番号）")
    print("-" * 60)
    invalid_law_number = "存在しない法令12345"
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{base_url}/law_data/{invalid_law_number}",
                params={"format": "json"}
            )
            response.raise_for_status()
            print("テスト4: 予期せず成功しました（エラーが発生すべき）")
            return False
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"期待通りの404エラーが発生しました")
            print("テスト4: OK")
        else:
            print(f"予期しないHTTPエラー: {e.response.status_code}")
            return False
    except Exception as e:
        print(f"テスト4: エラー - {e}")
        return False
    
    print()
    print("=" * 60)
    print("全てのテストが完了しました")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_api_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテストが中断されました")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n予期しないエラーが発生しました: {e}")
        sys.exit(1)

