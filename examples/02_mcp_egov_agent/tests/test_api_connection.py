#!/usr/bin/env python3
"""
e-Gov API v2 接続テスト

APIへの実接続を確認し、主要な機能をテストします。
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib
egov_client_module = importlib.import_module('examples.02_mcp_egov_agent.agents.egov_client')
EGovAPIClient = egov_client_module.EGovAPIClient
EGovAPIError = egov_client_module.EGovAPIError


def test_health_check(client: EGovAPIClient):
    """疎通確認テスト"""
    print("=" * 60)
    print("テスト1: API疎通確認")
    print("=" * 60)
    
    try:
        result = client.health_check()
        if result:
            print("結果: OK - APIに正常に接続できました")
        else:
            print("結果: NG - API接続に失敗しました")
        return result
    except Exception as e:
        print(f"結果: エラー - {e}")
        return False


def test_keyword_search(client: EGovAPIClient):
    """キーワード検索テスト"""
    print("\n" + "=" * 60)
    print("テスト2: キーワード検索")
    print("=" * 60)
    
    test_keywords = [
        "個人情報保護",
        "デジタル庁",
        "民法"
    ]
    
    for keyword in test_keywords:
        print(f"\nキーワード: '{keyword}'")
        try:
            result = client.search_by_keyword(keyword)
            laws = result.get("items", [])
            print(f"  結果: {len(laws)}件の法令が見つかりました")
            
            if laws:
                # 最初の1件を表示
                first_law = laws[0]
                law_info = first_law.get("law_info", {})
                print(f"  最初の法令: {law_info.get('law_title', '不明')}")
                print(f"  法令番号: {law_info.get('law_num', '不明')}")
        
        except EGovAPIError as e:
            print(f"  エラー: {e}")
            return False
        except Exception as e:
            print(f"  予期しないエラー: {e}")
            return False
    
    return True


def test_get_law_data(client: EGovAPIClient):
    """法令本文取得テスト"""
    print("\n" + "=" * 60)
    print("テスト3: 法令本文取得")
    print("=" * 60)
    
    # テスト対象の法令番号
    test_law_numbers = [
        "昭和二十二年法律第六十七号",  # 民法
        "平成十五年法律第五十七号",    # 個人情報の保護に関する法律
    ]
    
    for law_num in test_law_numbers:
        print(f"\n法令番号: '{law_num}'")
        try:
            result = client.get_law_data(law_num, response_format="json")
            law_title = result.get("law_title", "不明")
            print(f"  法令名: {law_title}")
            print(f"  取得成功")
        
        except EGovAPIError as e:
            print(f"  エラー: {e}")
            # 一部のエラーは許容（法令番号の形式違いなど）
            if "404" in str(e):
                print(f"  注: 法令が見つからない可能性があります")
            else:
                return False
        except Exception as e:
            print(f"  予期しないエラー: {e}")
            return False
    
    return True


def test_get_laws(client: EGovAPIClient):
    """法令一覧取得テスト"""
    print("\n" + "=" * 60)
    print("テスト4: 法令一覧取得")
    print("=" * 60)
    
    try:
        # 法令種別を指定して一覧取得
        result = client.get_laws(law_type=["Act"])
        laws = result.get("items", [])
        print(f"結果: {len(laws)}件の法律が見つかりました")
        
        if laws:
            # 最初の3件を表示
            for i, law in enumerate(laws[:3], 1):
                law_info = law.get("law_info", {})
                print(f"  [{i}] {law_info.get('law_title', '不明')}")
        
        return True
    
    except EGovAPIError as e:
        print(f"エラー: {e}")
        return False
    except Exception as e:
        print(f"予期しないエラー: {e}")
        return False


def test_error_handling(client: EGovAPIClient):
    """エラーハンドリングテスト"""
    print("\n" + "=" * 60)
    print("テスト5: エラーハンドリング")
    print("=" * 60)
    
    # 不正な法令番号でテスト
    print("\n不正な法令番号でテスト...")
    try:
        result = client.get_law_data("存在しない法令番号12345")
        print("  予期せず成功しました（エラーが発生すべき）")
        return False
    except EGovAPIError as e:
        print(f"  期待通りのエラー: {e}")
        return True
    except Exception as e:
        print(f"  予期しないエラー: {e}")
        return False


def main():
    """メイン処理"""
    print("=" * 60)
    print("e-Gov API v2 接続テスト")
    print("=" * 60)
    print()
    
    # APIクライアントの初期化
    print("APIクライアントを初期化中...")
    client = EGovAPIClient(
        base_url="https://laws.e-gov.go.jp/api/2",
        timeout=30,
        max_retries=3
    )
    print("初期化完了")
    print()
    
    # テストの実行
    tests = [
        ("疎通確認", test_health_check),
        ("キーワード検索", test_keyword_search),
        ("法令本文取得", test_get_law_data),
        ("法令一覧取得", test_get_laws),
        ("エラーハンドリング", test_error_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func(client)
            results[test_name] = result
        except Exception as e:
            print(f"\n{test_name}テストで予期しないエラー: {e}")
            results[test_name] = False
    
    # 結果のサマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "OK" if result else "NG"
        print(f"{test_name}: {status}")
    
    print()
    
    all_passed = all(results.values())
    if all_passed:
        print("全てのテストが成功しました")
        return 0
    else:
        print("一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())

