#!/usr/bin/env python3
"""
selection.jsonのreferencesに含まれる法令IDに対応するmarkdownファイルを抽出するスクリプト

使用方法:
    python scripts/extract_lawqa_md.py --output datasets/lawqa_md
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


def extract_law_id_from_url(url: str) -> str | None:
    """URLから法令IDを抽出する"""
    if url.startswith("https://laws.e-gov.go.jp/law/"):
        return url.replace("https://laws.e-gov.go.jp/law/", "")
    return None


def find_md_files_for_law_id(law_id: str, md_dir: Path) -> list[Path]:
    """法令IDに対応するmarkdownファイルを検索"""
    pattern = f"{law_id}_*.md"
    return list(md_dir.glob(pattern))


def get_latest_version(md_files: list[Path]) -> Path | None:
    """最新版のmarkdownファイルを取得（ファイル名の日付部分で判断）"""
    if not md_files:
        return None
    
    # ファイル名から日付を抽出してソート
    # 形式: 323AC0000000025_20280613_505AC0000000053.md
    def extract_date(path: Path) -> str:
        parts = path.stem.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "00000000"
    
    # 日付でソートして最新を取得
    sorted_files = sorted(md_files, key=extract_date, reverse=True)
    return sorted_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="selection.jsonのreferencesに対応するmarkdownファイルを抽出"
    )
    parser.add_argument(
        "--selection-json",
        type=str,
        default="datasets/lawqa_jp/data/selection.json",
        help="selection.jsonのパス",
    )
    parser.add_argument(
        "--md-dir",
        type=str,
        default="datasets/egov_laws",
        help="markdownファイルのディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/lawqa_md",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="全バージョンを含める（デフォルト: 最新版のみ）",
    )
    args = parser.parse_args()

    selection_path = Path(args.selection_json)
    md_dir = Path(args.md_dir)
    output_dir = Path(args.output)

    # selection.jsonを読み込み
    with open(selection_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 全てのreferencesからユニークな法令IDを抽出
    law_ids = set()
    for sample in data["samples"]:
        refs = sample.get("references", [])
        for ref in refs:
            law_id = extract_law_id_from_url(ref)
            if law_id:
                law_ids.add(law_id)

    print(f"selection.jsonから {len(law_ids)} 個のユニークな法令IDを抽出")
    print(f"法令ID: {sorted(law_ids)}")

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各法令IDに対応するmarkdownを検索してコピー
    copied_files = []
    not_found = []
    law_id_to_files = defaultdict(list)

    for law_id in sorted(law_ids):
        md_files = find_md_files_for_law_id(law_id, md_dir)
        
        if not md_files:
            not_found.append(law_id)
            print(f"  警告: {law_id} に対応するmarkdownが見つかりません")
            continue

        if args.all_versions:
            # 全バージョンをコピー
            for md_file in md_files:
                dest = output_dir / md_file.name
                shutil.copy2(md_file, dest)
                copied_files.append(md_file.name)
                law_id_to_files[law_id].append(md_file.name)
        else:
            # 最新版のみコピー
            latest = get_latest_version(md_files)
            if latest:
                dest = output_dir / latest.name
                shutil.copy2(latest, dest)
                copied_files.append(latest.name)
                law_id_to_files[law_id].append(latest.name)

    # メタデータを保存
    metadata = {
        "source_selection_json": str(selection_path),
        "source_md_dir": str(md_dir),
        "total_law_ids": len(law_ids),
        "copied_files_count": len(copied_files),
        "not_found_count": len(not_found),
        "all_versions": args.all_versions,
        "law_ids": sorted(law_ids),
        "not_found_ids": not_found,
        "law_id_to_files": dict(law_id_to_files),
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n結果:")
    print(f"  コピーしたファイル数: {len(copied_files)}")
    print(f"  見つからなかった法令ID: {len(not_found)}")
    print(f"  出力ディレクトリ: {output_dir}")
    print(f"  メタデータ: {metadata_path}")

    # 法令IDごとのファイルを表示
    print(f"\n法令IDごとのファイル:")
    for law_id, files in sorted(law_id_to_files.items()):
        print(f"  {law_id}:")
        for f in files:
            print(f"    - {f}")


if __name__ == "__main__":
    main()

