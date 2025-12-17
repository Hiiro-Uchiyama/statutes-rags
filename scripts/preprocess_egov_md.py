#!/usr/bin/env python3
"""
e-Gov法令Markdownファイルを前処理してJSONL形式に変換するスクリプト

使用方法:
    python scripts/preprocess_egov_md.py --input-dir datasets/egov_laws --output-file data/egov_laws_md.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm


# 条文番号の正規表現パターン（漢数字対応）
ARTICLE_PATTERN = re.compile(
    r'^-\s*(第[一二三四五六七八九十百千]+条(?:の[一二三四五六七八九十百千]+)?)\s+'
)

# 項番号の正規表現パターン
PARAGRAPH_PATTERN = re.compile(
    r'^-\s*([一二三四五六七八九十]+|[０-９]+|[0-9]+)\s+'
)


def extract_law_id_from_filename(filename: str) -> str:
    """ファイル名から法令IDを抽出"""
    # 形式: 323AC0000000025_20280613_505AC0000000053.md
    parts = filename.replace(".md", "").split("_")
    if parts:
        return parts[0]
    return ""


def extract_article_number(line: str) -> Optional[str]:
    """行から条文番号を抽出（例: "- 第五条 ..." -> "第五条"）"""
    match = ARTICLE_PATTERN.match(line)
    if match:
        return match.group(1)
    return None


def extract_law_info_from_content(content: str) -> Dict[str, str]:
    """Markdownの内容から法令情報を抽出"""
    info = {
        "law_title": "",
        "law_num": "",
    }
    
    # 除外する見出しパターン（行の先頭から一致する場合のみ除外）
    # 注意: 「金融商品取引法第二章の六...」のような法令名は除外しない
    exclude_patterns = [
        "目次",
        "附則",
        "（定義）",
        "（趣旨）",
        "（目的）",
    ]
    # 章/節の見出しパターン（「第X章」「第X節」で始まり、法令名ではないもの）
    chapter_section_pattern = re.compile(r'^第[一二三四五六七八九十百]+[章節款目編](\s|$|　)')
    
    lines = content.split("\n")
    for i, line in enumerate(lines):
        # ## 金融商品取引法 のような見出しから法令名を取得
        if line.startswith("## ") and not info["law_title"]:
            title = line[3:].strip()
            
            # 除外パターンに完全一致する場合はスキップ
            if title in exclude_patterns:
                continue
            
            # 章/節の見出しパターンで始まる場合はスキップ
            # ただし「金融商品取引法第二章の六の規定...」のような法令名は除外しない
            if chapter_section_pattern.match(title):
                continue
            
            # 法令番号のみの行（「（昭和二十三年法律第二十五号）」など）はスキップ
            if title.startswith("（") and title.endswith("）") and "法律第" in title:
                continue
            
            info["law_title"] = title
        
        # ## （昭和二十三年法律第二十五号） のような形式から法令番号を取得
        if "法律第" in line or "政令第" in line or "府令第" in line:
            match = re.search(r'[（(]([^）)]+)[）)]', line)
            if match and not info["law_num"]:
                info["law_num"] = match.group(1)
    
    return info


def chunk_markdown(content: str, law_id: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """Markdownをチャンク化する（条文単位での分割に対応）"""
    chunks = []
    law_info = extract_law_info_from_content(content)
    
    lines = content.split("\n")
    current_chunk = []
    current_section = ""
    current_article = ""
    
    for line in lines:
        # 章/節の見出しを検出（## で始まる行）
        if line.startswith("## "):
            if current_chunk and len("\n".join(current_chunk)) > 50:
                chunks.append({
                    "law_id": law_id,
                    "law_title": law_info["law_title"],
                    "law_num": law_info["law_num"],
                    "section": current_section,
                    "article": current_article,
                    "text": "\n".join(current_chunk).strip()
                })
            current_section = line[3:].strip()
            current_chunk = [line]
            current_article = ""  # セクション変更時に条文をリセット
        
        # ### で始まる条文見出し（従来のパターン）
        elif line.startswith("### "):
            if current_chunk and len("\n".join(current_chunk)) > 50:
                chunks.append({
                    "law_id": law_id,
                    "law_title": law_info["law_title"],
                    "law_num": law_info["law_num"],
                    "section": current_section,
                    "article": current_article,
                    "text": "\n".join(current_chunk).strip()
                })
            current_article = line[4:].strip()
            current_chunk = [line]
        
        # 条文パターンを検出（- 第X条 で始まる行）
        elif (article_num := extract_article_number(line)):
            if current_chunk and len("\n".join(current_chunk)) > 50:
                chunks.append({
                    "law_id": law_id,
                    "law_title": law_info["law_title"],
                    "law_num": law_info["law_num"],
                    "section": current_section,
                    "article": current_article,
                    "text": "\n".join(current_chunk).strip()
                })
            current_article = article_num
            current_chunk = [line]
        
        else:
            current_chunk.append(line)
            
            # チャンクサイズが大きくなったら分割
            if len("\n".join(current_chunk)) > max_chunk_size:
                chunks.append({
                    "law_id": law_id,
                    "law_title": law_info["law_title"],
                    "law_num": law_info["law_num"],
                    "section": current_section,
                    "article": current_article,
                    "text": "\n".join(current_chunk).strip()
                })
                current_chunk = []
    
    # 残りのチャンクを追加
    if current_chunk and len("\n".join(current_chunk)) > 50:
        chunks.append({
            "law_id": law_id,
            "law_title": law_info["law_title"],
            "law_num": law_info["law_num"],
            "section": current_section,
            "article": current_article,
            "text": "\n".join(current_chunk).strip()
        })
    
    return chunks


def chunk_markdown_simple(content: str, law_id: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """シンプルなサイズベースのチャンキング"""
    chunks = []
    law_info = extract_law_info_from_content(content)
    
    # テキストを整形
    text = content.replace("\n\n", "\n").strip()
    
    # オーバーラップ付きでチャンキング
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        if len(chunk_text.strip()) > 50:  # 短すぎるチャンクをスキップ
            chunks.append({
                "law_id": law_id,
                "law_title": law_info["law_title"],
                "law_num": law_info["law_num"],
                "chunk_idx": chunk_idx,
                "text": chunk_text.strip()
            })
            chunk_idx += 1
        
        start = end - overlap
        if start >= len(text) - overlap:
            break
    
    return chunks


def process_markdown_file(md_path: Path, chunk_method: str = "section", chunk_size: int = 500) -> List[Dict[str, Any]]:
    """単一のMarkdownファイルを処理"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        law_id = extract_law_id_from_filename(md_path.name)
        
        if chunk_method == "section":
            return chunk_markdown(content, law_id, max_chunk_size=chunk_size)
        else:
            return chunk_markdown_simple(content, law_id, chunk_size=chunk_size)
    except Exception as e:
        print(f"エラー: {md_path} - {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="e-Gov法令Markdownファイルを前処理してJSONL形式に変換"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Markdownファイルのディレクトリ",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="出力JSONLファイルパス",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理するファイル数の上限",
    )
    parser.add_argument(
        "--chunk-method",
        type=str,
        choices=["section", "simple"],
        default="section",
        help="チャンキング方法（section: セクション/条単位, simple: サイズベース）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="チャンクの最大サイズ（文字数）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    # Markdownファイルを収集
    md_files = sorted(input_dir.glob("*.md"))
    
    if args.limit:
        md_files = md_files[:args.limit]
    
    print(f"処理するファイル数: {len(md_files)}")
    
    # 出力ディレクトリを作成
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for md_path in tqdm(md_files, desc="Processing"):
            chunks = process_markdown_file(md_path, args.chunk_method, args.chunk_size)
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
    
    print(f"\n完了:")
    print(f"  処理したファイル数: {len(md_files)}")
    print(f"  生成したチャンク数: {total_chunks}")
    print(f"  出力ファイル: {output_file}")


if __name__ == "__main__":
    main()

