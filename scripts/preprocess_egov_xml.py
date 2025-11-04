#!/usr/bin/env python3
"""
e-Gov法令XMLをJSONL形式に変換するスクリプト
法令名/条/項/号/本文を構造化して抽出
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from tqdm import tqdm


def extract_text_recursive(element) -> str:
    """XMLエレメントからテキストを再帰的に抽出"""
    text_parts = []
    if element.text:
        text_parts.append(element.text.strip())
    for child in element:
        text_parts.append(extract_text_recursive(child))
        if child.tail:
            text_parts.append(child.tail.strip())
    return " ".join(filter(None, text_parts))


def parse_article(article_elem, law_title: str, law_num: str) -> List[Dict[str, Any]]:
    """条文をパースして構造化データを抽出"""
    chunks = []
    
    article_num = article_elem.get("Num", "")
    article_caption = ""
    
    caption_elem = article_elem.find(".//ArticleCaption")
    if caption_elem is not None:
        article_caption = extract_text_recursive(caption_elem)
    
    article_title_elem = article_elem.find(".//ArticleTitle")
    article_title = extract_text_recursive(article_title_elem) if article_title_elem is not None else f"第{article_num}条"
    
    paragraphs = article_elem.findall(".//Paragraph")
    
    if not paragraphs:
        text = extract_text_recursive(article_elem)
        if text:
            chunks.append({
                "law_title": law_title,
                "law_num": law_num,
                "article": article_num,
                "article_caption": article_caption,
                "article_title": article_title,
                "paragraph": None,
                "item": None,
                "text": text
            })
        return chunks
    
    for para in paragraphs:
        para_num = para.get("Num", "")
        
        items = para.findall(".//Item")
        
        if items:
            for item in items:
                item_num = item.get("Num", "")
                item_text = extract_text_recursive(item)
                
                if item_text:
                    chunks.append({
                        "law_title": law_title,
                        "law_num": law_num,
                        "article": article_num,
                        "article_caption": article_caption,
                        "article_title": article_title,
                        "paragraph": para_num if para_num else None,
                        "item": item_num,
                        "text": item_text
                    })
        else:
            para_text = extract_text_recursive(para)
            if para_text:
                chunks.append({
                    "law_title": law_title,
                    "law_num": law_num,
                    "article": article_num,
                    "article_caption": article_caption,
                    "article_title": article_title,
                    "paragraph": para_num if para_num else None,
                    "item": None,
                    "text": para_text
                })
    
    return chunks


def parse_law_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """単一の法令XMLファイルをパースしてチャンクのリストを返す"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        law_num_elem = root.find(".//LawNum")
        law_num = extract_text_recursive(law_num_elem) if law_num_elem is not None else ""
        
        law_title_elem = root.find(".//LawTitle")
        law_title = extract_text_recursive(law_title_elem) if law_title_elem is not None else ""
        
        if not law_title:
            law_title = xml_path.stem
        
        chunks = []
        
        articles = root.findall(".//Article")
        for article in articles:
            chunks.extend(parse_article(article, law_title, law_num))
        
        return chunks
    
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []


def process_directory(input_dir: Path, output_file: Path, limit: int = None):
    """ディレクトリ内の全XMLファイルを処理してJSONLファイルに出力"""
    xml_files = list(input_dir.rglob("*.xml"))
    
    if limit:
        xml_files = xml_files[:limit]
    
    print(f"Found {len(xml_files)} XML files to process")
    
    total_chunks = 0
    
    with output_file.open("w", encoding="utf-8") as out_f:
        for xml_path in tqdm(xml_files, desc="Processing XML files"):
            chunks = parse_law_xml(xml_path)
            for chunk in chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
    
    print(f"Processed {len(xml_files)} files, generated {total_chunks} chunks")
    print(f"Output saved to: {output_file}")


def main():
    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description="e-Gov法令XMLをJSONLに変換")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "datasets" / "egov_laws",
        help="入力XMLディレクトリ"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=project_root / "data" / "egov_laws.jsonl",
        help="出力JSONLファイル"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理するファイル数の上限"
    )
    
    args = parser.parse_args()
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    process_directory(args.input_dir, args.output_file, args.limit)


if __name__ == "__main__":
    main()
