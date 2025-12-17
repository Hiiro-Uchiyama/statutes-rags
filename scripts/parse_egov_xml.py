#!/usr/bin/env python3
"""
e-Gov法令XMLパーサー

XMLから条文単位でテキストを抽出し、JSONL形式で出力する。
コンテキストと同様の構造化された形式で出力。

使用方法:
    python scripts/parse_egov_xml.py --input-dir datasets/egov_laws_ --output-file data/lawqa_xml_chunks.jsonl --law-ids 323AC0000000025,335AC0000000145
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from xml.etree import ElementTree as ET
from tqdm import tqdm


def extract_text_from_element(element: ET.Element) -> str:
    """
    XML要素からテキストを再帰的に抽出
    """
    texts = []
    
    if element.text:
        texts.append(element.text.strip())
    
    for child in element:
        child_text = extract_text_from_element(child)
        if child_text:
            texts.append(child_text)
        if child.tail:
            texts.append(child.tail.strip())
    
    return ' '.join(filter(None, texts))


def parse_sentence(sentence: ET.Element) -> str:
    """Sentence要素からテキストを抽出"""
    return extract_text_from_element(sentence)


def parse_paragraph(paragraph: ET.Element) -> Dict[str, Any]:
    """
    Paragraph要素をパース
    
    Returns:
        {"num": "1", "text": "...", "items": [...]}
    """
    result = {
        "num": paragraph.get("Num", ""),
        "text": "",
        "items": []
    }
    
    # ParagraphSentence
    for ps in paragraph.findall(".//ParagraphSentence"):
        sentences = []
        for sent in ps.findall("Sentence"):
            sentences.append(parse_sentence(sent))
        result["text"] = " ".join(sentences)
    
    # Items (号)
    for item in paragraph.findall("Item"):
        item_num = item.get("Num", "")
        item_title = ""
        item_sentences = []
        
        for it in item.findall("ItemTitle"):
            item_title = extract_text_from_element(it)
        
        for sent in item.findall(".//ItemSentence/Sentence"):
            item_sentences.append(parse_sentence(sent))
        
        result["items"].append({
            "num": item_num,
            "title": item_title,
            "text": " ".join(item_sentences)
        })
    
    return result


def parse_article(article: ET.Element) -> Dict[str, Any]:
    """
    Article要素をパース
    
    Returns:
        {"num": "1", "title": "...", "caption": "...", "paragraphs": [...]}
    """
    result = {
        "num": article.get("Num", ""),
        "title": "",
        "caption": "",
        "paragraphs": []
    }
    
    # ArticleTitle (条文番号)
    for at in article.findall("ArticleTitle"):
        result["title"] = extract_text_from_element(at)
    
    # ArticleCaption (見出し)
    for ac in article.findall("ArticleCaption"):
        result["caption"] = extract_text_from_element(ac)
    
    # Paragraphs
    for para in article.findall("Paragraph"):
        result["paragraphs"].append(parse_paragraph(para))
    
    return result


def parse_section(section: ET.Element, section_type: str = "Section") -> Dict[str, Any]:
    """
    Section/Chapter等の要素をパース
    """
    result = {
        "type": section_type,
        "num": section.get("Num", ""),
        "title": "",
        "articles": [],
        "subsections": []
    }
    
    # Title
    title_tag = f"{section_type}Title"
    for t in section.findall(title_tag):
        result["title"] = extract_text_from_element(t)
    
    # Articles
    for art in section.findall("Article"):
        result["articles"].append(parse_article(art))
    
    # Subsections
    subsection_types = ["Subsection", "Division"]
    for sub_type in subsection_types:
        for sub in section.findall(sub_type):
            result["subsections"].append(parse_section(sub, sub_type))
    
    return result


def parse_xml_file(xml_path: Path) -> Dict[str, Any]:
    """
    XMLファイルをパースして法令情報を抽出
    
    注意: XMLは複雑なネスト構造を持つため、再帰的に全ての条文を抽出する
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    result = {
        "law_num": "",
        "law_title": "",
        "law_id": xml_path.stem.split("_")[0],
        "chapters": [],
        "articles": []
    }
    
    # LawNum
    law_num = root.find(".//LawNum")
    if law_num is not None:
        result["law_num"] = extract_text_from_element(law_num)
    
    # LawTitle
    law_title = root.find(".//LawTitle")
    if law_title is not None:
        result["law_title"] = extract_text_from_element(law_title)
    
    # 全てのArticleを再帰的に抽出（ネスト構造に関係なく）
    for art in root.findall(".//Article"):
        result["articles"].append(parse_article(art))
    
    return result


def format_article_text(law_title: str, article: Dict[str, Any], law_num: str = "") -> str:
    """
    条文をコンテキストと同様の形式でフォーマット
    
    Format:
    ## 法令名
    ### 第X条
    #### 第Y項
    条文内容...
    ##### 第Z号
    号の内容...
    """
    lines = []
    
    # 法令名
    lines.append(f"## {law_title}")
    
    # 条文番号
    article_title = article.get("title", "") or f"第{article.get('num', '')}条"
    lines.append(f"### {article_title}")
    
    # 見出し（キャプション）
    if article.get("caption"):
        lines.append(f"({article['caption']})")
    
    # 項
    for para in article.get("paragraphs", []):
        para_num = para.get("num", "")
        para_text = para.get("text", "")
        
        if para_num:
            lines.append(f"#### 第{para_num}項")
        
        if para_text:
            lines.append(para_text)
        
        # 号
        for item in para.get("items", []):
            item_num = item.get("num", "")
            item_title = item.get("title", "")
            item_text = item.get("text", "")
            
            if item_num:
                lines.append(f"##### 第{item_num}号")
            if item_title:
                lines.append(item_title)
            if item_text:
                lines.append(item_text)
    
    return "\n".join(lines)


def extract_all_articles(law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    法令データから全ての条文を抽出
    """
    articles = []
    law_title = law_data.get("law_title", "")
    law_num = law_data.get("law_num", "")
    law_id = law_data.get("law_id", "")
    
    # 直接のArticle
    for art in law_data.get("articles", []):
        articles.append({
            "law_id": law_id,
            "law_title": law_title,
            "law_num": law_num,
            "article_num": art.get("num", ""),
            "article_title": art.get("title", ""),
            "article_caption": art.get("caption", ""),
            "article_data": art,
            "formatted_text": format_article_text(law_title, art, law_num)
        })
    
    # Chapter/Section内のArticle
    def extract_from_sections(sections, chapter_title=""):
        for section in sections:
            section_title = section.get("title", "")
            full_title = f"{chapter_title} {section_title}".strip()
            
            for art in section.get("articles", []):
                articles.append({
                    "law_id": law_id,
                    "law_title": law_title,
                    "law_num": law_num,
                    "chapter": full_title,
                    "article_num": art.get("num", ""),
                    "article_title": art.get("title", ""),
                    "article_caption": art.get("caption", ""),
                    "article_data": art,
                    "formatted_text": format_article_text(law_title, art, law_num)
                })
            
            # 再帰的にサブセクションを処理
            extract_from_sections(section.get("subsections", []), full_title)
    
    extract_from_sections(law_data.get("chapters", []))
    
    return articles


def find_latest_xml(law_id: str, base_dir: Path) -> Optional[Path]:
    """
    法令IDに対応する最新のXMLファイルを検索
    """
    matching_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(law_id)]
    
    if not matching_dirs:
        return None
    
    # 日付でソートして最新を取得
    def extract_date(d):
        parts = d.name.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "00000000"
    
    matching_dirs.sort(key=extract_date, reverse=True)
    latest_dir = matching_dirs[0]
    
    # ディレクトリ内のXMLファイルを検索
    xml_files = list(latest_dir.glob("*.xml"))
    if xml_files:
        return xml_files[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(description="e-Gov法令XMLパーサー")
    parser.add_argument("--input-dir", type=str, required=True, help="XMLディレクトリ")
    parser.add_argument("--output-file", type=str, required=True, help="出力JSONLファイル")
    parser.add_argument("--law-ids", type=str, help="処理する法令IDのカンマ区切りリスト")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    # 法令IDのリスト
    if args.law_ids:
        law_ids = [lid.strip() for lid in args.law_ids.split(",")]
    else:
        # 全ディレクトリから法令IDを抽出
        law_ids = list(set(d.name.split("_")[0] for d in input_dir.iterdir() if d.is_dir()))
    
    print(f"処理する法令数: {len(law_ids)}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_articles = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for law_id in tqdm(law_ids, desc="Processing laws"):
            xml_path = find_latest_xml(law_id, input_dir)
            
            if not xml_path:
                print(f"警告: {law_id} のXMLが見つかりません")
                continue
            
            try:
                law_data = parse_xml_file(xml_path)
                articles = extract_all_articles(law_data)
                
                for art in articles:
                    # formatted_text以外のarticle_dataを除去（サイズ削減）
                    output = {
                        "law_id": art["law_id"],
                        "law_title": art["law_title"],
                        "law_num": art["law_num"],
                        "article_num": art["article_num"],
                        "article_title": art["article_title"],
                        "text": art["formatted_text"]
                    }
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")
                    total_articles += 1
                    
            except Exception as e:
                print(f"エラー: {law_id} - {e}")
    
    print(f"\n完了:")
    print(f"  法令数: {len(law_ids)}")
    print(f"  条文数: {total_articles}")
    print(f"  出力: {output_file}")


if __name__ == "__main__":
    main()

