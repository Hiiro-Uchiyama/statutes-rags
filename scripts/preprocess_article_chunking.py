#!/usr/bin/env python3
"""
条文単位の精密チャンキングスクリプト

コンテキストと同様の形式で条文を抽出:
## 法令名
### 第X条
#### 第Y項
条文内容...

使用方法:
    python scripts/preprocess_article_chunking.py --input-dir datasets/lawqa_md --output-file data/lawqa_article_chunks.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm


# 条文パターン（漢数字）
ARTICLE_PATTERN = re.compile(
    r'第([一二三四五六七八九十百千〇]+(?:の[一二三四五六七八九十百千〇]+)?)条'
)

# 項パターン
PARAGRAPH_PATTERN = re.compile(
    r'^-?\s*([一二三四五六七八九十０-９0-9]+)\s+'
)

# 号パターン
ITEM_PATTERN = re.compile(
    r'^-?\s*第?([一二三四五六七八九十０-９0-9]+)号\s*'
)


def extract_law_info(content: str) -> Dict[str, str]:
    """法令の基本情報を抽出"""
    info = {
        "law_title": "",
        "law_num": "",
    }
    
    lines = content.split("\n")
    for line in lines:
        # ## 金融商品取引法 のような見出しから法令名を取得
        if line.startswith("## ") and not info["law_title"]:
            title = line[3:].strip()
            # 除外パターン
            if title in ["目次", "附則", "（定義）", "（趣旨）", "（目的）"]:
                continue
            # 章/節見出しを除外
            if re.match(r'^第[一二三四五六七八九十百]+[章節款目編]', title):
                continue
            info["law_title"] = title
        
        # 法令番号を取得
        if "法律第" in line or "政令第" in line or "府令第" in line:
            match = re.search(r'[（(]([^）)]+)[）)]', line)
            if match and not info["law_num"]:
                info["law_num"] = match.group(1)
    
    return info


def find_article_boundaries(content: str) -> List[Tuple[int, str]]:
    """
    条文の境界位置と条番号を検出
    
    Returns:
        [(position, article_number), ...]
    """
    boundaries = []
    
    # パターン1: "- 第X条" (行頭)
    pattern1 = re.compile(r'^-\s*(第[一二三四五六七八九十百千〇]+(?:の[一二三四五六七八九十百千〇]+)?条)', re.MULTILINE)
    
    # パターン2: "### 第X条" (マークダウン見出し)
    pattern2 = re.compile(r'^###\s*(第[一二三四五六七八九十百千〇]+(?:の[一二三四五六七八九十百千〇]+)?条)', re.MULTILINE)
    
    # パターン3: "（見出し）\n- 第X条"
    pattern3 = re.compile(r'（[^）]+）\s*\n-\s*(第[一二三四五六七八九十百千〇]+(?:の[一二三四五六七八九十百千〇]+)?条)', re.MULTILINE)
    
    for pattern in [pattern1, pattern2, pattern3]:
        for match in pattern.finditer(content):
            boundaries.append((match.start(), match.group(1)))
    
    # 位置でソート、重複除去
    boundaries = sorted(set(boundaries), key=lambda x: x[0])
    
    return boundaries


def extract_article_content(content: str, start_pos: int, end_pos: int) -> str:
    """条文の内容を抽出し、整形する"""
    text = content[start_pos:end_pos].strip()
    
    # 次の条文の見出しを除去
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # 次の条文見出しが始まったら停止
        if re.match(r'^-\s*第[一二三四五六七八九十百千〇]+(?:の[一二三四五六七八九十百千〇]+)?条\s', line):
            if clean_lines:  # 最初の条文は含める
                break
        clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()


def parse_article_structure(article_text: str) -> Dict[str, Any]:
    """
    条文の階層構造を解析
    
    Returns:
        {
            "paragraphs": [
                {"num": "1", "text": "...", "items": [{"num": "1", "text": "..."}, ...]},
                ...
            ]
        }
    """
    structure = {"paragraphs": []}
    
    lines = article_text.split('\n')
    current_paragraph = None
    current_items = []
    
    for line in lines:
        # 項の検出
        para_match = re.match(r'^-?\s*([０-９0-9]+|[一二三四五六七八九十]+)\s+(.+)', line)
        if para_match:
            if current_paragraph:
                current_paragraph["items"] = current_items
                structure["paragraphs"].append(current_paragraph)
            current_paragraph = {
                "num": para_match.group(1),
                "text": para_match.group(2)
            }
            current_items = []
            continue
        
        # 号の検出
        item_match = re.match(r'^-?\s*([一二三四五六七八九十]+|[０-９0-9]+)\s*号?\s*(.+)', line)
        if item_match and current_paragraph:
            current_items.append({
                "num": item_match.group(1),
                "text": item_match.group(2)
            })
    
    if current_paragraph:
        current_paragraph["items"] = current_items
        structure["paragraphs"].append(current_paragraph)
    
    return structure


def chunk_by_article(content: str, law_id: str, min_chunk_size: int = 100) -> List[Dict[str, Any]]:
    """
    条文単位でチャンキング
    
    各チャンクはコンテキストと同様の形式:
    ## 法令名
    ### 第X条
    条文内容...
    
    注意: 短すぎるチャンク（見出しのみ等）は次のチャンクとマージ
    """
    chunks = []
    law_info = extract_law_info(content)
    
    # 条文境界を検出
    boundaries = find_article_boundaries(content)
    
    if not boundaries:
        # 条文が見つからない場合は全体を1チャンクに
        chunks.append({
            "law_id": law_id,
            "law_title": law_info["law_title"],
            "law_num": law_info["law_num"],
            "article": "",
            "text": content.strip(),
            "chunk_type": "full"
        })
        return chunks
    
    # 各条文をチャンクとして抽出
    pending_text = ""  # 短いチャンクを蓄積
    pending_article = ""
    
    for i, (start_pos, article_num) in enumerate(boundaries):
        # 次の条文の開始位置（または終端）
        if i + 1 < len(boundaries):
            end_pos = boundaries[i + 1][0]
        else:
            end_pos = len(content)
        
        article_text = extract_article_content(content, start_pos, end_pos)
        
        # 短すぎるチャンクは蓄積して次にマージ
        if len(article_text) < min_chunk_size:
            if pending_text:
                pending_text += "\n" + article_text
            else:
                pending_text = article_text
                pending_article = article_num
            continue
        
        # 蓄積されたテキストがあれば先頭に追加
        if pending_text:
            article_text = pending_text + "\n" + article_text
            if not pending_article:
                pending_article = article_num
            pending_text = ""
        
        # フォーマット済みテキストを作成
        formatted_text = f"## {law_info['law_title']}\n### {article_num}\n{article_text}"
        
        chunks.append({
            "law_id": law_id,
            "law_title": law_info["law_title"],
            "law_num": law_info["law_num"],
            "article": article_num,
            "text": article_text,
            "formatted_text": formatted_text,
            "chunk_type": "article"
        })
        pending_article = ""
    
    # 残りの蓄積テキストを最後のチャンクに追加
    if pending_text and chunks:
        chunks[-1]["text"] += "\n" + pending_text
        chunks[-1]["formatted_text"] += "\n" + pending_text
    elif pending_text:
        # チャンクがない場合は新規作成
        chunks.append({
            "law_id": law_id,
            "law_title": law_info["law_title"],
            "law_num": law_info["law_num"],
            "article": pending_article or "",
            "text": pending_text,
            "formatted_text": f"## {law_info['law_title']}\n{pending_text}",
            "chunk_type": "article"
        })
    
    return chunks


def chunk_by_article_paragraph(content: str, law_id: str) -> List[Dict[str, Any]]:
    """
    条+項単位でチャンキング（より細かい粒度）
    """
    chunks = []
    law_info = extract_law_info(content)
    
    # まず条単位で分割
    article_chunks = chunk_by_article(content, law_id)
    
    for article_chunk in article_chunks:
        if article_chunk.get("chunk_type") == "full":
            chunks.append(article_chunk)
            continue
        
        article_text = article_chunk["text"]
        article_num = article_chunk["article"]
        
        # 項に分割
        # パターン: "- 1 ", "- 一 ", "#### 第1項"
        para_pattern = re.compile(
            r'^(?:-\s*|####\s*第?)([一二三四五六七八九十０-９0-9]+)(?:項|\s)',
            re.MULTILINE
        )
        
        para_matches = list(para_pattern.finditer(article_text))
        
        if len(para_matches) <= 1:
            # 項が1つ以下なら条単位のまま
            chunks.append(article_chunk)
        else:
            # 項ごとに分割
            for i, match in enumerate(para_matches):
                para_num = match.group(1)
                start = match.start()
                end = para_matches[i + 1].start() if i + 1 < len(para_matches) else len(article_text)
                
                para_text = article_text[start:end].strip()
                
                if len(para_text) < 20:
                    continue
                
                formatted_text = f"## {law_info['law_title']}\n### {article_num}\n#### 第{para_num}項\n{para_text}"
                
                chunks.append({
                    "law_id": law_id,
                    "law_title": law_info["law_title"],
                    "law_num": law_info["law_num"],
                    "article": article_num,
                    "paragraph": para_num,
                    "text": para_text,
                    "formatted_text": formatted_text,
                    "chunk_type": "paragraph"
                })
    
    return chunks


def process_file(md_path: Path, chunk_level: str = "article") -> List[Dict[str, Any]]:
    """単一ファイルを処理"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # ファイル名から法令IDを抽出
        law_id = md_path.stem.split("_")[0]
        
        if chunk_level == "paragraph":
            return chunk_by_article_paragraph(content, law_id)
        else:
            return chunk_by_article(content, law_id)
            
    except Exception as e:
        print(f"Error processing {md_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="条文単位の精密チャンキング")
    parser.add_argument("--input-dir", type=str, required=True, help="入力ディレクトリ")
    parser.add_argument("--output-file", type=str, required=True, help="出力JSONLファイル")
    parser.add_argument("--chunk-level", choices=["article", "paragraph"], default="article",
                       help="チャンク粒度: article=条単位, paragraph=条+項単位")
    parser.add_argument("--limit", type=int, default=None, help="処理ファイル数の上限")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    md_files = sorted(input_dir.glob("*.md"))
    if args.limit:
        md_files = md_files[:args.limit]
    
    print(f"処理ファイル数: {len(md_files)}")
    print(f"チャンク粒度: {args.chunk_level}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for md_path in tqdm(md_files, desc="Chunking"):
            chunks = process_file(md_path, args.chunk_level)
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
    
    print(f"\n完了:")
    print(f"  処理ファイル数: {len(md_files)}")
    print(f"  生成チャンク数: {total_chunks}")
    print(f"  出力: {output_file}")


if __name__ == "__main__":
    main()

