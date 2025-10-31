"""
XML前処理のテスト
"""
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocess_egov_xml import (
    extract_text_recursive,
    parse_article,
    parse_law_xml
)


@pytest.mark.unit
@pytest.mark.xmlparse
def test_extract_text_recursive(sample_xml_file):
    """テキスト抽出のテスト"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(sample_xml_file)
    root = tree.getroot()
    
    law_title = root.find(".//LawTitle")
    assert law_title is not None
    
    text = extract_text_recursive(law_title)
    assert "博物館法" in text


@pytest.mark.unit
@pytest.mark.xmlparse
def test_parse_law_xml(sample_xml_file):
    """法令XMLパースのテスト"""
    chunks = parse_law_xml(sample_xml_file)
    
    # チャンクが生成されること
    assert len(chunks) > 0
    
    # 各チャンクが必要なフィールドを持つこと
    for chunk in chunks:
        assert "law_title" in chunk
        assert "law_num" in chunk
        assert "article" in chunk
        assert "text" in chunk
    
    # 法令名が正しいこと
    assert chunks[0]["law_title"] == "博物館法"
    
    # 条文番号が正しいこと
    assert chunks[0]["article"] == "1"
    assert chunks[1]["article"] == "2"


@pytest.mark.unit
@pytest.mark.xmlparse
def test_parse_article_structure(sample_xml_file):
    """条文構造のパースのテスト"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(sample_xml_file)
    root = tree.getroot()
    
    articles = root.findall(".//Article")
    assert len(articles) >= 2
    
    # 第1条のパース
    chunks_1 = parse_article(articles[0], "博物館法", "昭和二十六年法律第二百八十五号")
    assert len(chunks_1) > 0
    assert chunks_1[0]["article"] == "1"
    assert chunks_1[0]["article_caption"] == "（目的）"
    
    # 第2条のパース（複数項を持つ）
    chunks_2 = parse_article(articles[1], "博物館法", "昭和二十六年法律第二百八十五号")
    assert len(chunks_2) >= 2  # 2つの項がある


@pytest.mark.integration
@pytest.mark.xmlparse
def test_process_directory_integration(tmp_path, sample_xml_file):
    """ディレクトリ処理の統合テスト"""
    from scripts.preprocess_egov_xml import process_directory
    
    # 入力ディレクトリを作成
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # サンプルXMLをコピー
    import shutil
    xml_dir = input_dir / sample_xml_file.parent.name
    xml_dir.mkdir()
    shutil.copy(sample_xml_file, xml_dir / sample_xml_file.name)
    
    # 出力ファイル
    output_file = tmp_path / "output.jsonl"
    
    # 処理実行
    process_directory(input_dir, output_file)
    
    # 出力ファイルが作成されること
    assert output_file.exists()
    
    # 出力ファイルが読み込めること
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        
        # 各行がJSONとしてパースできること
        for line in lines:
            data = json.loads(line)
            assert "law_title" in data
            assert "text" in data


@pytest.mark.unit
@pytest.mark.xmlparse
def test_empty_xml_handling():
    """空のXMLの処理テスト"""
    empty_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Law>
    <LawNum>テスト</LawNum>
    <LawBody>
        <LawTitle>テスト法</LawTitle>
    </LawBody>
</Law>"""
    
    import xml.etree.ElementTree as ET
    from io import StringIO
    
    tree = ET.parse(StringIO(empty_xml_content))
    root = tree.getroot()
    
    articles = root.findall(".//Article")
    assert len(articles) == 0  # 条文がない


@pytest.mark.unit
@pytest.mark.xmlparse
def test_paragraph_and_item_extraction(sample_xml_file):
    """項と号の抽出テスト"""
    chunks = parse_law_xml(sample_xml_file)
    
    # 第2条の第2項をチェック
    article_2_chunks = [c for c in chunks if c["article"] == "2"]
    assert len(article_2_chunks) >= 2
    
    # 項番号が正しく設定されること
    paragraphs = [c["paragraph"] for c in article_2_chunks if c["paragraph"]]
    assert "1" in paragraphs or "2" in paragraphs
