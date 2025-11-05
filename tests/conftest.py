"""
pytest設定とフィクスチャ
"""
import sys
from pathlib import Path
import json
import tempfile
import shutil

import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root_path():
    """プロジェクトルートのパス"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root_path):
    """テストデータディレクトリ"""
    return project_root_path / "tests" / "test_data"


@pytest.fixture
def sample_xml_content():
    """サンプルXMLコンテンツ"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Law Era="Showa" Lang="ja" LawType="Act" Num="285" Year="26">
    <LawNum>昭和二十六年法律第二百八十五号</LawNum>
    <LawBody>
        <LawTitle Kana="はくぶつかんほう">博物館法</LawTitle>
        <MainProvision>
            <Chapter Num="1">
                <ChapterTitle>第一章　総則</ChapterTitle>
                <Article Num="1">
                    <ArticleCaption>（目的）</ArticleCaption>
                    <ArticleTitle>第一条</ArticleTitle>
                    <Paragraph Num="1">
                        <ParagraphNum/>
                        <ParagraphSentence>
                            <Sentence>この法律は、博物館の設置及び運営に関して必要な事項を定める。</Sentence>
                        </ParagraphSentence>
                    </Paragraph>
                </Article>
                <Article Num="2">
                    <ArticleCaption>（定義）</ArticleCaption>
                    <ArticleTitle>第二条</ArticleTitle>
                    <Paragraph Num="1">
                        <ParagraphNum/>
                        <ParagraphSentence>
                            <Sentence>この法律において「博物館」とは、資料を収集し、保管し、展示する機関をいう。</Sentence>
                        </ParagraphSentence>
                    </Paragraph>
                    <Paragraph Num="2">
                        <ParagraphNum>２</ParagraphNum>
                        <ParagraphSentence>
                            <Sentence>前項の資料は、歴史、芸術、民俗、産業、自然科学等に関するものとする。</Sentence>
                        </ParagraphSentence>
                    </Paragraph>
                </Article>
            </Chapter>
        </MainProvision>
    </LawBody>
</Law>"""


@pytest.fixture
def sample_xml_file(tmp_path, sample_xml_content):
    """サンプルXMLファイル"""
    xml_dir = tmp_path / "test_law_123"
    xml_dir.mkdir()
    xml_file = xml_dir / "test_law_123.xml"
    xml_file.write_text(sample_xml_content, encoding="utf-8")
    return xml_file


@pytest.fixture
def sample_jsonl_data():
    """サンプルJSONLデータ"""
    return [
        {
            "law_title": "博物館法",
            "law_num": "昭和二十六年法律第二百八十五号",
            "article": "1",
            "article_caption": "（目的）",
            "article_title": "第一条",
            "paragraph": "1",
            "item": None,
            "text": "この法律は、博物館の設置及び運営に関して必要な事項を定める。"
        },
        {
            "law_title": "博物館法",
            "law_num": "昭和二十六年法律第二百八十五号",
            "article": "2",
            "article_caption": "（定義）",
            "article_title": "第二条",
            "paragraph": "1",
            "item": None,
            "text": "この法律において「博物館」とは、資料を収集し、保管し、展示する機関をいう。"
        },
        {
            "law_title": "個人情報保護法",
            "law_num": "平成十五年法律第五十七号",
            "article": "27",
            "article_caption": "（第三者提供の制限）",
            "article_title": "第二十七条",
            "paragraph": "1",
            "item": None,
            "text": "個人情報取扱事業者は、次に掲げる場合を除くほか、あらかじめ本人の同意を得ないで、個人データを第三者に提供してはならない。"
        }
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path, sample_jsonl_data):
    """サンプルJSONLファイル"""
    jsonl_file = tmp_path / "test_data.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in sample_jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return jsonl_file


@pytest.fixture
def temp_index_dir(tmp_path):
    """一時インデックスディレクトリ"""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    yield index_dir
    # クリーンアップ
    if index_dir.exists():
        shutil.rmtree(index_dir)


@pytest.fixture
def mock_config():
    """モック設定"""
    from app.core.rag_config import RAGConfig, EmbeddingConfig, LLMConfig, RetrieverConfig, RerankerConfig
    
    return RAGConfig(
        embedding=EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 小さいモデルでテスト
            dimension=384
        ),
        llm=LLMConfig(
            provider="ollama",
            model_name="qwen3:8b",
            temperature=0.1,
            max_tokens=512
        ),
        retriever=RetrieverConfig(
            retriever_type="vector",
            top_k=3,
            use_mmr=False,
            mmr_lambda=0.5,
            bm25_tokenizer="simple"  # テスト用に確実に動作するトークナイザーを指定
        ),
        reranker=RerankerConfig(
            enabled=False,
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            top_n=2
        ),
        vector_store_path="/tmp/test_index",
        data_path="/tmp/test_data.jsonl"
    )


@pytest.fixture
def sample_lawqa_data():
    """サンプルlawqa_jpデータ"""
    return [
        {
            "question": "博物館法の目的は何ですか？",
            "answer": "博物館の設置及び運営に関して必要な事項を定めることです。",
            "law_references": ["博物館法第1条"]
        },
        {
            "question": "個人情報の第三者提供には何が必要ですか？",
            "answer": "原則として、あらかじめ本人の同意が必要です。",
            "law_references": ["個人情報保護法第27条"]
        }
    ]
