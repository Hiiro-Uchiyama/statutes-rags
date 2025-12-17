"""
日本語法令における数字表記の正規化ユーティリティ

アラビア数字 ↔ 漢数字の相互変換を行う
- クエリ前処理: 「第21条」→「第二十一条」
- 検索用: 両方の表記でマッチング可能に
"""
import re
from typing import Tuple, List


# アラビア数字 → 漢数字変換テーブル
ARABIC_TO_KANJI = {
    '0': '〇', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}

# 漢数字 → アラビア数字変換テーブル
KANJI_TO_ARABIC = {
    '〇': '0', '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
    '十': '10', '百': '100', '千': '1000'
}


def arabic_to_kanji_number(num: int) -> str:
    """
    アラビア数字を漢数字に変換
    
    Args:
        num: 変換する数字（0-9999）
    
    Returns:
        漢数字表記
    
    Examples:
        21 → 二十一
        164 → 百六十四
        1234 → 千二百三十四
    """
    if num == 0:
        return '〇'
    
    result = ''
    
    # 千の位
    if num >= 1000:
        thousands = num // 1000
        if thousands == 1:
            result += '千'
        else:
            result += ARABIC_TO_KANJI[str(thousands)] + '千'
        num %= 1000
    
    # 百の位
    if num >= 100:
        hundreds = num // 100
        if hundreds == 1:
            result += '百'
        else:
            result += ARABIC_TO_KANJI[str(hundreds)] + '百'
        num %= 100
    
    # 十の位
    if num >= 10:
        tens = num // 10
        if tens == 1:
            result += '十'
        else:
            result += ARABIC_TO_KANJI[str(tens)] + '十'
        num %= 10
    
    # 一の位
    if num > 0:
        result += ARABIC_TO_KANJI[str(num)]
    
    return result


def kanji_to_arabic_number(kanji: str) -> int:
    """
    漢数字をアラビア数字に変換
    
    Args:
        kanji: 漢数字表記
    
    Returns:
        数値
    
    Examples:
        二十一 → 21
        百六十四 → 164
    """
    result = 0
    current = 0
    
    for char in kanji:
        if char == '千':
            if current == 0:
                current = 1
            result += current * 1000
            current = 0
        elif char == '百':
            if current == 0:
                current = 1
            result += current * 100
            current = 0
        elif char == '十':
            if current == 0:
                current = 1
            result += current * 10
            current = 0
        elif char in KANJI_TO_ARABIC:
            val = KANJI_TO_ARABIC[char]
            if val.isdigit():
                current = int(val)
    
    result += current
    return result


def normalize_article_numbers(text: str, to_kanji: bool = True) -> str:
    """
    テキスト内の条文番号を正規化
    
    Args:
        text: 入力テキスト
        to_kanji: True=漢数字に変換, False=アラビア数字に変換
    
    Returns:
        正規化されたテキスト
    
    Examples:
        「第21条」→「第二十一条」(to_kanji=True)
        「第二十一条」→「第21条」(to_kanji=False)
    """
    if to_kanji:
        # アラビア数字 → 漢数字
        # 第X条、第X項、第X号のパターン
        patterns = [
            (r'第(\d+)条', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}条'),
            (r'第(\d+)項', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}項'),
            (r'第(\d+)号', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}号'),
            (r'第(\d+)款', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}款'),
            (r'第(\d+)編', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}編'),
            (r'第(\d+)章', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}章'),
            (r'第(\d+)節', lambda m: f'第{arabic_to_kanji_number(int(m.group(1)))}節'),
        ]
        
        for pattern, replacer in patterns:
            text = re.sub(pattern, replacer, text)
        
        # 「の38」のような接続も変換
        text = re.sub(r'の(\d+)', lambda m: f'の{arabic_to_kanji_number(int(m.group(1)))}', text)
        
    else:
        # 漢数字 → アラビア数字
        # 漢数字のパターン
        kanji_pattern = r'第([一二三四五六七八九十百千〇]+)([条項号款編章節])'
        
        def kanji_replacer(m):
            kanji_num = m.group(1)
            suffix = m.group(2)
            arabic_num = kanji_to_arabic_number(kanji_num)
            return f'第{arabic_num}{suffix}'
        
        text = re.sub(kanji_pattern, kanji_replacer, text)
        
        # 「の三十八」のような接続も変換
        text = re.sub(r'の([一二三四五六七八九十百千〇]+)', 
                     lambda m: f'の{kanji_to_arabic_number(m.group(1))}', text)
    
    return text


def get_both_notations(text: str) -> Tuple[str, str]:
    """
    テキストの両方の数字表記を取得
    
    Args:
        text: 入力テキスト
    
    Returns:
        (漢数字版, アラビア数字版) のタプル
    """
    kanji_version = normalize_article_numbers(text, to_kanji=True)
    arabic_version = normalize_article_numbers(text, to_kanji=False)
    return kanji_version, arabic_version


def extract_article_references(text: str) -> List[str]:
    """
    テキストから条文参照を抽出
    
    Args:
        text: 入力テキスト
    
    Returns:
        条文参照のリスト（漢数字表記）
    
    Examples:
        「第21条により」→ [「第二十一条」]
    """
    # まず漢数字に正規化
    normalized = normalize_article_numbers(text, to_kanji=True)
    
    # 条文参照を抽出
    pattern = r'第[一二三四五六七八九十百千〇]+(?:の[一二三四五六七八九十百千〇]+)?条(?:の[一二三四五六七八九十百千〇]+)?'
    matches = re.findall(pattern, normalized)
    
    return list(set(matches))


# テスト
if __name__ == '__main__':
    test_cases = [
        "金融商品取引法第21条により、損害賠償責任",
        "第24条の規定",
        "第164条に基づく",
        "第27条の38の規定",
        "第1項第2号",
    ]
    
    print("=== 数字正規化テスト ===")
    for text in test_cases:
        normalized = normalize_article_numbers(text, to_kanji=True)
        print(f"変換前: {text}")
        print(f"変換後: {normalized}")
        print()
    
    print("=== 条文参照抽出テスト ===")
    for text in test_cases:
        refs = extract_article_references(text)
        print(f"入力: {text}")
        print(f"抽出: {refs}")
        print()

