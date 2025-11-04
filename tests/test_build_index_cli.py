"""scripts/build_index.py の CLI パーサーに関するテスト"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_batch_size_hyphen_argument():
    """--batch-size オプションでバッチサイズを指定できることを確認"""
    import scripts.build_index as build_index

    parser = build_index.build_arg_parser()
    args = parser.parse_args(["--batch-size", "128"])

    assert args.batch_size == 128


def test_batch_size_underscore_argument():
    """--batch_size オプションも後方互換で受け付けることを確認"""
    import scripts.build_index as build_index

    parser = build_index.build_arg_parser()
    args = parser.parse_args(["--batch_size", "64"])

    assert args.batch_size == 64

