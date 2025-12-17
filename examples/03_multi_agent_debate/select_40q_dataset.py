#!/usr/bin/env python3
"""examples/03_multi_agent_debate/select_40q_dataset.py

140問の法令QA（例: datasets/lawqa_jp/data/selection.json）から、評価用に40問を抽出するユーティリティ。

デフォルト方針:
- references の e-Gov URL を law_list.json で法令タイトルにマッピング
- 法令タイトルごとに「最低1問」＋「残りは出現比率に比例」する割当（在庫でクリップ）
- 各法令内では、問題タイプ（正しい/誤り/組み合わせ/その他）の不足分を優先して貪欲に選択
- 乱数シードで再現可能（Pythonの hash() は使わず、SHA256で安定化）

出力:
- --output に指定したJSON（トップレベルは {"samples": [...]} ）
- 標準出力に、割当/実際の分布/選択した元インデックス（0-based）を表示
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ANSWER_LABELS = {"a", "b", "c", "d"}


def _project_root() -> Path:
    # examples/03_multi_agent_debate/ から見て3階層上がプロジェクトルート
    return Path(__file__).resolve().parent.parent.parent


def _resolve_from_root(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_samples(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported dataset format. Expected {'samples': [...]} or a list.")


def _question_text(sample: Dict[str, Any]) -> str:
    return str(sample.get("問題文") or sample.get("question") or "")


def _question_type(question: str) -> str:
    q = question or ""
    if "組み合わせ" in q or "組合せ" in q:
        return "組み合わせ"
    # 「誤っているものを…」等（ざっくり）
    if ("誤" in q) and ("もの" in q or "記述" in q or "選" in q):
        return "誤り選択"
    if "正しい" in q:
        return "正しい選択"
    return "その他"


def _map_law_title(sample: Dict[str, Any], url_to_title: Dict[str, str]) -> Optional[str]:
    refs = sample.get("references") or []
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        return None
    for u in refs:
        if u in url_to_title:
            return url_to_title[u]
    return None


def _validate_sample(sample: Dict[str, Any]) -> bool:
    # 最低限: 問題文/選択肢/正解ラベルがある
    q = _question_text(sample).strip()
    if not q:
        return False
    choices = sample.get("選択肢") or sample.get("choices")
    if not choices:
        return False
    out = (sample.get("output") or sample.get("answer") or "").strip().lower()
    if out not in ANSWER_LABELS:
        return False
    return True


def _stable_int_seed(*parts: str) -> int:
    """文字列部品から安定した 32-bit seed を生成"""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def _stable_shuffle(items: List[int], seed: int) -> List[int]:
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    return items


def _cap_and_redistribute(
    alloc: Dict[str, int], counts: Dict[str, int], target_total: int
) -> Dict[str, int]:
    """在庫(counts)を超えた割当をクリップし、余りを在庫に余裕があるカテゴリへ配り直す。"""

    items = list(alloc.keys())

    # まずクリップして excess を集計
    excess = 0
    for k in items:
        if alloc[k] > counts.get(k, 0):
            excess += alloc[k] - counts[k]
            alloc[k] = counts[k]

    # 余りを再配分
    while excess > 0:
        slack = [k for k in items if alloc[k] < counts.get(k, 0)]
        if not slack:
            break
        for k in slack:
            if excess <= 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                excess -= 1

    # 合計が target_total からズレた場合の微調整（念のため）
    total = sum(alloc.values())
    if total < target_total:
        slack = [k for k in items if alloc[k] < counts.get(k, 0)]
        i = 0
        while total < target_total and slack:
            k = slack[i % len(slack)]
            if alloc[k] < counts[k]:
                alloc[k] += 1
                total += 1
            i += 1
    elif total > target_total:
        over = total - target_total
        for k in sorted(items, key=lambda x: alloc[x], reverse=True):
            if over <= 0:
                break
            take = min(over, alloc[k])
            alloc[k] -= take
            over -= take

    return alloc


def _quota_by_proportion(
    counts: Dict[str, int],
    target_total: int,
    *,
    min_each: int = 1,
) -> Dict[str, int]:
    """最低 min_each を確保しつつ、残りを(在庫-min_each)に比例配分（Largest remainder）。"""

    items = [k for k, v in counts.items() if v > 0]
    if not items:
        return {}

    if target_total <= 0:
        return {k: 0 for k in items}

    base = {k: min_each for k in items}
    remaining = target_total - min_each * len(items)

    if remaining < 0:
        # target_total が小さすぎる場合は多い順で切る
        items_sorted = sorted(items, key=lambda k: counts[k], reverse=True)
        return {k: (1 if i < target_total else 0) for i, k in enumerate(items_sorted)}

    weights = {k: max(counts[k] - min_each, 0) for k in items}
    W = sum(weights.values())

    alloc = dict(base)
    if remaining == 0:
        return _cap_and_redistribute(alloc, counts, target_total)

    if W == 0:
        for i, k in enumerate(items):
            if i < remaining:
                alloc[k] += 1
        return _cap_and_redistribute(alloc, counts, target_total)

    fracs: List[Tuple[float, str]] = []
    used = 0
    for k in items:
        exact = remaining * (weights[k] / W)
        add = int(math.floor(exact))
        alloc[k] += add
        used += add
        fracs.append((exact - add, k))

    leftover = remaining - used
    for _, k in sorted(fracs, reverse=True)[:leftover]:
        alloc[k] += 1

    return _cap_and_redistribute(alloc, counts, target_total)


def select_subset(
    samples: List[Dict[str, Any]],
    url_to_title: Dict[str, str],
    *,
    target_n: int = 40,
    seed: int = 42,
    min_per_law: int = 1,
    require_mapped_law: bool = True,
) -> Tuple[List[int], Dict[str, Any]]:
    """選択する。

    Returns:
        selected_indices: 元データ(samples)に対するインデックス（0-based）
        report: 集計情報
    """

    # 有効データだけに絞る（元インデックスを保持）
    valid_indices: List[int] = []
    skipped_invalid = 0
    skipped_no_law = 0

    for i, s in enumerate(samples):
        if not _validate_sample(s):
            skipped_invalid += 1
            continue
        if require_mapped_law and (_map_law_title(s, url_to_title) is None):
            skipped_no_law += 1
            continue
        valid_indices.append(i)

    # law_title / qtype を付与
    items_all: List[Tuple[int, str, str]] = []
    for i in valid_indices:
        s = samples[i]
        law = _map_law_title(s, url_to_title) or "(no-law)"
        qt = _question_type(_question_text(s))
        items_all.append((i, law, qt))

    # ---- 必須タイプ（例: 「組み合わせ」）を先に確保 ----
    required_indices: List[int] = []
    combo_candidates = [idx for idx, _, qt in items_all if qt == "組み合わせ"]
    if combo_candidates:
        combo_candidates = _stable_shuffle(
            combo_candidates,
            _stable_int_seed(str(seed), "required", "組み合わせ"),
        )
        required_indices.append(combo_candidates[0])

    required_set = set(required_indices)
    items: List[Tuple[int, str, str]] = [t for t in items_all if t[0] not in required_set]
    remaining_target_n = max(target_n - len(required_indices), 0)

    # law 別にプール（required を除いた残りで作る）
    by_law: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for idx, law, qt in items:
        by_law[law].append((idx, qt))

    laws_sorted = sorted(by_law.keys(), key=lambda l: len(by_law[l]), reverse=True)

    # law 割当（required を除いた残り枠に対して）
    law_counts = {l: len(by_law[l]) for l in laws_sorted}
    law_quota = _quota_by_proportion(law_counts, remaining_target_n, min_each=max(min_per_law, 0))

    # type の全体目標（分布に比例）
    type_counts = Counter(qt for _, _, qt in items)
    type_quota = _quota_by_proportion(dict(type_counts), remaining_target_n, min_each=0)
    if type_counts.get("組み合わせ", 0) > 0 and type_quota.get("組み合わせ", 0) == 0:
        type_quota["組み合わせ"] = 1
        dec = max((k for k in type_quota.keys() if k != "組み合わせ"), key=lambda k: type_quota[k], default=None)
        if dec and type_quota[dec] > 0:
            type_quota[dec] -= 1

    # law内の候補順を安定シャッフル
    law_pools: Dict[str, Dict[str, List[int]]] = {}
    for law in laws_sorted:
        pool: Dict[str, List[int]] = defaultdict(list)
        for idx, qt in by_law[law]:
            pool[qt].append(idx)
        for qt in list(pool.keys()):
            s = _stable_int_seed(str(seed), law, qt)
            pool[qt] = _stable_shuffle(pool[qt], s)
        law_pools[law] = dict(pool)

    selected: List[int] = []
    selected_set = set()
    selected_type_counts: Counter = Counter()

    def remaining_need(qt: str) -> int:
        return max(type_quota.get(qt, 0) - selected_type_counts.get(qt, 0), 0)

    def pick_from_law(law: str) -> Optional[int]:
        pool = law_pools.get(law, {})
        qtypes = sorted(pool.keys(), key=lambda t: remaining_need(t), reverse=True)
        for qt in qtypes:
            while pool[qt]:
                cand = pool[qt].pop(0)
                if cand not in selected_set:
                    selected_set.add(cand)
                    selected_type_counts[qt] += 1
                    return cand
        return None

    # 1) law_quota を満たす（残り枠まで）
    for law in laws_sorted:
        q = law_quota.get(law, 0)
        for _ in range(q):
            if len(selected) >= remaining_target_n:
                break
            cand = pick_from_law(law)
            if cand is None:
                break
            selected.append(cand)

    # 2) 足りなければ全lawから補充
    if len(selected) < remaining_target_n:
        i = 0
        guard = 0
        while len(selected) < remaining_target_n and guard < 200000:
            guard += 1
            law = laws_sorted[i % len(laws_sorted)]
            i += 1
            cand = pick_from_law(law)
            if cand is not None:
                selected.append(cand)
            # 進展がなければ打ち切り
            if guard > 5000 and len(selected) < remaining_target_n:
                break

    # required を足して最終化
    selected = sorted(required_indices + selected)

    report = {
        "target_n": target_n,
        "seed": seed,
        "total_samples": len(samples),
        "valid_samples": len(valid_indices),
        "skipped_invalid": skipped_invalid,
        "skipped_no_law": skipped_no_law,
        "law_counts": dict(Counter(law for _, law, _ in items_all)),
        "law_quota": law_quota,
        "type_counts": dict(Counter(qt for _, _, qt in items_all)),
        "type_quota": type_quota,
        "required_indices": required_indices,
        "selected_law_counts": dict(
            Counter(_map_law_title(samples[i], url_to_title) or "(no-law)" for i in selected)
        ),
        "selected_type_counts": dict(
            Counter(_question_type(_question_text(samples[i])) for i in selected)
        ),
        "selected_indices": selected,
    }

    return selected, report


def main() -> int:
    root = _project_root()

    parser = argparse.ArgumentParser(description="140問から40問を抽出（再現可能）")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection.json"),
        help="入力データセット（default: datasets/lawqa_jp/data/selection.json）",
    )
    parser.add_argument(
        "--law-list",
        type=Path,
        default=None,
        help="law_list.json のパス（省略時: input と同ディレクトリの law_list.json）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection_40.json"),
        help="出力データセット（default: datasets/lawqa_jp/data/selection_40.json）",
    )
    parser.add_argument(
        "--output-remaining",
        dest="output_remaining",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection_100.json"),
        help="40問を除いた残り（100問）の出力先（default: datasets/lawqa_jp/data/selection_100.json）",
    )
    parser.add_argument("--n", type=int, default=40, help="抽出する問題数（default: 40）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（default: 42）")
    parser.add_argument(
        "--min-per-law",
        type=int,
        default=1,
        help="法令タイトルごとの最低抽出数（default: 1）",
    )

    args = parser.parse_args()

    input_path = _resolve_from_root(args.input, root)
    output_path = _resolve_from_root(args.output, root)
    output_remaining_path = _resolve_from_root(args.output_remaining, root)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    law_list_path = _resolve_from_root(args.law_list, root) if args.law_list else (input_path.parent / "law_list.json")
    if not law_list_path.exists():
        raise SystemExit(f"law_list.json not found: {law_list_path}")

    dataset = _load_json(input_path)
    samples = _extract_samples(dataset)

    law_list = _load_json(law_list_path)
    if not isinstance(law_list, list):
        raise SystemExit("law_list.json must be a list of {'title','url'} objects")
    url_to_title = {
        d["url"]: d.get("title", "")
        for d in law_list
        if isinstance(d, dict) and "url" in d
    }

    selected_indices, report = select_subset(
        samples,
        url_to_title,
        target_n=args.n,
        seed=args.seed,
        min_per_law=args.min_per_law,
        require_mapped_law=True,
    )

    out_samples = [samples[i] for i in selected_indices]
    out_obj = {"samples": out_samples}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # 残り（除外）100問を出力
    selected_set = set(selected_indices)
    remaining_indices = [i for i in range(len(samples)) if i not in selected_set]
    remaining_obj = {"samples": [samples[i] for i in remaining_indices]}

    output_remaining_path.parent.mkdir(parents=True, exist_ok=True)
    output_remaining_path.write_text(
        json.dumps(remaining_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # レポート表示
    print("=== selection report ===")
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"remaining_output: {output_remaining_path}")
    print(
        "selected: {}/{} (valid={}, skipped_invalid={}, skipped_no_law={})".format(
            len(selected_indices),
            report["total_samples"],
            report["valid_samples"],
            report["skipped_invalid"],
            report["skipped_no_law"],
        )
    )
    print(f"remaining: {len(remaining_indices)}/{len(samples)}")

    print("--- law quota / selected ---")
    for law, quota in sorted(report["law_quota"].items(), key=lambda x: (-x[1], x[0])):
        sel = report["selected_law_counts"].get(law, 0)
        avail = report["law_counts"].get(law, 0)
        print(f"{law}: quota={quota}, selected={sel}, avail={avail}")

    print("--- type quota / selected ---")
    for t, quota in sorted(report["type_quota"].items(), key=lambda x: (-x[1], x[0])):
        sel = report["selected_type_counts"].get(t, 0)
        avail = report["type_counts"].get(t, 0)
        print(f"{t}: quota={quota}, selected={sel}, avail={avail}")

    print("--- selected original indices (0-based) ---")
    print(", ".join(map(str, report["selected_indices"])))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
