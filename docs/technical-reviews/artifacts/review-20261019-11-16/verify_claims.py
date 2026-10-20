"""以最短路徑與直接算例核對文字距離，不量測模型品質。"""

import ast
import hashlib
import heapq
import itertools
import json
import platform
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.natural_concepts import edit_distance, text_error_report


def shortest_path(reference, prediction):
    """在前綴配對的有向圖上，以 Dijkstra 搜尋最低編輯成本。"""
    start = (0, 0)
    goal = (len(reference), len(prediction))
    costs = {start: 0}
    parents = {}
    frontier = [(0, *start)]
    while frontier:
        cost, i, j = heapq.heappop(frontier)
        if cost != costs[(i, j)]:
            continue
        if (i, j) == goal:
            path = []
            node = goal
            while node != start:
                previous, action = parents[node]
                path.append({"from": list(previous), "to": list(node), "action": action})
                node = previous
            return cost, path[::-1]
        edges = []
        if i < len(reference):
            edges.append(((i + 1, j), 1, "補入正確答案的字"))
        if j < len(prediction):
            edges.append(((i, j + 1), 1, "刪除預測的字"))
        if i < len(reference) and j < len(prediction):
            different = reference[i] != prediction[j]
            edges.append(((i + 1, j + 1), int(different), "替換" if different else "保留"))
        for node, increment, action in edges:
            candidate = cost + increment
            if candidate < costs.get(node, float("inf")):
                costs[node] = candidate
                parents[node] = ((i, j), action)
                heapq.heappush(frontier, (candidate, *node))
    raise AssertionError("前綴圖終點不可達")


def main():
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/11.16.md"
    text = frozen.read_text()
    chapter = (ROOT / "course/chapters/11.md").read_bytes()
    section = re.search(rb"^## 11\.16 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
    assert section == frozen.read_bytes(), "目前小節必須與凍結文字相同"
    fences = re.findall(r"```python\n(.*?)```", text, re.S)
    implementation = ast.parse((ROOT / "tiny_perceptron/natural_concepts.py").read_text())
    function = next(n for n in implementation.body if isinstance(n, ast.FunctionDef) and n.name == "edit_distance")
    excerpt = ast.parse(fences[1]).body[0]
    assert ast.dump(function) == ast.dump(excerpt), "教材節錄必須與實作相同"
    namespace = {}
    exec(compile(ast.Module(body=[excerpt], type_ignores=[]), str(frozen), "exec"), namespace)
    without_docstring = ast.parse(fences[1]).body[0]
    without_docstring.body.pop(0)
    plain_namespace = {}
    exec(compile(ast.Module(body=[without_docstring], type_ignores=[]), str(frozen), "exec"), plain_namespace)

    cases = [
        ("今天去臺北", "今天去台北", 1, 0.2),
        ("1010", "10100", 1, 0.25),
        ("1010", "010", 1, 0.25),
        ("今天去臺北", "今天去臺", 1, 0.2),
        ("今天去臺北", "今天去台", 2, 0.4),
        ("今日休館明日開放", "明日開放今日休館", 6, 0.75),
        ("ab", "ba", 2, 1.0),
        ("", "", 0, None),
        ("", "甲", 1, None),
        ("甲", "", 1, 1.0),
    ]
    records = []
    for reference, prediction, expected_edits, expected_cer in cases:
        report = text_error_report(reference, prediction)
        distance, path = shortest_path(reference, prediction)
        assert report["edits"] == distance == expected_edits, "距離必須符合最短路徑與預期"
        assert report["cer"] == expected_cer, "CER 必須符合正確答案分母"
        assert report["exact"] == (reference == prediction), "完整比對必須符合字串相等"
        records.append({"report": report, "shortest_path_cost": distance, "optimal_path": path})

    strings = ["".join(x) for size in range(5) for x in itertools.product("01", repeat=size)]
    comparisons = 0
    for reference in strings:
        for prediction in strings:
            oracle = shortest_path(reference, prediction)[0]
            assert edit_distance(reference, prediction) == namespace["edit_distance"](reference, prediction) == plain_namespace["edit_distance"](reference, prediction) == oracle, "實作、節錄與獨立最短路徑必須一致"
            comparisons += 1

    reference, prediction = cases[5][:2]
    matches = [i for i, (a, b) in enumerate(zip(reference, prediction), 1) if a == b]
    raw = text_error_report("今天去臺北", "今天去台北")
    normalized = text_error_report("今天去臺北".replace("臺", "台"), "今天去台北".replace("臺", "台"))
    receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_text())
    render = next(item for item in receipt["records"] if item["page_id"] == "11.16")
    png = ROOT / render["png"]["path"]
    assert hashlib.sha256(png.read_bytes()).hexdigest() == render["png"]["sha256"], "PNG 必須符合算繪收據"
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-16/verify_claims.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform(), "unicode_database": unicodedata.unidata_version},
        "source_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "excerpt_matches_repository_ast": True,
        "cases": records,
        "binary_strings": len(strings),
        "binary_pairs_checked": comparisons,
        "docstring_removal_preserves_results": True,
        "swapped_lines": {"same_character_set": set(reference) == set(prediction), "same_character_multiset": Counter(reference) == Counter(prediction), "lengths": [len(reference), len(prediction)], "matching_positions": matches, "aligned_replacements": len(reference) - len(matches), "distance": shortest_path(reference, prediction)[0]},
        "unicode": {char: {"codepoint": f"U+{ord(char):04X}", "python_length": len(char), "utf8_bytes": list(char.encode("utf-8")), "nfc": unicodedata.normalize("NFC", char)} for char in "臺台"},
        "manual_normalization": {"raw_cer": raw["cer"], "mapped_cer": normalized["cer"], "mapping": "臺 → 台，套用於正確答案與預測"},
        "python_helpers": {"range_list": list(range(6)), "enumerate_from_one": list(enumerate("甲乙", 1)), "false_integer_addition": 0 + False, "true_integer_addition": 0 + True, "min_example": min(3, 1, 2)},
        "render_receipt": render,
        "result": "全部斷言通過；只核對確定性的文字距離與圖檔版本。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
