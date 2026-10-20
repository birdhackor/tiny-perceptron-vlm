"""獨立核對逐字轉寫、正規化與空參考答案的邊界。"""

import hashlib
import itertools
import json
import os
import platform
import re
import sys
import unicodedata
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.natural_concepts import edit_distance, text_error_report


def shortest_edit_paths(start, alphabet, maximum_length):
    """以單步插入、刪除、替換建圖，廣度優先搜尋最短距離。"""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        value = queue.popleft()
        candidates = set()
        for index in range(len(value)):
            candidates.add(value[:index] + value[index + 1 :])
            for char in alphabet:
                candidates.add(value[:index] + char + value[index + 1 :])
        if len(value) < maximum_length:
            for index in range(len(value) + 1):
                for char in alphabet:
                    candidates.add(value[:index] + char + value[index:])
        for candidate in sorted(candidates):
            if candidate not in distances:
                distances[candidate] = distances[value] + 1
                queue.append(candidate)
    return distances


def main():
    expected_cases = [
        ("今天去臺北", "今天去台北", False, 1, 5, 0.2),
        ("今天去臺北", "今天去臺", False, 1, 5, 0.2),
        ("今天去臺北", "今天去台", False, 2, 5, 0.4),
        ("今天去臺北", "今天去臺北", True, 0, 5, 0.0),
        ("今天去臺北", "", False, 5, 5, 1.0),
        ("", "", True, 0, 0, None),
        ("", "熟悉的文字", False, 5, 0, None),
        ("Ａ", "A", False, 1, 1, 1.0),
        ("文 字！", "文字", False, 2, 4, 0.5),
        ("汉字", "漢字", False, 1, 2, 0.5),
    ]
    cases = []
    for reference, prediction, exact, edits, count, cer in expected_cases:
        observed = text_error_report(reference, prediction)
        expected = {"reference": reference, "prediction": prediction,
                    "exact": exact, "edits": edits,
                    "reference_characters": count, "cer": cer}
        if observed != expected:
            raise AssertionError(f"轉寫核對不符：{reference!r} / {prediction!r}")
        cases.append({"expected": expected, "observed": observed,
                      "prediction_characters": len(prediction)})

    alphabet = ("台", "臺", "北")
    strings = ["".join(chars) for size in range(4)
               for chars in itertools.product(alphabet, repeat=size)]
    comparisons = 0
    for prediction in strings:
        distances = shortest_edit_paths(prediction, alphabet, 3)
        for reference in strings:
            expected = distances[reference]
            observed = edit_distance(reference, prediction)
            if expected != observed:
                raise AssertionError(f"編輯距離不符：{reference!r} / {prediction!r}")
            comparisons += 1

    normalization = []
    for original, expected in [("Ａ", "A"), ("臺", "臺"), ("台", "台"),
                               ("今天去臺北", "今天去臺北"), ("汉字", "汉字")]:
        observed = unicodedata.normalize("NFKC", original)
        if observed != expected:
            raise AssertionError(f"正規化核對不符：{original!r}")
        normalization.append({"input": original, "expected": expected,
                              "observed": observed,
                              "decomposition": [unicodedata.decomposition(c) for c in original]})

    freeze_path = "docs/course-review-20261019/freeze-01/sources/20.10.md"
    current = (ROOT / "course/chapters/20.md").read_bytes()
    section = re.search(rb"(?ms)^## 20\.10 .*?(?=^## |\Z)", current).group()
    frozen = (ROOT / freeze_path).read_bytes()
    if section != frozen:
        raise AssertionError("目前小節與凍結版本不同")
    record = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-10/verify_text.py",
        "executed_at": datetime.now(UTC).isoformat(),
        "environment": {"python": sys.version, "torch": str(torch.__version__),
                        "unicode_database": unicodedata.unidata_version,
                        "device": "cpu", "platform": platform.platform()},
        "result": "全部預期相符",
        "cases": cases,
        "independent_distance_check": {
            "method": "以單步編輯建圖後做廣度優先搜尋，不使用動態規劃遞迴式。",
            "scope": "字母表為台、臺、北，所有長度不超過三個碼點的字串；最短路徑不必超過兩端最大長度。",
            "strings": len(strings), "ordered_pairs": comparisons, "mismatches": 0},
        "normalization": normalization,
        "reference_measurement": {"text": "今天去臺北", "codepoints": len("今天去臺北"),
                                  "utf8_bytes": len("今天去臺北".encode()),
                                  "codepoint_names": [f"U+{ord(c):04X}" for c in "今天去臺北"]},
        "source_match": {"frozen_path": freeze_path, "current_equals_frozen": True,
                         "section_sha256": hashlib.sha256(section).hexdigest()},
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ["tiny_perceptron/natural_concepts.py", freeze_path,
                                      "docs/course-review-20261019/freeze-04.json"]},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
