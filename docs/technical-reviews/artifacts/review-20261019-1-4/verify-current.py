"""核對凍結全文差異、算例與 Counter 繼承的字典介面。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "frozen-example", "exec"), namespace)
    return namespace, stdout.getvalue()


root = Path(__file__).resolve().parents[4]
snapshot = root / "docs/course-review-20261019/freeze-02/sources/1.4.md"
previous_snapshot = root / "docs/course-review-20261019/freeze-01/sources/1.4.md"
raw = snapshot.read_bytes()
previous_raw = previous_snapshot.read_bytes()
source_hash = digest(raw)
assert source_hash == "dda1987857b27d366f722b8c939b9ea23306646fbef1d6eacb98926af6ec5471"
assert digest(previous_raw) == "4436f03e544b3b36ee6408e1c61a60a8357530f5c2eec637791329336bd6eeb0"
before = "所以字典的寫法都能照用"
after = "所以本節用到的 `.values()`、`.items()` 都能照用"
assert previous_raw.count(before.encode()) == 1
assert previous_raw.replace(before.encode(), after.encode()) == raw
freeze_path = root / "docs/course-review-20261019/freeze-02.json"
freeze_hash = digest(freeze_path.read_bytes())
assert freeze_hash == "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1"
page = next(p for p in json.loads(freeze_path.read_text())["pages"] if p["page_id"] == "1.4")
assert page["figures_sha256"] == {}
assert page["source_sha256"] == source_hash
current = (root / "course/chapters/01.md").read_bytes()
current_section = re.search(rb"(?ms)^## 1\.4 .*?(?=^## |\Z)", current).group()
assert current_section == raw
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
previous_code = re.search(r"```python\n(.*?)```", previous_raw.decode(), re.S).group(1)
assert code == previous_code

previous_report_path = root / "docs/course-review-20261019/reviews/rv1019_technical_1-4/technical-1-4-original-01.json"
assert digest(previous_report_path.read_bytes()) == "f513ed6f41cf051ad361685589be9252f1720cb2c4af89a0eb28bee3f3251175"
previous_report = json.loads(previous_report_path.read_text())
previous_artifacts = []
for artifact in previous_report["artifacts"]:
    actual = digest((root / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    previous_artifacts.append({"path": artifact["path"], "sha256": actual})

namespace, original_stdout = execute(code)
counts, probability = namespace["counts"], namespace["probability"]
assert counts == {"貓": 3, "狗": 1, "鳥": 1}
assert namespace["total"] == 5
assert probability == {"貓": 0.6, "狗": 0.2, "鳥": 0.2}
assert list(counts.values()) == [3, 1, 1]
assert list(counts.items()) == [("貓", 3), ("狗", 1), ("鳥", 1)]
assert Counter.values is dict.values
assert Counter.items is dict.items
assert list(counts.values()) == list(dict(counts).values())
assert list(counts.items()) == list(dict(counts).items())
assert isinstance(counts, dict)
assert counts["魚"] == 0
assert "魚" not in counts
assert 1e-6 == 0.000001
assert abs(-0.2) == 0.2
exact = {key: Fraction(value, 5) for key, value in counts.items()}
assert sum(exact.values()) == 1
assert max(probability, key=probability.get) == "貓"
assert all(exact[key] > 0 for key in ("狗", "鳥"))
exercise_code = code.replace('text = "貓貓貓狗鳥"', 'text = "貓貓貓狗狗狗鳥"')
exercise, exercise_stdout = execute(exercise_code)
assert exercise["counts"] == {"貓": 3, "狗": 3, "鳥": 1}
assert exercise["total"] == 7
assert exercise["probability"]["貓"] == 3 / 7
assert round(3 / 7, 3) == 0.429
assert Fraction(3, 7) < Fraction(3, 5)

documents = []
for url, anchors, markers, quotes, note in [
    (
        "https://docs.python.org/3.13/library/collections.html",
        ["collections.Counter", "collections.Counter.fromkeys", "collections.Counter.update"],
        ["The usual dictionary methods are available", "except for two which work differently for counters"],
        ["The usual dictionary methods are available"],
        "已讀 Counter 類別、缺鍵讀取、一般字典方法與兩個例外。values 和 items 不在例外中。",
    ),
    (
        "https://docs.python.org/3.13/library/stdtypes.html",
        ["dict.values", "dict.items"],
        ["Return a new view", "values", "items"],
        ["Return a new view"],
        "已讀 dict.values 與 dict.items 的定義及 dictionary views 說明；前者提供值，後者提供鍵值對。",
    ),
]:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", "--write-out", "\nCURL_EFFECTIVE_URL:%{url_effective}\n", url]
    result = subprocess.run(command, capture_output=True, check=True)
    original, effective = result.stdout.rsplit(b"\nCURL_EFFECTIVE_URL:", 1)
    html = original.decode()
    parser = Text()
    parser.feed(html)
    plain = re.sub(r"\s+", " ", " ".join(parser.parts))
    for anchor in anchors:
        assert f'id="{anchor}"' in html
    for marker in markers:
        assert marker in plain
    title = unescape(re.search(r"<title>(.*?)</title>", html, re.S).group(1))
    assert "3.13.16" in title
    documents.append({
        "url": url, "effective_url": effective.decode().strip(), "title": title,
        "accessed_at_utc": datetime.now(UTC).isoformat(), "command_argv": command,
        "exit_code": result.returncode, "original_sha256": digest(original),
        "original_bytes": len(original), "locators": anchors, "short_quotes": quotes,
        "inspection_note": note, "full_original_saved": False,
    })

report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-4/verify-current.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU", "float_mant_dig": str(sys.float_info.mant_dig)},
    "source_sha256": source_hash, "freeze_sha256": freeze_hash,
    "current_matches_freeze": True, "figure_sha256": {},
    "only_text_change": {"before": before, "after": after, "rest_identical": True, "python_fence_identical": True},
    "previous_evidence_integrity": previous_artifacts,
    "original": {"stdout": original_stdout, "counts": dict(counts), "total": namespace["total"], "probability": probability, "sum": sum(probability.values()), "exact_rationals": {key: str(value) for key, value in exact.items()}, "values": list(counts.values()), "items": list(counts.items())},
    "exercise": {"stdout": exercise_stdout, "counts": dict(exercise["counts"]), "total": exercise["total"], "cat_probability": exercise["probability"]["貓"], "rounded_three_places": round(3 / 7, 3)},
    "floating_point": {"cat_float_ratio": str(Fraction.from_float(probability["貓"])), "cat_exact_ratio": "3/5", "cat_representation_error": str(Fraction.from_float(probability["貓"]) - Fraction(3, 5)), "sum_error": abs(sum(probability.values()) - 1), "tolerance": 1e-6, "absolute_value_example": abs(-0.2)},
    "selection": {"greedy": max(probability, key=probability.get), "sampling_intervals": {"貓": "[0, 3/5)", "狗": "[3/5, 4/5)", "鳥": "[4/5, 1)"}, "interpretation": "抽選機率以區間長度解析核對，未做抽樣頻率實驗。"},
    "counter_interface": {"is_dict": isinstance(counts, dict), "values_inherited_from_dict": Counter.values is dict.values, "items_inherited_from_dict": Counter.items is dict.items, "values_equal_plain_dict": list(counts.values()) == list(dict(counts).values()), "items_equal_plain_dict": list(counts.items()) == list(dict(counts).items()), "missing_subscript": counts["魚"]},
    "original_documents": documents,
    "result": "凍結版本、全文差異、原例、練習與 Counter 字典介面核對的全部斷言通過。",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
