"""9.15 審閱者的短算例核對；沒有訓練或品質評測。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices


def confidence(probabilities):
    n = len(probabilities)
    return (max(probabilities) - 1 / n) / (1 - 1 / n)


body = lesson_slices((ROOT / "course/chapters/09.md").read_text())["9.15"]
code = re.search(r"```python\n(.*?)```", body, re.S).group(1)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(code, {})
assert captured.getvalue() == "最大機率 0.6 集中度 0.4\n"

cases = [
    ("正文三選一", [0.6, 0.2, 0.2], Fraction(2, 5)),
    ("同最大值二選一", [0.6, 0.4], Fraction(1, 5)),
    ("均勻", [1 / 3, 1 / 3, 1 / 3], Fraction(0)),
    ("全部集中", [1.0, 0.0, 0.0], Fraction(1)),
    ("練習重排", [0.2, 0.6, 0.2], Fraction(2, 5)),
    ("錯誤也可很集中之機制示意", [0.99, 0.005, 0.005], Fraction(197, 200)),
]
results = []
for name, probabilities, exact in cases:
    observed = confidence(probabilities)
    assert abs(observed - float(exact)) <= 1e-12
    results.append(
        {
            "case": name,
            "probabilities": probabilities,
            "expected_exact": str(exact),
            "observed": observed,
            "argmax_index": probabilities.index(max(probabilities)),
            "tolerance": "absolute <= 1e-12",
        }
    )

try:
    confidence([1.0])
except ZeroDivisionError as error:
    singleton = type(error).__name__
else:
    raise AssertionError("單一候選應出現除零")

output = {
    "reviewer_task": "/root/tech_20261016_9_15",
    "command": "uv run python docs/technical-reviews/artifacts/revision-20261016/9.15-choice-check.py",
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "CPU; Python scalar arithmetic only",
    },
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "actual_lesson_code": code,
    "actual_lesson_stdout": captured.getvalue(),
    "cases": results,
    "singleton_result": singleton,
    "result": "教材原碼及六組透明算例通過；n=1出現ZeroDivisionError。",
    "scope": "僅核對公式、原碼與語義重排；非訓練成果、校準效果或服務介面驗收。",
}
target = ROOT / "docs/technical-reviews/artifacts/revision-20261016/9.15-execution.json"
target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
