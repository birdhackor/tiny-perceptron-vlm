"""Independent check of the frozen 9.16 cost example; no model training."""
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
import hashlib
import json
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

source = ROOT / "course/chapters/09.md"
body = lesson_slices(source.read_bytes().decode("utf-8"))["9.16"]
freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-08.json").read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.16")
source_sha = hashlib.sha256(body.encode()).hexdigest()
assert source_sha == page["source_sha256"]
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
stream = StringIO()
with redirect_stdout(stream):
    exec(compile(blocks[0], "course/chapters/09.md#9.16:block0", "exec"), {})
actual = stream.getvalue()
expected = "0.6 2.0 澄清\n0.8 1.0 澄清\n0.9 0.5 回答\n"
assert actual == expected, repr(actual)

# Exact rational arithmetic independently checks the displayed cost formula.
rows = []
for p in [Fraction(3, 5), Fraction(4, 5), Fraction(9, 10)]:
    cost = 5 * (1 - p)
    rows.append({"p": str(p), "answer_cost": str(cost),
                 "action": "回答" if cost < 1 else "澄清"})
assert [r["answer_cost"] for r in rows] == ["2", "1", "1/2"]
threshold_one = 1 - Fraction(1, 5)
threshold_half = 1 - Fraction(1, 2) / 5
assert threshold_one == Fraction(4, 5)
assert threshold_half == Fraction(9, 10)
# The authored epsilon prevents binary rounding at p=.8 from selecting 回答.
tie_float = 5 * (1 - .8)
assert tie_float < 1
assert not (tie_float < 1 - 1e-9)
receipt = {
    "lesson_id": "9.16", "source_sha256": source_sha,
    "command": "uv run python docs/technical-reviews/artifacts/revision-20261016/9.16-verify.py",
    "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                    "device": "CPU; standard library only", "platform": platform.platform()},
    "code_sha256": hashlib.sha256(blocks[0].encode()).hexdigest(),
    "stdout": actual, "expected_stdout": expected, "exact_rational_rows": rows,
    "answer_if_p_strictly_greater_than": {"clarification_cost_1": str(threshold_one),
                                          "clarification_cost_0.5": str(threshold_half)},
    "floating_tie_cost": repr(tie_float),
    "tolerance_scope": "The 1e-9 guard enforces clarify at the represented tie; costs within 1e-9 of 1 are also treated as ties. The three displayed p values match the exact decision.",
    "result": "All assertions passed; code output and exact rational cost thresholds match the lesson.",
    "limits": "Toy deterministic expected-cost calculation only; no trained model, calibration test, service action, GPU claim, or product-quality result."
}
out = ROOT / "docs/technical-reviews/artifacts/revision-20261016/9.16-execution.json"
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
