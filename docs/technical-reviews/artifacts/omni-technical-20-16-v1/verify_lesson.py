"""CPU-only verification of the exact frozen 20.16 embedded program."""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
page = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == "20.16")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
body = raw.decode("utf-8")
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
code = blocks[0]
chapter = (ROOT / page["source"]).read_text()
current = re.search(r"^## 20\.16 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
assert current.encode() == raw
assert page["figures_sha256"] == {}
assert not re.findall(r"!\[[^\]]*\]\([^)]*\)", body)
notebook = ROOT / page["notebook"]
assert hashlib.sha256(notebook.read_bytes()).hexdigest() == page["notebook_sha256"]
nb = json.loads(notebook.read_bytes())
assert any("".join(c["source"]).strip() == code.strip() for c in nb["cells"] if c["cell_type"] == "code")


def execute(text):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(text, str(snapshot) + ":embedded-python", "exec"), namespace)
    return namespace, output.getvalue()


original, stdout = execute(code)
assert stdout == (
    "逐題 4 / 6\n兩題都對 1 / 3\n接受 5 / 6 錯誤 2\n"
    "coverage 0.833\naccepted risk 0.4\n"
)
assert original["pairs"] == {"p1": [True, True], "p2": [True, False], "p3": [False, True]}
assert original["correct"] == 4 and original["paired"] == 1
assert len(original["accepted"]) == 5 and original["errors"] == 2
assert original["risk"] == 2 / 5
row_details = []
for family, variant, logits, gold in original["rows"]:
    p = original["probabilities"](logits)
    pred = max(range(len(p)), key=lambda i: p[i])
    assert abs(sum(p) - 1.0) < 1e-15
    row_details.append(dict(family=family, variant=variant, raw_logits=logits, gold=gold,
                            probabilities=p, pred=pred, correct=pred == gold,
                            accepted=max(p) >= original["tau"] and pred != 2))
exercise_code = code.replace("T, tau = 1.25, 0.85", "T, tau = 1.25, 0.95")
assert exercise_code != code
exercise, exercise_stdout = execute(exercise_code)
assert not exercise["accepted"] and exercise["risk"] is None
assert "接受 0 / 6 錯誤 0\ncoverage 0.0\naccepted risk None\n" in exercise_stdout
# A high-confidence none candidate still fails the exact original acceptance predicate.
p = original["probabilities"]([-1.0, 0.0, 3.0])
pred = max(range(len(p)), key=lambda i: p[i])
assert max(p) >= original["tau"] and pred == 2
assert not (max(p) >= original["tau"] and pred != 2)

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-20-16-v1/verify_lesson.py",
    "environment": {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
                    "platform": platform.platform(), "device": "CPU; standard library only; no model inference"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "source_snapshot": {"path": page["snapshot"], "sha256": page["source_sha256"]},
    "embedded_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "notebook": {"path": page["notebook"], "sha256": page["notebook_sha256"], "same_embedded_program": True},
    "stdout": stdout,
    "rows": row_details,
    "aggregate": {"correct": original["correct"], "questions": len(original["rows"]),
                  "paired": original["paired"], "families": len(original["pairs"]),
                  "accepted": len(original["accepted"]), "errors": original["errors"],
                  "coverage": len(original["accepted"]) / len(original["rows"]), "risk": original["risk"]},
    "exercise_tau_0_95": {"only_change": "T, tau = 1.25, 0.85 -> T, tau = 1.25, 0.95", "stdout": exercise_stdout,
                          "accepted": len(exercise["accepted"]), "risk": exercise["risk"]},
    "none_guard_probe": {"raw_logits": [-1.0, 0.0, 3.0], "pred": pred, "p_max": max(p), "accepted": False},
    "scope": "Exact frozen embedded arithmetic, source/notebook identity and explicit synthetic exercise only. No mature-model output or quality result."
}
target = ARTIFACT / "embedded-execution.json"
with target.open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "stdout": stdout, "exercise_stdout": exercise_stdout, "environment": result["environment"]}, ensure_ascii=False))
