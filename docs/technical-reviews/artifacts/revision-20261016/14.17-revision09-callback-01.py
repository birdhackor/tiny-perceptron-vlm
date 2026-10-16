"""Informed same-reviewer recheck; preserve the original review and evidence."""

import ast
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices


root = Path(__file__).resolve().parents[4]
lesson_id = "14.17"
expected = "77c0a911ebb7b5ddbcfabe649af4ac8c8c13671b0de9fa36c90610e7d6748910"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(freeze_name):
    freeze = json.loads((root / freeze_name).read_text())
    page = next(p for p in freeze["pages"] if p["page_id"] == lesson_id)
    raw = (root / page["snapshot"]).read_bytes()
    assert sha(raw) == page["source_sha256"]
    return page, raw.decode()


old_page, old_body = snapshot("docs/course-revision-20261016/freeze-08.json")
new_page, new_body = snapshot("docs/course-revision-20261016/freeze-09.json")
current = lesson_slices((root / "course/chapters/14.md").read_bytes().decode())[lesson_id]
assert current == new_body
assert sha(new_body.encode()) == expected
pattern = r"```python\n(.*?)```"
old_codes = re.findall(pattern, old_body, flags=re.S)
new_codes = re.findall(pattern, new_body, flags=re.S)
assert len(old_codes) == len(new_codes) == 1
assert re.sub(pattern, "<PYTHON>", old_body, flags=re.S) == re.sub(
    pattern, "<PYTHON>", new_body, flags=re.S
)
assert ast.dump(ast.parse(old_codes[0])) == ast.dump(ast.parse(new_codes[0]))
assert old_page["figures_sha256"] == new_page["figures_sha256"] == {}


def run(code):
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "14.17-informed-callback", "exec"), namespace)
    rows = []
    for round_id, p in enumerate(namespace["predictions"], start=1):
        assert list(p.shape) == [2]
        assert p.dtype == torch.float32
        assert bool(torch.all(p >= 0))
        assert abs(float(p.sum()) - 1) <= 1e-7
        rows.append({
            "round": round_id,
            "probabilities": p.tolist(),
            "argmax": int(p.argmax()),
            "confidence": float(p.max()),
            "correct_against_offline_truth_1": int(p.argmax()) == 1,
        })
    return {"stdout": output.getvalue(), "rows": rows}


old_run = run(old_codes[0])
new_run = run(new_codes[0])
assert old_run == new_run
rows = new_run["rows"]
assert [r["argmax"] for r in rows] == [0, 0]
assert not any(r["correct_against_offline_truth_1"] for r in rows)
confidence_stop = next(r for r in rows if r["confidence"] > 0.9)
agreement_stop = next(
    rows[i] for i in range(1, len(rows)) if rows[i]["argmax"] == rows[i - 1]["argmax"]
)
assert confidence_stop["round"] == 1
assert agreement_stop["round"] == 2
assert not confidence_stop["correct_against_offline_truth_1"]
assert not agreement_stop["correct_against_offline_truth_1"]
evaluation_sentence = "需要獨立校準與停止規則評估，記下每個規則答錯幾題、用了幾輪及多少時間。"
assert evaluation_sentence in new_body

print(json.dumps({
    "lesson_id": lesson_id,
    "reviewer_task": "/root/tech_20261016_14_17",
    "initial_task_context": "fresh",
    "callback_context": "informed, same original reviewer; not a new blind reading",
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/14.17-revision09-callback-01.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": "cpu",
    },
    "old_snapshot": old_page["snapshot"],
    "old_source_sha256": old_page["source_sha256"],
    "new_snapshot": new_page["snapshot"],
    "new_source_sha256": expected,
    "current_source_matches_new_snapshot": True,
    "outside_python_bytes_unchanged": True,
    "python_ast_unchanged": True,
    "figures_unchanged": True,
    "diff": "".join(difflib.unified_diff(old_body.splitlines(True), new_body.splitlines(True))),
    "executed_new_lesson_code": new_codes[0],
    "old_and_new_outputs_equal": True,
    "new_run": new_run,
    "confidence_threshold_0_9_stop": confidence_stop,
    "two_consecutive_argmax_agreement_stop": agreement_stop,
    "evaluation_sentence": evaluation_sentence,
    "evaluation_scope": "Text requires error counts, rounds and time. It does not define a coverage denominator, abstention protocol, evaluation dataset or empirical stopping performance. These are not claimed as already specified or measured.",
    "author_kernel_execution09_used_as_own_execution": False,
    "result": "All assertions passed: only decimal literal spelling changed; identical AST and outputs; confidence stop at round 1 and agreement stop at round 2 are both wrong.",
}, ensure_ascii=False, indent=2))
