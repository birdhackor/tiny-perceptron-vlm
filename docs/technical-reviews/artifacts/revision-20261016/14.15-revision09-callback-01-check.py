"""Informed original-reviewer callback: compare snapshots and execute freeze-09."""

import ast
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

from scripts.reading_time import lesson_slices


root = Path(__file__).resolve().parents[4]


def load_page(freeze):
    manifest = json.loads((root / freeze).read_text(encoding="utf-8"))
    page = next(page for page in manifest["pages"] if page["page_id"] == "14.15")
    raw = (root / page["snapshot"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
    return page, raw.decode("utf-8")


previous, old = load_page("docs/course-revision-20261016/freeze-08.json")
current, new = load_page("docs/course-revision-20261016/freeze-09.json")
live = dict(lesson_slices((root / "course/chapters/14.md").read_text(encoding="utf-8")))["14.15"]
assert live == new
old_blocks = re.findall(r"```python\n(.*?)```", old, re.S)
new_blocks = re.findall(r"```python\n(.*?)```", new, re.S)
assert len(old_blocks) == len(new_blocks) == 1
assert old.replace("probability_correct = .9 if state > 0 else .1", "probability_correct = 0.9 if state > 0 else 0.1") == new
assert re.sub(r"```python\n.*?```", "<python>", old, flags=re.S) == re.sub(r"```python\n.*?```", "<python>", new, flags=re.S)
assert previous["figures_sha256"] == current["figures_sha256"] == {}
old_ast = ast.dump(ast.parse(old_blocks[0]), include_attributes=False)
new_ast = ast.dump(ast.parse(new_blocks[0]), include_attributes=False)
assert old_ast == new_ast

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(new_blocks[0], "freeze09-14.15-block-0", "exec"), {})
observed = capture.getvalue()
expected = "1 0.9\n2 0.1\n3 0.9\n4 0.1\n5 0.9\n6 0.1\n"
assert observed == expected
derived = [0.9 if (-1) ** (r + 1) > 0 else 0.1 for r in range(1, 7)]
assert derived == [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
supervised = {1, 3, 5}
assert [r for r in range(1, 7) if min(supervised) <= r <= max(supervised) and r not in supervised] == [2, 4]
assert 6 > max(supervised)
assert 2 in (supervised | {2}) and 4 not in (supervised | {2})

print(json.dumps({
    "schema_version": 1,
    "lesson_id": "14.15",
    "reviewer_task": "/root/tech_20261016_14_15",
    "review_mode": "informed_callback",
    "initial_task_context": "fresh",
    "callback_context": "informed; original reviewer, not a new blind review",
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/14.15-revision09-callback-01-check.py",
    "environment": {"python": platform.python_version(), "device": "cpu", "execution": "ordinary Python; no training or kernel-level benchmark"},
    "old_snapshot": previous["snapshot"],
    "old_source_sha256": previous["source_sha256"],
    "new_snapshot": current["snapshot"],
    "new_source_sha256": current["source_sha256"],
    "live_source_matches_new_snapshot": live == new,
    "diff": "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=previous["snapshot"], tofile=current["snapshot"])),
    "outside_code_identical": True,
    "only_change": "Python literal spelling .9/.1 -> 0.9/0.1",
    "figure_sha256": {},
    "ast_equal": old_ast == new_ast,
    "new_code_sha256": hashlib.sha256(new_blocks[0].encode("utf-8")).hexdigest(),
    "stdout": observed,
    "expected_stdout": expected,
    "exact_match": observed == expected,
    "derived_probabilities": derived,
    "depth_set_checks": "S={1,3,5}; in-range unsupervised={2,4}; 6>5; after adding 2, S={1,2,3,5} and 4 remains unsupervised",
    "result": "All assertions passed. The six-round sign-flip remains a nonmonotone counterexample. Literal normalization changes neither AST nor values. Existing conceptual sources and terminal-gradient evidence remain applicable because their statements are unchanged.",
    "scope": "Personally executed the freeze-09 snippet and these snapshot/AST/value checks. Did not run or claim authorship of the separate author kernel-execution09 / 12-kernel execution."
}, ensure_ascii=False, indent=2))
