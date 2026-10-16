"""原技術審閱者對 freeze09 的知情複查；不覆寫首次審閱證據。"""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch

from scripts.reading_time import figure_hashes, lesson_slices


root = Path(__file__).resolve().parents[4]
freeze_path = root / "docs/course-revision-20261016/freeze-09.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
entry = next(page for page in freeze["pages"] if page["page_id"] == "9.14")
snapshot_path = root / entry["snapshot"]
body = snapshot_path.read_text(encoding="utf-8")
source_path = root / "course/chapters/09.md"
current = lesson_slices(source_path.read_text(encoding="utf-8"))["9.14"]
old_path = root / "docs/course-revision-20261016/freeze-02/sources/9.14.md"
old = old_path.read_text(encoding="utf-8")
digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
assert body == current
assert digest(body) == entry["source_sha256"]

pattern = r"```python\n(.*?)```"
old_blocks = re.findall(pattern, old, re.S)
new_blocks = re.findall(pattern, body, re.S)
assert len(old_blocks) == len(new_blocks) == 1
old_ast = ast.dump(ast.parse(old_blocks[0]), include_attributes=False)
new_ast = ast.dump(ast.parse(new_blocks[0]), include_attributes=False)
assert old_ast == new_ast
assert re.sub(pattern, "<python>", old, flags=re.S) == re.sub(
    pattern, "<python>", body, flags=re.S
)
figures = figure_hashes(root, source_path, body)
assert figures == entry["figures_sha256"] == {}

namespace = {}
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(new_blocks[0], entry["snapshot"], "exec"), namespace)
assert namespace["y"].tolist() == [1.0, 0.0, 0.0]
assert namespace["y"].dtype == torch.float32
assert namespace["y"].device.type == "cpu"
rows = []
for line, expected_ce, expected_brier in zip(
    captured.getvalue().splitlines(),
    [0.2231435513142097, 1.6094379124341003],
    [0.06, 1.14],
    strict=True,
):
    label_ce, ce, label_brier, brier = line.split()
    assert label_ce == "CE" and label_brier == "Brier"
    assert abs(float(ce) - expected_ce) < 1e-6
    assert abs(float(brier) - expected_brier) < 1e-6
    rows.append({"CE": float(ce), "Brier_sum": float(brier),
                 "expected_CE": expected_ce, "expected_Brier_sum": expected_brier})
assert len(rows) == 2

result = {
    "lesson_id": "9.14",
    "reviewer_task": "/root/tech_20261016_9_14",
    "initial_review_context": "fresh",
    "this_review_context": "informed",
    "new_blind_review": False,
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/9.14-revision09-callback01-check.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "dtype": "torch.float32", "platform": platform.platform()},
    "freeze_path": str(freeze_path.relative_to(root)),
    "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    "snapshot": entry["snapshot"],
    "old_snapshot": str(old_path.relative_to(root)),
    "old_source_sha256": digest(old),
    "source_sha256": digest(body),
    "figure_sha256": figures,
    "version_checks": {"current_matches_snapshot": True, "prose_unchanged": True,
                       "python_block_count": 1, "python_ast_unchanged": True,
                       "python_ast_sha256": digest(new_ast)},
    "original_snippet_stdout": captured.getvalue(),
    "cases": rows,
    "result": "PASS: freeze09 snapshot personally executed verbatim; both CE and Brier values agree within 1e-6.",
    "scope": "Reviewer ran only this deterministic CPU arithmetic example, without training. Author kernel-execution09 was not run by this reviewer and is not used as reviewer execution evidence.",
}
destination = root / "docs/technical-reviews/artifacts/revision-20261016/9.14-revision09-callback01-execution.json"
with destination.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
