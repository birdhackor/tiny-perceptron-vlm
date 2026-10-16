"""原14.16技術審閱者的知情callback；不覆寫原始證據，不訓練模型。"""

import ast
import contextlib
import difflib
import hashlib
import io
import json
import math
import platform
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import figure_hashes, lesson_slices

old_path = ROOT / "docs/course-revision-20261016/freeze-01/sources/14.16.md"
new_path = ROOT / "docs/course-revision-20261016/freeze-09/sources/14.16.md"
old = old_path.read_text(encoding="utf-8")
new = new_path.read_text(encoding="utf-8")
current = lesson_slices((ROOT / "course/chapters/14.md").read_text(encoding="utf-8"))["14.16"]
pattern = r"```python\n(.*?)```"
old_code = re.search(pattern, old, re.S)[1]
new_code = re.search(pattern, new, re.S)[1]
old_ast = ast.dump(ast.parse(old_code), include_attributes=False)
new_ast = ast.dump(ast.parse(new_code), include_attributes=False)
assert old_ast == new_ast
assert re.sub(pattern, "[PYTHON_CODE]", old, flags=re.S) == re.sub(pattern, "[PYTHON_CODE]", new, flags=re.S)
assert current == new
assert figure_hashes(ROOT, ROOT / "course/chapters/14.md", new) == {}

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    runpy.run_path(str(ROOT / "docs/technical-reviews/artifacts/revision-20261016/14.16-check.py"))
rerun = json.loads(capture.getvalue())
assert rerun["source_sha256"] == hashlib.sha256(new.encode()).hexdigest()
assert rerun["original_snippet"] == new_code
assert rerun["original_snippet_stdout"] == "早輪 0.2 後輪 0.4 懲罰 0.2\n早輪 0.5 後輪 0.4 懲罰 0.0\n"

# 訓練點的單調約束並未限制未見輸入：以下是同一函數在兩個不同輸入的查表反例。
# 每列兩輪均為合法二元預測，正解皆為類別0；不聲稱這是訓練結果。
generalization_counterexample = []
for label, probs in [("training_input", [0.6, 0.8]), ("unseen_input", [0.8, 0.6])]:
    losses = [-math.log(p) for p in probs]
    generalization_counterexample.append({
        "input": label, "gold": 0, "gold_probability_per_round": probs,
        "CE_per_round": losses, "delta_CE": losses[1] - losses[0],
        "monotonicity_penalty": max(0.0, losses[1] - losses[0]),
    })
assert generalization_counterexample[0]["monotonicity_penalty"] == 0.0
assert generalization_counterexample[1]["monotonicity_penalty"] > 0.0

print(json.dumps({
    "lesson_id": "14.16", "reviewer_task": "/root/tech_20261016_14_16",
    "initial_review_context": "fresh", "callback_context": "informed",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/14.16-revision09-callback-check.py > docs/technical-reviews/artifacts/revision-20261016/14.16-revision09-callback-execution.json",
    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": "CPU; builtins/math/ast only; no GPU, notebook kernels or model training"},
    "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
    "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
    "current_equals_new_snapshot": current == new,
    "prose_outside_python_unchanged": True, "python_ast_equal": True,
    "python_ast_sha256": hashlib.sha256(new_ast.encode()).hexdigest(),
    "figures": {},
    "actual_diff": "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(old_path.relative_to(ROOT)), tofile=str(new_path.relative_to(ROOT)))),
    "current_lesson_verification_rerun": rerun,
    "training_vs_unseen_counterexample": generalization_counterexample,
    "scope": "作者kernel-execution09未由本審閱者執行或代稱自己的結果。本次只直接執行CPU透明機制算例，保留訓練軟懲罰與任意新題逐題單調保證的差別。",
    "result": "All assertions passed; only float spelling, spaces and blank lines differ; six original claims rechecked."
}, ensure_ascii=False, indent=2))
