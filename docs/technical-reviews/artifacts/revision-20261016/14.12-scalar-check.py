"""Review-only checks for the two hand-written updates in lesson 14.12."""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

from scripts.reading_time import lesson_slices


root = Path(__file__).resolve().parents[4]
body = lesson_slices((root / "course/chapters/14.md").read_text(encoding="utf-8"))["14.12"]
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
stdout = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(stdout):
    exec(compile(code, "course/chapters/14.md#14.12", "exec"), namespace)

expected_stdout = "1 2.0\n2 3.0\n3 3.5\n4 3.75\n"
assert stdout.getvalue() == expected_stdout
assert namespace["e"] == 4.0
assert namespace["state"] == 3.75


def trajectory(coefficient, rounds):
    target, state = 4.0, 0.0
    states, errors = [], []
    for _ in range(rounds):
        state = state + coefficient * (target - state)
        states.append(state)
        errors.append(target - state)
    return {"coefficient": coefficient, "states": states, "signed_errors": errors}


half = trajectory(0.5, 4)
double = trajectory(2.0, 8)
assert half["states"] == [2.0, 3.0, 3.5, 3.75]
assert half["signed_errors"] == [2.0, 1.0, 0.5, 0.25]
assert double["states"] == [8.0, 0.0] * 4
assert double["signed_errors"] == [-4.0, 4.0] * 4

print(
    json.dumps(
        {
            "lesson_id": "14.12",
            "reviewer_task": "/root/tech_20261016_14_12",
            "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "command": "PYTHONPATH=. uv run python docs/technical-reviews/artifacts/revision-20261016/14.12-scalar-check.py",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": "CPU; Python scalar arithmetic only",
                "dependencies": "Python standard library and repository lesson_slices; no torch used",
            },
            "original_lesson_code": code,
            "original_lesson_stdout": stdout.getvalue(),
            "original_lesson_expected_stdout": expected_stdout,
            "half_gap_update": half,
            "exercise_coefficient_2": double,
            "result": "All exact-value assertions passed. This executes only the hand-written scalar update, not a learned Transformer or training run.",
        },
        ensure_ascii=False,
        indent=2,
    )
)
