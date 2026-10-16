"""Informed callback: compare 14.12 freeze 08/09 and execute freeze 09 scalars."""

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
lesson_id = "14.12"
pages = {}
bodies = {}
for revision in ("08", "09"):
    freeze = json.loads(
        (root / f"docs/course-revision-20261016/freeze-{revision}.json").read_text()
    )
    pages[revision] = next(page for page in freeze["pages"] if page["page_id"] == lesson_id)
    bodies[revision] = (root / pages[revision]["snapshot"]).read_text()
    assert hashlib.sha256(bodies[revision].encode()).hexdigest() == pages[revision]["source_sha256"]

assert bodies["09"] == lesson_slices((root / "course/chapters/14.md").read_text())[lesson_id]
assert bodies["09"] == bodies["08"].replace("state + .5 *", "state + 0.5 *")
codes = {
    revision: re.search(r"```python\n(.*?)```", body, re.S)[1]
    for revision, body in bodies.items()
}
ast_equal = ast.dump(ast.parse(codes["08"])) == ast.dump(ast.parse(codes["09"]))
assert ast_equal
outside_code_equal = re.sub(r"```python\n.*?```", "<code>", bodies["08"], flags=re.S) == re.sub(
    r"```python\n.*?```", "<code>", bodies["09"], flags=re.S
)
assert outside_code_equal
assert pages["08"]["figures_sha256"] == pages["09"]["figures_sha256"] == {}


def execute(code):
    captured = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(captured):
        exec(compile(code, "freeze-09/sources/14.12.md", "exec"), namespace)
    return captured.getvalue(), namespace


stdout, namespace = execute(codes["09"])
assert stdout == "1 2.0\n2 3.0\n3 3.5\n4 3.75\n"
assert namespace["e"] == 4.0 and namespace["state"] == 3.75
states = [float(line.split()[1]) for line in stdout.splitlines()]
errors = [4.0 - state for state in states]
assert errors == [2.0, 1.0, 0.5, 0.25]

exercise = codes["09"].replace("0.5 *", "2.0 *").replace("range(1, 5)", "range(1, 9)")
exercise_stdout, _ = execute(exercise)
exercise_states = [float(line.split()[1]) for line in exercise_stdout.splitlines()]
exercise_errors = [4.0 - state for state in exercise_states]
assert exercise_states == [8.0, 0.0] * 4
assert exercise_errors == [-4.0, 4.0] * 4

print(
    json.dumps(
        {
            "lesson_id": lesson_id,
            "reviewer_task": "/root/tech_20261016_14_12",
            "review_context": "informed callback by the original technical reviewer; initial review was fresh",
            "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/14.12-revision09-check.py",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": "CPU; Python scalar arithmetic only",
                "dependencies": "Python standard library and repository lesson_slices; no torch used",
            },
            "source_sha256_before": pages["08"]["source_sha256"],
            "source_sha256_after": pages["09"]["source_sha256"],
            "snapshot_after": pages["09"]["snapshot"],
            "source_matches_snapshot": True,
            "outside_code_equal": outside_code_equal,
            "python_ast_equal": ast_equal,
            "figure_sha256": pages["09"]["figures_sha256"],
            "unified_diff": "".join(
                difflib.unified_diff(
                    bodies["08"].splitlines(keepends=True),
                    bodies["09"].splitlines(keepends=True),
                    fromfile="freeze-08:14.12",
                    tofile="freeze-09:14.12",
                )
            ),
            "executed_lesson_code": codes["09"],
            "lesson_stdout": stdout,
            "states": states,
            "signed_errors": errors,
            "executed_exercise_code": exercise,
            "exercise_stdout": exercise_stdout,
            "exercise_states": exercise_states,
            "exercise_signed_errors": exercise_errors,
            "result": "All assertions passed: the sole edit is .5 to 0.5; surrounding text and AST are unchanged; freeze09 code and coefficient-2 exercise both match the stated numbers. This reviewer executed only these scalar examples, not the author's 12 notebook kernels or any training.",
        },
        ensure_ascii=False,
        indent=2,
    )
)
