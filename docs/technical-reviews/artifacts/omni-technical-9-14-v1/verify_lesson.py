"""Execute the exact frozen 9.14 example and its stated exercise on CPU."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("execution.json")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.14")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(blocks) == 1
original = blocks[0]
needle = "torch.tensor([0.2, 0.7, 0.1])]"
assert original.count(needle) == 1
exercise = original.replace(
    needle,
    "torch.tensor([0.2, 0.7, 0.1]), torch.tensor([1 / 3, 1 / 3, 1 / 3])]",
)


def run(code):
    scope = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, str(page["snapshot"]), "exec"), scope)
    assert scope["y"].shape == (3,)
    assert scope["y"].dtype == torch.float32
    assert scope["y"].device.type == "cpu"
    return stream.getvalue()


original_stdout = run(original)
exercise_stdout = run(exercise)
rows = []
for line, expected in zip(
    exercise_stdout.splitlines(),
    [(0.2231, 0.06), (1.6094, 1.14), (1.0986, 0.6667)],
    strict=True,
):
    _, ce, _, brier = line.split()
    observed = [float(ce), float(brier)]
    assert abs(observed[0] - expected[0]) <= 5e-5
    assert abs(observed[1] - expected[1]) <= 5e-5
    rows.append({"expected_rounded": expected, "observed": observed, "absolute_tolerance": 5e-5})

# The following small checks supplement the literal example; they are not model
# training or evidence that any real model is calibrated.
p = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float64)
q = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
y = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
soft_ce = float(-(q * p.log()).sum())
assert math.isclose(soft_ce, float(F.cross_entropy(p.log(), q)), abs_tol=1e-12)
assert math.isclose(float(-(y * p.log()).sum()), float(-p[0].log()), abs_tol=1e-12)
binary = []
for target in [0.0, 1.0]:
    value = float(
        F.binary_cross_entropy(
            torch.tensor(0.8, dtype=torch.float64), torch.tensor(target, dtype=torch.float64)
        )
    )
    expected = -math.log(0.8 if target == 1 else 0.2)
    assert math.isclose(value, expected, abs_tol=1e-12)
    binary.append({"target_yes": target, "p_yes": 0.8, "loss": value})

proper_example = []
truth = [0.7, 0.3]
for predicted in [[0.7, 0.3], [0.9, 0.1], [0.5, 0.5]]:
    ce = -sum(t * math.log(v) for t, v in zip(truth, predicted, strict=True))
    brier = sum(
        truth[k] * sum((v - float(j == k)) ** 2 for j, v in enumerate(predicted)) for k in range(2)
    )
    proper_example.append({"truth": truth, "predicted": predicted, "expected_ce": ce, "expected_brier_sum": brier})
assert proper_example[0]["expected_ce"] < min(row["expected_ce"] for row in proper_example[1:])
assert proper_example[0]["expected_brier_sum"] < min(row["expected_brier_sum"] for row in proper_example[1:])

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-14-v1/verify_lesson.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "example_dtype": "torch.float32",
        "supplementary_dtype": "torch.float64",
        "default_dtype": str(torch.get_default_dtype()),
    },
    "source_snapshot": page["snapshot"],
    "source_sha256": page["source_sha256"],
    "source_freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
    "exact_embedded_code": original,
    "exact_embedded_code_sha256": hashlib.sha256(original.encode()).hexdigest(),
    "original_stdout": original_stdout,
    "exercise_change": "Append torch.tensor([1 / 3, 1 / 3, 1 / 3]) to the same loop; all other lines unchanged.",
    "exercise_code_sha256": hashlib.sha256(exercise.encode()).hexdigest(),
    "exercise_stdout": exercise_stdout,
    "comparison": rows,
    "supplementary_checks": {"soft_q_ce": soft_ce, "binary": binary, "proper_binary_example": proper_example},
    "result": "All literal-example, exercise and supplementary assertions passed.",
    "scope": "Deterministic CPU loss arithmetic only. No learned model, optimizer update, calibration fit, empirical quality claim, GPU or paid API.",
}
with OUT.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"result": result["result"], "original_stdout": original_stdout, "exercise_stdout": exercise_stdout}, ensure_ascii=False))
