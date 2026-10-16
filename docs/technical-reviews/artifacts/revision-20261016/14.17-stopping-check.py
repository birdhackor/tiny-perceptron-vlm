"""Execute the frozen lesson's artificial counterexample; no model training."""

import contextlib
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
body = lesson_slices((root / "course/chapters/14.md").read_bytes().decode())["14.17"]
digest = hashlib.sha256(body.encode()).hexdigest()
assert digest == "0c88d27a6d9451a0a2617e07e9ccc60034cb567b59327e80bd0e8aa0e464ef62"
code = re.findall(r"```python\n(.*?)```", body, flags=re.S)[0]
namespace = {}
out = io.StringIO()
with contextlib.redirect_stdout(out):
    exec(compile(code, "course/chapters/14.md#14.17", "exec"), namespace)
predictions = namespace["predictions"]
truth = 1
rows = []
for round_id, p in enumerate(predictions, start=1):
    assert p.shape == (2,)
    assert bool(torch.all(p >= 0))
    assert abs(float(p.sum()) - 1) < 1e-7
    rows.append({
        "round": round_id,
        "shape": list(p.shape),
        "dtype": str(p.dtype),
        "probabilities": p.tolist(),
        "argmax": int(p.argmax()),
        "confidence": float(p.max()),
        "correct": int(p.argmax()) == truth,
    })
first_stop = next(row for row in rows if row["confidence"] > 0.9)
assert [row["argmax"] for row in rows] == [0, 0]
assert all(not row["correct"] for row in rows)
assert first_stop["round"] == 1 and not first_stop["correct"]
print(json.dumps({
    "lesson_id": "14.17",
    "source_sha256": digest,
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/14.17-stopping-check.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": "cpu",
    },
    "executed_exact_lesson_code": code,
    "lesson_stdout": out.getvalue(),
    "offline_truth": truth,
    "rows": rows,
    "threshold": 0.9,
    "first_stop_round": first_stop["round"],
    "first_stop_correct": first_stop["correct"],
    "result": "All assertions passed; both argmax values are 0, and the first confidence-based stop is wrong.",
    "scope": "Artificial two-round classification probabilities only; no training, calibration fit, deployment, GPU timing, or model quality measurement.",
}, ensure_ascii=False, indent=2))
