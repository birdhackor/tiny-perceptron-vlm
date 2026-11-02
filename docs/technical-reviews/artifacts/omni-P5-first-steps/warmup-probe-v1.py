"""Reviewer-owned CPU checks of the frozen first-steps page; no formal models."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v1/sources/first-steps.md"
body_bytes = BODY.read_bytes()
assert hashlib.sha256(body_bytes).hexdigest() == "781bc5ccb7bf8f0c716e753f643ef64b20f3cc6baf28dc92008b9ca5e9185516"
assert body_bytes == (ROOT / "course/first-steps.md").read_bytes()
torch.set_num_threads(1)
torch.manual_seed(42)
rows = []
namespace = {}
for section in re.split(r"(?=^## W\.)", body_bytes.decode(), flags=re.M)[1:]:
    heading = section.splitlines()[0]
    for index, code in enumerate(re.findall(r"```python\n(.*?)```", section, re.S)):
        output = StringIO()
        with redirect_stdout(output):
            exec(compile(code, f"{heading}:block{index}", "exec"), namespace)
        rows.append({"heading": heading, "block": index, "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "stdout": output.getvalue()})

checks = {}
def check(name, observed, expected):
    assert observed == expected, (name, observed, expected)
    checks[name] = {"observed": observed, "expected": expected}

animals = ["鳥", "貓", "狗"]
check("W2-exercise-list", [animals[0], len(animals), len([f"看到 {x}" for x in animals])], ["鳥", 3, 3])
record = {"animal": "貓", "caption": "另一段文字"}
check("W2-caption-does-not-change-list", animals, ["鳥", "貓", "狗"])
def first(items):
    return items[0]
check("W2-function", first(animals), "鳥")
scores_list = [5, 9, 7]
table_examples = {
    "slices": [[5, 2, 7][:-1], [5, 2, 7][1:]],
    "range": list(range(3)),
    "set": len(set("貓狗貓")),
    "unicode": [ord("狗"), ord("貓")],
    "sorted": [sorted([3, 1, 2]), sorted([3, 1, 2], reverse=True), sorted(set("貓狗貓"))],
    "keys": [max(["貓", "小花貓"], key=len), max(range(3), key=lambda i: scores_list[i])],
    "zip": list(zip(["貓", "狗"], [3, 1])),
    "dict_zip": dict(zip(["貓", "狗"], [0, 1])),
    "enumerate": list(enumerate(["貓", "狗"])),
    "reversed": list(reversed(["貓", "狗", "鳥"])),
    "join": ["".join(["貓", "看", "狗"]), "、".join(["貓", "狗"])],
    "comprehensions": [[x * 2 for x in [1, 2, 3]], sum(x * 2 for x in [1, 2, 3])],
    "next": [next(x for x in [3, 8, 12] if x > 5), next((x for x in [3] if x > 5), None)],
    "f_string": f"看到{'貓'}",
    "floor": [7 // 2, -7 // 2],
}
expected_table = {"slices": [[5, 2], [2, 7]], "range": [0, 1, 2], "set": 2, "unicode": [29399, 35987], "sorted": [[1, 2, 3], [3, 2, 1], ["狗", "貓"]], "keys": ["小花貓", 1], "zip": [("貓", 3), ("狗", 1)], "dict_zip": {"貓": 0, "狗": 1}, "enumerate": [(0, "貓"), (1, "狗")], "reversed": ["鳥", "狗", "貓"], "join": ["貓看狗", "貓、狗"], "comprehensions": [[2, 4, 6], 12], "next": [8, None], "f_string": "看到貓", "floor": [3, -4]}
check("W2-entire-reference-table", table_examples, expected_table)

scores = torch.tensor([[60.0, 70.0, 80.0], [80.0, 90.0, 100.0]])
check("W3-conversions", [list(tuple(scores.shape)), scores[0].tolist(), scores[1, 2].item(), scores.tolist()], [[2, 3], [60., 70., 80.], 100., [[60., 70., 80.], [80., 90., 100.]]])
check("W3-means", [scores.mean(1).tolist(), scores.mean(0).tolist(), scores.mean(-1).tolist()], [[70., 90.], [70., 80., 90.], [70., 90.]])
check("W3-reshape-round-trip", scores.reshape(6).reshape(2, 3).tolist(), scores.tolist())
check("W3-BTD-index", list(torch.zeros(2, 4, 3)[0].shape), [4, 3])
scores[0, 2] = 110.
check("W3-exercise", [scores.mean(1).tolist(), scores.mean(0).tolist()], [[80., 90.], [70., 80., 105.]])
for name, call in [("W3-item-multiple", lambda: scores.item()), ("W3-reshape-eight", lambda: scores.reshape(8))]:
    try:
        call()
    except RuntimeError as error:
        checks[name] = {"exception": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError(name)

check("W4-costs", [round(-math.log(p), 3) for p in [.9, .5, .1, .25]], [.105, .693, 2.303, 1.386])
assert math.isclose(-math.log(.5 * .5), -math.log(.5) - math.log(.5), abs_tol=1e-15)
checks["W4-product"] = {"joint_probability": .25, "cost": -math.log(.25), "sum": -math.log(.5)-math.log(.5)}
try:
    math.log(0)
except ValueError as error:
    checks["W4-log-zero"] = {"exception": type(error).__name__, "message": str(error)}
from torch.nn import functional as F
logits = torch.tensor([[2., -1.]])
check("W4-softmax-total", F.softmax(logits, dim=-1).sum().item(), 1.)
assert torch.allclose(F.cross_entropy(logits, torch.tensor([0])), -F.log_softmax(logits, -1)[0, 0])
checks["W4-cross-entropy"] = {"raw_logits": logits.tolist(), "target": 0, "cross_entropy": F.cross_entropy(logits, torch.tensor([0])).item(), "negative_log_probability": -F.log_softmax(logits, -1)[0, 0].item()}

recipes = namespace["recipes"]
layer = namespace["layer"]
check("W5-manual-matmul", namespace["result"].tolist(), [[40., 1.], [80., 6.]])
check("W5-linear", layer(recipes).tolist(), [[45., 1.], [85., 6.]])
with torch.no_grad():
    layer.bias.copy_(torch.tensor([8., 0.]))
check("W5-bias-exercise", layer(recipes).tolist(), [[48., 1.], [88., 6.]])
try:
    recipes @ torch.ones(2, 2)
except RuntimeError as error:
    checks["W5-incompatible-shape"] = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("Expected inner-dimension mismatch")

check("W6-exact-hand-values", [(1.-3.)**2, round((1.1-3.)**2, 2), round(((1.1-3.)**2-4.)/.1, 1)], [4., 3.61, -3.9])
checks["W6-derivative-limit"] = [{"h": h, "difference_quotient": (((1.+h-3.)**2)-4.)/h} for h in [.1, .01, .001, .0001]]
for lr, expected_w, expected_loss in [(.1, 1.4, 2.56), (.6, 3.4, .16), (1.1, 5.4, 5.76)]:
    w = torch.tensor(1., requires_grad=True)
    loss = (w - 3.).square()
    loss.backward()
    check(f"W6-backward-no-update-{lr}", [w.item(), loss.item(), w.grad.item()], [1., 4., -4.])
    with torch.no_grad():
        w -= lr * w.grad
    actual_loss = (w - 3.).square().item()
    assert abs(w.item()-expected_w) < 1e-6 and abs(actual_loss-expected_loss) < 2e-6
    checks[f"W6-update-{lr}"] = {"w": w.item(), "loss": actual_loss, "expected_w": expected_w, "expected_loss": expected_loss, "absolute_tolerance": 2e-6}
w = torch.tensor(1., requires_grad=True)
(w-3).square().backward()
(w-3).square().backward()
check("W6-gradient-accumulation", w.grad.item(), -8.)
w.grad = None
(w-3).square().backward()
check("W6-cleared-gradient", w.grad.item(), -4.)

old_cwd = Path.cwd()
try:
    path_results = []
    for folder in [ROOT, ROOT / "notebooks"]:
        os.chdir(folder)
        path_results.append({"cwd": str(Path.cwd()), "pyproject": Path("pyproject.toml").exists(), "tiny_perceptron": Path("tiny_perceptron").exists()})
    check("W7-relative-paths", [[x["pyproject"], x["tiny_perceptron"]] for x in path_results], [[True, True], [False, False]])
    checks["W7-cwds"] = path_results
    bootstrap = next("".join(c["source"]) for c in json.loads((ROOT/"notebooks/01/1.1.ipynb").read_bytes())["cells"] if c["cell_type"] == "code")
    local_ns = {}
    exec(compile(bootstrap, "1.1-local-bootstrap", "exec"), local_ns)
    check("W1-bootstrap-from-notebooks", str(local_ns["root"]), str(ROOT))
finally:
    os.chdir(old_cwd)
notebook = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_bytes())
lesson_code = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and 'text = "貓看狗，狗看貓。"' in "".join(c["source"]))
for sentence in ["貓看狗，狗看貓。", "鳥看狗，狗看鳥。"]:
    ns = {}
    with redirect_stdout(StringIO()) as output:
        exec(lesson_code.replace("貓看狗，狗看貓。", sentence), ns)
    check("W1-restored-" + sentence, ns["restored"], sentence)
    checks["W1-output-" + sentence] = output.getvalue()

print(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": hashlib.sha256(body_bytes).hexdigest(), "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "CPU", "threads": "1"}, "exact_fenced_python_blocks": rows, "supplementary_checks": checks, "limits": "Artificial inline warmup computations only; no training, final-family inference, downloads, Modal, dispatch, or publishing."}, ensure_ascii=False, indent=2))
