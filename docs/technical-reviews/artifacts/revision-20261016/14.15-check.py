"""Independent, offline mechanism checks for frozen lesson 14.15; no training."""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices


root = Path(__file__).resolve().parents[4]
body = dict(lesson_slices((root / "course/chapters/14.md").read_text(encoding="utf-8")))["14.15"]
code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(code, "frozen-14.15-block-0", "exec"), {})
stdout = capture.getvalue()
expected = "1 0.9\n2 0.1\n3 0.9\n4 0.1\n5 0.9\n6 0.1\n"
assert stdout == expected, (stdout, expected)

# Same scalar update at every step, with only a terminal loss. This checks
# the existence of the indirect path, without claiming universal nonzero grads.
w = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
states = [torch.tensor(0.0, dtype=torch.float64)]
for _ in range(5):
    state = w * states[-1] + 1.0
    state.retain_grad()
    states.append(state)
loss = (states[5] - 2.0).square()
loss.backward()
expected_grads = {"2": -0.015625, "4": -0.0625}
observed_grads = {str(i): states[i].grad.item() for i in (2, 4)}
assert observed_grads == expected_grads

supervised = {1, 3, 5}
interior_unseen = [i for i in range(1, 7) if min(supervised) < i < max(supervised) and i not in supervised]
exterior = [i for i in range(1, 7) if i > max(supervised)]
assert interior_unseen == [2, 4]
assert exterior == [6]

print(json.dumps({
    "schema_version": 1,
    "lesson_id": "14.15",
    "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    "command": "PYTHONPATH=. uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/14.15-check.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "torch.float64 (gradient example)"},
    "authored_snippet": {"code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(), "stdout": stdout, "expected_stdout": expected, "exact_match": stdout == expected},
    "terminal_only_gradient_check": {"update": "s_i = w*s_(i-1)+1", "s0": 0.0, "w": w.item(), "states": [s.item() for s in states], "loss": "(s5-2)^2", "loss_value": loss.item(), "direct_loss_depths": [5], "observed_state_grads": observed_grads, "expected_state_grads": expected_grads, "w_grad": w.grad.item(), "derivation": "dL/ds_i=2*(s5-2)*w**(5-i); dL/ds2=2*(-1/16)*(1/2)^3=-1/64; dL/ds4=2*(-1/16)*(1/2)=-1/16", "scope": "A differentiable untruncated graph. No intermediate direct loss; no parameter update or training. Nonzero gradients in this example do not imply nonzero gradients for every model/input."},
    "depth_sets": {"directly_supervised": sorted(supervised), "in_range_unseen": interior_unseen, "above_max": exterior, "after_adding_2": [1, 2, 3, 5], "4_remains_unsupervised": 4 not in (supervised | {2})},
    "result": "All assertions passed. The shared sign-flip counterexample and terminal-only gradient path agree with the lesson. No model-quality, GPU timing or memory result is claimed."
}, ensure_ascii=False, indent=2))
