"""W.4 independent CPU verification; executes the frozen inline code verbatim."""

from contextlib import redirect_stdout
from datetime import UTC, datetime
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4.json"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "W.4")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
assert page["source_sha256"] == "6335a9ec2d690e6dbf31ca894ae716e49f434f2255460481cadd35c7edb0303d"
assert page["figures_sha256"] == {}
body = raw.decode("utf-8")
assert not re.search(r"!\[|<img\b|<svg\b", body)
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
code = blocks[0]


def run_inline(text):
    namespace = {}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(text, str(snapshot) + "#inline", "exec"), namespace)
    return output.getvalue(), namespace


original_stdout, namespace = run_inline(code)
assert original_stdout == "0.9 0.105\n0.5 0.693\n0.1 2.303\n"
assert namespace["cost"] == -math.log(0.1)
exercise_code = code.replace("[0.9, 0.5, 0.1]", "[0.9, 0.5, 0.1, 0.25]")
assert exercise_code != code
exercise_stdout, _ = run_inline(exercise_code)
assert exercise_stdout == original_stdout + "0.25 1.386\n"

values = [{"p": p, "cost": -math.log(p), "rounded": round(-math.log(p), 3)}
          for p in [0.9, 0.5, 0.1, 0.25]]
assert -math.log(0.25) > -math.log(0.5)
joint = 0.5 * 0.5
joint_cost = -math.log(joint)
sum_cost = -math.log(0.5) - math.log(0.5)
assert math.isclose(joint_cost, sum_cost, rel_tol=0, abs_tol=1e-15)
inverse_pairs = []
for a in [-2.0, -1.0, -0.5, 0.0, 0.5, 2.0]:
    p = math.e ** a
    actual = math.log(p)
    assert math.isclose(actual, a, rel_tol=0, abs_tol=1e-15)
    inverse_pairs.append({"a": a, "e_power_a": p, "log_of_result": actual})
assert math.isclose(math.e ** 2, math.e * math.e, rel_tol=0, abs_tol=1e-15)
assert math.isclose(math.e ** -1, 1 / math.e, rel_tol=0, abs_tol=1e-15)
try:
    math.log(0)
except ValueError as exc:
    zero_error = {"type": type(exc).__name__, "message": str(exc)}
else:
    raise AssertionError("math.log(0) unexpectedly returned")

torch.set_num_threads(1)
logits = torch.tensor([[math.log(0.9), math.log(0.1)],
                       [math.log(0.5), math.log(0.5)],
                       [math.log(0.1), math.log(0.9)]], dtype=torch.float64, device="cpu")
targets = torch.tensor([0, 0, 0], dtype=torch.int64, device="cpu")
with torch.no_grad():
    probabilities = F.softmax(logits, dim=1)
    loss_none = F.cross_entropy(logits, targets, reduction="none")
    direct = -probabilities[torch.arange(3), targets].log()
    loss_mean = F.cross_entropy(logits, targets)
    wrong_pre_softmax = F.cross_entropy(probabilities, targets, reduction="none")
assert torch.allclose(probabilities.sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-15, rtol=0)
assert torch.allclose(loss_none, direct, atol=1e-15, rtol=0)
assert torch.allclose(loss_mean, direct.mean(), atol=1e-15, rtol=0)
assert not torch.allclose(wrong_pre_softmax, direct, atol=1e-3, rtol=0)

pkg = Path(torch.__file__).parent
official_cache = Path("/workspace/work/reference/omni-technical-w-4-v1")
source_matches = []
for downloaded, installed in [("torch-functional.py", pkg / "nn/functional.py"),
                              ("torch-loss.py", pkg / "nn/modules/loss.py"),
                              ("torch-activation.py", pkg / "nn/modules/activation.py")]:
    same = (official_cache / downloaded).read_bytes() == installed.read_bytes()
    assert same
    source_matches.append({"cache_file": downloaded, "installed": str(installed),
                           "identical": same, "sha256": hashlib.sha256(installed.read_bytes()).hexdigest()})

record = {
    "executed_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-4-v1/verify.py",
    "environment": {"python": sys.version, "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version, "platform": platform.platform(),
                    "device": "cpu", "dtype": "Python float; torch.float64; torch.int64 targets",
                    "threads": "1"},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "snapshot": {"path": page["snapshot"], "sha256": hashlib.sha256(raw).hexdigest(),
                 "figure_refs": [], "inline_python_blocks": len(blocks)},
    "inline_original": {"code": code, "inputs": [0.9, 0.5, 0.1], "stdout": original_stdout,
                        "unrounded_final_cost_after_print": namespace["cost"]},
    "exercise": {"edit": "append 0.25 to the inline list", "code": exercise_code, "stdout": exercise_stdout,
                 "cost_0_25_gt_cost_0_5": True},
    "negative_log_values": values,
    "e_and_powers": {"e": math.e, "e_squared": math.e ** 2, "e_times_e": math.e * math.e,
                     "e_power_minus_1": math.e ** -1, "one_divided_by_e": 1 / math.e,
                     "inverse_pairs": inverse_pairs},
    "independence_example": {"p1": 0.5, "p2": 0.5, "joint_probability": joint,
                             "joint_cost": joint_cost, "sum_of_unrounded_costs": sum_cost,
                             "absolute_error": abs(joint_cost - sum_cost)},
    "math_log_zero": zero_error,
    "cross_entropy_example": {"logits": logits.tolist(), "targets": targets.tolist(),
                              "target_shape": list(targets.shape), "input_shape": list(logits.shape),
                              "probabilities": probabilities.tolist(), "row_sums": probabilities.sum(1).tolist(),
                              "reduction_none": loss_none.tolist(), "direct_negative_log_p": direct.tolist(),
                              "default_mean": loss_mean.item(), "direct_mean": direct.mean().item(),
                              "incorrect_probability_input_losses": wrong_pre_softmax.tolist(),
                              "settings": "weight=None, ignore_index=-100 (no target ignored), label_smoothing=0.0"},
    "installed_vs_official_source": source_matches,
    "result": "PASS: exact frozen inline output, appended-0.25 exercise, powers/log API, log(0), and categorical cross_entropy checks passed",
    "limits": "Deterministic CPU examples; no training, quality, speed or GPU claim. Real logarithm identities are mathematical statements, not a floating-point exactness guarantee."
}
print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
