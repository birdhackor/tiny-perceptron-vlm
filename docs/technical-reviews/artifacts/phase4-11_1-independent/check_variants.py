"""Bounded CPU checks for 11.1; no training, generation, download or saved model."""
import contextlib
import hashlib
import io
import json
import platform
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
first = (HERE / "original-fences/fence-1.py").read_text()
zero = (HERE / "original-fences/fence-2.py").read_text()
namespace = {}
baseline_stdout = io.StringIO()
with contextlib.redirect_stdout(baseline_stdout):
    exec(compile(first, "original-fence-1", "exec"), namespace)
projector, a, b = (namespace[k] for k in ("projector", "a", "b"))
baseline = namespace["difference"].item()
diff = projector(a) - projector(b)
analytic_norm = diff.square().sum().sqrt().item()
assert tuple(projector.weight.shape) == (8, 4)
assert tuple(projector.bias.shape) == (8,)
assert tuple(projector(a).shape) == (1, 8)
assert baseline > 0 and abs(baseline - analytic_norm) < 1e-6
exec(compile(zero, "original-fence-2-after-difference", "exec"), namespace)
stale = namespace["difference"].item()
fresh = (projector(a) - projector(b)).norm().item()
assert stale == baseline and fresh == 0.0
assert torch.equal(projector(a), projector(b))
assert torch.equal(projector(a), projector.bias.unsqueeze(0))

insert_at = first.index("difference =")
before_code = first[:insert_at] + zero + first[insert_at:]
(HERE / "variant-zero-before.py").write_text(before_code)
before_namespace = {}
before_stdout = io.StringIO()
with contextlib.redirect_stdout(before_stdout):
    exec(compile(before_code, "variant-zero-before", "exec"), before_namespace)
assert before_namespace["difference"].item() == 0.0
assert "False" in before_stdout.getvalue()

hand = nn.Linear(4, 1)
with torch.no_grad():
    hand.weight.copy_(torch.tensor([[1.0, 2.0, 0.0, 0.0]]))
    hand.bias.fill_(0.5)
hand_values = [hand(x).item() for x in (a, b)]
assert hand_values == [0.5, 3.5]
hand_delta = (hand(a) - hand(b)).norm().item()
assert hand_delta == 3.0

# A nonzero connector can also erase this specific input difference.
# This is a counterexample to treating nonzero weights as proof of sensitivity.
with torch.no_grad():
    hand.weight.copy_(torch.tensor([[1.0, -1.0, 0.0, 0.0]]))
nullspace_delta = (hand(a) - hand(b)).norm().item()
assert hand.weight.abs().sum().item() > 0 and nullspace_delta == 0.0

# Linear operates on the feature axis while preserving batch/position axes.
torch.manual_seed(0)
wide = nn.Linear(4, 8)
batched = wide(torch.stack((a, b), dim=1))
assert tuple(batched.shape) == (1, 2, 8)
assert wide.weight.grad is None and wide.bias.grad is None

result = {
    "environment": {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu", "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()),
    },
    "original_fence_hashes": {
        name: hashlib.sha256((HERE / "original-fences" / name).read_bytes()).hexdigest()
        for name in ("fence-1.py", "fence-2.py")
    },
    "baseline": {"stdout": baseline_stdout.getvalue(), "norm": baseline,
                 "analytic_l2_norm": analytic_norm, "output_shape": [1, 8],
                 "weight_shape": [8, 4], "bias_shape": [8]},
    "zero_after": {"saved_difference": stale, "recomputed_difference": fresh,
                   "outputs_equal_bias": True},
    "zero_before": {"stdout": before_stdout.getvalue(), "difference": 0.0},
    "hand_affine_row": {"weight": [1, 2, 0, 0], "bias": 0.5,
                        "outputs": hand_values, "l2_difference": hand_delta},
    "nonzero_weight_nullspace_control": {"weight": [1, -1, 0, 0], "difference": nullspace_delta},
    "axis_control": {"input_shape": [1, 2, 4], "output_shape": list(batched.shape),
                     "axes": ["batch", "position", "feature"]},
    "coverage": ["imports and assignment", "manual_seed and Linear initialization",
                 "zeros/ones", "forward affine calculation and shape/tuple",
                 "subtraction and default norm over all elements", "item and > comparison",
                 "no_grad context and in-place zero_", "old tensor versus recomputed forward",
                 "print output", "manual affine example", "extra position axis"],
    "scope": "Two artificial inputs and deterministic affine controls. No image model, labels, training, evaluation, optimizer, backward, GPU or model file. Norm is an unnormalised dimensionless L2 length over eight feature values, not a success rate. No empirical dataset/token/step denominator exists in 11.1.",
}
(HERE / "variants-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
