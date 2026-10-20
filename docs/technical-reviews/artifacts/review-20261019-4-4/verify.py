import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.modern import DenseFFN as LibraryDenseFFN

frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/4.4.md"
raw = frozen.read_bytes()
namespace = {}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    for block in re.findall(r"```python\n(.*?)```", raw.decode(), re.S):
        exec(compile(block, str(frozen), "exec"), namespace)
layer, x, y = (namespace[key] for key in ("layer", "x", "y"))
before = {name: value.detach().clone() for name, value in layer.state_dict().items()}
shapes = [list(x.shape), list(layer.up(x).shape), list(F.gelu(layer.up(x)).shape), list(y.shape)]
same_positions = torch.equal(y[0, 0], y[0, 1])
x[0, 1, 0] = 2
changed = layer(x)
exercise = [torch.equal(changed[0, 0], y[0, 0]), torch.equal(changed[0, 1], y[0, 1])]
assert exercise == [True, False]
values = [1.0, -1.0, -0.5]
exact = [value * 0.5 * (1 + math.erf(value / math.sqrt(2))) for value in values]
actual = F.gelu(torch.tensor(values, dtype=torch.float64)).tolist()
rounded_expected = [0.841, -0.159, -0.154]
assert max(abs(a - b) for a, b in zip(actual, exact)) < 1e-14
assert max(abs(a - b) for a, b in zip(actual, rounded_expected)) < 0.0005
branches = {}
for activation in ("gelu", "relu2", "swiglu"):
    module = LibraryDenseFFN(4, activation=activation)
    h = module.up(x)
    if activation == "gelu":
        h = F.gelu(h)
    elif activation == "relu2":
        h = F.relu(h).square()
    else:
        h = F.silu(module.gate(x)) * h
    branches[activation] = {"shape": list(module(x).shape), "formula_matches": torch.equal(module(x), module.down(h))}
    assert branches[activation]["formula_matches"]
torch.manual_seed(42)
recreated = namespace["DenseFFN"](4)
seed_recreates_parameters = all(torch.equal(before[key], recreated.state_dict()[key]) for key in before)
assert seed_recreates_parameters
assert same_positions and all(torch.equal(value, layer.state_dict()[key]) for key, value in before.items())
assert all(parameter.grad is None for parameter in layer.parameters())
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-4/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": "float32 model; float64 GELU calculation"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "frozen_code_stdout": stream.getvalue(),
    "shape_path": shapes,
    "identical_position_outputs": same_positions,
    "exercise_equal_before_after": exercise,
    "exercise_changed_output": changed.detach().tolist(),
    "gelu": {"inputs": values, "formula": "x * (1 + erf(x / sqrt(2))) / 2", "hand_formula_evaluated": exact, "torch_float64": actual, "printed_approximations": rounded_expected, "max_formula_error": max(abs(a - b) for a, b in zip(actual, exact))},
    "bias_shapes": {"up": list(layer.up.bias.shape), "down": list(layer.down.bias.shape)},
    "hidden_default": layer.up.out_features,
    "hidden_explicit": namespace["DenseFFN"](4, hidden=10).up.out_features,
    "hidden_zero_truthiness": namespace["DenseFFN"](4, hidden=0).up.out_features,
    "library_activation_branches": branches,
    "seed_recreates_parameters": seed_recreates_parameters,
    "parameters_unchanged_after_forward": all(torch.equal(value, layer.state_dict()[key]) for key, value in before.items()),
    "parameter_gradients_remain_none": all(parameter.grad is None for parameter in layer.parameters()),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
