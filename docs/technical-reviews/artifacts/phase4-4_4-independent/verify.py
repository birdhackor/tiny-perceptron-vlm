"""Bounded independent CPU checks of lesson 4.4; no training or external I/O."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.modern import DenseFFN

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
out = Path(__file__).resolve().parent
namespace = {"__name__": "__main__"}
fence = (out / "original/fence-1.py").read_bytes()
print("ORIGINAL FENCE")
exec(compile(fence, "original/fence-1.py", "exec"), namespace)
layer, x, y = (namespace[name] for name in ("layer", "x", "y"))
before = {name: value.detach().clone() for name, value in layer.named_parameters()}

print("INDEPENDENT CHECKS")
up = layer.up(x)
activated = F.gelu(up)
manual_up = x @ layer.up.weight.T + layer.up.bias
manual_activated = 0.5 * manual_up * (1 + torch.erf(manual_up / math.sqrt(2)))
manual_output = manual_activated @ layer.down.weight.T + layer.down.bias
manual_difference = (manual_output - y).abs().max().item()
assert up.shape == activated.shape == (1, 2, 16)
assert y.shape == (1, 2, 4)
assert layer.gate is None and layer.activation == "gelu"
assert torch.equal(y[:, 0], y[:, 1])
assert torch.allclose(manual_output, y, rtol=1e-5, atol=1e-6)

# The stated exercise changes just feature 0 of position 1.
x[0, 1, 0] = 2
changed = layer(x)
position0_diff = (changed[:, 0] - y[:, 0]).abs().max().item()
position1_diff = (changed[:, 1] - y[:, 1]).abs().max().item()
assert torch.equal(changed[:, 0], y[:, 0])
assert not torch.allclose(changed[:, 1], y[:, 1])
# Each position's computation also agrees with applying the module to that row alone.
rowwise = torch.stack([layer(x[0, i]) for i in range(2)])[None]
assert torch.allclose(rowwise, changed, rtol=1e-5, atol=1e-6)
# Permuting position order only permutes outputs; the module has no position lookup.
permuted = layer(x.flip(1))
assert torch.equal(permuted, changed.flip(1))

# GELU scaling is checked from the original formula, independently of the kernel.
inputs = torch.tensor([-3., -1., -0.5, 0., 0.5, 1., 3.], dtype=torch.float64)
analytic = torch.tensor([v * (1 + math.erf(v / math.sqrt(2))) / 2 for v in inputs.tolist()], dtype=torch.float64)
gelu = F.gelu(inputs)
relu = F.relu(inputs)
gelu_difference = (gelu - analytic).abs().max().item()
assert torch.allclose(gelu, analytic, rtol=0, atol=1e-14)
assert (gelu[:3] < 0).all() and torch.equal(relu[:3], torch.zeros(3, dtype=torch.float64))
retained_fraction = [(1 + math.erf(v / math.sqrt(2))) / 2 for v in inputs.tolist()]
assert all(a < b for a, b in zip(retained_fraction, retained_fraction[1:]))

# Repeating initialization under the same runtime seed reproduces parameters.
torch.manual_seed(42)
repeat = DenseFFN(4)
assert all(torch.equal(repeat.state_dict()[name], value) for name, value in before.items())
assert all(parameter.grad is None for parameter in layer.parameters())
assert all(torch.equal(before[name], value) for name, value in layer.named_parameters())

facts = {
    "axis_order": ["batch", "position", "feature"],
    "dimension_unit": "number of feature slots (not token count)",
    "input_shape": list(x.shape), "up_shape": list(up.shape),
    "activated_shape": list(activated.shape), "output_shape": list(y.shape),
    "up_weight_shape": list(layer.up.weight.shape), "down_weight_shape": list(layer.down.weight.shape),
    "identical_position_max_abs_difference": (y[:, 0] - y[:, 1]).abs().max().item(),
    "manual_formula_max_abs_difference": manual_difference,
    "manual_formula_tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "perturbation": "x[0,1,0]: 1 -> 2; all other input features unchanged",
    "changed_output": changed.detach().tolist(),
    "unchanged_position0_max_abs_difference": position0_diff,
    "changed_position1_max_abs_difference": position1_diff,
    "rowwise_max_abs_difference": (rowwise - changed).abs().max().item(),
    "position_permutation_equivariance_exact": True,
    "gelu_inputs": inputs.tolist(), "gelu_outputs": gelu.tolist(), "relu_outputs": relu.tolist(),
    "gelu_retained_fraction_phi": retained_fraction,
    "gelu_vs_math_erf_max_abs_difference": gelu_difference,
    "gelu_formula_tolerance": {"atol": 1e-14, "rtol": 0},
    "parameter_gradients_all_none": True, "parameters_unchanged": True,
    "same_runtime_seed_reinitialization_exact": True,
    "sample_scope": "one batch, two positions, four features; seven scalar GELU probes; no quality metric or statistical denominator",
    "backward_calls": 0, "optimizer_steps": 0,
}
environment = {
    "python": sys.version, "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
    "device": str(x.device), "dtype": str(x.dtype), "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform(),
    "torch_threads": str(torch.get_num_threads()),
    "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["tiny_perceptron/modern.py", "tiny_perceptron/model.py", "docs/review-tools/section_facts.py"]},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "fence_sha256": hashlib.sha256(fence).hexdigest(),
}
(out / "variation-results.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
(out / "variation-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(facts, ensure_ascii=False, indent=2))
print("PASS: exact source fence and all bounded CPU assertions")
