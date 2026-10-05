"""Bounded independent CPU checks for section 4.3, including intentional perturbations."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None
torch.set_default_device("cpu")
environment = {"python": sys.version, "torch": str(torch.__version__),
               "torch_git_version": str(torch.version.git_version), "platform": platform.platform(),
               "device": "cpu", "threads": "1", "cuda_build": str(torch.version.cuda)}
(ROOT / "probe-environment.json").write_text(json.dumps(environment, indent=2) + "\n")

def scalar_row(row, eps=1e-5, gamma=1., beta=0.):
    # Independent Python arithmetic, with denominator equal to the feature count.
    m = sum(row) / len(row)
    v = sum((value - m) ** 2 for value in row) / len(row)
    sd = math.sqrt(v)
    output = [gamma * (value - m) / math.sqrt(v + eps) + beta for value in row]
    return {"mean": m, "variance": v, "standard_deviation_without_epsilon": sd,
            "feature_denominator": len(row), "output": output,
            "output_variance_expected": gamma ** 2 * v / (v + eps)}

rows = [[1., 2., 3.], [10., 20., 30.]]
x = torch.tensor([rows])
layer = nn.LayerNorm(3)
weights_before = {key: value.detach().clone() for key, value in layer.state_dict().items()}
y = layer(x)
expected = [scalar_row(row) for row in rows]
manual = torch.tensor([[item["output"] for item in expected]])
assert y.shape == (1, 2, 3)
assert torch.allclose(y, manual, atol=1e-6, rtol=0)
assert torch.allclose(y.mean(-1), torch.zeros(1, 2), atol=1e-6, rtol=0)
assert torch.allclose(y.var(-1, unbiased=False), y.var(-1, correction=0), atol=0, rtol=0)
assert torch.allclose(y.var(-1, unbiased=False), torch.tensor([[item["output_variance_expected"] for item in expected]]), atol=2e-7, rtol=0)
assert torch.equal(layer.weight, torch.ones(3)) and torch.equal(layer.bias, torch.zeros(3))
assert layer.eps == 1e-5

shift = x.clone()
shift[0, 1] += 100
shifted = layer(shift)
translation_equal = [torch.allclose(y[0, i], shifted[0, i]) for i in range(2)]
assert translation_equal == [True, True]
single = x.clone()
single[0, 1, 0] += 100
single_y = layer(single)
single_equal = [torch.allclose(y[0, i], single_y[0, i]) for i in range(2)]
assert single_equal == [True, False]

with torch.no_grad():
    layer.weight.fill_(2)
    layer.bias.fill_(1)
affine_y = layer(x)
assert torch.allclose(affine_y, 2*y + 1, atol=1e-6, rtol=0)
assert torch.allclose(affine_y.mean(-1), torch.ones(1, 2), atol=1e-6, rtol=0)
assert torch.allclose(affine_y.var(-1, correction=0), 4*y.var(-1, correction=0), atol=1e-6, rtol=0)

ordinary = nn.LayerNorm(3)
constant_y = ordinary(torch.full((1,2,3), 100.))
assert torch.isfinite(constant_y).all() and torch.equal(constant_y, torch.zeros_like(constant_y))
cross_batch = torch.tensor([rows, [[8., 2., -5.], [-1., 6., 4.]]])
batch_y = ordinary(cross_batch)
cross_batch[1] += torch.tensor([[200., 2., 1.], [-500., 4., 0.]])
batch_changed_y = ordinary(cross_batch)
assert torch.equal(batch_y[0], batch_changed_y[0])

wrong = nn.LayerNorm((2,3), elementwise_affine=False)
wrong_y = wrong(x)
wrong_shift_y = wrong(shift)
wrong_axis_early_change = (wrong_y[0,0] - wrong_shift_y[0,0]).abs().max().item()
assert wrong_axis_early_change > 0.1

shared = nn.LayerNorm(3)
identical = torch.tensor([[[1.,2.,3.],[1.,2.,3.]], [[1.,2.,3.],[1.,2.,3.]]])
with torch.no_grad():
    shared.weight.copy_(torch.tensor([1.,2.,3.]))
    shared.bias.copy_(torch.tensor([4.,5.,6.]))
shared_y = shared(identical)
assert shared.weight.shape == (3,) and shared.bias.shape == (3,)
assert torch.equal(shared_y[0,0], shared_y[0,1]) and torch.equal(shared_y[0,0], shared_y[1,0])
assert y.requires_grad and all(torch.equal(weights_before[key], ordinary.state_dict()[key]) for key in weights_before)

installed_file = Path(inspect.getfile(nn.LayerNorm))
installed_raw = installed_file.read_bytes()
(ROOT / "torch-normalization-installed-local.py").write_bytes(installed_raw)
official_raw = (ROOT / "torch-normalization-installed-commit.py").read_bytes()
installed_sha = hashlib.sha256(installed_raw).hexdigest()
official_sha = hashlib.sha256(official_raw).hexdigest()
assert installed_sha == official_sha
results = {"axis_order": ["batch", "position", "feature"], "input_shape": list(x.shape),
           "epsilon": layer.eps, "independent_arithmetic": expected,
           "float32_output": y.detach().tolist(), "mean": y.mean(-1).detach().tolist(),
           "variance_denominator_N": y.var(-1, unbiased=False).detach().tolist(),
           "max_manual_error": (y-manual).abs().max().item(),
           "tenfold_scale_max_difference": (y[0,0]-y[0,1]).abs().max().item(),
           "whole_row_shift_equal": translation_equal,
           "whole_row_shift_max_difference": (y-shifted).abs().max().item(),
           "single_feature_shift_equal": single_equal,
           "single_feature_shift_output": single_y.detach().tolist(),
           "gamma_2_beta_1_mean": affine_y.mean(-1).detach().tolist(),
           "gamma_2_beta_1_variance": affine_y.var(-1, correction=0).detach().tolist(),
           "constant_finite_zero": True, "other_batch_preserved_exactly": True,
           "wrong_axis_early_change": wrong_axis_early_change,
           "affine_parameter_shape": list(shared.weight.shape), "affine_shared_across_positions_and_batch": True,
           "no_backward_or_optimizer_step": True,
           "installed_source_provenance": {"path": str(installed_file), "sha256": installed_sha,
                                           "official_commit_sha256": official_sha, "identical": True}}
(ROOT / "probe-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
