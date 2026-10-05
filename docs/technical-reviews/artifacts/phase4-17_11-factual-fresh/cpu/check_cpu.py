"""Independent bounded CPU verification of 17.11; no training or weight files."""
import ast
import hashlib
import json
import platform
import sys
from fractions import Fraction as R
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.quantization import QuantizedLinear, quantize_symmetric

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent / "original"
torch.set_default_device("cpu")
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def mm(a, b):
    return [[sum(v * z for v, z in zip(row, col)) for col in zip(*b)] for row in a]


def trans(a):
    return [list(x) for x in zip(*a)]


def sub(a, b):
    return [[x - y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def mae(a, b):
    return sum(abs(v) for row in sub(a, b) for v in row) / sum(map(len, a))


xr = [[R(7, 10), R(1, 5)], [R(-7, 10), R(6, 5)]]
wr = xr
qr = [[R(1, 2), R(0)], [R(-1, 2), R(1)]]
reference_r, weight_r, both_r = mm(xr, trans(wr)), mm(xr, trans(qr)), mm(qr, trans(qr))
assert reference_r == [[R(53, 100), R(-1, 4)], [R(-1, 4), R(193, 100)]]
assert weight_r == [[R(7, 20), R(-3, 20)], [R(-7, 20), R(31, 20)]]
assert both_r == [[R(1, 4), R(-1, 4)], [R(-1, 4), R(5, 4)]]
assert mae(reference_r, weight_r) == R(19, 100)
assert mae(reference_r, both_r) == R(6, 25)
exact = {
    "reference": [[str(v) for v in row] for row in reference_r],
    "weight_only": [[str(v) for v in row] for row in weight_r],
    "both": [[str(v) for v in row] for row in both_r],
    "absolute_error_sums": ["19/25 = 0.76", "24/25 = 0.96"],
    "MAE_denominator": 4,
    "MAE": ["19/100 = 0.19", "6/25 = 0.24"],
}

x = torch.tensor([[0.7, 0.2], [-0.7, 1.2]], dtype=torch.float64)
w = x.clone()
q = lambda a, scale=0.5: (a / scale).round() * scale
qx, qw = q(x), q(w)
reference, weight_only, both = x @ w.T, x @ qw.T, qx @ qw.T
dw, dx = qw - w, qx - x
weight_error = x @ dw.T
activation_increment = dx @ qw.T
torch.testing.assert_close(both - reference, weight_error + activation_increment, atol=1e-12, rtol=0)
torch.testing.assert_close(activation_increment, both - weight_only, atol=1e-12, rtol=0)
for i, j in [(0, 1), (1, 0)]:
    assert abs(weight_error[i, j].item()) > 0.09
    assert abs((weight_error + activation_increment)[i, j].item()) < 1e-12
exercise = x.clone()
exercise[0] = torch.tensor([1.0, 0.0], dtype=torch.float64)
torch.testing.assert_close((exercise @ qw.T)[0], (q(exercise) @ qw.T)[0], atol=0, rtol=0)
assert (q(exercise) @ qw.T)[0].tolist() == [0.5, -0.5]

xf = x.float()
qxf = q(xf)
official_fake = torch.fake_quantize_per_tensor_affine(xf, 0.5, 0, -128, 127)
assert torch.equal(qxf, official_fake)
assert qxf.dtype == xf.dtype == torch.float32 and qxf.numel() * qxf.element_size() == 16
tie_input = torch.tensor([-0.25, 0.25, 0.75, 1.25])
assert q(tie_input).tolist() == [-0.0, 0.0, 1.0, 1.0]
different_scale = x @ q(w, 0.25).T

source = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    source.weight.copy_(xf)
quantized = QuantizedLinear(source, bits=4)
captures = []
real_linear = F.linear


def capture_linear(a, weight, bias=None):
    captures.append({"input_identity": a is xf, "dtype": str(a.dtype), "shape": list(a.shape)})
    assert a is xf and torch.equal(a, xf)
    return real_linear(a, weight, bias)


with patch.object(F, "linear", capture_linear):
    packed_result = quantized(xf)
namespace = {"nn": nn, "F": SimpleNamespace(linear=capture_linear), "quantize_symmetric": quantize_symmetric}
contract_raw = (OUT / "original-qat-contract.py").read_bytes()
exec(compile(contract_raw, "compression.py@5af615e:_QATLinear/_qat_layers", "exec"), namespace)
qat_layer = namespace["_QATLinear"](source, bits=4)
qat_result = qat_layer(xf)
torch.testing.assert_close(packed_result, qat_result, atol=0, rtol=0)
assert len(captures) == 2 and all(x["input_identity"] for x in captures)
activation_ranges = [[float(a.min()), float(a.max())] for a in (xf, xf * 3)]
assert activation_ranges[0] != activation_ranges[1]

json_checks = []
for stem, names in [("quantization", ["packed4", "packed8"]), ("qat", ["initial_ptq4", "matched_ptq4", "qat_packed4"])]:
    path = ORIGINAL / (stem + "-result.json")
    data = json.loads(path.read_bytes())
    pointers = ["/revision", "/device", "/torch_version", "/seed", "/code_sha256/tiny_perceptron~1quantization.py", "/code_sha256/scripts~1course_experiments~1compression.py"]
    assert data["revision"] == "5af615e5d7c9642afee800390fa072257f895d0c"
    assert data["code_sha256"]["tiny_perceptron/quantization.py"] == hashlib.sha256((ORIGINAL / "quantization.py").read_bytes()).hexdigest()
    assert data["code_sha256"]["scripts/course_experiments/compression.py"] == hashlib.sha256((ORIGINAL / "compression-original.py").read_bytes()).hexdigest()
    summaries = []
    for name in names:
        storage = data["results"]["runs"][name]["storage"]
        pointers += [f"/results/runs/{name}/storage/{k}" for k in ["packed_linear_modules", "retained_float_modules", "buffers"]]
        assert len(storage["packed_linear_modules"]) == 13
        assert all(k.endswith((".values", ".scale", ".bias")) for k in storage["buffers"])
        summaries.append({"run": name, "packed_linear_modules": storage["packed_linear_modules"], "retained_float_modules": storage["retained_float_modules"], "buffer_keys": list(storage["buffers"])})
    activation_flag = data["results"].get("activation_quantization", "field absent: scope checked from original implementation")
    if stem == "qat":
        pointers.append("/results/activation_quantization")
        assert activation_flag is False
    json_checks.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pointers": pointers, "activation_quantization": activation_flag, "runs": summaries})

results = {
    "exact_derivation": exact,
    "shapes": {"x": list(x.shape), "w": list(w.shape), "outputs": list(reference.shape)},
    "axes": "input row = input example; weight row = output feature; last dimension contracts two input features",
    "units": "output value units; MAE averages four output cells, not two input examples",
    "weight_error": weight_error.tolist(),
    "activation_increment": activation_increment.tolist(),
    "exercise_first_row": (q(exercise) @ qw.T)[0].tolist(),
    "round_half_even_variant": q(tie_input).tolist(),
    "different_weight_scale_output": different_scale.tolist(),
    "simulation_dtype": str(qxf.dtype),
    "simulation_storage_bytes_per_tensor": qxf.numel() * qxf.element_size(),
    "official_fake_quant_agrees_with_unclipped_values": True,
    "PTQ_QAT_forward_capture": captures,
    "PTQ_QAT_equal_output": packed_result.tolist(),
    "activation_range_input_change": activation_ranges,
    "raw_result_scope": json_checks,
    "not_run": "No complete model, GPU, training, score reevaluation, model/data download, or weight save/load.",
}
environment = {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "CPU", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())}
(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"exact": exact, "exercise": results["exercise_first_row"], "captures": captures, "raw_result_runs_checked": [len(x["runs"]) for x in json_checks], "checks": "all assertions passed"}, ensure_ascii=False, indent=2))
