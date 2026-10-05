"""Bounded CPU checks for 17.7; no downloads, training, or checkpoint writes."""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import torch
from torch import nn

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from tiny_perceptron.quantization import QuantizedLinear, unpack_int4

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(177)
out = Path(__file__).resolve().parent
environment = {
    "python": sys.version,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "threads": str(torch.get_num_threads()),
    "quantization_sha256": hashlib.sha256((root / "tiny_perceptron/quantization.py").read_bytes()).hexdigest(),
}
print("environment", json.dumps(environment))

original = (out / "original-run/fence-1.py").read_text()
print("unmodified fence with only bits=4 replaced by bits=8")
variant = original.replace("QuantizedLinear(layer, bits=4)", "QuantizedLinear(layer, bits=8)")
assert variant != original
(out / "fence-8.py").write_text(variant)
exec(compile(variant, "fence-8.py", "exec"), {})

rows = torch.tensor([[0.7, 0.2], [-0.7, 1.2]])
rows_exact = [[Fraction(7, 10), Fraction(1, 5)], [Fraction(-7, 10), Fraction(6, 5)]]
records = []
for bits in (4, 8):
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(rows)
    old_optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    compressed = QuantizedLinear(layer, bits=bits)
    q = unpack_int4(compressed.values, compressed.shape) if bits == 4 else compressed.values
    maximum = 2 ** (bits - 1) - 1
    exact_scales = [max(map(abs, r)) / maximum for r in rows_exact]
    expected_q = [[round(w / s) for w in r] for r, s in zip(rows_exact, exact_scales)]
    exact_restored = [[v * s for v in r] for r, s in zip(expected_q, exact_scales)]
    expected_weight = torch.tensor([[float(v) for v in r] for r in exact_restored])
    assert q.tolist() == expected_q
    assert compressed.scale.shape == (2, 1)
    torch.testing.assert_close(q * compressed.scale, expected_weight, atol=1e-7, rtol=0)
    for x in [torch.tensor([[1.0, 1.0]]), torch.eye(2), torch.tensor([[2.0, -1.0]])]:
        y = compressed(x)
        torch.testing.assert_close(y, x @ expected_weight.T, atol=2e-7, rtol=0)
        assert x.dtype == torch.float32 and y.dtype == torch.float32
    expected_bytes = 10 if bits == 4 else 12
    assert compressed.storage_bytes() == expected_bytes
    assert compressed.values.dtype == (torch.uint8 if bits == 4 else torch.int8)
    assert not list(compressed.named_parameters())
    assert all(not b.requires_grad for b in compressed.buffers())
    assert not any(p is b for g in old_optimizer.param_groups for p in g["params"] for b in compressed.buffers())
    differentiable_input = torch.tensor([[1.0, 1.0]], requires_grad=True)
    compressed(differentiable_input).sum().backward()
    assert differentiable_input.grad is not None
    assert all(b.grad is None for b in compressed.buffers())
    assert not list(compressed.parameters())
    record = {
        "bits": bits,
        "values": compressed.values.tolist(),
        "values_dtype": str(compressed.values.dtype),
        "q": q.tolist(),
        "shape": compressed.shape,
        "scale_shape": list(compressed.scale.shape),
        "scale": compressed.scale.tolist(),
        "restored": (q * compressed.scale).tolist(),
        "exact_scales": [str(s) for s in exact_scales],
        "exact_output_for_ones": [str(sum(r)) for r in exact_restored],
        "float_output_for_ones": compressed(torch.ones((1, 2))).tolist(),
        "named_buffers": {n: {"shape": list(b.shape), "dtype": str(b.dtype), "bytes": b.numel() * b.element_size(), "requires_grad": b.requires_grad} for n, b in compressed.named_buffers()},
        "storage_bytes": compressed.storage_bytes(),
        "parameter_count": sum(p.numel() for p in compressed.parameters()),
        "input_gradient_exists": differentiable_input.grad is not None,
        "old_optimizer_references_original_weight": old_optimizer.param_groups[0]["params"][0] is layer.weight,
    }
    print("bounded_variant", json.dumps(record))
    records.append(record)
    biased = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        biased.weight.copy_(rows)
        biased.bias.copy_(torch.tensor([0.3, -0.1]))
    bias_compressed = QuantizedLinear(biased, bits=bits)
    assert bias_compressed.bias.dtype == torch.float32
    assert bias_compressed.bias.numel() * bias_compressed.bias.element_size() == 8
    assert bias_compressed.storage_bytes() == expected_bytes + 8
    torch.testing.assert_close(bias_compressed(torch.ones((1, 2))), compressed(torch.ones((1, 2))) + biased.bias.detach(), atol=1e-7, rtol=0)
    half_output = compressed(torch.ones((1, 2), dtype=torch.float16))
    assert half_output.dtype == torch.float16
    print("bias/half variant", bits, "bias_storage_bytes", bias_compressed.storage_bytes(), "half_output_dtype", str(half_output.dtype))

raw_path = out / "sources/quantization-original.json"
raw = raw_path.read_bytes()
original_raw_path = root / "docs/course-experiments/results/quantization.json"
assert hashlib.sha256(raw).digest() == hashlib.sha256(original_raw_path.read_bytes()).digest()
data = json.loads(raw)
config = data["results"]["teacher_provenance"]["config"]
assert config["norm"] == "layer" and not config["rotary"] and not config["tied"]
width, layers = config["width"], config["layers"]
retained_parts = {
    "embedding": config["vocab_size"] * width,
    "position": config["max_length"] * width,
    "normalization_weight_and_bias": (2 * layers + 1) * 2 * width,
}
retained_count = sum(retained_parts.values())
expected_names = ["embedding", "position"] + [f"blocks.{i}.norm{j}" for i in range(layers) for j in (1, 2)] + ["final_norm"]
storage_checks = []
for run in ("packed4", "packed8"):
    storage = data["results"]["runs"][run]["storage"]
    assert storage["retained_float_modules"] == expected_names
    assert storage["float_parameter_count"] == retained_count == 25728
    assert storage["parameter_tensor_bytes"] == retained_count * 4 == 102912
    buffers = storage["buffers"]
    assert sum(buffers.values()) == storage["buffer_tensor_bytes"]
    assert storage["parameter_tensor_bytes"] + storage["buffer_tensor_bytes"] == storage["tensor_bytes"]
    bias_buffers = {k: v for k, v in buffers.items() if k.endswith(".bias")}
    expected_bias = {f"blocks.{i}.ffn.{n}.bias": v for i in range(layers) for n, v in (("up", 4 * width * 4), ("down", width * 4))}
    assert bias_buffers == expected_bias
    assert len(storage["packed_linear_modules"]) == 6 * layers + 1 == 13
    storage_checks.append({"run": run, "float_parameter_count": retained_count, "parameter_tensor_bytes": retained_count * 4, "buffer_tensor_bytes": sum(buffers.values()), "bias_buffers": bias_buffers, "bias_bytes": sum(bias_buffers.values()), "packed_linear_module_count": len(storage["packed_linear_modules"])})

pointer_manifest = {
    "raw_json_sha256": hashlib.sha256(raw).hexdigest(),
    "copied_full_original": True,
    "retained_parameter_elements": retained_parts,
    "pointers_actually_inspected": [
        "/schema_version", "/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version",
        "/results/teacher_provenance/config", "/results/teacher_provenance/sha256", "/results/training/final_sha256",
        "/results/same_fp32_source", "/results/reloaded_packed_checkpoints_before_evaluation",
        *[f"/code_sha256/{p.replace('~','~0').replace('/','~1')}" for p in ("tiny_perceptron/quantization.py", "tiny_perceptron/model.py", "scripts/course_experiments/compression.py")],
        *[f"/results/runs/{r}/storage/{k}" for r in ("packed4", "packed8") for k in ("float_parameter_count", "parameter_tensor_bytes", "buffer_tensor_bytes", "tensor_bytes", "retained_float_modules", "packed_linear_modules", "buffers")],
    ],
    "storage_checks": storage_checks,
    "restriction": "Existing raw storage fields only; no validation/test/timing scores, GPU, complete training, model/data downloads, or retained neural weights.",
}
print("raw_storage_check", json.dumps(pointer_manifest))
(out / "cpu-observations.json").write_text(json.dumps({"environment": environment, "variants": records, "raw_storage": pointer_manifest}, indent=2) + "\n")
print("all bounded CPU assertions passed")
