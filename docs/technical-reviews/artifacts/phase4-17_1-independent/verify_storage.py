"""Bounded CPU checks of original 17.1 fence and existing raw storage measurements."""
from pathlib import Path
import copy
import hashlib
import io
import json
import os
import platform
import sys
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear, pack_int4, replace_linear_layers, unpack_int4

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
torch.set_num_threads(1)
torch.manual_seed(1701)
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "platform": platform.platform(),
    "threads": str(torch.get_num_threads()),
    "cwd": str(Path.cwd()),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-17_1-independent/verify_storage.py",
    "offline_env": {k: os.environ.get(k, "unset") for k in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"]},
}
print("ENVIRONMENT", json.dumps(environment))
print("ORIGINAL FENCE")
namespace = {"__name__": "__main__"}
exec(compile((BASE / "original-fence.py").read_bytes(), "course/chapters/17.md#17.1:original-fence", "exec"), namespace)
assert tuple(namespace["weight"].shape) == (64, 32)

checks = []
for rows in [64, 128]:
    tensor = torch.zeros(rows, 32)
    for dtype, per_value in [(torch.float32, 4), (torch.float16, 2), (torch.int8, 1)]:
        changed = tensor.to(dtype)
        observed = {"rows": rows, "columns": 32, "dtype": str(dtype), "count": changed.numel(), "element_bytes": changed.element_size(), "tensor_bytes": changed.numel() * changed.element_size(), "storage_bytes": changed.untyped_storage().nbytes(), "all_zero": bool(torch.all(changed == 0))}
        assert changed.numel() == rows * 32
        assert changed.element_size() == per_value
        assert observed["tensor_bytes"] == observed["storage_bytes"] == rows * 32 * per_value
        assert observed["all_zero"]
        checks.append(observed)
print("ZERO TABLE AND 128-ROW VARIANT", json.dumps(checks))
cast = torch.tensor([0.7, -0.7]).to(torch.int8)
assert cast.tolist() == [0, 0]
print("DIRECT CAST", cast.tolist(), "is_quantized", cast.is_quantized)
assert not cast.is_quantized

values = torch.arange(-8, 8, dtype=torch.int8)
packed = pack_int4(values)
assert packed.numel() == 8 and packed.element_size() == 1
assert torch.equal(unpack_int4(packed, values.shape), values)
print("INT4 CODES", values.tolist(), "PACKED BYTES", packed.tolist())

raw = (BASE / "inputs/docs/course-experiments/results/quantization.json").read_bytes()
original_raw = (ROOT / "docs/course-experiments/results/quantization.json").read_bytes()
assert hashlib.sha256(raw).digest() == hashlib.sha256(original_raw).digest()
record = json.loads(raw)
config = record["results"]["teacher_provenance"]["config"]
model = TinyLM(ModelConfig(**config))
linear = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
parameter_count = sum(p.numel() for p in model.parameters())
linear_weights = sum(layer.weight.numel() for _, layer in linear)
linear_biases = sum(0 if layer.bias is None else layer.bias.numel() for _, layer in linear)
scale_count = sum(layer.out_features for _, layer in linear)
retained = parameter_count - linear_weights - linear_biases
assert (parameter_count, linear_weights, linear_biases, retained, scale_count) == (141568, 115200, 640, 25728, 1416)
print("MODEL CONFIG", json.dumps(config))
print("LINEAR SHAPES", json.dumps([{ "name": name, "weight_shape": list(layer.weight.shape), "bias_count": 0 if layer.bias is None else layer.bias.numel(), "output_axis_scales": layer.out_features } for name, layer in linear]))
print("COUNTS", json.dumps({"logical_parameters":parameter_count,"quantized_weight_elements":linear_weights,"biases_retained_fp32":linear_biases,"other_retained_fp32":retained,"output_axis_scales":scale_count}))
empirical_checks = []
for name, bits in [("fp32", 32), ("packed4", 4), ("packed8", 8)]:
    candidate = model if bits == 32 else replace_linear_layers(copy.deepcopy(model), bits)
    parameter_bytes = sum(p.numel() * p.element_size() for p in candidate.parameters())
    buffers = {n: b.numel() * b.element_size() for n, b in candidate.named_buffers()}
    buffer_bytes = sum(buffers.values())
    storage = record["results"]["runs"][name]["storage"]
    assert parameter_bytes == storage["parameter_tensor_bytes"]
    assert buffer_bytes == storage["buffer_tensor_bytes"]
    assert buffers == storage["buffers"]
    assert parameter_bytes + buffer_bytes == storage["tensor_bytes"]
    assert storage["parameter_count"] == parameter_count
    assert not storage["optimizer_in_deployment_file"]
    filename = Path(storage["checkpoint"]).name
    artifact = next(a for a in record["artifacts"] if a["path"] == filename)
    assert storage["file_bytes"] == artifact["bytes"]
    assert storage["file_overhead_bytes"] == storage["file_bytes"] - storage["tensor_bytes"]
    if bits != 32:
        assert sum(v for k, v in buffers.items() if k.endswith(".values")) == linear_weights * bits // 8
        assert sum(v for k, v in buffers.items() if k.endswith(".scale")) == scale_count * 4
        assert sum(v for k, v in buffers.items() if k.endswith(".bias")) == linear_biases * 4
        assert len([layer for layer in candidate.modules() if isinstance(layer, QuantizedLinear)]) == 13
        assert sum(p.numel() for p in candidate.parameters()) == retained
    empirical_checks.append({"variant":name,"logical_parameter_count":parameter_count,"parameter_bytes":parameter_bytes,"buffer_bytes":buffer_bytes,"tensor_bytes":storage["tensor_bytes"],"original_reported_file_bytes":storage["file_bytes"],"original_reported_file_sha256":artifact["sha256"],"original_reported_overhead":storage["file_overhead_bytes"],"weights_executed_or_loaded":False})
print("ORIGINAL MEASUREMENT ACCOUNTING", json.dumps(empirical_checks))
training = record["results"]["training"]
assert training["optimizer_updates"] == training["steps"] == 120
assert training["weights_changed"] and training["initialization_sha256"] != training["final_sha256"]
print("EXISTING TRAINING PROVENANCE ONLY", json.dumps({k:training[k] for k in ["steps","optimizer_updates","weights_changed","initialization_sha256","final_sha256"]}))
assert retained * 4 + linear_biases * 4 == 105472
assert scale_count * 4 == 5664
assert linear_weights // 2 + (retained + linear_biases) * 4 + scale_count * 4 == 168736
assert linear_weights + (retained + linear_biases) * 4 + scale_count * 4 == 226336
assert linear_weights * 4 + (retained + linear_biases) * 4 == 566272

# In-memory zero-tensor serialization: container behavior, no saved weights/artifacts.
temporary = io.BytesIO()
torch.save({"example": torch.zeros(64, 32)}, temporary)
with zipfile.ZipFile(temporary) as archive:
    members = archive.namelist()
assert any(name.endswith("data.pkl") for name in members)
assert len(temporary.getvalue()) > 8192
print("IN-MEMORY ZERO SERIALIZATION", json.dumps({"tensor_bytes":8192,"container_bytes":len(temporary.getvalue()),"members":members,"supports_private_file_size":False}))

imports = {}
for name, module in sorted(sys.modules.items()):
    file = getattr(module, "__file__", None)
    if name.startswith("tiny_perceptron") and file and Path(file).is_file():
        file = Path(file).resolve()
        imports[name] = {"path":str(file.relative_to(ROOT)),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
environment["repository_modules"] = imports
(BASE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(BASE / "storage-verification.json").write_text(json.dumps({"zero_tables":checks,"direct_cast":cast.tolist(),"model_counts":{"parameters":parameter_count,"linear_weights":linear_weights,"linear_biases":linear_biases,"retained_other":retained,"scales":scale_count},"existing_measurement_checks":empirical_checks,"all_assertions_passed":True,"scope":"Original code fence and bounded storage arithmetic only. No original/private weights loaded, no training, no scoring, no GPU."},indent=2)+"\n")
print("ALL STORAGE ASSERTIONS PASSED")
