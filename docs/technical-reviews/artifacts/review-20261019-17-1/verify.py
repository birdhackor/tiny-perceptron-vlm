"""核對型別轉換、容量算例與實驗紀錄；所有 tensor 留在記憶體。"""

import copy
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers
from tiny_perceptron.training import restore_checkpoint_rng, save_checkpoint
from scripts.course_experiments import compression
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_default_device("cpu")
torch.manual_seed(42)
record_path = ROOT / "docs/course-experiments/results/quantization.json"
record = json.loads(record_path.read_text())
results = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
    },
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-1/verify.py",
    "repository_inputs": {
        name: sha(ROOT / name)
        for name in (
            "tiny_perceptron/model.py",
            "tiny_perceptron/quantization.py",
            "tiny_perceptron/training.py",
            "scripts/course_experiments/compression.py",
            "docs/course-experiments/results/quantization.json",
        )
    },
}

tables = []
for rows in (64, 128):
    weight = torch.zeros(rows, 32)
    for dtype in (torch.float32, torch.float16, torch.int8):
        converted = weight.to(dtype)
        tables.append({
            "rows": rows,
            "dtype": str(dtype),
            "numel": converted.numel(),
            "element_bytes": converted.element_size(),
            "tensor_bytes": converted.numel() * converted.element_size(),
            "is_original": converted is weight,
            "shares_storage": converted.data_ptr() == weight.data_ptr(),
            "all_zero": bool((converted == 0).all()),
        })
results["tables"] = tables
results["int8_range"] = {"min": torch.iinfo(torch.int8).min, "max": torch.iinfo(torch.int8).max}
results["fraction_conversion"] = {
    "input": [-1.7, -0.7, 0.7, 1.7],
    "int8": torch.tensor([-1.7, -0.7, 0.7, 1.7]).to(torch.int8).tolist(),
}
results["float_representations"] = {
    str(dtype): {
        "value": torch.tensor(0.7, dtype=dtype).item(),
        "absolute_error": abs(torch.tensor(0.7, dtype=dtype).item() - 0.7),
    }
    for dtype in (torch.float32, torch.float16)
}

config = ModelConfig(**record["results"]["teacher_provenance"]["config"])
model = TinyLM(config).eval()
linear_weights = sum(layer.weight.numel() for layer in model.modules() if isinstance(layer, nn.Linear))
linear_biases = sum(layer.bias.numel() for layer in model.modules() if isinstance(layer, nn.Linear) and layer.bias is not None)
total = sum(p.numel() for p in model.parameters())
results["model_counts"] = {
    "config": record["results"]["teacher_provenance"]["config"],
    "total": total,
    "linear_weights": linear_weights,
    "linear_biases": linear_biases,
    "other_parameters": total - linear_weights - linear_biases,
    "all_fp32_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
}
results["packed_counts"] = {}
for bits in (4, 8):
    candidate = replace_linear_layers(copy.deepcopy(model), bits)
    params = sum(p.numel() for p in candidate.parameters())
    buffers = {name: x.numel() * x.element_size() for name, x in candidate.named_buffers()}
    storage = record["results"]["runs"][f"packed{bits}"]["storage"]
    scales = sum(x.scale.numel() for x in candidate.modules() if isinstance(x, QuantizedLinear))
    biases = sum(x.bias.numel() for x in candidate.modules() if isinstance(x, QuantizedLinear) and x.bias is not None)
    tensor_bytes = sum(p.numel() * p.element_size() for p in candidate.parameters()) + sum(buffers.values())
    results["packed_counts"][str(bits)] = {
        "retained_float_parameters": params,
        "bias_buffer_elements": biases,
        "scale_elements": scales,
        "scale_dtypes": sorted({str(x.scale.dtype) for x in candidate.modules() if isinstance(x, QuantizedLinear)}),
        "bias_dtypes": sorted({str(x.bias.dtype) for x in candidate.modules() if isinstance(x, QuantizedLinear) and x.bias is not None}),
        "weight_value_bytes": sum(value for name, value in buffers.items() if name.endswith(".values")),
        "scale_bytes": sum(value for name, value in buffers.items() if name.endswith(".scale")),
        "retained_float_bytes": params * 4 + biases * 4,
        "tensor_bytes": tensor_bytes,
        "buffers_match_record": buffers == storage["buffers"],
        "tensor_bytes_match_record": tensor_bytes == storage["tensor_bytes"],
        "logical_parameter_count": params + linear_weights + biases,
    }
    assert buffers == storage["buffers"]
    assert tensor_bytes == storage["tensor_bytes"]
results["recorded_sizes"] = {
    name: {key: run["storage"][key] for key in ("checkpoint", "parameter_count", "tensor_bytes", "file_bytes", "optimizer_in_deployment_file")}
    for name, run in record["results"]["runs"].items()
    if name in ("fp32", "packed4", "packed8")
}
results["recorded_training_provenance"] = {
    "revision": record["revision"],
    "environment": {key: record[key] for key in ("python_version", "torch_version", "device", "gpu", "seed")},
    "teacher_steps": record["results"]["teacher_provenance"]["steps"],
    "further_updates": record["results"]["training"]["optimizer_updates"],
    "weights_changed": record["results"]["training"]["weights_changed"],
    "initialization_sha256": record["results"]["training"]["initialization_sha256"],
    "final_sha256": record["results"]["training"]["final_sha256"],
}

captured = {}


def capture(payload, path):
    captured.update(payload)


with patch("tiny_perceptron.training.torch.save", side_effect=capture), patch.object(Path, "replace"):
    save_checkpoint(Path(__file__).parent / "memory-only.pt", model, metadata={"purpose": "記憶體內的格式核對"})
before = torch.rand(5)
restore_checkpoint_rng(captured)
after = torch.rand(5)
results["checkpoint_payload"] = {
    "keys": sorted(captured),
    "optimizer_is_none": captured["optimizer"] is None,
    "rng_not_in_model_state": not any("rng" in name for name in captured["model"]),
    "restored_random_sequence_matches": torch.equal(before, after),
    "save_interception": "torch.save 接收真實 save_checkpoint 的 payload；Path.replace 被攔截，未建立權重檔。",
}
results["packed_payloads"] = {}
for bits in (4, 8):
    packed_payload = {}

    def capture_packed(payload, path):
        packed_payload.update(payload)

    def load_captured(path, device):
        candidate = replace_linear_layers(TinyLM(ModelConfig(**packed_payload["config"])), bits)
        candidate.load_state_dict(packed_payload["model"])
        return candidate, packed_payload

    with patch("scripts.course_experiments.compression.torch.save", side_effect=capture_packed), patch(
        "scripts.course_experiments.compression.load_checkpoint", side_effect=load_captured
    ):
        compression._packed(SimpleNamespace(output=str(Path(__file__).parent), device="cpu"), model, "memory-only", bits)
    results["packed_payloads"][str(bits)] = {
        "keys": sorted(packed_payload),
        "optimizer_is_none": packed_payload["optimizer"] is None,
        "has_rng_keys": any("rng" in key for key in packed_payload),
        "metadata_keys": sorted(packed_payload["metadata"]),
        "scope": "攔截實際 _packed 的 torch.save；load_checkpoint 改以記憶體 payload 重建，未建立或重測部署檔案。",
    }
stream = io.BytesIO()
torch.save(model.state_dict(), stream)
results["in_memory_serialization"] = {
    "state_dict_tensor_bytes": total * 4,
    "serialized_bytes": len(stream.getvalue()),
    "scope": "只核對容器另有開銷；隨機初始化、CPU 與不同 metadata 不能重現既有實驗檔案 bytes。",
}
assert total == 141568
assert linear_weights == 115200
assert linear_biases == 640
assert tables[0]["is_original"] is True
assert results["fraction_conversion"]["int8"] == [-1, 0, 0, 1]
assert results["checkpoint_payload"]["restored_random_sequence_matches"]
results["result"] = "容量、型別範圍、截斷、模型分拆、刻度與 RNG 核對相符；to(float32) 回傳原 tensor，推翻無條件建立新 tensor 的敘述。"
print(json.dumps(results, ensure_ascii=False, indent=2))
