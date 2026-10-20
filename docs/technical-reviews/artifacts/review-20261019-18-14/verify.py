"""核對模型數字儲存、固定版本的逐題推論，以及直接讀取的原始來源。"""

import copy
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.compression import _evaluate
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers, unpack_int4
from tiny_perceptron.training import load_checkpoint

torch.set_num_threads(1)
torch.manual_seed(123)
torch.set_default_device("cpu")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def curl(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, command


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def source_record(spec):
    identifier, url, version, locator, excerpt = spec
    raw, command = curl(url)
    parser = Text()
    parser.feed(raw.decode("utf-8"))
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    if excerpt not in text:
        raise ValueError("原文缺少預期摘錄：" + identifier)
    return {
        "id": identifier,
        "url": url,
        "version": version,
        "locator": locator,
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "command": command,
        "original_bytes": len(raw),
        "original_sha256": sha(raw),
        "excerpt": excerpt,
        "checked_original": True,
    }


specs = [
    (
        "distillation",
        "https://ar5iv.labs.arxiv.org/html/1503.02531",
        "arXiv:1503.02531v1；ar5iv HTML",
        "Distillation，教師 soft target 與正確標籤的混合目標",
        "use a weighted average of two different objective functions",
    ),
    (
        "controlled_students",
        "https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html",
        "PyTorch Tutorials 2.14.0+cu130；讀取日期與全文 SHA 見本收據",
        "Cross-entropy runs，兩支小網路的相同初始化",
        "we need to make sure these two networks have the exact same initialization.",
    ),
    (
        "numel",
        "https://docs.pytorch.org/docs/2.14/generated/torch.numel.html",
        "PyTorch 2.14",
        "torch.numel API",
        "Returns the total number of elements in the input tensor.",
    ),
    (
        "element_size",
        "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.element_size.html",
        "PyTorch 2.14",
        "Tensor.element_size API",
        "Returns the size in bytes of an individual element.",
    ),
    (
        "mae",
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.L1Loss.html",
        "PyTorch 2.14",
        "L1Loss，逐元素絕對誤差與 mean reduction",
        "Creates a criterion that measures the mean absolute error (MAE)",
    ),
    (
        "calibration",
        "https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.quantize.html",
        "PyTorch 2.14",
        "quantize API；run_fn 是獨立的 calibration 階段",
        "Quantize the input float model with post training static quantization.",
    ),
]
with ThreadPoolExecutor(max_workers=6) as pool:
    external = list(pool.map(source_record, specs))


def tensors(model):
    return {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }


def payload(model):
    return sum(t.numel() * t.element_size() for t in tensors(model).values())


teacher = TinyLM(ModelConfig(width=16, layers=2))
student = TinyLM(ModelConfig(width=8, layers=1))
student_state = {name: value.clone() for name, value in student.state_dict().items()}
q4 = replace_linear_layers(copy.deepcopy(student), bits=4)
q8 = replace_linear_layers(copy.deepcopy(student), bits=8)
linear_inventory = []
for name, layer in student.named_modules():
    if isinstance(layer, nn.Linear):
        quantized = q4.get_submodule(name)
        linear_inventory.append({
            "name": name,
            "weight_shape": list(layer.weight.shape),
            "weight_elements": layer.weight.numel(),
            "fp32_weight_bytes": layer.weight.numel() * layer.weight.element_size(),
            "packed4_bytes": quantized.values.numel() * quantized.values.element_size(),
            "scale_elements": quantized.scale.numel(),
            "scale_bytes": quantized.scale.numel() * quantized.scale.element_size(),
            "scale_dtype": str(quantized.scale.dtype),
            "bias_bytes": 0 if layer.bias is None else layer.bias.numel() * layer.bias.element_size(),
        })
        assert quantized.values.numel() == math.ceil(layer.weight.numel() / 2)
        assert quantized.scale.shape == (layer.weight.shape[0], 1)
        assert quantized.scale.dtype == torch.float32
        assert quantized.bias is None or torch.equal(quantized.bias, layer.bias)

retained_names = []
for name, layer in student.named_modules():
    if isinstance(layer, (nn.Embedding, nn.LayerNorm)):
        for field, value in layer.named_parameters(recurse=False):
            key = name + "." + field
            assert torch.equal(q4.state_dict()[key], value)
            retained_names.append(key)

weight_elements = sum(item["weight_elements"] for item in linear_inventory)
scale_elements = sum(item["scale_elements"] for item in linear_inventory)
remaining = payload(student) - weight_elements * 4
toy = {
    "teacher_config": vars(teacher.config),
    "student_config": vars(student.config),
    "teacher_parameter_elements": sum(t.numel() for t in teacher.parameters()),
    "student_parameter_elements": sum(t.numel() for t in student.parameters()),
    "teacher_payload": payload(teacher),
    "student_payload": payload(student),
    "packed4_payload": payload(q4),
    "packed8_payload": payload(q8),
    "quantized_to_teacher_ratio": payload(q4) / payload(teacher),
    "quantized_to_student_ratio": payload(q4) / payload(student),
    "linear_inventory": linear_inventory,
    "weight_elements": weight_elements,
    "scale_elements": scale_elements,
    "retained_float_bytes": remaining,
    "packed4_equation": [remaining, weight_elements // 2, scale_elements * 4],
    "packed8_equation": [remaining, weight_elements, scale_elements * 4],
    "retained_float_parameters_equal": retained_names,
    "original_student_unchanged": all(torch.equal(value, student.state_dict()[name]) for name, value in student_state.items()),
    "no_training_performed": True,
}
assert [payload(teacher), payload(student), payload(q4), payload(q8)] == [67840, 24416, 15680, 17120]
assert [weight_elements, scale_elements, remaining] == [2880, 336, 12896]
assert round(toy["quantized_to_teacher_ratio"], 4) == 0.2311

layer = q4.output
x = torch.randn(2, 3, 8)
reconstructed = unpack_int4(layer.values, layer.shape).float() * layer.scale
actual = layer(x)
reference = torch.nn.functional.linear(x, reconstructed)
assert torch.equal(actual, reference)
toy["reference_forward"] = {
    "input_dtype": str(x.dtype),
    "reconstructed_weight_dtype": str(reconstructed.dtype),
    "output_dtype": str(actual.dtype),
    "equal_to_fp32_linear": bool(torch.equal(actual, reference)),
    "weight_mae": (student.output.weight - reconstructed).abs().mean().item(),
}

result_path = ROOT / "docs/course-experiments/results/distillation.json"
result_raw = result_path.read_bytes()
result = json.loads(result_raw)
attributes = result["results"]["tasks"]["attributes"]
test_samples = attributes["teacher_test"]["generated_samples"]
records = [{"family": s["family"], "question": s["question"], "answer": s["expected"]} for s in test_samples]
release_path = ROOT / "docs/course-experiments/public-releases/distillation.json"
release_raw = release_path.read_bytes()
manifest = json.loads(release_raw)["release"]["public_manifest"]
selected = [
    ("teacher", "sft-teacher.pt", attributes["teacher_storage"], attributes["teacher_test"]),
    ("ce", "sft-w32-ce.pt", attributes["runs"]["w32_ce"]["storage"], attributes["runs"]["w32_ce"]["test"]),
    ("kd", "sft-w32-ce_kl.pt", attributes["runs"]["w32_ce_kl"]["storage"], attributes["runs"]["w32_ce_kl"]["test"]),
    ("packed4", "sft-w32-ce_kl-packed4.pt", attributes["runs"]["w32_ce_kl_packed4"]["storage"], attributes["runs"]["w32_ce_kl_packed4"]["test"]),
]
models = {}
public_replay = {}
for name, filename, reported_storage, reported_test in selected:
    entry = next(f for f in manifest["files"] if f.get("output") == filename)
    url = "https://huggingface.co/" + manifest["repo"] + "/resolve/" + manifest["revision"] + "/" + entry["path"]
    raw, command = curl(url)
    assert sha(raw) == entry["sha256"]
    assert len(raw) == entry["bytes"]
    model, state = load_checkpoint(io.BytesIO(raw), "cpu")
    model.eval()
    models[name] = model
    observed = _evaluate(model, records, "cpu", tokens=24)
    assert payload(model) == reported_storage["tensor_bytes"]
    assert observed["correct"] == reported_test["correct"]
    assert observed["examples"] == reported_test["examples"] == 10
    assert observed["supervised_tokens"] == reported_test["supervised_tokens"] == 69
    assert all(a["generated_ids"] == b["generated_ids"] for a, b in zip(observed["generated_samples"], reported_test["generated_samples"], strict=True))
    public_replay[name] = {
        "url": url,
        "command": command,
        "sha256": sha(raw),
        "public_file_bytes": len(raw),
        "experiment_file_bytes": reported_storage["file_bytes"],
        "config": state["config"],
        "format_version": state["format_version"],
        "tensor_bytes": payload(model),
        "reported_storage": reported_storage,
        "observed_test": observed,
        "all_generated_ids_match_experiment": True,
    }

repacked = replace_linear_layers(copy.deepcopy(models["kd"]), bits=4)
identical = {
    key: torch.equal(value, models["packed4"].state_dict()[key])
    for key, value in repacked.state_dict().items()
}
repack_differences = {
    key: {
        "dtype": str(value.dtype),
        "different_elements": int((value != models["packed4"].state_dict()[key]).sum()),
        "maximum_absolute_difference": float((value.double() - models["packed4"].state_dict()[key].double()).abs().max()),
        "maximum_relative_difference": float(((value.double() - models["packed4"].state_dict()[key].double()).abs() / models["packed4"].state_dict()[key].double().abs().clamp_min(1e-30)).max()),
    }
    for key, value in repacked.state_dict().items() if not identical[key]
}
assert all(key.endswith(".scale") for key in repack_differences)
assert all(item["maximum_absolute_difference"] < 1e-8 for item in repack_differences.values())
assert all(item["maximum_relative_difference"] < 1.3e-7 for item in repack_differences.values())
ce_training = attributes["runs"]["w32_ce"]["training"]
kd_training = attributes["runs"]["w32_ce_kl"]["training"]
matched_fields = ["initialization_sha256", "steps", "optimizer_updates", "batch_size", "learning_rate", "batch_plan_sha256", "training_examples", "effective_supervised_tokens"]
assert all(ce_training[key] == kd_training[key] for key in matched_fields)

source_paths = [
    "tiny_perceptron/model.py", "tiny_perceptron/quantization.py",
    "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
    "tiny_perceptron/training.py", "scripts/course_experiments/compression.py",
    "docs/course-experiments/results/distillation.json",
    "docs/course-experiments/public-releases/distillation.json",
]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-14/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads())},
    "external_sources": external,
    "repository_sources": {path: sha((ROOT / path).read_bytes()) for path in source_paths},
    "toy": toy,
    "experiment": {
        "revision": result["revision"], "seed": result["seed"], "python": result["python_version"],
        "torch": result["torch_version"], "device": result["device"], "gpu": result["gpu"],
        "data": attributes["data"], "test_records": records,
        "matched_student_training": {key: ce_training[key] for key in matched_fields},
        "ce_objective": ce_training["objective"], "kd_objective": kd_training["objective"],
        "teacher_cache": attributes["teacher_cache"],
        "hard_target_generation": {key: value for key, value in attributes["hard_target_generation"].items() if key != "audit"},
    },
    "public_replay": public_replay,
    "repacked_kd_equals_public_packed4": identical,
    "repack_differences": repack_differences,
    "repack_tolerance": {"integer_codes_and_retained_floats": "完全相同", "scale_absolute": "< 1e-8", "scale_relative": "< 1.3e-7"},
    "scope": "只重跑 CPU 推論與儲存核對；未重訓教師或學生，未量速度或峰值 RAM，未外推模型品質。權重僅在記憶體讀取，收據保留固定 HTTPS URL、版本與 SHA-256。",
    "result": "全部斷言通過",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
