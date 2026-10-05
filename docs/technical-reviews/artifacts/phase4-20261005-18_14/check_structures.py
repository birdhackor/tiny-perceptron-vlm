"""Bounded CPU inspection of random structures and existing raw measurements.

No model scores are recomputed. No training, download, or weight-file write.
"""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys

import torch

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(model):
    parameters = {n: t.numel() * t.element_size() for n, t in model.named_parameters()}
    buffers = {n: t.numel() * t.element_size() for n, t in model.named_buffers()}
    return {"parameters": parameters, "buffers": buffers,
            "parameter_bytes": sum(parameters.values()), "buffer_bytes": sum(buffers.values()),
            "total_bytes": sum(parameters.values()) + sum(buffers.values())}


def formula(width, layers, bits=None):
    # vocabulary=264; positional table=128; two LayerNorm per block + final norm.
    if bits is None:
        return 4 * (658 * width + layers * (12 * width ** 2 + 9 * width))
    floating_embedding_norm = 4 * (394 + 4 * layers) * width
    ffn_bias = 4 * 5 * layers * width
    per_output_scales = 4 * (9 * layers * width + 264)
    linear_weight_elements = 12 * layers * width ** 2 + 264 * width
    packed_weights = linear_weight_elements * bits // 8
    return floating_embedding_norm + ffn_bias + per_output_scales + packed_weights


structures = {}
for width, layers in [(16, 2), (8, 1), (64, 2), (32, 1)]:
    torch.manual_seed(91)
    model = TinyLM(ModelConfig(width=width, layers=layers, heads=2 if width >= 32 else 1))
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    key = f"width{width}_layers{layers}"
    structures[key] = payload(model)
    assert structures[key]["total_bytes"] == formula(width, layers)
    if layers == 1:
        for bits in (4, 8):
            quantized = replace_linear_layers(copy.deepcopy(model), bits=bits)
            storage = payload(quantized)
            assert storage["total_bytes"] == formula(width, layers, bits)
            assert all(torch.equal(initial[n], p) for n, p in model.named_parameters())
            assert not any(isinstance(m, torch.nn.Linear) for m in quantized.modules())
            assert len([m for m in quantized.modules() if isinstance(m, QuantizedLinear)]) == 7
            assert all(p.dtype == torch.float32 for p in quantized.parameters())
            assert all(b.dtype == (torch.uint8 if bits == 4 else torch.int8)
                       if n.endswith("values") else b.dtype == torch.float32
                       for n, b in quantized.named_buffers())
            ids = torch.tensor([[8, 9, 10]])
            with torch.inference_mode():
                out = quantized(ids)["logits"]
            assert out.dtype == torch.float32 and out.shape == (1, 3, 264)
            structures[f"{key}_packed{bits}"] = storage

assert structures["width16_layers2"]["total_bytes"] == 67840
assert structures["width8_layers1"]["total_bytes"] == 24416
assert structures["width8_layers1_packed4"]["total_bytes"] == 15680
assert structures["width8_layers1_packed8"]["total_bytes"] == 17120
assert round(15680 / 67840, 4) == 0.2311
assert 15680 > 24416 / 8

# Ordinary API checks cover scalar, multidimensional, empty, and byte tensors.
api_cases = []
for t, count, size in [(torch.tensor(3.), 1, 4), (torch.zeros(2, 3), 6, 4),
                       (torch.empty(0), 0, 4), (torch.zeros(5, dtype=torch.uint8), 5, 1)]:
    assert t.numel() == count and torch.numel(t) == count and t.element_size() == size
    api_cases.append({"shape": list(t.shape), "dtype": str(t.dtype),
                      "numel": t.numel(), "element_size": t.element_size()})

# MAE uses all tensor elements, not parameter tables or samples as denominator.
original = torch.tensor([[1., -2.], [3., -4.]])
restored = torch.tensor([[0.5, -1.5], [3., -3.]])
mae = float((restored - original).abs().mean())
assert mae == 0.5 and mae == float(torch.nn.L1Loss()(restored, original))

raw = json.loads((BASE / "evidence/distillation-original.json").read_bytes())
assert raw["revision"] == "5af615e5d7c9642afee800390fa072257f895d0c"
assert digest(BASE / "code/compression-at-experiment-revision.py") == raw["code_sha256"]["scripts/course_experiments/compression.py"]
for name in ["model", "quantization", "attention", "modern", "data"]:
    assert digest(ROOT / f"tiny_perceptron/{name}.py") == raw["code_sha256"][f"tiny_perceptron/{name}.py"]

attrs = raw["results"]["tasks"]["attributes"]
tok = ByteTokenizer()
teacher_config = attrs["teacher_provenance"]["config"]
assert (teacher_config["width"], teacher_config["layers"], teacher_config["tied"]) == (64, 2, False)
assert attrs["student_layers"] == 1
assert attrs["data"]["counts"] == {"train": 45, "validation": 5, "test": 10}
assert attrs["data"]["family_intersections"] == 0

records = [("teacher", attrs["teacher_storage"], attrs["teacher_test"])]
records += [(name, attrs["runs"][name]["storage"], attrs["runs"][name]["test"])
            for name in ["w32_ce", "w32_ce_kl", "w32_ce_kl_packed4"]]
observed = []
pointer_reads = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/code_sha256",
                 "/results/tasks/attributes/teacher_provenance/config",
                 "/results/tasks/attributes/data", "/results/tasks/attributes/student_layers"]
for name, storage, test in records:
    expected_payload = 566272 if name == "teacher" else 64160 if name.endswith("packed4") else 134528
    assert storage["tensor_bytes"] == storage["parameter_tensor_bytes"] + storage["buffer_tensor_bytes"] == expected_payload
    assert storage["file_bytes"] - storage["tensor_bytes"] == storage["file_overhead_bytes"]
    samples = test["generated_samples"]
    assert len(samples) == test["examples"] == 10
    exact, eos, completed = 0, 0, 0
    for sample in samples:
        ids = sample["generated_ids"]
        has_eos = tok.eos_id in ids
        answer = ids[:ids.index(tok.eos_id)] if has_eos else ids
        hit = answer == tok.encode(sample["expected"])
        assert tok.decode(answer) == sample["generated"]
        assert hit == sample["exact"] and has_eos == sample["ended_with_eos"]
        exact += hit
        eos += has_eos
        completed += hit and has_eos
    assert exact == test["correct"] and exact / 10 == test["exact_match"]
    assert completed == test["completed_correct"] and completed / 10 == test["completed_exact_match"]
    assert eos == test["eos_count"] and eos / 10 == test["eos_rate"]
    assert test["supervised_tokens"] == 69 and test["max_new_tokens"] == 24
    observed.append({"name": name, "tensor_bytes": expected_payload, "correct": exact,
                     "examples": 10, "eos_count": eos, "supervised_tokens": 69})
    prefix = "/results/tasks/attributes/teacher_" if name == "teacher" else f"/results/tasks/attributes/runs/{name}/"
    pointer_reads += [prefix + "storage/{parameter_tensor_bytes,buffer_tensor_bytes,tensor_bytes,file_bytes,file_overhead_bytes,forward,retained_float_modules,packed_linear_modules}",
                      prefix + "test/{correct,examples,exact_match,completed_correct,completed_exact_match,eos_count,eos_rate,supervised_tokens,max_new_tokens,exact_match_definition,generated_samples}"]

ce = attrs["runs"]["w32_ce"]["training"]
kl = attrs["runs"]["w32_ce_kl"]["training"]
for field in ["steps", "optimizer_updates", "batch_size", "learning_rate", "initialization_sha256",
              "batch_plan_sha256", "effective_supervised_tokens", "training_examples"]:
    assert ce[field] == kl[field]
    pointer_reads += [f"/results/tasks/attributes/runs/{name}/training/{field}" for name in ["w32_ce", "w32_ce_kl"]]
assert ce["steps"] == ce["optimizer_updates"] == 400
assert ce["effective_supervised_tokens"] == 44985 and ce["training_examples"] == 45
assert ce["alpha"] == 0 and kl["alpha"] == 0.5 and kl["temperature"] == 2.0
for name in ["w32_ce", "w32_ce_kl"]:
    pointer_reads += [f"/results/tasks/attributes/runs/{name}/training/{field}" for field in ["objective", "alpha", "temperature"]]

before = attrs["runs"]["w32_ce_kl"]["test"]["generated_samples"][2]
after = attrs["runs"]["w32_ce_kl_packed4"]["test"]["generated_samples"][2]
assert (before["family"], before["question"], before["expected"]) == (after["family"], after["question"], after["expected"])
assert before["generated"] == "blue" and before["exact"] is True
assert after["generated"] == "ble" and after["exact"] is False

environment = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
               "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
               "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
               "threads": str(torch.get_num_threads()), "installed_module_sha256": digest(Path(inspect.getfile(torch.nn.Module))),
               "installed_loss_sha256": digest(Path(inspect.getfile(torch.nn.L1Loss)))}
assert environment["installed_module_sha256"] == digest(BASE / "sources/torch-module-original.py")
assert environment["installed_loss_sha256"] == digest(BASE / "sources/torch-loss-original.py")
result = {"environment": environment, "structure_payloads": structures, "ordinary_api_cases": api_cases,
          "ratio": 15680 / 67840, "rounded_ratio": round(15680 / 67840, 4), "mae_hand_example": mae,
          "existing_result_reaggregation": observed, "changed_sample": {"before": before, "after": after},
          "matched_training": {k: ce[k] for k in ["steps", "batch_size", "learning_rate", "effective_supervised_tokens", "training_examples", "initialization_sha256", "batch_plan_sha256"]},
          "read_pointers": pointer_reads, "existing_result_device": raw["device"],
          "existing_result_torch": raw["torch_version"], "score_rerun": False, "new_training": False,
          "new_weight_files": False, "raw_result_sha256": digest(BASE / "evidence/distillation-original.json")}
(BASE / "cpu-checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "all assertions passed", "structure_bytes": {k: v["total_bytes"] for k, v in structures.items()},
                  "existing_results": observed, "blue_to_ble": True, "rounded_ratio": result["rounded_ratio"],
                  "mae_hand_example": mae, "device": "cpu", "score_rerun": False}, ensure_ascii=False, indent=2))
