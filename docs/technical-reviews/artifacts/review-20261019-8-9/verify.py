"""查核矩陣還原、縮放、合併，以及既有實驗紀錄的數字與範圍。"""

import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.behavior import (
    _adapter_state,
    _add_lora,
    _merge_lora,
    _restore_adapter,
)
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-section-code", "exec"), namespace)
    return namespace, output.getvalue()


torch.set_num_threads(1)
torch.set_default_device("cpu")
freeze_path = ROOT / "docs/course-review-20261019/freeze-01/sources/8.9.md"
frozen = freeze_path.read_text(encoding="utf-8")
code = re.search(r"```python\n(.*?)```", frozen, re.S).group(1)
original, original_stdout = execute(code)
omission = '    layer.a.copy_(adapter_a["a"])\n'
assert code.count(omission) == 1
missing_a, missing_stdout = execute(code.replace(omission, ""))
assert "0.1" in missing_stdout and "False" in missing_stdout
layer = original["layer"]
adapter = original["adapter_a"]
original_ids = [id(layer.a), id(layer.b)]
assert original_stdout == "切回短答套的B平均 0.1\n短答套兩個矩陣都恢復 True\n"
assert torch.equal(layer.a, adapter["a"]) and torch.equal(layer.b, adapter["b"])
assert layer.a.data_ptr() != adapter["a"].data_ptr()
assert layer.b.data_ptr() != adapter["b"].data_ptr()
assert not adapter["a"].requires_grad and not adapter["b"].requires_grad
with torch.no_grad():
    layer.a.fill_(0.25)
    layer.a.copy_(adapter["a"])
assert original_ids == [id(layer.a), id(layer.b)]

with torch.no_grad():
    delta = layer.b @ layer.a * (layer.alpha / layer.rank)
    scaled = copy.deepcopy(layer)
    scaled.alpha = 4
    doubled = scaled.b @ scaled.a * (scaled.alpha / scaled.rank)
    assert torch.equal(doubled, 2 * delta)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 0.0, 2.0, 0.25]])
    separate_output = layer(x)
    merged = copy.deepcopy(layer.base)
    merged.weight.copy_(layer.merged_weight())
    merged_difference = float((separate_output - merged(x)).abs().max())
    assert merged_difference < 1e-6
    duplicated = LoRALinear(copy.deepcopy(merged), rank=layer.rank, alpha=layer.alpha)
    duplicated.a.copy_(layer.a)
    duplicated.b.copy_(layer.b)
    twice_delta_error = float((duplicated.merged_weight() - (layer.base.weight + 2 * delta)).abs().max())
    duplicate_effect = float((duplicated(x) - merged(x)).abs().max())
    assert twice_delta_error < 1e-6 and duplicate_effect > 0
    different_base = copy.deepcopy(layer)
    different_base.base.weight.add_(1)
    different_base_difference = float((different_base(x) - layer(x)).abs().max())
    assert different_base_difference > 0

mean_left = torch.tensor([0.0, 0.0])
mean_right = torch.tensor([1.0, -1.0])
assert mean_left.mean().item() == mean_right.mean().item()
assert not torch.equal(mean_left, mean_right)

torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=16)).eval()
base_state = {name: value.clone() for name, value in model.state_dict().items()}
names = _add_lora(model, rank=2, alpha=2)
with torch.no_grad():
    for part in model.modules():
        if isinstance(part, LoRALinear):
            part.b.fill_(0.1)
state_a = _adapter_state(model)
with torch.no_grad():
    for part in model.modules():
        if isinstance(part, LoRALinear):
            part.a.fill_(0.2)
            part.b.fill_(-0.1)
state_b = _adapter_state(model)
probe = torch.tensor([[1, 3, 57, 51, 57, 69, 71, 2, 4]])
with torch.no_grad():
    _restore_adapter(model, state_a)
    logits_a = model(probe)["logits"].clone()
    _restore_adapter(model, state_b)
    logits_b = model(probe)["logits"].clone()
    merged_model = _merge_lora(model).eval()
    logits_merged = merged_model(probe)["logits"]
    _restore_adapter(model, state_a)
    logits_restored = model(probe)["logits"]
assert torch.equal(logits_a, logits_restored)
assert not torch.equal(logits_a, logits_b)
assert not any(isinstance(part, LoRALinear) for part in merged_model.modules())
for name, value in base_state.items():
    adapter_name = name
    for part_name in names:
        if name.startswith(part_name + "."):
            adapter_name = part_name + ".base." + name[len(part_name) + 1 :]
            break
    assert torch.equal(value, model.state_dict()[adapter_name])

record_path = ROOT / "docs/course-experiments/results/lora.json"
record = json.loads(record_path.read_bytes())
result = record["results"]
assert result["a_b_a_max_difference"] == 0
assert result["adapter_a_b_logit_difference"] > 0
assert result["merge_max_difference"] < 0.0001
assert f'{result["merge_max_difference"]:.8f}' == "0.00000668"
assert ByteTokenizer().decode(probe[0].tolist()) == "1+1=?"
assert probe.shape == (1, 9)

source_paths = [
    "tiny_perceptron/alignment.py",
    "tiny_perceptron/adapters.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "scripts/course_experiments/behavior.py",
    "course/training.md",
    "docs/course-experiments/results/lora.json",
]
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-9/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "dtype": str(layer.a.dtype),
        "platform": platform.platform(),
        "threads": str(torch.get_num_threads()),
    },
    "source_sha256": digest(freeze_path),
    "source_files": {path: digest(ROOT / path) for path in source_paths},
    "original_stdout": original_stdout,
    "omitted_a_restore_stdout": missing_stdout,
    "matrix_checks": {
        "a_shape": list(layer.a.shape),
        "b_shape": list(layer.b.shape),
        "adapter_a_values": {name: value.tolist() for name, value in adapter.items()},
        "adapter_b_values": {name: value.tolist() for name, value in original["adapter_b"].items()},
        "clones_have_separate_storage": True,
        "no_grad_clones_require_grad": False,
        "copy_preserves_parameter_identity": True,
        "alpha_doubled_delta_exactly": True,
        "mean_counterexample": {"left": mean_left.tolist(), "right": mean_right.tolist(), "same_mean": True, "equal": False},
        "merged_output_max_difference": merged_difference,
        "duplicate_merge_twice_delta_max_error": twice_delta_error,
        "duplicate_adapter_output_max_change": duplicate_effect,
        "same_shape_other_base_output_max_change": different_base_difference,
    },
    "untrained_model_mechanism": {
        "seed": 42,
        "width": 8,
        "layers": 1,
        "rank": 2,
        "alpha": 2,
        "inserted_layers": names,
        "probe_ids": probe.tolist(),
        "logits_shape": list(logits_a.shape),
        "a_b_a_max_difference": float((logits_a - logits_restored).abs().max()),
        "a_b_max_difference": float((logits_a - logits_b).abs().max()),
        "merge_max_difference": float((logits_b - logits_merged).abs().max()),
        "base_parameters_unchanged": True,
        "merged_has_no_lora_layers": True,
        "scope": "手設矩陣；沒有訓練文風，不能充當既有 GPU 實驗的重新量測。",
    },
    "recorded_experiment_audit": {
        "record_path": str(record_path.relative_to(ROOT)),
        "record_sha256": digest(record_path),
        "revision": record["revision"],
        "environment": {key: record[key] for key in ["device", "gpu", "seed", "python_version", "torch_version", "step_scale", "evidence_status"]},
        "base_sha256": result["base_sha256"],
        "rank": result["rank"],
        "alpha": result["alpha"],
        "scaling": result["scaling"],
        "a_b_a_max_difference": result["a_b_a_max_difference"],
        "adapter_a_b_logit_difference": result["adapter_a_b_logit_difference"],
        "merge_max_difference": result["merge_max_difference"],
        "rounded_merge_max_difference": f'{result["merge_max_difference"]:.8f}',
        "probe_question": ByteTokenizer().decode(probe[0].tolist()),
        "probe_ids": probe.tolist(),
        "probe_count": 1,
        "positions": 9,
        "vocabulary": 264,
        "compared_logits": 2376,
        "mask": "run_lora 的 probe forward 沒有 valid 遮罩，max 涵蓋全部位置與詞表。",
        "merge_tolerance": 0.0001,
        "training": {name: {key: run["training"][key] for key in ["steps", "effective_tokens", "trainable_parameters"]} for name, run in result["runs"].items()},
        "data": {name: run["data"] for name, run in result["runs"].items()},
        "current_code_matches_recorded_sha256": {path: digest(ROOT / path) == recorded for path, recorded in record["code_sha256"].items() if path in source_paths},
        "scope": "讀取已保存的完成紀錄並核對量測程式；本次沒有重跑訓練或 NVIDIA L4 推論。",
    },
    "result": "所有斷言通過。",
}
with (DIRECTORY / "verification.json").open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
