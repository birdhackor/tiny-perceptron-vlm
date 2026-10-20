import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN

torch.manual_seed(42)
torch.set_num_threads(1)
torch.set_default_device("cpu")

content = torch.tensor([2.0, 2.0, 2.0])
gate = torch.tensor([-2.0, 0.0, 2.0])
factor = F.silu(gate)
manual = [x / (1 + math.exp(-x)) for x in gate.tolist()]
assert torch.allclose(factor, torch.tensor(manual), rtol=1e-6, atol=1e-7)
assert torch.allclose(factor, torch.tensor([-0.2384, 0.0, 1.7616]), rtol=0, atol=5e-5)
assert torch.allclose(content * factor, torch.tensor([-0.4768, 0.0, 3.5232]), rtol=0, atol=5e-5)
exercise = {}
for values in ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]):
    exercise[str(values)] = (content * F.silu(torch.tensor(values))).tolist()
assert exercise[str([0.0, 0.0, 0.0])] == [0.0, 0.0, 0.0]
assert abs(exercise[str([2.0, 0.0, 0.0])][0] - 3.5232) < 5e-5

layers = {}
for activation in ("gelu", "swiglu"):
    layer = DenseFFN(8, 16, activation=activation)
    x = torch.randn(2, 3, 8)
    h = layer.up(x)
    h = F.gelu(h) if activation == "gelu" else F.silu(layer.gate(x)) * h
    manual_forward = layer.down(h)
    assert torch.equal(layer(x), manual_forward)
    tables = (layer.up, layer.down) if activation == "gelu" else (layer.up, layer.gate, layer.down)
    weights = sum(m.weight.numel() for m in tables)
    biases = sum(m.bias.numel() for m in tables)
    layers[activation] = {
        "weight_shapes": [list(m.weight.shape) for m in tables],
        "bias_shapes": [list(m.bias.shape) for m in tables],
        "weight_count": weights,
        "bias_count": biases,
        "parameters": sum(p.numel() for p in layer.parameters()),
        "input_shape": list(x.shape),
        "output_shape": list(layer(x).shape),
        "explicit_forward_exact": bool(torch.equal(layer(x), manual_forward)),
    }
    assert layers[activation]["parameters"] == weights + biases
assert layers["gelu"]["weight_count"] == 256
assert layers["swiglu"]["weight_count"] == 384
assert layers["gelu"]["parameters"] == 280
assert layers["swiglu"]["parameters"] == 424

path = ROOT / "docs/course-experiments/results/modern.json"
record = json.loads(path.read_bytes())
experiments = {}
expected_nll = {"baseline": {"validation": 2.26331, "test": 2.29163}, "swiglu": {"validation": 2.25444, "test": 2.28009}}
for name, activation, count, expected_ms in (("baseline", "gelu", 141568, 13.062), ("swiglu", "swiglu", 174848, 13.336)):
    v = record["results"]["variants"][name]
    model = TinyLM(ModelConfig(**v["model"]["config"]))
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == count == v["model"]["parameters"]
    assert model.config.activation == activation
    widths = [block.ffn.up.out_features for block in model.blocks]
    assert widths == [256, 256]
    training = v["training"]
    assert training["steps"] == training["optimizer_updates"] == training["requested_steps"] == 240
    assert training["skipped_updates"] == 0
    ms = 1000 * training["warm_step_median_seconds"]
    assert abs(ms - expected_ms) <= 0.0005
    heldout = {}
    for split in ("validation", "test"):
        h = {k: value for k, value in v["heldout"][split].items() if k != "samples"}
        recomputed = h["nll_sum"] / h["effective_tokens"]
        assert abs(h["nll"] - recomputed) <= 1e-12
        assert abs(h["nll"] - expected_nll[name][split]) <= 5e-6
        heldout[split] = {**h, "recomputed_nll": recomputed}
    experiments[name] = {
        "config": v["model"]["config"],
        "measured_parameters": parameters,
        "hidden_widths": widths,
        "heldout": heldout,
        "training": {k: training[k] for k in ("steps", "optimizer_updates", "skipped_updates", "batch_size", "learning_rate", "effective_tokens")},
        "warm_step_median_milliseconds": ms,
        "timing_warmup_steps": 3,
        "timing_measured_steps": training["steps"] - 3,
    }

assert experiments["baseline"]["training"] == experiments["swiglu"]["training"]
for split in ("validation", "test"):
    assert experiments["swiglu"]["heldout"][split]["nll"] < experiments["baseline"]["heldout"][split]["nll"]
assert 174848 - 141568 == 2 * (64 * 256 + 256)
assert 3 * 8 * 16 == 2 * 8 * 24

paths = [
    "tiny_perceptron/modern.py",
    "tiny_perceptron/model.py",
    "scripts/course_experiments/architecture.py",
    "scripts/course_experiments/common.py",
    "docs/course-experiments/results/modern.json",
    "docs/review-tools/section_facts.py",
]
current_sha = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
for p in ("tiny_perceptron/modern.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py"):
    assert record["code_sha256"][p] == current_sha[p]

historical_revision = "a5133f444e0dca2347635f74f8ac83a27bb203d4"
historical_command = ["git", "show", historical_revision + ":scripts/course_experiments/architecture.py"]
historical = subprocess.run(historical_command, capture_output=True, check=True).stdout
historical_sha = hashlib.sha256(historical).hexdigest()
assert historical_sha == record["code_sha256"]["scripts/course_experiments/architecture.py"]
relevant_names = ("_train", "run_modern", "_clone_config", "_text_dataset", "_steps", "_heldout")


def relevant_functions(source):
    tree = ast.parse(source)
    return {node.name: ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in relevant_names}


historical_functions = relevant_functions(historical.decode())
current_functions = relevant_functions((ROOT / "scripts/course_experiments/architecture.py").read_text())
assert historical_functions == current_functions

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-5/verify_facts.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "seed": "42", "threads": "1"},
    "result": "全部斷言通過；僅重算算例、建模計數與核對既有實驗紀錄，沒有重新訓練或量測 GPU 時間。",
    "sigmoid_minus_two": 1 / (1 + math.exp(2)),
    "manual_silu": manual,
    "factor": factor.tolist(),
    "elementwise_product": (content * factor).tolist(),
    "exercise": exercise,
    "layers": layers,
    "matched_weight_example": {"width": 8, "ordinary_hidden": 24, "swiglu_hidden": 16, "ordinary_weights": 384, "swiglu_weights": 384},
    "experimental_record_sha256": current_sha["docs/course-experiments/results/modern.json"],
    "original_experiment_environment": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu")},
    "dataset": record["results"]["dataset"],
    "experiments": experiments,
    "current_source_sha256": current_sha,
    "recorded_code_verification": {
        "current_model_and_common_hashes_match": True,
        "architecture_current_full_hash_matches": False,
        "historical_command_argv": historical_command,
        "historical_source_sha256": historical_sha,
        "historical_full_hash_matches": True,
        "relevant_function_names": list(relevant_names),
        "relevant_functions_identical": True,
    },
    "limitations": "既有實驗是 seed 42 的單次 NVIDIA L4 短訓；本機 CPU 驗證不構成 GPU 效能重現，也不證明一般品質優勢。"
}, ensure_ascii=False, indent=2))
