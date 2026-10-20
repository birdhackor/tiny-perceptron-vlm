"""核對 expert 合併算例、梯度與實驗紀錄；僅使用 CPU。"""

import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.modern import MoEFFN


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def close(actual, expected):
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=1e-6)


def gradient(normalize):
    logits = torch.tensor([math.log(2), 0.0], dtype=torch.float64, requires_grad=True)
    selected = logits.softmax(-1).topk(1).values
    gate = selected / selected.sum() if normalize else selected
    output = gate.sum() * 2
    loss = output.square()
    derivative = torch.autograd.grad(loss, logits)[0]
    return {"gate": gate.tolist(), "output": output.item(), "loss": loss.item(), "gradient": derivative.tolist()}


class FixedExpert(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values))

    def forward(self, x):
        return self.values.expand(len(x), -1)


torch.set_num_threads(1)
torch.manual_seed(42)
weights = torch.tensor([0.7, 0.2])
outputs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
normalized = weights / weights.sum()
direct = weights @ outputs
mixed = normalized @ outputs
swapped = normalized @ outputs.flip(0)
both_swapped = normalized.flip(0) @ outputs.flip(0)
close(direct, [0.7, 0.4])
close(normalized, [7 / 9, 2 / 9])
close(mixed, [7 / 9, 4 / 9])
close(swapped, [2 / 9, 14 / 9])
close(both_swapped, [7 / 9, 4 / 9])
assert list(torch.cat(tuple(outputs)).shape) == [4]
assert list(mixed.shape) == [2]
assert float(weights[0]) == 0.699999988079071

gate_raw, gate_normalized = gradient(False), gradient(True)
assert abs(gate_raw["gradient"][0] - 32 / 27) < 1e-12
assert gate_normalized["gradient"] == [0.0, 0.0]

module_results = {}
for k in (1, 2):
    layer = MoEFFN(2, experts=3, top_k=k)
    with torch.no_grad():
        layer.router.weight.copy_(torch.tensor([[math.log(0.1), 0], [math.log(0.7), 0], [math.log(0.2), 0]]))
    layer.experts = nn.ModuleList([FixedExpert([9.0, 9.0]), FixedExpert([1.0, 0.0]), FixedExpert([0.0, 2.0])])
    x = torch.tensor([[[1.0, 0.0]]])
    output, _, selected = layer(x)
    close(output.flatten(), [0.7, 0.0] if k == 1 else [7 / 9, 4 / 9])
    output.square().sum().backward()
    module_results[str(k)] = {
        "output": output.tolist(), "shape": list(output.shape), "selected": selected.tolist(),
        "router_gradient_norm": layer.router.weight.grad.norm().item(),
    }
    assert module_results[str(k)]["router_gradient_norm"] > 0

run_path = ROOT / "docs/course-experiments/results/moe.json"
run_raw = run_path.read_bytes()
run = json.loads(run_raw)
variants = {}
for name, value in run["results"]["variants"].items():
    training = value["training"]
    assert training["optimizer_updates"] == training["steps"] == training["requested_steps"] == 180
    assert training["skipped_updates"] == 0
    variants[name] = {
        "config": value["model"]["config"],
        "training": {key: training[key] for key in (
            "requested_steps", "steps", "optimizer_updates", "skipped_updates", "effective_tokens",
            "batch_size", "learning_rate", "auxiliary_weight", "scaler_enabled", "status",
        )},
        "initial_training_nll": training["initial"]["nll"],
        "final_training_nll": training["final"]["nll"],
    }

code_checks = {}
for name in ("tiny_perceptron/modern.py", "tiny_perceptron/model.py", "scripts/course_experiments/architecture.py"):
    code_checks[name] = {"current_sha256": sha((ROOT / name).read_bytes()), "run_sha256": run["code_sha256"][name]}
    code_checks[name]["matches_run"] = code_checks[name]["current_sha256"] == code_checks[name]["run_sha256"]
assert code_checks["tiny_perceptron/modern.py"]["matches_run"]
assert code_checks["tiny_perceptron/model.py"]["matches_run"]

historical_path = "scripts/course_experiments/architecture.py"
historical_ref = "a5133f444e0dca2347635f74f8ac83a27bb203d4"
historical = subprocess.run(["git", "show", historical_ref + ":" + historical_path], capture_output=True, check=True).stdout
assert sha(historical) == run["code_sha256"][historical_path]
text = historical.decode()
tree = ast.parse(text)
historical_excerpts = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_train", "run_moe"):
        lines = ast.get_source_segment(text, node).splitlines()
        needles = ("scaler.step(optimizer)", "updates += 1", '"optimizer_updates": updates', "for k in (1, 2):",
                   'plans.append((f"top{k}_aux{weight:g}"', "training = _train(")
        historical_excerpts[node.name] = [line.strip() for line in lines if any(needle in line for needle in needles)]

unavailable = subprocess.run(["git", "show", run["revision"] + ":" + historical_path], capture_output=True, text=True)

freeze_path = ROOT / "docs/course-review-20261019/freeze-04.json"
freeze = json.loads(freeze_path.read_bytes())
page = next(page for page in freeze["pages"] if page["page_id"] == "15.5")
assert sha((ROOT / page["snapshot"]).read_bytes()) == page["source_sha256"]
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
figure = next(item for item in receipt["records"] if item["page_id"] == "15.5")
assert sha((ROOT / figure["figure"]).read_bytes()) == figure["frozen_svg"]["sha256"]
assert sha((ROOT / figure["png"]["path"]).read_bytes()) == figure["png"]["sha256"]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-5/verify_combine.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "result": "所有斷言通過",
    "numeric": {
        "exact_weights": [str(Fraction(7, 9)), str(Fraction(2, 9))], "direct": direct.tolist(),
        "normalized": normalized.tolist(), "mixed": mixed.tolist(), "swapped": swapped.tolist(),
        "both_swapped": both_swapped.tolist(), "float32_point7": float(weights[0]),
        "printed_round4": [direct.round(decimals=4).tolist(), normalized.round(decimals=4).tolist(), mixed.round(decimals=4).tolist()],
        "selected_mass": weights.sum().item(), "mixture_width": mixed.numel(), "concatenated_width": torch.cat(tuple(outputs)).numel(),
    },
    "top1_gradients": {"raw": gate_raw, "normalized": gate_normalized},
    "repository_moe": module_results,
    "experiment_record": {
        "path": run_path.relative_to(ROOT).as_posix(), "sha256": sha(run_raw),
        "environment": {key: run[key] for key in ("python_version", "torch_version", "gpu", "device", "seed")},
        "dataset": run["results"]["dataset"], "variants": variants,
        "code_checks": code_checks,
        "matching_historical_script": {"git_ref": historical_ref, "path": historical_path, "sha256": sha(historical), "excerpts": historical_excerpts},
        "recorded_revision_lookup": {"revision": run["revision"], "exit_code": unavailable.returncode, "stderr": unavailable.stderr},
    },
    "freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_path.read_bytes()), "page": page},
    "figure_receipt": figure,
    "local_api_docs": {
        "torch.matmul": "Matrix product of two tensors.",
        "nn.LayerNorm": "Applies Layer Normalization over a mini-batch of inputs.",
        "matmul_excerpt_present": "Matrix product of two tensors." in torch.matmul.__doc__,
        "layernorm_excerpt_present": "Applies Layer Normalization over a mini-batch of inputs." in nn.LayerNorm.__doc__,
        "note": "本機安裝版本的官方 API 說明另與版本化公開文件及實際運算核對。",
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
