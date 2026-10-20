"""核對成本算例、圖捕捉時機與既有 GPU 原始量測。"""

import ast
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[3]
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ARTIFACT / "torch-cache")
os.environ["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "0"
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.modern import DenseFFN, MoEFFN


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.manual_seed(42)
torch.set_num_threads(1)
captured = []


def recording_backend(graph, inputs):
    captured.append(str(graph.graph))
    return graph.forward


torch._dynamo.reset()
layer = torch.nn.Linear(4, 3).eval()
x = torch.ones(2, 4)
compiled = torch.compile(layer, backend=recording_backend, dynamic=False)
counts = [len(captured)]
with torch.no_grad():
    reference = layer(x)
    actual = compiled(x)
    counts.append(len(captured))
    repeated = compiled(x.clone())
    counts.append(len(captured))
    changed = compiled(torch.ones(3, 4))
    counts.append(len(captured))
assert counts == [0, 1, 1, 2], "捕捉次數不符預期"
assert actual.shape == (2, 3)
assert torch.equal(reference, actual) and torch.equal(reference, repeated)
assert changed.shape == (3, 3)

torch._dynamo.reset()
eager_compiled = torch.compile(layer, backend="eager")
with torch.no_grad():
    eager_actual = eager_compiled(x)
assert torch.equal(reference, eager_actual)

moe = MoEFFN(2, experts=2, top_k=1).eval()
with torch.no_grad():
    moe.router.weight.copy_(torch.eye(2))
groups = []
for inputs in (torch.tensor([[2., 0.], [3., 0.]]), torch.tensor([[2., 0.], [0., 3.]])):
    with torch.no_grad():
        _, _, chosen = moe(inputs)
    groups.append(torch.bincount(chosen.flatten(), minlength=2).tolist())
assert groups == [[2, 0], [1, 1]], "expert 分組不符預期"

raw_report = ROOT / "docs/course-experiments/results/efficiency.json"
report = json.loads(raw_report.read_bytes())
measured = report["results"]["compile"]
stats = {}
for route in ("eager", "compiled"):
    item = measured[route]
    values = item["samples_seconds"]
    median = statistics.median(values)
    assert len(values) == item["measured_calls"] == 9
    assert item["warmup_calls"] == 3 and item["synchronized"]
    assert median == sorted(values)[4] == item["median_seconds"]
    stats[route] = {
        "sorted_seconds": sorted(values),
        "median_seconds": median,
        "median_ms": median * 1000,
        "rounded_ms": format(median * 1000, ".6f"),
        "mean_seconds": statistics.mean(values),
    }
assert stats["eager"]["rounded_ms"] == "0.125070"
assert stats["compiled"]["rounded_ms"] == "0.319508"
saving = stats["eager"]["median_seconds"] - stats["compiled"]["median_seconds"]
assert saving < 0 and measured["estimated_calls_to_amortize_first_call"] is None
assert format(measured["first_compiled_call_seconds"], ".3f") == "7.717"
assert format(measured["output_max_error"], ".5e") == "7.15256e-07"
assert format(measured["worker_wall_seconds"], ".3f") == "16.255"
assert measured["backend"] == "inductor" and measured["fullgraph"] and not measured["dynamic"]
assert measured["shape"] == [4, 16, 64] and measured["dtype"] == "torch.float32"
assert measured["unique_graphs"] == 1 and measured["graph_breaks"] == {}
assert measured["inductor_generated_kernel_count"] == 1
assert "peak_allocated_bytes" not in measured
config = report["results"]["models"]["mha"]["model"]["config"]
assert config["heads"] == 4 and config["experts"] == 0 and config["width"] == 64

plain, optimized = Decimal("0.01"), Decimal("0.006")
scenarios = []
for setup in (Decimal("2"), Decimal("4")):
    scenarios.append({
        "setup": str(setup),
        "per_call_saving": str(plain - optimized),
        "break_even_calls": int(setup / (plain - optimized)),
        "costs": [
            {"calls": n, "plain_seconds": str(n * plain), "compiled_seconds": str(setup + n * optimized)}
            for n in (100, 500, 1000)
        ],
    })

code = ROOT / "scripts/course_experiments/architecture.py"
tree = ast.parse(code.read_text())
names = ("_sync", "_benchmark", "_compile_worker", "_compile_probe", "run_efficiency")
functions = {
    node.name: ast.get_source_segment(code.read_text(), node)
    for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
}
assert set(functions) == set(names)
stored = Path(__file__).with_name("section-execution.json")
section = json.loads(stored.read_text())
environment = json.loads(section["copied_files"]["environment.json"]["text"])
freeze = ROOT / "docs/course-review-20261019/freeze-04/sources/16.11.md"
assert sha(freeze) == "9bb1280a28276b2d94c5c6dd300fc1274d890549e43c4caea775e30941eee81f"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-11b/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform(), "seed": "42", "num_threads": "1"},
    "section_run": {"source_sha256": sha(freeze), "guard_events": environment["guard_events"], "stdout": section["copied_files"]["stdout.txt"]["text"], "stderr": section["copied_files"]["stderr.txt"]["text"]},
    "interface_probe": {"shape": list(actual.shape), "max_error": float((reference - actual).abs().max()), "eager_max_error": float((reference - eager_actual).abs().max()), "backend_calls_after_wrapper_first_repeat_shape_change": counts, "captured_graphs": captured},
    "moe_probe": {"input_shape": [2, 2], "top_k": 1, "groups": groups},
    "cost_scenarios": scenarios,
    "historical_experiment": {"path": raw_report.relative_to(ROOT).as_posix(), "sha256": sha(raw_report), "gpu": report["gpu"], "python": report["python_version"], "torch": report["torch_version"], "seed": report["seed"], "reported_revision": report["revision"], "mha_config": config, "compile": measured, "recomputed": stats, "per_call_saving_seconds": saving, "peak_memory_scope": report["peak_memory_scope"], "code_sha256": report["code_sha256"][code.relative_to(ROOT).as_posix()]},
    "inspected_current_code": {"path": code.relative_to(ROOT).as_posix(), "sha256": sha(code), "functions": functions},
    "limitations": ["本回合未重跑 L4；對既有 JSON 的樣本、設定、數值與程式邊界進行稽核。", "主程序的峰值紀錄明確排除子程序；編譯結果沒有子程序峰值。", "capturing backend 只驗證捕捉與重用時機，未聲稱測得 Inductor 效益。"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
