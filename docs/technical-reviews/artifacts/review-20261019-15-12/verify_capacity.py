"""核對容量算例、實際分派與既有訓練紀錄；不進行訓練。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.modern import MoEFFN

torch.set_num_threads(1)
torch.manual_seed(0)
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/15.12.md"
raw = frozen.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "444d0cd9b4248707955566ccf5e17050fcc7810955675b20a7b5fa0f1c9904d8"
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
cases = []
for expression, factor, expected_capacity in (("2 / 3", Fraction(2, 3), 2), ("1", Fraction(1), 3), ("2", Fraction(2), 6)):
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(code.replace("capacity_factor = 2 / 3", "capacity_factor = " + expression), namespace)
    chosen = namespace["chosen"]
    count = Counter(chosen.tolist())
    expected_counts = [count[i] for i in range(namespace["experts"])]
    exact_capacity = math.ceil(factor * len(chosen) * namespace["k"] / namespace["experts"])
    expected_overflow = [max(value - exact_capacity, 0) for value in expected_counts]
    assert namespace["capacity"] == exact_capacity == expected_capacity
    assert namespace["counts"].tolist() == expected_counts == [5, 1]
    assert namespace["overflow"].tolist() == expected_overflow
    assert chosen.tolist() == [0, 0, 0, 1, 0, 0]
    cases.append({
        "capacity_factor_expression": expression,
        "capacity": exact_capacity,
        "counts": expected_counts,
        "counts_minus_capacity": (namespace["counts"] - exact_capacity).tolist(),
        "overflow": expected_overflow,
        "overflow_total": sum(expected_overflow),
        "dispatch_denominator": len(chosen) * namespace["k"],
        "overflow_rate": sum(expected_overflow) / (len(chosen) * namespace["k"]),
        "chosen_unchanged": True,
        "stdout": stream.getvalue(),
        "dtype": str(chosen.dtype),
    })

dispatch_runs = []
for top_k, crowded in ((1, False), (1, True), (2, False)):
    layer = MoEFFN(2, experts=2, top_k=top_k, hidden=3).eval()
    with torch.no_grad():
        layer.router.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    first = [2.0] * 6 if crowded else [2.0, 2.0, 2.0, -2.0, 2.0, 2.0]
    x = torch.tensor(list(zip(first, range(6), strict=True))).reshape(1, 6, 2)
    received = [0, 0]

    def receiver(index):
        def collect(module, args):
            received[index] += len(args[0])
        return collect

    hooks = [expert.register_forward_pre_hook(receiver(index)) for index, expert in enumerate(layer.experts)]
    with torch.no_grad():
        output, _, selected = layer(x)
    for hook in hooks:
        hook.remove()
    flat = x.reshape(-1, 2)
    with torch.no_grad():
        weights, indices = layer.router(flat).softmax(-1).topk(top_k, dim=-1)
        if top_k > 1:
            weights /= weights.sum(-1, keepdim=True)
        reference = torch.zeros_like(flat)
        for row in range(len(flat)):
            for slot in range(top_k):
                reference[row] += weights[row, slot] * layer.experts[int(indices[row, slot])](flat[row])
    expected_received = torch.bincount(selected.flatten(), minlength=2).tolist()
    assert received == expected_received
    assert sum(received) == len(flat) * top_k
    torch.testing.assert_close(output.reshape_as(reference), reference, rtol=1e-6, atol=1e-6)
    dispatch_runs.append({
        "top_k": top_k,
        "all_tokens_choose_same_expert": crowded,
        "chosen": selected.tolist(),
        "expert_received_rows": received,
        "dispatch_denominator": len(flat) * top_k,
        "max_abs_difference_from_per_token_reference": (output.reshape_as(reference) - reference).abs().max().item(),
    })

record_path = ROOT / "docs/course-experiments/results/moe.json"
record = json.loads(record_path.read_bytes())
source_audit = {}
for path in ("tiny_perceptron/modern.py", "tiny_perceptron/model.py", "scripts/course_experiments/architecture.py"):
    content = (ROOT / path).read_bytes()
    pinned = subprocess.run(["git", "show", record["revision"] + ":" + path], capture_output=True, text=True)
    current_sha = hashlib.sha256(content).hexdigest()
    source_audit[path] = {
        "current_sha256": current_sha,
        "experiment_record_sha256": record["code_sha256"][path],
        "matches_experiment_record": current_sha == record["code_sha256"][path],
        "pinned_read_command": ["git", "show", record["revision"] + ":" + path],
        "pinned_read_exit_code": pinned.returncode,
        "pinned_read_stderr": pinned.stderr,
    }
assert source_audit["tiny_perceptron/modern.py"]["matches_experiment_record"]
assert source_audit["tiny_perceptron/model.py"]["matches_experiment_record"]

metric_matches = []

def visit(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = path + "." + key
            if any(word in key.lower() for word in ("drop", "capacity", "overflow", "fallback", "rerout")):
                metric_matches.append({"path": item_path, "value": item})
            visit(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            visit(item, path + f"[{index}]")

visit(record)
assert not metric_matches
variants = {}
for name, entry in record["results"]["variants"].items():
    routing = entry["validation_routing"]
    variants[name] = {
        "config": entry["model"]["config"],
        "steps": entry["training"]["steps"],
        "heldout_splits": list(entry["heldout"]),
        "routing_effective_input_tokens": routing["effective_input_tokens"] if routing else None,
        "dispatch_denominators": [layer["dispatch_denominator"] for layer in routing["layers"]] if routing else [],
    }
assert len(variants) == 7
assert all(entry["steps"] == 180 for entry in variants.values())

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-12/verify_capacity.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads()), "seed": "0"},
    "result": "所有容量、分派與紀錄核對均通過",
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "capacity_cases": cases,
    "dropless_dispatch": dispatch_runs,
    "experiment_record_audit": {
        "path": record_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "revision": record["revision"],
        "recorded_environment": {key: record[key] for key in ("device", "seed", "torch_version", "python_version", "gpu")},
        "variants": variants,
        "capacity_or_drop_metric_keys": metric_matches,
        "source_audit": source_audit,
        "scope": "僅審查現有紀錄與相同雜湊的模型分派實作；未重跑訓練。紀錄中的完整實驗 revision 在目前本機 Git 無法讀取，現行 architecture.py 與紀錄的雜湊不同。",
    },
}, ensure_ascii=False, indent=2))
