"""核對短序列 profiler 的數字、事件來源與既有量測配置。"""

import ast
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


run = ROOT / "outputs/reviewer-tools/runs/20261020T064524015915Z"
copied = {}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py"):
    raw = (run / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "text": raw.decode("utf-8")}
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/16.md#16.1" --execute',
    "exit_code": 0,
    "copied_files": copied,
})

environment = {
    "python": sys.version,
    "torch": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "device": "cpu",
    "platform": platform.platform(),
    "threads": "1",
    "cuda_available": str(torch.cuda.is_available()),
}
torch.set_num_threads(1)
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/16.1.md").read_bytes()
chapter = (ROOT / "course/chapters/16.md").read_bytes()
section = re.search(rb"(?ms)^## 16\.1 .*?(?=^## |\Z)", chapter).group()
assert section == frozen
assert sha(section) == "f6ddbd3982533283fb2ab45d4911bbec4bcd494f78f69a1ec75785f321858bea"

runs = []
for length in (3, 12):
    for repetition in range(3):
        torch.manual_seed(42)
        model = TinyLM(ModelConfig(width=8)).eval()
        ids = torch.tensor([[1, 2, 3]]) if length == 3 else torch.arange(1, 13)[None]
        with torch.no_grad():
            for _ in range(3):
                model(ids)
            with profile(activities=[ProfilerActivity.CPU]) as profiler:
                for _ in range(5):
                    result = model(ids)
        averages = profiler.key_averages()
        events = {
            event.key: {
                "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_total_us": event.cpu_time_total,
                "cpu_avg_us": event.cpu_time,
            }
            for event in averages
        }
        assert result["logits"].shape == (1, length, 264)
        assert events["aten::mm"]["calls"] == 25
        assert events["aten::addmm"]["calls"] == 10
        assert events["aten::index_select"]["calls"] == 10
        assert events["aten::_softmax"]["calls"] == 5
        assert events["aten::gelu"]["calls"] == 5
        assert events["aten::repeat_interleave"]["calls"] == 10
        assert all(abs(e["cpu_avg_us"] * e["calls"] - e["cpu_total_us"]) < 1e-8 for e in events.values())
        raw_events = profiler.events()
        self_formula_errors = [
            abs(e.self_cpu_time_total - (e.cpu_time_total - sum(c.cpu_time_total for c in e.cpu_children)))
            for e in raw_events if not e.is_async and str(e.device_type) == "DeviceType.CPU"
        ]
        assert max(self_formula_errors, default=0) < 1e-8
        parameters = [
            {"name": n, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype), "element_size": p.element_size()}
            for n, p in model.named_parameters()
        ]
        count = sum(p["numel"] for p in parameters)
        size = sum(p["numel"] * p["element_size"] for p in parameters)
        assert count == 6104 and size == 24416
        runs.append({
            "length": length, "repetition": repetition + 1, "seed": 42,
            "ids": ids.tolist(), "input_shape": list(ids.shape), "input_dtype": str(ids.dtype),
            "logits_shape": list(result["logits"].shape), "parameters": parameters,
            "parameter_count": count, "parameter_bytes": size,
            "events": events, "self_cpu_sum_us": sum(e["self_cpu_us"] for e in events.values()),
            "cpu_total_sum_us": sum(e["cpu_total_us"] for e in events.values()),
            "table_top20": averages.table(sort_by="self_cpu_time_total", row_limit=20),
            "self_formula_max_error_us": max(self_formula_errors, default=0),
            "observed_parent_child_pairs": dict(Counter(
                e.cpu_parent.key + " -> " + e.key for e in raw_events
                if e.cpu_parent is not None and e.key in {
                    "aten::mm", "aten::addmm", "aten::bmm", "aten::index_select", "aten::_softmax", "aten::clone"
                }
            )),
        })

torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8)).eval()
linear_calls = Counter()
handles = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        def count_call(module, inputs, output, name=name):
            linear_calls[name] += 1
        handles.append(module.register_forward_hook(count_call))
with torch.no_grad():
    model(torch.tensor([[1, 2, 3]]))
for handle in handles:
    handle.remove()
linear_modules = {n: {"bias": m.bias is not None, "calls_per_forward": linear_calls[n]} for n, m in model.named_modules() if isinstance(m, nn.Linear)}
assert sum(not d["bias"] for d in linear_modules.values()) == 5
assert sum(d["bias"] for d in linear_modules.values()) == 2

base = torch.arange(6).reshape(2, 3)
views = {"view": base.view(3, 2), "unsqueeze": base.unsqueeze(0), "as_strided": base.as_strided((3, 2), (1, 3))}
assert all(t.untyped_storage().data_ptr() == base.untyped_storage().data_ptr() for t in views.values())
assert torch.equal(base.new_zeros((2, 3)), torch.zeros((2, 3), dtype=base.dtype))
factories = {"empty": torch.empty((2, 3)), "new_empty": base.new_empty((2, 3)), "empty_like": torch.empty_like(base)}
assert all(t.shape == (2, 3) for t in factories.values())
optimizer = torch.optim.AdamW(model.parameters())
loss = model(torch.tensor([[1, 2, 3]]))["logits"].sum()
loss.backward()
gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
optimizer.step()
state_bytes = sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
assert gradient_bytes > 0 and state_bytes > 0

architecture = ROOT / "scripts/course_experiments/architecture.py"
text = architecture.read_text(encoding="utf-8")
tree = ast.parse(text)
functions = {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
timing_nodes = []
reports = []
for name in ("efficiency", "flash_probe"):
    path = ROOT / "docs/course-experiments/results" / (name + ".json")
    data = json.loads(path.read_bytes())
    old_command = ["git", "show", data["revision"] + ":scripts/course_experiments/architecture.py"]
    old_result = subprocess.run(old_command, check=False, capture_output=True, text=True)
    comparison = None
    if old_result.returncode == 0:
        old_text = old_result.stdout
        old_tree = ast.parse(old_text)
        old_functions = {node.name: ast.get_source_segment(old_text, node) for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        compare_names = ("_sync", "_benchmark", "_train", "_packing_probe") if name == "efficiency" else ("_sync", "_flash_measure")
        comparison = {n: functions[n] == old_functions[n] for n in compare_names}
    def walk(value, locator):
        if isinstance(value, dict):
            if "samples_seconds" in value:
                samples = value["samples_seconds"]
                assert len(samples) == value["measured_calls"] == 9
                assert value["warmup_calls"] == 3 and value["synchronized"] is True
                assert statistics.median(samples) == value["median_seconds"]
                timing_nodes.append({"path": str(path.relative_to(ROOT)), "locator": locator, "warmup_calls": value["warmup_calls"], "measured_calls": value["measured_calls"], "synchronized": value["synchronized"], "samples_seconds": samples, "median_seconds": value["median_seconds"]})
            for key, child in value.items():
                walk(child, locator + "." + key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, locator + f"[{i}]")
    walk(data["results"], "$.results")
    reports.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()), "revision": data["revision"], "device": data["device"], "torch_version": data["torch_version"], "gpu": data["gpu"], "historical_source_lookup": {"command": old_command, "exit_code": old_result.returncode, "stderr": old_result.stderr}, "timing_helpers_unchanged_from_recorded_revision": comparison})

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-1/verify-runtime.py",
    "environment": environment, "status": "passed", "source_sha256": sha(section),
    "profile_config": {"warmup": 3, "measured_forwards": 5, "independent_runs_per_length": 3, "grad_enabled": False, "activities": ["CPU"], "seed": 42},
    "runs": runs, "linear_modules": linear_modules,
    "parameter_derivation": "264*8 + 128*8 + 4*(8*8) + (8*32+32) + (32*8+8) + 3*(2*8) + 8*264 = 6104; 6104*4 = 24416",
    "tensor_checks": {"storage_shared": list(views), "zeros_verified": True, "empty_factory_shapes_verified_without_assuming_values": list(factories)},
    "training_storage": {"gradient_bytes": gradient_bytes, "optimizer_state_tensor_bytes": state_bytes, "note": "單次CPU反向傳播及AdamW更新，僅核對確實存在額外狀態，沒有量峰值。"},
    "numeric_checks": {"MiB_bytes": 2**20, "median_even": statistics.median([1, 2, 3, 20]), "median_odd_nine": statistics.median([9, 1, 7, 3, 5, 2, 4, 6, 8]), "mean_example": statistics.mean([1, 2, 3, 20])},
    "saved_experiment_reports": reports, "saved_timing_records_checked": timing_nodes,
    "timing_source_functions": {n: functions[n] for n in ("_sync", "_benchmark", "_flash_measure")},
    "training_median_expressions": [line.strip() for line in text.splitlines() if "statistics.median(latencies[3:]" in line],
    "limitations": "本次只有CPU執行；既有GPU紀錄的樣本數與中位數已重算，沒有重跑GPU或長時間訓練。既有紀錄的revision若不在本機Git物件庫，不能宣稱其程式與現行檔逐字相同。時間起伏不構成速度或瓶頸結論。",
}
assert result["numeric_checks"]["median_even"] == 2.5
save("runtime-verification.json", result)
print(json.dumps({"status": "passed", "runs": len(runs), "parameter_count": 6104, "parameter_bytes": 24416, "saved_timing_records_checked": len(timing_nodes), "gradient_bytes": gradient_bytes, "optimizer_state_tensor_bytes": state_bytes}, ensure_ascii=False))
