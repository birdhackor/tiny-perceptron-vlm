"""16.1 獨立技術核對；直接擷取凍結教材程式，僅做 CPU 前向與紀錄核算。"""

import ast
import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RAW = Path("/workspace/work/reference/omni-technical-technical_16_1-v1")
COMMAND = (
    "uv run --extra cpu python "
    "docs/technical-reviews/artifacts/omni-technical-technical_16_1-v1/audit.py"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


packet_path = ROOT / (
    "docs/course-revision-20261029-omni/reviews/technical-stage/"
    "dispatch-briefs-v9/technical_16_1.json"
)
packet = json.loads(packet_path.read_text())
freeze = json.loads((ROOT / packet["source_freeze"]).read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "16.1")
snapshot = ROOT / page["snapshot"]
assert digest(snapshot) == page["source_sha256"] == packet["page"]["source_sha256"]
assert page["figures_sha256"] == {}
body = snapshot.read_text()
current = re.search(
    r"^## 16\.1 .*?(?=^## |\Z)",
    (ROOT / page["source"]).read_text(),
    re.M | re.S,
).group()
assert current == body
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
code = blocks[0]
expected_counts = {
    "aten::mm": 25,
    "aten::addmm": 10,
    "aten::linear": 35,
    "aten::repeat_interleave": 10,
    "aten::index_select": 10,
    "aten::_softmax": 5,
    "aten::gelu": 5,
}
runs, raw_runs = [], []
original_stdout = None
first_model = None
for length in (3, 12):
    inline = code
    if length == 12:
        assert code.count("ids = torch.tensor([[1, 2, 3]])") == 1
        inline = code.replace(
            "ids = torch.tensor([[1, 2, 3]])", "ids = torch.arange(1, 13)[None]"
        )
    for repeat in range(1, 4):
        namespace = {"__name__": "__inline_lesson_audit__"}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(inline, str(snapshot) + ":python-block", "exec"), namespace)
        if original_stdout is None:
            original_stdout = stdout.getvalue()
            first_model = namespace["model"]
        prof = namespace["p"]
        averages = prof.key_averages()
        averages.table(row_limit=-1, time_unit="us")
        total_self = sum(e.self_cpu_time_total for e in averages)
        selected = {}
        all_events = []
        for event in averages:
            assert event.cpu_time == event.cpu_time_total / event.count
            assert event.self_cpu_percent == f"{100 * event.self_cpu_time_total / total_self:.2f}%"
            assert event.total_cpu_percent == f"{100 * event.cpu_time_total / total_self:.2f}%"
            record = {
                "name": event.key,
                "count": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_total_us": event.cpu_time_total,
                "cpu_avg_us": event.cpu_time,
                "self_cpu_percent": event.self_cpu_percent,
                "cpu_total_percent": event.total_cpu_percent,
            }
            all_events.append(record)
            if event.key in expected_counts:
                selected[event.key] = record
        for name, count in expected_counts.items():
            assert selected[name]["count"] == count, (name, selected)
        model, result = namespace["model"], namespace["result"]
        shape = tuple(result["logits"].shape)
        assert shape == (1, length, 264)
        assert not result["logits"].requires_grad
        assert all(p.grad is None for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == 6104
        assert sum(p.numel() * p.element_size() for p in model.parameters()) == 24416
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        assert prof.activities == {torch.profiler.ProfilerActivity.CPU}
        raw_events = list(prof.events())
        assert all(str(e.device_type) == "DeviceType.CPU" for e in raw_events)
        parent_errors = [
            abs(e.self_cpu_time_total - (e.cpu_time_total - sum(c.cpu_time_total for c in e.cpu_children)))
            for e in raw_events
            if not e.is_async
        ]
        assert max(parent_errors) == 0.0
        run = {
            "length": length,
            "repeat": repeat,
            "logits_shape": list(shape),
            "parameter_elements": 6104,
            "parameter_bytes": 24416,
            "total_self_cpu_us": total_self,
            "sum_cpu_total_us": sum(e.cpu_time_total for e in averages),
            "selected": selected,
            "top10": [e.key for e in sorted(averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:10]],
            "verified_avg_and_percent_rows": len(averages),
            "max_parent_child_formula_error_us": max(parent_errors),
            "cache_shapes": [[list(k.shape), list(v.shape)] for k, v in result["cache"]],
        }
        runs.append(run)
        raw_runs.append({"length": length, "repeat": repeat, "events": all_events})

parameters = [
    {"name": n, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype), "element_size": p.element_size()}
    for n, p in first_model.named_parameters()
]
linears = [
    {"name": n, "input": m.in_features, "output": m.out_features, "bias": m.bias is not None}
    for n, m in first_model.named_modules()
    if isinstance(m, torch.nn.Linear)
]
assert sum(not m["bias"] for m in linears) == 5
assert sum(m["bias"] for m in linears) == 2

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0], [6.0]])
assert torch.mm(a, b).tolist() == [[17.0], [39.0]]
assert torch.matmul(a, b).tolist() == [[17.0], [39.0]]
assert torch.addmm(torch.ones(2, 1), a, b).tolist() == [[18.0], [40.0]]
container = torch.tensor([1.0, 2.0, 3.0, 4.0])
assert container.new_zeros(3).tolist() == [0.0, 0.0, 0.0]
assert list(torch.empty(2, 3).shape) == [2, 3]
assert list(container.new_empty(2, 3).shape) == [2, 3]
assert list(torch.empty_like(container).shape) == [4]
views = [container.view(2, 2), container.unsqueeze(0), container.as_strided((2, 2), (2, 1))]
assert all(v.untyped_storage().data_ptr() == container.untyped_storage().data_ptr() for v in views)
assert torch.equal(container.repeat_interleave(1), container)
assert list(torch.arange(1, 13)[None].shape) == [1, 12]
assert torch.arange(1, 13).tolist() == list(range(1, 13))

efficiency_path = ROOT / "docs/course-experiments/results/efficiency.json"
efficiency = json.loads(efficiency_path.read_text())
historical_path = RAW / "historical-scripts__course_experiments__architecture.py"
assert digest(historical_path) == efficiency["code_sha256"]["scripts/course_experiments/architecture.py"]
assert efficiency["step_scale"] == 1.0
historical_tree = ast.parse(historical_path.read_text())
current_path = ROOT / "scripts/course_experiments/architecture.py"
current_tree = ast.parse(current_path.read_text())
historical_functions = {n.name: n for n in historical_tree.body if isinstance(n, ast.FunctionDef)}
current_functions = {n.name: n for n in current_tree.body if isinstance(n, ast.FunctionDef)}
for name in ("_sync", "_benchmark", "_train"):
    assert ast.dump(historical_functions[name], include_attributes=False) == ast.dump(current_functions[name], include_attributes=False)
namespace = {"torch": torch, "statistics": statistics, "time": time}
for name in ("_sync", "_benchmark"):
    exec(compile(ast.Module(body=[current_functions[name]], type_ignores=[]), str(current_path), "exec"), namespace)
calls = []
benchmark_demo = namespace["_benchmark"](lambda: calls.append(1), torch.device("cpu"))
assert len(calls) == 12
assert benchmark_demo["warmup_calls"] == 3 and benchmark_demo["measured_calls"] == 9
assert benchmark_demo["median_seconds"] == statistics.median(benchmark_demo["samples_seconds"])
fit_node = current_functions["_train"]
median_node = next(n for n in ast.walk(fit_node) if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "warm_step_median_seconds" for k in n.keys))
index = next(i for i, k in enumerate(median_node.keys) if isinstance(k, ast.Constant) and k.value == "warm_step_median_seconds")
expression = ast.Expression(body=median_node.values[index])
training_median_demo = eval(compile(expression, str(current_path) + ":_train-return", "eval"), {"statistics": statistics, "latencies": [100, 200, 300, 1, 2, 3, 4]})
assert training_median_demo == 2.5
historical_timings = []


def inspect_timings(value, path="/results"):
    if isinstance(value, dict):
        if "samples_seconds" in value and "median_seconds" in value:
            samples = value["samples_seconds"]
            assert len(samples) == value["measured_calls"] == 9
            assert value["warmup_calls"] == 3 and value["synchronized"] is True
            recalculated = statistics.median(samples)
            error = abs(recalculated - value["median_seconds"])
            assert error < 1e-15
            historical_timings.append({"path": path, "warmup_calls": 3, "measured_calls": 9, "median_seconds": recalculated, "median_error_seconds": error})
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                inspect_timings(child, path + "/" + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if isinstance(child, (dict, list)):
                inspect_timings(child, path + "/" + str(i))


inspect_timings(efficiency["results"])
assert len(historical_timings) == 10
assert statistics.median([1, 2, 3, 20]) == 2.5
assert statistics.median(list(range(1, 10))) == 5
assert 2**20 == 1048576
raw_path = RAW / "profile-aggregates.json"
raw_path.write_text(json.dumps(raw_runs, ensure_ascii=False, indent=2) + "\n")
receipt = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_16_1",
    "command": COMMAND,
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "CPU", "threads": str(torch.get_num_threads()), "platform": platform.platform()},
    "packet_sha256": digest(packet_path),
    "source_freeze": packet["source_freeze"],
    "source_sha256": digest(snapshot),
    "embedded_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "method": "原樣擷取凍結程式；短輸入與只替換 ids 的練習各完整執行三輪；未訓練，未執行 GPU。",
    "original_stdout": original_stdout,
    "runs": runs,
    "parameters": parameters,
    "linears": linears,
    "api_checks": {"mm_and_matmul": [[17.0], [39.0]], "addmm_bias_ones": [[18.0], [40.0]], "new_zeros": [0.0, 0.0, 0.0], "empty_shapes_only_no_uninitialized_value_read": True, "view_unsqueeze_as_strided_share_storage": True, "repeat_interleave_one_equals_original": True, "arange_1_13": list(range(1, 13)), "none_axis_shape": [1, 12]},
    "benchmark_code_check": {"historical_revision": efficiency["revision"], "historical_architecture_sha256": digest(historical_path), "current_architecture_sha256": digest(current_path), "sync_benchmark_train_ast_unchanged": True, "cpu_actual_function_calls": len(calls), "warmup_calls": 3, "measured_calls": 9, "training_median_expression_input": [100, 200, 300, 1, 2, 3, 4], "training_median_expression_result": training_median_demo, "scope": "執行實際 CPU benchmark 函式與訓練回傳的中位數算式；無訓練及無 GPU 實測。"},
    "existing_efficiency_record_check": {"path": str(efficiency_path.relative_to(ROOT)), "sha256": digest(efficiency_path), "recorded_device": efficiency["device"], "recorded_gpu": efficiency["gpu"], "recorded_torch": efficiency["torch_version"], "seed": efficiency["seed"], "step_scale": efficiency["step_scale"], "timings": historical_timings, "scope": "核既有紀錄的十組九次樣本、暖機與同步欄位並重算中位數；未獨立重跑或證明原 GPU 執行。"},
    "arithmetic": {"parameter_sum": "2112+1024+256+512+40+48+2112=6104", "FP32_bytes": "6104*4=24416", "MiB_bytes": 2**20, "even_sample_median": 2.5, "nine_ordered_values_median": 5, "four_sample_mean": (1 + 2 + 3 + 20) / 4},
    "raw_aggregates": {"path": str(raw_path), "sha256": digest(raw_path), "bytes": raw_path.stat().st_size},
    "result": "全部斷言成立；結果僅支持指定 CPU 設定的 profiler 與算例，以及既有 GPU 紀錄／測量程式的可核算部分。",
}
output_path = HERE / "execution.json"
with output_path.open("x") as file:
    file.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"artifact": str(output_path.relative_to(ROOT)), "sha256": digest(output_path), "runs": len(runs), "historical_timing_medians_recalculated": len(historical_timings), "result": receipt["result"]}, ensure_ascii=False))
