"""核對凍結算例、原始量測紀錄與編譯介面；不重跑 GPU 實驗。"""

import ast
import hashlib
import json
import os
import platform
import statistics
import subprocess
from decimal import Decimal
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ART / "compiler-cache")
os.environ["TRITON_CACHE_DIR"] = str(ART / "triton-cache")

import torch

from tiny_perceptron.modern import DenseFFN, MoEFFN


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def functions(text, names):
    lines = text.splitlines()
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            result[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return result


torch.set_num_threads(1)
torch.manual_seed(42)
record_path = ROOT / "docs/course-experiments/results/efficiency.json"
record = json.loads(record_path.read_bytes())
measured = record["results"]["compile"]
checks = {}

# 十進位精確重算假設，避免把二進位浮點誤差當成成本差異。
plain, compiled = Decimal("0.01"), Decimal("0.006")
checks["cost_scenarios"] = []
for setup in (Decimal("2"), Decimal("4")):
    saving = plain - compiled
    breakeven = setup / saving
    costs = {
        str(n): {"plain": str(plain * n), "compiled": str(setup + compiled * n)}
        for n in (100, 500, 1000)
    }
    checks["cost_scenarios"].append(
        {"setup": str(setup), "saving": str(saving), "breakeven_calls": str(breakeven), "costs": costs}
    )
assert checks["cost_scenarios"][0]["breakeven_calls"] == "5E+2", "準備成本的回本次數不符"
assert checks["cost_scenarios"][1]["breakeven_calls"] == "1E+3", "練習的回本次數不符"

# 自訂 backend 只計數並交回原運算圖，用來核對延後準備與快取重用。
backend_calls = []


def counting_backend(graph, example_inputs):
    backend_calls.append({"graph_nodes": len(list(graph.graph.nodes))})
    return graph.forward


torch._dynamo.reset()
layer = torch.nn.Linear(4, 3).eval()
wrapped = torch.compile(layer, backend=counting_backend, dynamic=False)
counts = [len(backend_calls)]
with torch.no_grad():
    x = torch.ones(2, 4)
    reference = layer(x)
    first = wrapped(x)
    counts.append(len(backend_calls))
    second = wrapped(x)
    counts.append(len(backend_calls))
    changed_shape = wrapped(torch.ones(3, 4))
    counts.append(len(backend_calls))
assert counts == [0, 1, 1, 2], "backend 的呼叫時機與快取重用不符"
assert torch.equal(first, reference) and torch.equal(second, reference), "介面輸出不一致"
checks["compile_interface"] = {
    "backend_counts_after_wrap_first_repeat_new_shape": counts,
    "shape": list(first.shape),
    "changed_shape": list(changed_shape.shape),
    "max_error": float((first - reference).abs().max()),
    "backend": "只回傳 graph.forward 的計數 backend；不建立速度結論",
}

# 固定 router，直接觀察不同輸入在各 expert 形成的資料列數。
moe = MoEFFN(2, experts=2, top_k=1).eval()
with torch.no_grad():
    moe.router.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
inputs = [torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]), torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])]
dispatch = []
with torch.no_grad():
    for x in inputs:
        _, _, chosen = moe(x)
        dispatch.append([int((chosen == i).sum()) for i in range(2)])
assert dispatch == [[2, 0], [1, 1]], "router 分派資料列數不符"
dense = DenseFFN(2).eval()
with torch.no_grad():
    dense_shapes = [list(dense(x).shape) for x in inputs]
checks["dispatch"] = {"input_shapes": [list(x.shape) for x in inputs], "expert_rows": dispatch, "dense_output_shapes": dense_shapes}

# 從九個原始量測值重新排序、取第五個值，並核對秒轉毫秒。
medians = {}
for name in ("eager", "compiled"):
    values = measured[name]["samples_seconds"]
    ordered = sorted(values)
    assert len(values) == 9, "量測樣本數不符"
    assert statistics.median(values) == ordered[4] == measured[name]["median_seconds"], "原始中位數不符"
    assert measured[name]["warmup_calls"] == 3 and measured[name]["synchronized"], "暖機與同步條件不符"
    medians[name] = {
        "sorted_seconds": ordered,
        "fifth_seconds": ordered[4],
        "mean_seconds": statistics.mean(values),
        "milliseconds_six_decimals": f"{ordered[4] * 1000:.6f}",
    }
assert medians["eager"]["milliseconds_six_decimals"] == "0.125070", "eager 毫秒值不符"
assert medians["compiled"]["milliseconds_six_decimals"] == "0.319508", "compiled 毫秒值不符"
saving = measured["eager"]["median_seconds"] - measured["compiled"]["median_seconds"]
assert saving < 0 and measured["estimated_calls_to_amortize_first_call"] is None, "未省時的回本欄位不符"
checks["measured_recalculation"] = {
    "medians": medians,
    "per_call_saving_seconds": saving,
    "first_seconds_three_decimals": f"{measured['first_compiled_call_seconds']:.3f}",
    "worker_seconds_three_decimals": f"{measured['worker_wall_seconds']:.3f}",
    "max_error_six_significant_digits": f"{measured['output_max_error']:.6g}",
    "break_even": None,
    "scope": "重算既存 L4 紀錄，沒有在本機重跑 GPU 量測",
}

# 原始紀錄有完整程式 SHA；依該 SHA 找到真正量測版本，避免只靠 revision 欄位。
code_path = "scripts/course_experiments/architecture.py"
expected_sha = record["code_sha256"][code_path]
matched = None
revisions = git("rev-list", "--all", "--", code_path).decode().splitlines()
for revision in revisions:
    raw = git("show", f"{revision}:{code_path}")
    if sha(raw) == expected_sha:
        matched = {"commit": revision, "sha256": sha(raw), "text": raw.decode()}
        break
assert matched is not None, "未找到與原始量測程式雜湊相符的 Git 版本"
names = {"_sync", "_benchmark", "_compile_worker", "_compile_probe"}
old_functions = functions(matched["text"], names)
current_raw = (ROOT / code_path).read_bytes()
current_functions = functions(current_raw.decode(), names)
checks["measurement_code"] = {
    "path": code_path,
    "recorded_sha256": expected_sha,
    "matched_commit": matched["commit"],
    "current_sha256": sha(current_raw),
    "selected_functions_equal_current": old_functions == current_functions,
    "functions_at_measured_version": old_functions,
    "revision_field_is_not_code_freeze": "量測紀錄的 revision 所指 commit 沒有此路徑；改以 code_sha256 逐檔核對歷史版本",
}

result = {
    "result": "全部核對成立",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-11c/verify.py",
    "record": {
        "path": record_path.relative_to(ROOT).as_posix(),
        "sha256": sha(record_path.read_bytes()),
        "version": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu")},
        "model_config": record["results"]["models"]["mha"]["model"]["config"],
        "compile": measured,
        "peak_memory_scope": record["peak_memory_scope"],
    },
    "checks": checks,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
