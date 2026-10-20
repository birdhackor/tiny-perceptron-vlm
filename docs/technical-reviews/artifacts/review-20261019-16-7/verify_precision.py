"""離線核對精度算例、資料分母與已保存的實驗數值。"""

import ast
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


torch.set_num_threads(1)
record_path = "docs/course-experiments/results/precision.json"
record = json.loads((ROOT / record_path).read_bytes())
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-7/verify_precision.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "platform": platform.platform(),
        "cuda_available": str(torch.cuda.is_available()),
    },
    "inputs": {record_path: sha(record_path)},
    "scope": "CPU算例實跑；L4數據只重算既有紀錄，不重跑GPU訓練或計時。",
}

b = torch.tensor([[0.5, 1.0], [1.5, -1.0]])
cases = {
    "original": torch.tensor([[1.001, 2.002], [3.003, 4.004]]),
    "times_ten": torch.tensor([[1.001, 2.002], [3.003, 4.004]]) * 10,
    "integer_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
}
result["matrix_cases"] = {}
for name, a in cases.items():
    before = a.clone()
    full = a @ b
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low = a @ b
    error = (full - low.float()).abs()
    expected = {"original": 0.0075, "times_ten": 0.075, "integer_values": 0.0}[name]
    observed = round(error.max().item(), 4)
    assert observed == expected
    assert a.dtype == torch.float32 and low.dtype == torch.bfloat16
    assert torch.equal(before, a)
    result["matrix_cases"][name] = {
        "input_dtype": str(a.dtype),
        "output_dtype": str(low.dtype),
        "input_unchanged": torch.equal(before, a),
        "rounded_input": a.bfloat16().float().tolist(),
        "full": full.tolist(),
        "low": low.float().tolist(),
        "absolute_errors": error.tolist(),
        "max_error": error.max().item(),
        "rounded_max_error": observed,
        "roundtrip_restores_input": torch.equal(a.bfloat16().float(), a),
    }

left = Decimal("1.001") * Decimal("0.5") + Decimal("2.002") * Decimal("1.5")
right = Decimal("1.001") - Decimal("2.002")
bottom = Decimal("3.003") * Decimal("0.5") + Decimal("4.004") * Decimal("1.5")
assert (left, right, bottom) == (Decimal("3.5035"), Decimal("-1.001"), Decimal("7.5075"))
result["decimal_products"] = [str(left), str(right), str(bottom)]
result["formats"] = {}
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    info = torch.finfo(dtype)
    result["formats"][str(dtype)] = {
        "bits": info.bits,
        "bytes": torch.empty((), dtype=dtype).element_size(),
        "epsilon": info.eps,
        "max": info.max,
        "smallest_normal": info.tiny,
    }
assert torch.finfo(torch.float16).max == 65504
assert torch.finfo(torch.float16).eps < torch.finfo(torch.bfloat16).eps
result["underflow_and_special_values"] = {
    "small_fp16": torch.tensor(1e-8, dtype=torch.float16).item(),
    "same_small_bf16": torch.tensor(1e-8, dtype=torch.bfloat16).item(),
    "zero_div_zero_is_nan": bool(torch.isnan(torch.tensor(0.0) / torch.tensor(0.0))),
    "overflow_is_inf": bool(torch.isinf(torch.tensor(1e10).half())),
    "zero_is_finite": bool(torch.isfinite(torch.tensor(0.0))),
}

x = torch.tensor(4.0, requires_grad=True)
normal_gradient = torch.autograd.grad(x * x, x)[0].item()
x = torch.tensor(4.0, requires_grad=True)
scaled_gradient = torch.autograd.grad(1024 * x * x, x)[0].item()
assert scaled_gradient == normal_gradient * 1024
result["scaled_gradient"] = {
    "normal": normal_gradient,
    "scaled": scaled_gradient,
    "scale": 1024,
    "unscaled": scaled_gradient / 1024,
}
result["scaler_steps"] = []
for value in (2.0, float("inf"), float("nan")):
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024)
    scaler.scale(p * value).backward()
    before = p.item()
    scaler.step(optimizer)
    scaler.update()
    skipped = p.item() == before
    assert skipped == (not math.isfinite(value))
    result["scaler_steps"].append({
        "gradient_kind": "finite" if math.isfinite(value) else ("inf" if math.isinf(value) else "nan"),
        "parameter_before": before,
        "parameter_after": p.item(),
        "skipped": skipped,
        "scale_after": scaler.get_scale(),
    })

data = split_records(generate_records("attributes-sft"), seed=42)
examples = {name: text_examples(rows, "sft", 128) for name, rows in data.items()}
families = {name: {row["family"] for row in rows} for name, rows in data.items()}
assert all(not families[a] & families[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
result["dataset"] = {}
for name, rows in data.items():
    digest = records_sha256(rows)
    assert digest == record["results"]["dataset"][name]["sha256"]
    targets = [int((labels != IGNORE).sum()) for _, labels in examples[name]]
    assert targets == [len(row["messages"][-1]["content"].encode()) + 1 for row in rows]
    result["dataset"][name] = {
        "records": len(rows),
        "families": sorted(families[name]),
        "sha256": digest,
        "effective_targets": sum(targets),
        "targets_per_record": targets,
        "largest_input_length": max(len(x) for x, _ in examples[name]),
    }
sampler = random.Random(42)
effective = 0
for _ in range(200):
    batch = sampler.choices(examples["train"], k=16)
    effective += sum(int((labels != IGNORE).sum()) for _, labels in batch)
assert effective == 22493
result["sampled_training_targets"] = effective

model = TinyLM(ModelConfig(width=64, layers=2, heads=1, max_length=128))
parameter_count = sum(p.numel() for p in model.parameters())
parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
assert (parameter_count, parameter_bytes) == (141568, 566272)
result["local_model"] = {"parameters": parameter_count, "parameter_bytes": parameter_bytes}
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
optimizer.zero_grad(set_to_none=True)
sum(p.sum() for p in model.parameters()).backward()
optimizer.step()
result["local_model"]["parameter_dtypes_after_adamw"] = sorted({str(p.dtype) for p in model.parameters()})
result["local_model"]["optimizer_state_dtypes"] = sorted({
    str(value.dtype) for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)
})

tok = ByteTokenizer()
result["recorded_environment"] = {k: record[k] for k in ("device", "gpu", "torch_version", "python_version", "seed", "step_scale", "status")}
result["variants"] = {}
expected_rows = {
    "fp32": ["0.47748", "0.36272", 2, 7, "11.673", "2.444", ["33.086", "79.261", "46.175"]],
    "bf16": ["0.83820", "0.32709", 1, 6, "14.414", "2.835", ["65.374", "75.882", "10.509"]],
    "fp16": ["1.17649", "0.71923", 1, 4, "14.280", "2.740", ["65.232", "75.743", "10.510"]],
}
for name, variant in record["results"]["variants"].items():
    training, heldout, inference = (variant[key] for key in ("training", "heldout", "inference"))
    assert training["steps"] == training["optimizer_updates"] == training["requested_steps"] == 200
    assert training["skipped_updates"] == 0 and training["effective_tokens"] == effective
    assert training["batch_size"] == 16 and training["learning_rate"] == 0.003
    assert training["all_parameters_finite"] and variant["logits_finite"]
    assert all(math.isfinite(h["loss"]) and h["gradients_finite"] for h in training["history"])
    assert training["scaler_enabled"] == (name == "fp16")
    assert variant["weights_dtype"] == "torch.float32"
    assert variant["model"]["parameters"] == parameter_count
    assert variant["model"]["parameter_bytes"] == parameter_bytes
    assert inference["warmup_calls"] == 3 and inference["measured_calls"] == len(inference["samples_seconds"]) == 9
    assert statistics.median(inference["samples_seconds"]) == inference["median_seconds"]
    checked_heldout = {}
    for split, report in heldout.items():
        assert len(report["samples"]) == report["records"] == len(data[split])
        assert report["effective_tokens"] == result["dataset"][split]["effective_targets"]
        assert math.isclose(report["nll_sum"] / report["effective_tokens"], report["nll"], rel_tol=1e-14)
        exact, eos = 0, 0
        for row, sample in zip(data[split], report["samples"], strict=True):
            assert row["messages"][:-1] == sample["messages"]
            assert row["messages"][-1]["content"] == sample["expected"]
            ids = sample["generated_ids"]
            assert len(ids) <= 32
            raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            assert tok.decode(raw) == sample["generated"]
            assert (raw == tok.encode(sample["expected"])) == sample["exact"]
            assert (tok.eos_id in ids) == sample["eos"]
            exact += int(sample["exact"])
            eos += int(sample["eos"])
        assert exact == report["matches"]
        assert eos / len(data[split]) == report["eos_rate"] == 1.0
        checked_heldout[split] = {
            "nll": report["nll"], "nll_sum": report["nll_sum"],
            "effective_targets": report["effective_tokens"], "matches": exact,
            "records": len(data[split]), "eos_ended": eos,
            "first_sample": report["samples"][0],
        }
    memory_keys = ("memory_allocated_before_bytes", "peak_memory_allocated_bytes", "peak_additional_allocated_bytes")
    memory = [training[key] for key in memory_keys]
    assert memory[1] - memory[0] == memory[2]
    row = [f"{heldout['validation']['nll']:.5f}", f"{heldout['test']['nll']:.5f}",
           heldout["validation"]["matches"], heldout["test"]["matches"],
           f"{training['warm_step_median_seconds'] * 1000:.3f}",
           f"{inference['median_seconds'] * 1000:.3f}", [f"{value / 2**20:.3f}" for value in memory]]
    assert row == expected_rows[name]
    result["variants"][name] = {
        "page_table_values": row, "memory_bytes": dict(zip(memory_keys, memory, strict=True)),
        "heldout_recalculated": checked_heldout, "steps": training["steps"],
        "optimizer_updates": training["optimizer_updates"], "skipped_updates": training["skipped_updates"],
        "scaler_enabled": training["scaler_enabled"], "weights_dtype": variant["weights_dtype"],
        "observed_logits_dtype": variant["observed_logits_dtype"],
        "sampled_history_losses_finite": True, "all_parameters_finite": training["all_parameters_finite"],
        "logits_finite": variant["logits_finite"], "history": training["history"],
        "inference_median_recalculated": statistics.median(inference["samples_seconds"]),
    }
result["memory_start_difference_mib"] = {
    name: (variant["training"]["memory_allocated_before_bytes"] - record["results"]["variants"]["fp32"]["training"]["memory_allocated_before_bytes"]) / 2**20
    for name, variant in record["results"]["variants"].items() if name != "fp32"
}

historical_commit = "a5133f444e0dca2347635f74f8ac83a27bb203d4"
path = "scripts/course_experiments/architecture.py"
raw = subprocess.run(["git", "show", f"{historical_commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
historical_hash = hashlib.sha256(raw).hexdigest()
assert historical_hash == record["code_sha256"][path]
before = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
after = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((ROOT / path).read_bytes()).body if isinstance(node, ast.FunctionDef)}
functions = ("_sync", "_description", "_sft_dataset", "_data_report", "_runtime", "_steps", "_amp", "_forward", "_nll", "_train", "_heldout", "_benchmark", "run_precision")
assert all(before[name] == after[name] for name in functions)
result["implementation_provenance"] = {
    "historical_commit": historical_commit,
    "historical_file_sha256": historical_hash,
    "current_file_sha256": sha(path),
    "functions_with_identical_ast": list(functions),
    "note": "實驗紀錄的整檔SHA與歷史程式相符；現行檔雖不同，本節使用的函式AST逐一相同。",
}
for path in ("scripts/course_experiments/common.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py"):
    assert sha(path) == record["code_sha256"][path]
    result["inputs"][path] = sha(path)
for path in ("scripts/course_experiments/architecture.py", "scripts/prepare_data.py", "tiny_perceptron/training.py"):
    result["inputs"][path] = sha(path)
result["result"] = "所有斷言通過"
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
