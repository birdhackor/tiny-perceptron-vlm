"""Bounded CPU checks and inspection of existing raw L4 measurements; no retraining."""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
from decimal import Decimal
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import TinyLM, ModelConfig
from scripts.course_experiments.common import text_examples

assert str(torch.__version__) == "2.14.1+cpu" and torch.version.cuda is None
torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)
env = {"python": sys.version, "torch": str(torch.__version__),
       "torch_git_version": str(torch.version.git_version), "device": "cpu",
       "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
       "threads": str(torch.get_num_threads()), "default_dtype": str(torch.get_default_dtype()),
       "network_used_by_verification": "false", "original_GPU_training_repeated": "false"}

# Execute only the original fence and the two specified data substitutions.
original = (BASE / "original-execution/fence-1.py").read_text()
matrix = {}
replacements = {
    "original": None,
    "times10": "a = torch.tensor([[1.001, 2.002], [3.003, 4.004]]) * 10",
    "integers": "a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])",
}
for name, replacement in replacements.items():
    code = original if replacement is None else original.replace(
        "a = torch.tensor([[1.001, 2.002], [3.003, 4.004]])", replacement)
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    (BASE / ("variant-" + name + ".py")).write_text(code)
    full, low, a = namespace["full"], namespace["low"], namespace["a"]
    assert full.shape == low.shape == (2, 2)
    assert a.dtype == torch.float32 and low.dtype == torch.bfloat16
    matrix[name] = {"stdout": stdout.getvalue(), "fp32": full.tolist(),
                    "bf16": low.float().tolist(), "a_bf16": a.to(torch.bfloat16).float().tolist(),
                    "max_absolute_error": (full-low.float()).abs().max().item()}
assert abs(matrix["original"]["max_absolute_error"] - 0.0075) < 2e-7
assert matrix["times10"]["max_absolute_error"] > matrix["original"]["max_absolute_error"]
assert matrix["integers"]["max_absolute_error"] == 0
aa = [[Decimal("1.001"), Decimal("2.002")], [Decimal("3.003"), Decimal("4.004")]]
bb = [[Decimal("0.5"), Decimal("1")], [Decimal("1.5"), Decimal("-1")]]
exact = [[sum(aa[i][k]*bb[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
assert exact == [[Decimal("3.5035"), Decimal("-1.001")], [Decimal("7.5075"), Decimal("-1.001")]]

formats = {}
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    f = torch.finfo(dtype)
    formats[str(dtype)] = {"bits": f.bits, "bytes": torch.empty(1, dtype=dtype).element_size(),
                           "eps": f.eps, "max": f.max, "smallest_normal": f.tiny}
assert formats["torch.float16"]["max"] == 65504
assert formats["torch.float16"]["eps"] < formats["torch.bfloat16"]["eps"]
assert abs(formats["torch.bfloat16"]["max"] / 3.39e38 - 1) < 0.001
small = torch.tensor(1e-8)
underflow = {"unscaled_fp16": small.half().item(),
             "scaled_before_fp16_then_unscaled_fp32": ((small * 1024).half().float() / 1024).item(),
             "bf16": small.bfloat16().item()}
assert underflow["unscaled_fp16"] == 0
assert underflow["scaled_before_fp16_then_unscaled_fp32"] > 0
assert underflow["bf16"] > 0

# Scalar CPU GradScaler verifies scale/unscale and distinct attempted/skipped steps.
p = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([p], lr=0.1)
scaler = torch.amp.GradScaler("cpu", init_scale=1024)
scaler.scale((p-3).square()).backward()
scaled_grad = p.grad.item()
scaler.unscale_(optimizer)
unscaled_grad = p.grad.item()
scaler.step(optimizer)
scaler.update()
assert scaled_grad == -4096 and unscaled_grad == -4
assert abs(p.item()-1.4) < 1e-6
scaler_result = {"scaled_gradient": scaled_grad, "unscaled_gradient": unscaled_grad,
                 "parameter_after_finite_step": p.item(), "injected_nonfinite_cases": []}
for value in (float("inf"), float("nan")):
    optimizer.zero_grad(set_to_none=True)
    before = p.item()
    scale_before = scaler.get_scale()
    scaler.scale((p-3).square()).backward()
    p.grad.fill_(value)
    scaler.step(optimizer)
    scaler.update()
    assert p.item() == before and scaler.get_scale() == scale_before/2
    scaler_result["injected_nonfinite_cases"].append({"gradient": str(value), "parameter_unchanged": True,
        "scale_before": scale_before, "scale_after": scaler.get_scale()})

special = torch.tensor([0.0, float("inf"), float("nan")])
assert torch.isfinite(special).tolist() == [True, False, False]
assert math.isnan((torch.tensor(0.0) / torch.tensor(0.0)).item())
assert math.isinf(torch.tensor(1e5).half().item())

# Single bounded optimizer-state construction; this is not the existing model experiment.
mini = torch.nn.Linear(2, 1)
adam = torch.optim.AdamW(mini.parameters(), lr=0.003)
with torch.autocast("cpu", dtype=torch.bfloat16):
    logits = mini(torch.ones(1, 2))
    loss = logits.float().square().sum()
loss.backward()
adam.step()
adam_state = [{"parameter_dtype": str(p.dtype), "exp_avg_dtype": str(adam.state[p]["exp_avg"].dtype),
               "exp_avg_sq_dtype": str(adam.state[p]["exp_avg_sq"].dtype)} for p in mini.parameters()]
assert all(set(x.values()) == {"torch.float32"} for x in adam_state)

# Independently execute only pure original data methods found by AST.
namespace = {"json": json, "hashlib": hashlib, "random": random}
for fname, names in [("prepare_data.py", {"conversation", "generate_records"}),
                      ("common.py", {"split_records", "records_sha256"})]:
    tree = ast.parse((BASE / "repository" / fname).read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), fname+":selected-original-functions", "exec"), namespace)
parts = namespace["split_records"](namespace["generate_records"]("attributes-sft"), seed=42)
(BASE / "deterministically-reconstructed-dataset.json").write_text(json.dumps(parts, ensure_ascii=False, indent=2)+"\n")
result = json.loads((BASE / "precision-original-result.json").read_bytes())
data = result["results"]
assert result["revision"] == "48a4f3e912b483d70aee57c42c2aac226534a9a6"
assert data["source"] == "sft/model.pt" and data["seed"] == 42 and data["step_scale"] == 1
dataset = {}
for name, rows in parts.items():
    sha = namespace["records_sha256"](rows)
    assert sha == data["dataset"][name]["sha256"]
    assert len(rows) == data["dataset"][name]["records"]
    examples = text_examples(rows, "sft", 128)
    dataset[name] = {"records": len(rows), "sha256": sha,
                     "families": sorted({r["family"] for r in rows}),
                     "effective_targets": sum(int((y != IGNORE).sum()) for _, y in examples)}
assert [dataset[n]["records"] for n in ("train", "validation", "test")] == [45, 5, 10]
assert not (set(dataset["train"]["families"]) & set(dataset["validation"]["families"]))
assert not (set(dataset["train"]["families"]) & set(dataset["test"]["families"]))
assert dataset["validation"]["effective_targets"] == 36 and dataset["test"]["effective_targets"] == 69
training_examples = text_examples(parts["train"], "sft", 128)
sampler = random.Random(42)
sampled_targets = sum(sum(int((y != IGNORE).sum()) for _, y in sampler.choices(training_examples, k=16)) for _ in range(200))
assert sampled_targets == 22493

tok = ByteTokenizer()
metrics = {}
inspected_pointers = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu",
    "/step_scale", "/peak_memory_scope", "/code_sha256", "/artifacts",
    "/results/seed", "/results/source", "/results/step_scale", "/results/runtime", "/results/dataset"]
for name, v in data["variants"].items():
    t = v["training"]
    model = TinyLM(ModelConfig(**v["model"]["config"]))
    assert sum(p.numel() for p in model.parameters()) == v["model"]["parameters"] == 141568
    assert sum(p.numel()*p.element_size() for p in model.parameters()) == v["model"]["parameter_bytes"] == 566272
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert t["requested_steps"] == t["steps"] == t["optimizer_updates"] == 200
    assert t["skipped_updates"] == 0 and t["effective_tokens"] == sampled_targets
    assert t["all_requested_attempts_completed"] and not t["budget_exhausted"]
    assert t["batch_size"] == 16 and t["learning_rate"] == 0.003
    assert t["scaler_enabled"] == (name == "fp16")
    assert t["all_parameters_finite"] and v["logits_finite"]
    assert all(math.isfinite(h["loss"]) and h["gradients_finite"] for h in t["history"])
    assert v["weights_dtype"] == "torch.float32"
    assert v["observed_logits_dtype"] == {"fp32": "torch.float32", "bf16": "torch.bfloat16", "fp16": "torch.float16"}[name]
    inf = v["inference"]
    assert len(inf["samples_seconds"]) == inf["measured_calls"] == 9 and inf["warmup_calls"] == 3
    assert statistics.median(inf["samples_seconds"]) == inf["median_seconds"]
    assert inf["synchronized"]
    peak, before = t["peak_memory_allocated_bytes"], t["memory_allocated_before_bytes"]
    assert peak-before == t["peak_additional_allocated_bytes"]
    metrics[name] = {"training_ms": round(t["warm_step_median_seconds"]*1000, 3),
                     "inference_ms": round(inf["median_seconds"]*1000, 3),
                     "memory_mib_independently_rounded": [round(x/2**20, 3) for x in (before, peak, peak-before)],
                     "heldout": {}}
    for split, h in v["heldout"].items():
        samples = h["samples"]
        assert h["records"] == h["examples"] == len(samples) == len(parts[split])
        assert h["effective_tokens"] == dataset[split]["effective_targets"]
        assert abs(h["nll_sum"]/h["effective_tokens"]-h["nll"]) < 1e-12
        matches, ended = 0, 0
        for sample, row in zip(samples, parts[split]):
            expected = row["messages"][-1]["content"]
            assert sample["messages"] == row["messages"][:-1] and sample["expected"] == expected
            ids = sample["generated_ids"]
            assert len(ids) <= 32
            eos = tok.eos_id in ids
            raw = ids[:ids.index(tok.eos_id)] if eos else ids
            exact_match = raw == tok.encode(expected)
            assert exact_match == sample["exact"] and eos == sample["eos"]
            assert tok.decode(raw) == sample["generated"]
            matches += exact_match
            ended += eos
        assert matches == h["matches"] and matches/len(samples) == h["exact_match"]
        assert ended/len(samples) == h["eos_rate"] == 1
        metrics[name]["heldout"][split] = {"nll_rounded5": round(h["nll"], 5), "nll": h["nll"],
            "nll_sum": h["nll_sum"], "effective_targets": h["effective_tokens"],
            "matches": matches, "records": len(samples), "eos": ended,
            "first_expected": samples[0]["expected"], "first_generated": samples[0]["generated"]}
    prefix = "/results/variants/"+name
    inspected_pointers += [prefix+"/model", prefix+"/inference", prefix+"/weights_dtype",
        prefix+"/observed_logits_dtype", prefix+"/logits_finite"]
    inspected_pointers += [prefix+"/training/"+key for key in [
        "requested_steps", "all_requested_attempts_completed", "step_scale", "schedule", "learning_rate",
        "batch_size", "micro_batch_sizes", "gradient_clip_norm", "auxiliary_weight", "steps", "optimizer_updates",
        "skipped_updates", "budget_exhausted", "effective_tokens", "warm_step_median_seconds",
        "memory_allocated_before_bytes", "peak_memory_allocated_bytes", "peak_additional_allocated_bytes",
        "scaler_enabled", "all_parameters_finite", "history"]]
    for split in ("validation", "test"):
        inspected_pointers += [prefix+"/heldout/"+split+"/"+key for key in [
            "nll", "nll_sum", "effective_tokens", "examples", "exact_match", "matches", "records", "eos_rate", "samples"]]
assert metrics["fp32"]["heldout"]["test"]["first_generated"] == "circle"
assert metrics["fp16"]["heldout"]["test"]["first_generated"] == "circlow"
assert metrics["bf16"]["heldout"]["test"]["nll"] < metrics["fp32"]["heldout"]["test"]["nll"]
assert metrics["bf16"]["heldout"]["test"]["matches"] == metrics["fp32"]["heldout"]["test"]["matches"]-1
for name in ("bf16", "fp16"):
    assert metrics[name]["training_ms"] > metrics["fp32"]["training_ms"]
    assert metrics[name]["inference_ms"] > metrics["fp32"]["inference_ms"]

out = {"environment": env, "matrix": matrix, "exact_decimal_row_column_products": [[str(x) for x in row] for row in exact],
       "formats": formats, "underflow": underflow, "grad_scaler_cpu": scaler_result,
       "adamw_cpu_state": adam_state, "finite_mask_for_zero_inf_nan": torch.isfinite(special).tolist(),
       "dataset": dataset, "training_sampled_effective_targets": sampled_targets,
       "original_measurements_recomputed": metrics, "inspected_result_pointers": inspected_pointers,
       "all_assertions_passed": True, "scope": "No L4 run, no model checkpoints, no model reevaluation or retraining; pure data reconstruction, raw JSON arithmetic, original fence and bounded CPU mechanism checks."}
(BASE / "cpu-and-raw-results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
(BASE / "cpu-environment.json").write_text(json.dumps(env, indent=2)+"\n")
print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
