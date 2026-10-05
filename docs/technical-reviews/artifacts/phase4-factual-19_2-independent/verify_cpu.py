"""Independent, bounded CPU verification of section 19.2; no checkpoint I/O."""

import contextlib
import hashlib
import io
import json
import math
import platform
import runpy
import statistics
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, default_config, prepare_batch
from tiny_perceptron.data import IGNORE
from tiny_perceptron.modern import MoEFFN
from scripts.course_experiments.capstone_deployment import benchmark_training_step


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name, value):
    print(json.dumps({"check": name, "observed": value}, ensure_ascii=False, sort_keys=True))


assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
environment = {
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "threads": str(torch.get_num_threads()),
}
(HERE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
record("environment", environment)

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    runpy.run_path(str(HERE / "fence-1.py"), run_name="__main__")
original_output = stdout.getvalue()
print("ORIGINAL FENCE STDOUT\n" + original_output, end="")
expected_output = (
    "MoE 全部參數 328128\n"
    "邏輯active參數 195776 FP32權重bytes 1312512\n"
    "Dense 全部參數 129088\n"
    "邏輯active參數 129088 FP32權重bytes 516352\n"
)
assert original_output == expected_output

counts = {}
for dense, width in [(False, 64), (True, 64), (True, 80)]:
    c = replace(default_config(dense=dense), width=width)
    model = CapstoneModel(c)
    description = model.description()
    expert_count = 8 * width * width + 5 * width
    measured_ffn_count = sum(p.numel() for p in (
        model.language.blocks[0].ffn.experts[0] if not dense else model.language.blocks[0].ffn
    ).parameters())
    assert expert_count == measured_ffn_count
    total = sum(p.numel() for p in model.parameters())
    assert all(p.dtype == torch.float32 and p.element_size() == 4 for p in model.parameters())
    inactive = 0 if dense else c.layers * (c.experts - c.top_k) * expert_count
    assert description["parameters"] == total
    assert description["logical_active_parameters"] == total - inactive
    assert model.language.embedding.weight is not model.language.output.weight
    assert model.language.embedding.weight.data_ptr() != model.language.output.weight.data_ptr()
    counts[("Dense" if dense else "MoE") + str(width)] = {
        "parameters": total,
        "logical_active_parameters": total - inactive,
        "one_ffn_parameters": expert_count,
        "all_fp32_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "embedding_and_output_untied": True,
        "optimizer_updates": 0,
    }
assert counts["MoE64"]["parameters"] == 328128
assert counts["MoE64"]["logical_active_parameters"] == 195776
assert counts["Dense64"]["parameters"] == 129088
assert counts["Dense80"]["parameters"] == 189520
record("independent_parameter_and_byte_count", counts)

# Force a collapsed assignment with positive, identical input features.
router = MoEFFN(width=4, experts=4, top_k=2, hidden=8)
with torch.no_grad():
    router.router.weight.copy_(torch.tensor([[3., 2., 0., 0.], [2., 3., 0., 0.], [-3., -2., 0., 0.], [-2., -3., 0., 0.]]))
x = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]]])
y, aux, chosen = router(x)
assert chosen.tolist() == [[0, 1], [0, 1], [0, 1]]
y.square().sum().backward()
expert_has_task_gradient = [any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in e.parameters()) for e in router.experts]
assert expert_has_task_gradient == [True, True, False, False]
flat = x.reshape(-1, 4)
probabilities = router.router(flat).softmax(-1)
# The repo top-2 variant averages over token AND choice axes: 3*2 assignments.
load = torch.bincount(chosen.reshape(-1), minlength=4).float() / chosen.numel()
importance = probabilities.mean(0)
manual_aux = 4 * (load * importance).sum()
assert torch.allclose(aux, manual_aux, atol=1e-7, rtol=0)
router.zero_grad(set_to_none=True)
_, aux, _ = router(x)
aux.backward()
aux_router_gradient = float(router.router.weight.grad.abs().sum())
assert aux_router_gradient > 0
balanced_x = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.]]])
_, balanced_aux, balanced_chosen = router(balanced_x)
balanced_load = torch.bincount(balanced_chosen.reshape(-1), minlength=4).float() / balanced_chosen.numel()
assert torch.equal(balanced_load, torch.full((4,), 0.25))
assert abs(balanced_aux.item() - 1) < 1e-6 and aux.item() > balanced_aux.item()
record("routing_and_load_balance_variant", {
    "collapsed_selected_experts": chosen.tolist(),
    "expert_has_task_gradient": expert_has_task_gradient,
    "load_denominator_token_choice_assignments": int(chosen.numel()),
    "load": load.tolist(),
    "importance": importance.detach().tolist(),
    "computed_aux": aux.item(),
    "independent_aux": manual_aux.item(),
    "aux_router_gradient_abs_sum": aux_router_gradient,
    "balanced_load": balanced_load.tolist(),
    "balanced_aux": balanced_aux.item(),
    "optimizer_updates": 0,
})

# Vary padding, keeping all three real input positions identical.
model = CapstoneModel()
model.eval()
ids = torch.tensor([[10, 11, 0], [20, 0, 0]])
valid = torch.tensor([[True, True, False], [True, False, False]])
extended_ids = torch.cat([ids, torch.tensor([[17, 18], [19, 21]])], dim=1)
extended_valid = torch.cat([valid, torch.zeros((2, 2), dtype=torch.bool)], dim=1)
ffn_input_shapes = []
hooks = [b.ffn.register_forward_pre_hook(lambda module, args: ffn_input_shapes.append(list(args[0].shape))) for b in model.language.blocks]
with torch.no_grad():
    before = model(ids, valid=valid)
    after = model(extended_ids, valid=extended_valid)
for hook in hooks:
    hook.remove()
assert ffn_input_shapes == [[1, 3, 64]] * 4
max_logit_difference = float((before["logits"][valid] - after["logits"][:, :3][valid]).abs().max())
aux_difference = abs(before["auxiliary"].item() - after["auxiliary"].item())
assert max_logit_difference < 1e-6 and aux_difference < 1e-6
record("padding_exclusion_variant", {
    "real_tokens": int(valid.sum()),
    "ffn_input_shapes_for_two_layers_two_calls": ffn_input_shapes,
    "padding_positions_before": int((~valid).sum()),
    "padding_positions_after": int((~extended_valid).sum()),
    "max_real_logit_difference": max_logit_difference,
    "auxiliary_difference": aux_difference,
    "tolerance": "absolute < 1e-6",
})

splits, _ = build_dataset(seed=42)
raw = json.loads((HERE / "originals/docs/course-experiments/capstone-evidence/deployment/mechanism-benchmark.json").read_text())
batch, labels = prepare_batch(splits["train"][:24])
independent_measurements = {}
for name in ["moe", "dense80"]:
    measured = raw[name]
    assert len(measured["seconds"]) == measured["measured_iterations"] == 10
    assert measured["warmup_iterations"] == 3
    assert measured["batch_ids"] == [row["id"] for row in splits["train"][:24]]
    assert measured["batch_shape"] == list(batch["ids"].shape) == [24, 94]
    assert measured["optimizer_updates"] == 0 and measured["weights_unchanged"] is True
    median = statistics.median(measured["seconds"])
    mean = statistics.mean(measured["seconds"])
    assert math.isclose(median, measured["median_seconds"], abs_tol=1e-12, rel_tol=0)
    assert math.isclose(mean, measured["mean_seconds"], abs_tol=1e-12, rel_tol=0)
    extra = measured["cuda_peak_allocated_bytes"] - measured["cuda_baseline_allocated_bytes"]
    assert extra == measured["cuda_peak_extra_bytes"]
    expected_ms = {"moe": 21.464, "dense80": 11.338}[name]
    assert abs(median * 1000 - expected_ms) <= 0.0005
    independent_measurements[name] = {
        "median_seconds": median,
        "median_milliseconds": median * 1000,
        "mean_seconds": mean,
        "peak_minus_baseline_bytes": extra,
        "timed_iterations": len(measured["seconds"]),
        "untimed_warmup_iterations": measured["warmup_iterations"],
    }
assert raw["moe"]["cuda_peak_extra_bytes"] > raw["dense80"]["cuda_peak_extra_bytes"]
record("recompute_recorded_L4_measurements_not_GPU_rerun", {
    "measurements": independent_measurements,
    "batch_rows": len(splits["train"][:24]),
    "padded_batch_shape": list(batch["ids"].shape),
    "real_input_token_positions": int(batch["valid"].sum()),
    "supervised_target_positions": int((labels != IGNORE).sum()),
    "seconds_to_milliseconds": 1000,
    "median_formula": "average of sorted samples at indices 4 and 5 for ten samples",
    "rounding_tolerance_milliseconds": 0.0005,
})

# Execute the real benchmark helper on a tiny random model and two fixed rows.
# This verifies method behavior, not the original L4 timings or learned quality.
small = CapstoneModel(replace(default_config(), width=8, layers=1))
small.eval()
state_before = {k: v.detach().clone() for k, v in small.state_dict().items()}
probe = benchmark_training_step(small, splits["train"][:2], warmup=1, measured=2)
assert probe["weights_unchanged"] is True and probe["optimizer_updates"] == 0
assert len(probe["seconds"]) == 2 and probe["warmup_iterations"] == 1
assert all(torch.equal(v, small.state_dict()[k]) for k, v in state_before.items())
assert small.training is False and all(p.grad is None for p in small.parameters())
assert all(probe[k] is None for k in ["cuda_baseline_allocated_bytes", "cuda_peak_allocated_bytes", "cuda_peak_extra_bytes"])
record("bounded_CPU_benchmark_helper_variant", probe)

run = json.loads((HERE / "deployment-run-original.json").read_text())
manifest = json.loads((HERE / "originals/docs/course-experiments/capstone-evidence/deployment/review-manifest.json").read_text())
benchmark_file = HERE / "originals/docs/course-experiments/capstone-evidence/deployment/mechanism-benchmark.json"
receipts = [x for x in run["artifacts"] if x["path"] == "mechanism-benchmark.json"]
assert len(receipts) == 1 and receipts[0]["sha256"] == digest(benchmark_file)
assert receipts[0] == [x for x in manifest["files"] if x["path"] == "mechanism-benchmark.json"][0]
assert run["modal"]["run_id"] == manifest["run_id"] == "gha-37169529991-1"
assert run["revision"] == manifest["training_revision"] == "6ffc653199a71ad83afcee24aa8a2388122771bd"
assert run["gpu"] == "NVIDIA L4" and run["torch_version"] == "2.14.1+cu126"
for name in ["moe", "dense80"]:
    for key in ["parameters", "device", "dtype", "batch_ids", "batch_shape", "warmup_iterations", "measured_iterations", "seconds", "median_seconds", "mean_seconds", "optimizer_updates", "weights_unchanged", "cuda_baseline_allocated_bytes", "cuda_peak_allocated_bytes", "cuda_peak_extra_bytes"]:
        assert run["results"]["mechanism_benchmark"][name][key] == raw[name][key]
assert run["results"]["mechanism_benchmark"]["checkpoint_sha256"] == raw["checkpoint_sha256"]
checked_code = ["tiny_perceptron/capstone.py", "tiny_perceptron/model.py", "tiny_perceptron/modern.py", "scripts/course_experiments/capstone.py", "scripts/course_experiments/capstone_deployment.py", "scripts/course_experiments/run.py"]
for name in checked_code:
    assert digest(HERE / "frozen-run-code" / name) == run["code_sha256"][name] == digest(ROOT / name)
training = json.loads((HERE / "originals/docs/course-experiments/capstone-evidence/joint/train-report.json").read_text())
assert training["stage"] == "joint" and training["steps"] == training["requested_steps"] == 600
assert training["schedule_completed"] is True
assert training["inference_export"]["sha256"] == raw["checkpoint_sha256"]
assert len(training["history"]) == 11 and all(math.isfinite(h["auxiliary_before_update"]) for h in training["history"])
record("raw_provenance_and_version_links", {
    "run_id": run["modal"]["run_id"],
    "training_revision": run["revision"],
    "original_gpu": run["gpu"],
    "original_torch": run["torch_version"],
    "original_python": run["python_version"],
    "benchmark_file_receipt": receipts[0],
    "joint_checkpoint_sha256": raw["checkpoint_sha256"],
    "joint_training_steps": training["steps"],
    "joint_auxiliary_log_entries": len(training["history"]),
    "source_files_verified_against_frozen_run_hashes": checked_code,
})
print("ALL BOUNDED CPU ASSERTIONS PASSED")
