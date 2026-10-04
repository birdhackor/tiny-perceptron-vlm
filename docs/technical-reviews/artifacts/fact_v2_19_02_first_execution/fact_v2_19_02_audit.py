"""Replay 19.2 on CPU, retaining tensor evidence rather than a tracked checkpoint."""

import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from scripts.course_experiments.capstone_deployment import benchmark_training_step
from tiny_perceptron.capstone import CapstoneModel, build_dataset, default_config, evaluate_rows, load_capstone, prepare_batch
from tiny_perceptron.data import IGNORE, ByteTokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_02_"
ENV = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "1"}
torch.set_num_threads(1)
torch.manual_seed(42)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    p = OUT / (PREFIX + name)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_inventory(model):
    rows = []
    for name, tensor in model.state_dict().items():
        rows.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(),
                     "element_size": tensor.element_size(), "bytes": tensor.numel() * tensor.element_size(),
                     "finite": bool(torch.isfinite(tensor).all()), "sha256": tensor_sha(tensor)})
    return rows


def summary(evaluation):
    by_task = {}
    for record in evaluation["records"]:
        row = by_task.setdefault(record["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        row["count"] += 1
        row["action_correct"] += int(record["action_trace"]["eos"] and record["action_trace"]["raw"] == record["expected_action"])
        row["end_to_end_correct"] += int(record["action_correct"] and record["answer"] == record["expected_final"])
    result = {"count": len(evaluation["records"]), "action_correct": sum(r["action_correct"] for r in evaluation["records"]),
              "end_to_end_correct": sum(r["end_to_end_correct"] for r in evaluation["records"]), "by_task": by_task}
    assert all(result[k] == evaluation[k] for k in result)
    return result


section_file = OUT / (PREFIX + "section_19_2.md")
section = section_file.read_text(encoding="utf-8")
code = re.findall(r"```python\n(.*?)```", section, re.S)[0]
run_records = []
for name, program in [("original", code), ("exercise", code.replace("total * 4", "total * 16"))]:
    p = OUT / (PREFIX + name + ".py")
    p.write_text(program, encoding="utf-8")
    result = subprocess.run([sys.executable, str(p)], cwd=ROOT, capture_output=True, text=True, check=False)
    (OUT / (PREFIX + name + "_stdout.txt")).write_text(result.stdout, encoding="utf-8")
    (OUT / (PREFIX + name + "_stderr.txt")).write_text(result.stderr, encoding="utf-8")
    run_records.append({"kind": name, "command": f"PYTHONPATH=. .venv/bin/python {p.relative_to(ROOT)}",
                        "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                        "code_sha256": sha(p), "section_sha256": sha(section_file), "environment": ENV})
    assert result.returncode == 0
save("snippet_execution.json", run_records)

models = {"moe64": CapstoneModel(default_config()), "dense64": CapstoneModel(default_config(dense=True)),
          "dense80": CapstoneModel(replace(default_config(dense=True), width=80))}
counts = {}
for name, model in models.items():
    description = model.description()
    groups = Counter()
    for key, parameter in model.named_parameters():
        group = "experts" if ".experts." in key else "routers" if ".router." in key else "other"
        groups[group] += parameter.numel()
    counts[name] = {"description": description, "groups": dict(groups),
                    "bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                    "embedding_output_same_object": model.language.embedding.weight is model.language.output.weight,
                    "embedding_output_same_storage": model.language.embedding.weight.data_ptr() == model.language.output.weight.data_ptr(),
                    "parameter_tables": tensor_inventory(model)}
assert counts["moe64"]["description"]["parameters"] == 328128
assert counts["moe64"]["description"]["logical_active_parameters"] == 195776
assert counts["moe64"]["bytes"] == 1312512
assert counts["dense64"]["description"]["parameters"] == 129088
assert counts["dense80"]["description"]["parameters"] == 189520
assert not counts["moe64"]["embedding_output_same_storage"]

# This zero-lr initialization allocates Adam state; it performs no learning.
adam_model = models["moe64"]
before = tensor_inventory(adam_model)
adam = torch.optim.Adam(adam_model.parameters(), lr=0.0, amsgrad=False)
for parameter in adam_model.parameters():
    parameter.grad = torch.zeros_like(parameter)
adam.step()
state_bytes = Counter()
for state in adam.state.values():
    for key, tensor in state.items():
        state_bytes[key] += tensor.numel() * tensor.element_size()
gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in adam_model.parameters())
coarse_bytes = counts["moe64"]["bytes"] + gradient_bytes + state_bytes["exp_avg"] + state_bytes["exp_avg_sq"]
assert coarse_bytes == 328128 * 16
assert tensor_inventory(adam_model) == before
counts["adam_exercise"] = {"gradient_bytes": gradient_bytes, "state_bytes": dict(state_bytes),
                           "coarse_bytes_excluding_step_scalars": coarse_bytes, "weights_unchanged": True,
                           "limitation": "FP32 weights/grad/first+second moments only; excludes step scalars, allocator, activations, Python and framework."}
counts["tokenizer"] = {word: {"utf8_hex": word.encode("utf-8").hex(), "ids": ByteTokenizer().encode(word)}
                       for word in ["A", "中", "文", "𠀀"]}
save("numeric_audit.json", {"environment": ENV, "result": counts})

splits, manifest = build_dataset(42)
batch, labels = prepare_batch(splits["train"][:24])
denominators = {"batch_shape": list(batch["ids"].shape), "grid_positions": batch["ids"].numel(),
                "valid_input_positions": int(batch["valid"].sum()), "effective_answer_targets": int((labels != IGNORE).sum()),
                "sample_ids": [r["id"] for r in splits["train"][:24]], "seed": 42,
                "split_rows": {k: len(v) for k, v in splits.items()},
                "family_count": {k: len({r["family"] for r in v}) for k, v in splits.items()},
                "task_counts": {k: dict(Counter(r["task"] for r in v)) for k, v in splits.items()}}
assert denominators["batch_shape"] == [24, 94]
save("dataset_audit.json", {"manifest": manifest, "denominators": denominators})

original_checkpoint = ROOT / "outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt"
weights = {"environment": ENV, "original_path": str(original_checkpoint.relative_to(ROOT)), "original_available": original_checkpoint.exists()}
if original_checkpoint.exists():
    assert sha(original_checkpoint) == "8f7e85820bd70bf1f16e7ef789b11f2d10873cf26a2688d651e23136bd7c3b47"
    model, payload = load_capstone(original_checkpoint)
    weights["original"] = {"sha256": sha(original_checkpoint), "bytes": original_checkpoint.stat().st_size,
                           "non_tensor_payload": {k: v for k, v in payload.items() if k != "model"},
                           "tensor_inventory": tensor_inventory(model), "strict_load": True}
public = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
public_joint = next(item for item in public["models"] if item["id"] == "joint")
public_file = next(item for item in public_joint["files"] if item["output"] == "model.pt")
weights["public_request"] = {"repo_id": public["repo"], "revision": public["revision"], "filename": public_file["path"],
                             "token": False, "expected_sha256": public_file["sha256"], "expected_bytes": public_file["bytes"]}
try:
    downloaded = hf_hub_download(repo_id=public["repo"], filename=public_file["path"], revision=public["revision"], token=False,
                                 local_dir=ROOT / "outputs/technical-sources/fact_v2_19_02_public")
    assert sha(downloaded) == public_file["sha256"]
    public_model, public_payload = load_capstone(downloaded)
    if original_checkpoint.exists():
        equal = all(torch.equal(value, public_model.state_dict()[key]) for key, value in model.state_dict().items())
        assert equal
    else:
        model, payload = public_model, public_payload
        equal = None
    weights["public"] = {"sha256": sha(downloaded), "bytes": Path(downloaded).stat().st_size,
                         "non_tensor_payload": {k: v for k, v in public_payload.items() if k != "model"},
                         "tensor_inventory": tensor_inventory(public_model), "strict_load": True,
                         "all_tensors_equal_to_original": equal}
except Exception as error:
    weights["public_fetch_error"] = {"type": type(error).__name__, "message": str(error)}
    if not original_checkpoint.exists():
        save("weights_audit.json", weights)
        raise
save("weights_audit.json", weights)

# Directly rederive f_i and P_i over only the valid-token tensor seen by each FFN.
model.eval()
pad_records = []
ids = torch.tensor([[1, 52, 53, 54, 55], [1, 60, 61, 0, 0]])
valid = ids != 0
for added in [0, 4]:
    current_ids = torch.cat([ids, torch.zeros(2, added, dtype=torch.long)], dim=1)
    current_valid = torch.cat([valid, torch.zeros(2, added, dtype=torch.bool)], dim=1)
    observed = []

    def routing_hook(module, args, output):
        flat = args[0].detach().reshape(-1, model.config.width)
        _, auxiliary, chosen = output
        probabilities = module.router(flat).softmax(-1)
        selection_counts = torch.bincount(chosen.flatten(), minlength=4)
        load = selection_counts.float() / (len(flat) * module.top_k)
        importance = probabilities.mean(0)
        reconstructed = 4 * (load * importance).sum()
        observed.append({"ffn_input_shape": list(args[0].shape), "valid_tokens": len(flat),
                         "routing_assignments": chosen.numel(), "selection_counts": selection_counts.tolist(),
                         "load_f_i": load.tolist(), "importance_P_i": importance.detach().tolist(),
                         "auxiliary": float(auxiliary.detach()), "reconstructed_auxiliary": float(reconstructed.detach()),
                         "difference": float((auxiliary - reconstructed).abs().detach())})

    hooks = [block.ffn.register_forward_hook(routing_hook) for block in model.language.blocks]
    result = model(current_ids, valid=current_valid)
    for hook in hooks:
        hook.remove()
    if added == 0:
        original_logits = result["logits"][valid].detach().clone()
        original_auxiliary = result["auxiliary"].detach().clone()
    delta = float((result["logits"][current_valid] - original_logits).abs().max().detach())
    aux_delta = float((result["auxiliary"] - original_auxiliary).abs().detach())
    assert delta <= 1e-5 and aux_delta <= 1e-6
    assert all(row["valid_tokens"] == 8 and row["routing_assignments"] == 16 and row["difference"] <= 1e-6 for row in observed)
    model.zero_grad(set_to_none=True)
    result["auxiliary"].backward()
    router_grads = [float(block.ffn.router.weight.grad.norm()) for block in model.language.blocks]
    assert all(value > 0 for value in router_grads)
    pad_records.append({"added_pad_columns": added, "grid_shape": list(current_ids.shape), "valid_logits_max_difference": delta,
                        "auxiliary_difference": aux_delta, "layers": observed, "router_auxiliary_gradient_norms": router_grads})
save("pad_balance_audit.json", {"environment": ENV, "records": pad_records,
                               "formula": "N*sum_i f_i*P_i; f_i=count_i/(valid_tokens*top_k), P_i=mean full router softmax; coefficient .01 outside FFN; Switch original Eq5 is top-1, this code adapts to top-2 assignments."})

gpu = json.loads((ROOT / "docs/course-experiments/capstone-evidence/deployment/mechanism-benchmark.json").read_text())
deployment = json.loads((ROOT / "docs/course-experiments/results/capstone_deployment.json").read_text())
assert gpu == deployment["results"]["mechanism_benchmark"]
gpu_checks = {}
for name in ["moe", "dense80"]:
    report = gpu[name]
    assert report["batch_ids"] == denominators["sample_ids"]
    assert len(report["seconds"]) == report["measured_iterations"] == 10
    assert statistics.median(report["seconds"]) == report["median_seconds"]
    assert statistics.mean(report["seconds"]) == report["mean_seconds"]
    assert report["cuda_peak_allocated_bytes"] - report["cuda_baseline_allocated_bytes"] == report["cuda_peak_extra_bytes"]
    assert report["optimizer_updates"] == 0 and report["weights_unchanged"]
    gpu_checks[name] = {"median_ms": statistics.median(report["seconds"]) * 1000,
                        "extra_bytes": report["cuda_peak_allocated_bytes"] - report["cuda_baseline_allocated_bytes"],
                        "denominators": denominators, "raw_report": report}
save("gpu_report_audit.json", {"source_sha256": sha(ROOT / "docs/course-experiments/capstone-evidence/deployment/mechanism-benchmark.json"),
                              "checks": gpu_checks, "original_environment": {k: deployment[k] for k in ["gpu", "device", "python_version", "torch_version", "seed", "revision"]},
                              "not_retimed_on_gpu": True, "memory_scope": "CUDA allocated peak minus baseline; baseline includes live tensors and cloned state snapshots; peak includes warmup and measured loops; excludes reserved/driver/external allocations.",
                              "time_scope": "Forward, masked CE+.01 balance, backward and synchronized wait; zero_grad before timer; no optimizer, clipping, data preparation, model construction or upload."})

cpu = {"environment": ENV, "denominators": denominators,
       "moe": benchmark_training_step(model, splits["train"][:24]),
       "dense80": benchmark_training_step(models["dense80"], splits["train"][:24]),
       "limitation": "CPU replay demonstrates operations and weight invariance, and does not reproduce or substantiate the numerical L4 latency/memory peaks."}
save("cpu_mechanism_benchmark.json", cpu)

per_row = {}
for split, rel in [("validation", "docs/course-experiments/capstone-evidence/joint/validation.json"),
                   ("test", "docs/course-experiments/capstone-evidence/deployment/test-joint.json")]:
    historical = json.loads((ROOT / rel).read_text())
    assert {r["id"] for r in historical["records"]} == {r["id"] for r in splits[split]}
    historical_summary = summary(historical)
    fresh = evaluate_rows(model, splits[split])
    fresh_summary = summary(fresh)
    save("cpu_" + split + ".json", fresh)
    saved = {r["id"]: r for r in historical["records"]}
    differences = []
    for row in fresh["records"]:
        old = saved[row["id"]]
        changed = [key for key in ["action_correct", "end_to_end_correct", "answer"] if row[key] != old[key]]
        for key in ["action_trace", "final_trace"]:
            left, right = row[key], old[key]
            if (None if left is None else left["generated_ids"]) != (None if right is None else right["generated_ids"]):
                changed.append(key + ".generated_ids")
        if changed:
            differences.append({"id": row["id"], "changed_fields": changed})
    per_row[split] = {"historical_path": rel, "historical_sha256": sha(ROOT / rel),
                      "historical_recomputed_summary": historical_summary, "fresh_cpu_summary": fresh_summary,
                      "raw_per_row_differences": differences, "cpu_raw_path": str((OUT / (PREFIX + "cpu_" + split + ".json")).relative_to(ROOT))}
save("per_row_audit.json", {"environment": ENV, "result": per_row, "interpretation": "Coverage replay only; no claim that CPU or these narrow templates prove a general quality advantage over Dense."})

moe_result = json.loads((ROOT / "docs/course-experiments/results/moe.json").read_text())
variants = moe_result["results"]["variants"]
old_training = {}
for name in ["dense_active_top2", "top2_aux0.01"]:
    row = variants[name]
    old_training[name] = {"model": row["model"], "budget": row["budget"],
                          "training": row["training"], "heldout": {k: {j: v for j, v in value.items() if j != "samples"} for k, value in row["heldout"].items()}}
    assert row["training"]["effective_tokens"] == 337761
    assert row["training"]["steps"] == row["training"]["optimizer_updates"] == 180
ratio = variants["top2_aux0.01"]["training"]["warm_step_median_seconds"] / variants["dense_active_top2"]["training"]["warm_step_median_seconds"]
assert round(ratio, 2) == 2.28
save("moe_training_audit.json", {"environment": {k: moe_result[k] for k in ["gpu", "device", "python_version", "torch_version", "seed", "revision"]},
                                "branches": old_training, "ratio": ratio, "historical_report_sha256": sha(ROOT / "docs/course-experiments/results/moe.json"),
                                "source_version_snapshot": "docs/technical-reviews/artifacts/fact_v2_19_02_moe_run_architecture.py.txt",
                                "time_scope": "Synchronized forward/loss/backward, finite-gradient checks, clipping, scaler/AdamW update; latencies[3:] median =177 measurements after3 steps. Different experiment/config from capstone mechanism benchmark."})

result_inventory = []
for p in sorted((ROOT / "docs/course-experiments/results").glob("capstone_*.json")):
    d = json.loads(p.read_text())
    r = d["results"]
    result_inventory.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p),
                             "environment": {k: d.get(k) for k in ["device", "gpu", "python_version", "torch_version", "seed", "revision", "timing_scope"]},
                             "stage": {k: r[k] for k in ["stage", "steps", "requested_steps", "effective_tokens", "parameters", "objective", "parent_checkpoint_sha256", "validation_summary"] if k in r},
                             "current_code_sha_checks": {key: {"reported": value, "current": sha(ROOT / key), "equal": value == sha(ROOT / key)} for key, value in d["code_sha256"].items() if key in ["tiny_perceptron/capstone.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/attention.py", "scripts/course_experiments/capstone.py", "scripts/course_experiments/capstone_deployment.py"]}})
selection = json.loads((ROOT / "docs/course-experiments/capstone-selection.json").read_text())
assert selection["selected_stage"] == "joint"
assert selection["candidates"]["joint"]["checkpoint_sha256"] == gpu["checkpoint_sha256"]
save("experiment_inventory.json", {"results": result_inventory, "selection": selection,
                                   "selection_sha256": sha(ROOT / "docs/course-experiments/capstone-selection.json"),
                                   "reviewed_at": datetime.now(UTC).isoformat()})
save("audit_completion.json", {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_02_audit.py",
                               "environment": ENV, "section_sha256": sha(section_file), "status": "completed",
                               "script_sha256": sha(Path(__file__)), "completed_at": datetime.now(UTC).isoformat()})
print(json.dumps({"numeric": {k: v["description"] for k, v in counts.items() if "description" in v},
                  "pad_records": pad_records, "gpu_summary": {k: {j: v[j] for j in ["median_ms", "extra_bytes"]} for k, v in gpu_checks.items()},
                  "cpu_median_ms": {k: cpu[k]["median_seconds"] * 1000 for k in ["moe", "dense80"]},
                  "per_row": per_row, "training_ratio": ratio, "public_error": weights.get("public_fetch_error")}, ensure_ascii=False, indent=2))
