"""Fresh bounded CPU mechanisms and independent audit of original GPU records.

No training, GPU use, original checkpoint loading, installation or data download.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import hashlib
import inspect
import json
import math
import platform
import random
import re
import sys

import torch
from torch import nn

from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.capstone import CapstoneModel, build_dataset, default_config, save_capstone
from tiny_perceptron.capstone_quantization import load_quantized_capstone, quantizable_weights, quantize_capstone, tensor_bytes
from tiny_perceptron.modern import RMSNorm
from tiny_perceptron.quantization import QuantizedLinear, pack_int4, unpack_int4

torch.set_num_threads(2)
ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts/natural-v4-factual/19.10"
RESEARCH = ROOT / "outputs/natural-v4/factual-research/19.10"
RESEARCH.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

summary = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
            "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
            "threads": str(torch.get_num_threads())},
           "scope": "Own CPU mechanisms and independent recomputation of existing original GPU raw records; no own GPU training or inference-score replication."}
assert torch.__version__ == "2.14.1+cpu"
assert not torch.cuda.is_available()

# Execute the exact lesson block, then its only requested edit (bits=8).
section = OUT.joinpath("section.raw.md").read_text()
block = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
probes = []
for bits in (4, 8):
    namespace = {}
    exec(compile(block.replace("bits=4", f"bits={bits}"), "course/chapters/19.md#19.10", "exec"), namespace)
    layer, compressed, x, error = [namespace[k] for k in ("layer", "compressed", "x", "error")]
    counts = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                     "bytes": value.numel() * value.element_size()} for name, value in compressed.named_buffers()}
    with torch.no_grad():
        original, packed = layer(x), compressed(x)
    assert tuple(original.shape) == tuple(packed.shape) == (1, 4)
    assert not original.requires_grad and all(p.grad is None for p in layer.parameters())
    assert sum(p.numel() * p.element_size() for p in layer.parameters()) == 144
    assert compressed.storage_bytes() == (48 if bits == 4 else 64)
    assert counts["values"]["bytes"] == (16 if bits == 4 else 32)
    assert counts["scale"]["bytes"] == counts["bias"]["bytes"] == 16
    probes.append({"bits": bits, "fp32_bytes": 144, "storage_bytes": compressed.storage_bytes(),
                   "buffers": counts, "output_shape": list(original.shape), "max_output_error": error,
                   "original_output": original.tolist(), "quantized_output": packed.tolist(),
                   "no_parameter_gradients": True})
assert probes[1]["max_output_error"] < probes[0]["max_output_error"]
summary["lesson_block_and_exercise"] = probes
q = torch.tensor([-8, -1, 0, 1, 7, 2, 3], dtype=torch.int8)
assert pack_int4(q).tolist() == [112, 152, 175, 139]
assert torch.equal(unpack_int4(pack_int4(q), q.shape), q)
summary["packing_prerequisite"] = {"integers": q.tolist(), "bytes": pack_int4(q).tolist(), "lossless_integer_roundtrip": True}

splits, manifest = build_dataset(seed=42)
stored_data = load("docs/course-experiments/capstone-evidence/deployment/data.json")
assert splits == stored_data["splits"] and manifest == stored_data["manifest"]
summary["data"] = {"version": manifest["version"], "seed": manifest["seed"], "counts": manifest["counts"],
                   "split_sha256": manifest["sha256"], "family_intersections": {
                       a + "/" + b: len({r["family"] for r in splits[a]} & {r["family"] for r in splits[b]})
                       for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
                   }, "current_rules_equal_original_recorded_data": True}

# Initialize actual models only; serialize/reload our random models in ignored research.
architectures = []
for name, cfg, bit_options in [("moe", default_config(), (4, 8)),
                                ("student", replace(default_config(dense=True), width=48), (4,))]:
    torch.manual_seed(42)
    model = CapstoneModel(cfg)
    state = model.state_dict()
    init_sha = hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in state.values())).hexdigest()
    source = RESEARCH / (name + "-random.pt")
    save_capstone(source, model, stage="joint", step=0, metadata={})
    names = quantizable_weights(model)
    item = {"name": name, "parameters": sum(p.numel() for p in model.parameters()),
            "config": model.description()["config"], "float_tensor_bytes": tensor_bytes(state),
            "initial_state_sha256_cpu": init_sha, "quantizable_weights": names, "ptq": []}
    for bits in bit_options:
        destination = RESEARCH / f"{name}-random-int{bits}.pt"
        receipt = quantize_capstone(source, destination, bits)
        restored, payload = load_quantized_capstone(destination)
        independent_packed = sum((v["values"].numel() * v["values"].element_size()
                                 + v["scale"].numel() * v["scale"].element_size())
                                for v in payload["quantized"].values())
        independent_float = sum(t.numel() * t.element_size() for t in payload["model"].values())
        assert independent_packed + independent_float == receipt["tensor_bytes"]
        assert all(p.dtype == torch.float32 for p in restored.parameters())
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state.items() if k not in names)
        assert all(not k.endswith("router.weight") for k in names)
        item["ptq"].append({"bits": bits, "tensor_bytes": receipt["tensor_bytes"],
                            "quantized_values_bytes": sum(v["values"].numel() * v["values"].element_size() for v in payload["quantized"].values()),
                            "scale_bytes": sum(v["scale"].numel() * v["scale"].element_size() for v in payload["quantized"].values()),
                            "unquantized_float_bytes": independent_float,
                            "all_restored_parameters_float32": True, "unquantized_values_exact": True,
                            "current_random_container_bytes_not_original_run_bytes": destination.stat().st_size})
    architectures.append(item)
assert architectures[0]["parameters"] == 328128 and architectures[0]["float_tensor_bytes"] == 1312512
assert [r["tensor_bytes"] for r in architectures[0]["ptq"]] == [248864, 402720]
assert architectures[1]["parameters"] == 79920 and architectures[1]["float_tensor_bytes"] == 319680
assert architectures[1]["ptq"][0]["tensor_bytes"] == 91680
summary["architecture_and_ptq_mechanisms"] = architectures

# Independently compute the KD objective with Python math, checking masking and T² once.
s_values = [[[0.4, -0.3, 0.7], [1.1, -0.2, 0.3], [-0.9, 0.8, 0.2]]]
t_values = [[[0.3, 1.2, -0.4], [0.5, -0.7, 1.4], [0.1, 0.2, 0.9]]]
labels = torch.tensor([[-100, 2, 1]])
student = torch.tensor(s_values, requires_grad=True)
teacher = torch.tensor(t_values, requires_grad=True)
def probabilities(values, temperature):
    exponentials = [math.exp(v / temperature) for v in values]
    return [v / sum(exponentials) for v in exponentials]
ce_terms, kl_terms = [], []
for i, target in enumerate(labels[0].tolist()):
    if target == -100:
        continue
    p, q = probabilities(t_values[0][i], 2), probabilities(s_values[0][i], 2)
    ce_terms.append(-math.log(probabilities(s_values[0][i], 1)[target]))
    kl_terms.append(sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q)))
ce = sum(ce_terms) / 2
raw_kl = sum(kl_terms) / 2
expected = 0.5 * ce + 0.5 * 4 * raw_kl
actual = distillation_loss(student, teacher, labels, alpha=0.5, temperature=2)
assert abs(actual.item() - expected) < 1e-6
actual.backward()
assert teacher.grad is None and student.grad[0, 0].abs().max().item() == 0
summary["distillation_formula"] = {"temperature": 2, "alpha": 0.5, "valid_answer_positions": 2,
                                    "ce": ce, "raw_forward_kl": raw_kl, "T_squared": 4,
                                    "independent_expected": expected, "helper_observed": actual.item(),
                                    "tolerance": 1e-6, "teacher_gradient": str(teacher.grad),
                                    "ignored_position_gradient": student.grad[0, 0].tolist()}

# Show why the lesson's center-and-scale claim conflicts with actual RMSNorm.
probe_x = torch.tensor([[1.0, 3.0]])
rms_output = RMSNorm(2)(probe_x)
ln_output = nn.LayerNorm(2)(probe_x)
assert rms_output.mean().item() > 0.8 and abs(ln_output.mean().item()) < 1e-6
summary["normalization_counterexample"] = {"capstone_norm": default_config().norm, "input": probe_x.tolist(),
                                           "rms_output": rms_output.tolist(), "rms_mean": rms_output.mean().item(),
                                           "layernorm_output": ln_output.tolist(), "layernorm_mean": ln_output.mean().item(),
                                           "finding": "Actual capstone RMSNorm divides by RMS and does not subtract/recenter by the feature mean."}

deployment = load("docs/course-experiments/results/capstone_deployment.json")
student_report = load("docs/course-experiments/results/capstone_student.json")
joint = load("docs/course-experiments/results/capstone_joint.json")
dpo = load("docs/course-experiments/results/capstone_preference.json")
source_paths = ["tiny_perceptron/quantization.py", "tiny_perceptron/capstone_quantization.py",
                "tiny_perceptron/alignment.py", "tiny_perceptron/capstone.py", "tiny_perceptron/model.py",
                "tiny_perceptron/modern.py", "scripts/course_experiments/capstone_student.py",
                "scripts/course_experiments/capstone_deployment.py", "scripts/course_experiments/capstone.py"]
hash_records = {}
for path in source_paths:
    current = sha(path)
    originals = {name: report["code_sha256"].get(path) for name, report in [("deployment", deployment), ("student", student_report)]}
    assert all(digest == current for digest in originals.values() if digest is not None)
    hash_records[path] = {"current_fullfile_sha256": current, "original_run_code_sha256": originals}
summary["inspected_code_current_equals_original_run"] = hash_records
for report in [deployment, student_report, joint, dpo]:
    assert report["seed"] == 42 and report["step_scale"] == 1.0 and report["gpu"] == "NVIDIA L4"
    assert report["results"]["data_manifest"] == manifest
assert dpo["results"]["parent_checkpoint_sha256"] == joint["results"]["inference_export"]["sha256"]
assert student_report["results"]["teacher_checkpoint_sha256"] == dpo["results"]["inference_export"]["sha256"]
assert student_report["results"]["ptq"]["source_checkpoint_sha256"] == student_report["results"]["final_model"]["sha256"]
assert architectures[1]["initial_state_sha256_cpu"] == student_report["results"]["initial_student_state_sha256"]
summary["original_run_provenance"] = {"deployment": {k: deployment[k] for k in ["revision", "seed", "gpu", "device", "torch_version", "python_version", "step_scale", "timing_scope"]},
                                       "student": {k: student_report[k] for k in ["revision", "seed", "gpu", "device", "torch_version", "python_version", "step_scale", "timing_scope"]},
                                       "joint_teacher_sha": joint["results"]["inference_export"]["sha256"],
                                       "DPO_parent_matches_joint": True, "student_teacher_sha": student_report["results"]["teacher_checkpoint_sha256"],
                                       "student_teacher_is_DPO": True, "CPU_initial_state_matches_recorded_GPU_student_initial_state": True}

# Reconstruct the prescribed sampler sequence, without executing model updates.
grouped = defaultdict(list)
for row in splits["train"]:
    grouped[row["task"]].append(row)
tasks = sorted(grouped)
sampler_results = []
for branch in ("ce", "kd"):
    rng = random.Random(42 + 5000)
    sampled = [rng.choice(grouped[rng.choice(tasks)]) for _ in range(350 * 24)]
    count = sum(len(row["answer"].encode("utf-8")) + 1 for row in sampled)
    report = student_report["results"]["branches"][branch]
    assert count == report["effective_tokens"] == 145163
    assert report["steps"] == report["requested_steps"] == 350 and report["schedule_completed"]
    sequence_sha = hashlib.sha256("\n".join(row["id"] for row in sampled).encode()).hexdigest()
    sampler_results.append({"branch": branch, "steps": 350, "batch_size": 24, "sampled_rows": len(sampled),
                            "effective_answer_positions_recomputed": count, "prescribed_sequence_sha256": sequence_sha,
                            "original_training_seconds": report["seconds"], "recorded_objective": report["objective"]})
assert sampler_results[0]["prescribed_sequence_sha256"] == sampler_results[1]["prescribed_sequence_sha256"]
summary["student_budget_and_sampler"] = {"branches": sampler_results,
                                         "limit": "Prescribed sequence reconstructed from exact original runner code and seed; original GPU per-batch row IDs were not saved, so this is not a batch-log replay audit."}

# Independent byte decoding, action parsing and scoring from all recorded generated IDs.
def decode_trace(trace):
    ids = trace["generated_ids"]
    control = [i for i, value in enumerate(ids) if value < 8]
    ordinary = ids[:control[0]] if control else ids
    raw = bytes(value - 8 for value in ordinary).decode("utf-8", errors="replace")
    eos = bool(control and ids[control[0]] == 2)
    assert raw == trace["raw"] and eos == trace["eos"]
    assert all(value >= 8 for value in ordinary)
    if control:
        assert control == [len(ids) - 1]
    return raw, eos

def parse(raw, eos):
    if not eos:
        return None
    if raw.startswith("DIRECT:") and len(raw) > 7:
        return ("direct", raw[7:])
    if raw.startswith("ASK:") and len(raw) > 4:
        return ("ask", raw[4:])
    match = re.fullmatch(r"TOOL:([a-z_]+):([0-9]{1,3})\+([0-9]{1,3})", raw)
    return ("tool", match[1], int(match[2]), int(match[3])) if match else None

def audit_evaluation(path, split):
    evaluation = load(path)
    records = evaluation["records"]
    assert [r["id"] for r in records] == [r["id"] for r in splits[split]]
    by_task = defaultdict(lambda: {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
    for record, row in zip(records, splits[split], strict=True):
        raw, eos = decode_trace(record["action_trace"])
        action = parse(raw, eos)
        expected = row["answer"]
        expected_final = str(sum(int(v) for v in expected.split(":")[-1].split("+"))) if expected.startswith("TOOL:") else expected.split(":", 1)[1]
        assert record["expected_action"] == expected and record["expected_final"] == expected_final
        answer = None
        if action and action[0] in ("direct", "ask"):
            answer = action[1]
        if action and action[0] == "tool":
            runtime = record["runtime"]
            if row["available"] and action[1] == "calculator":
                assert runtime == {"status": "ok", "result": str(action[2] + action[3])}
                final_raw, final_eos = decode_trace(record["final_trace"])
                final_action = parse(final_raw, final_eos)
                if final_action and final_action[0] == "direct":
                    answer = final_action[1]
        action_correct = eos and raw == expected
        correct = action_correct and answer == expected_final
        assert record["answer"] == answer
        assert record["action_correct"] == action_correct and record["end_to_end_correct"] == correct
        by_task[row["task"]]["count"] += 1
        by_task[row["task"]]["action_correct"] += int(action_correct)
        by_task[row["task"]]["end_to_end_correct"] += int(correct)
    totals = {"count": len(records), "action_correct": sum(v["action_correct"] for v in by_task.values()),
              "end_to_end_correct": sum(v["end_to_end_correct"] for v in by_task.values()), "by_task": dict(by_task)}
    assert all(totals[k] == evaluation[k] for k in totals)
    return evaluation, totals

record_paths = {
    "joint": ("deployment/test-joint.json", "test"), "joint4": ("deployment/test-joint-ptq4.json", "test"),
    "joint8": ("deployment/test-joint-ptq8.json", "test"), "dpo": ("deployment/test-dpo.json", "test"),
    "dpo4": ("deployment/test-ptq4.json", "test"), "dpo8": ("deployment/test-ptq8.json", "test"),
    "ce": ("student/test-ce.json", "test"), "kd": ("student/test-kd.json", "test"),
    "kd4": ("student/test-kd-ptq4.json", "test"), "validation_ce": ("student/validation-ce.json", "validation"),
    "validation_kd": ("student/validation-kd.json", "validation")}
evaluations, totals = {}, {}
for identity, (tail, split) in record_paths.items():
    path = "docs/course-experiments/capstone-evidence/" + tail
    evaluations[identity], totals[identity] = audit_evaluation(path, split)
    wrapper = student_report if tail.startswith("student/") else deployment
    receipt = next(item for item in wrapper["artifacts"] if item["path"] == tail.split("/", 1)[1])
    assert sha(path) == receipt["sha256"] and Path(path).stat().st_size == receipt["bytes"]
    totals[identity].update(path=path, sha256=sha(path), matches_original_artifact_receipt=True)
summary["record_scores_independently_recomputed"] = totals
assert [totals[k]["end_to_end_correct"] for k in ["joint", "joint4", "joint8", "dpo", "dpo4", "dpo8"]] == [78] * 6
assert [totals[k]["end_to_end_correct"] for k in ["ce", "kd", "kd4", "validation_ce", "validation_kd"]] == [62, 61, 61, 59, 59]

questions = {r["id"]: r["user"] for r in splits["test"]}
differences = {}
for base, compressed, expected_count in [("joint", "joint4", 1), ("joint", "joint8", 0), ("dpo", "dpo4", 2), ("dpo", "dpo8", 0), ("kd", "kd4", 6)]:
    changed = []
    for a, b in zip(evaluations[base]["records"], evaluations[compressed]["records"], strict=True):
        assert a["id"] == b["id"]
        changes = [key for key in ("action_trace", "final_trace") if
                   (a[key] or {}).get("generated_ids") != (b[key] or {}).get("generated_ids")]
        if changes:
            changed.append({"id": a["id"], "question": questions[a["id"]], "task": a["task"], "traces_changed": changes,
                            "base_action": a["action_trace"]["raw"], "compressed_action": b["action_trace"]["raw"],
                            "base_final": (a["final_trace"] or {}).get("raw"), "compressed_final": (b["final_trace"] or {}).get("raw"),
                            "base_answer": a["answer"], "compressed_answer": b["answer"],
                            "base_correct": a["end_to_end_correct"], "compressed_correct": b["end_to_end_correct"]})
    assert len(changed) == expected_count
    assert all(not r["base_correct"] and not r["compressed_correct"] for r in changed)
    assert totals[base]["by_task"] == totals[compressed]["by_task"]
    differences[base + "/" + compressed] = {"rows_compared": 90, "changed_id_sequences": len(changed), "changed_records": changed}
summary["ptq_generation_id_differences"] = differences

ce_kd_score_differences = []
for a, b in zip(evaluations["ce"]["records"], evaluations["kd"]["records"], strict=True):
    if a["end_to_end_correct"] != b["end_to_end_correct"]:
        ce_kd_score_differences.append({"id": a["id"], "question": questions[a["id"]], "ce_request": a["action_trace"]["raw"],
                                      "ce_runtime": a["runtime"], "ce_final": (a["final_trace"] or {}).get("raw"),
                                      "kd_request": b["action_trace"]["raw"], "kd_runtime": b["runtime"],
                                      "kd_final": (b["final_trace"] or {}).get("raw")})
assert len(ce_kd_score_differences) == 1 and ce_kd_score_differences[0]["question"] == "1+8等於多少？"
summary["ce_kd_one_score_difference"] = ce_kd_score_differences
exports = deployment["results"]["public_stage_exports"]
summary["original_container_and_payload_bytes"] = {
    "joint": {"file": exports["joint"]["bytes"], "tensor": exports["joint"]["tensor_bytes"]},
    "joint4": {"file": exports["joint-int4"]["file_bytes"], "tensor": exports["joint-int4"]["tensor_bytes"]},
    "joint8": {"file": exports["joint-int8"]["file_bytes"], "tensor": exports["joint-int8"]["tensor_bytes"]},
    "ce": {"file": student_report["results"]["branches"]["ce"]["export"]["bytes"], "tensor": 79920 * 4},
    "kd": {"file": student_report["results"]["branches"]["kd"]["export"]["bytes"], "tensor": 79920 * 4},
    "kd4": {"file": student_report["results"]["ptq"]["file_bytes"], "tensor": student_report["results"]["ptq"]["tensor_bytes"]}}
assert summary["original_container_and_payload_bytes"] == {
    "joint": {"file": 1345023, "tensor": 1312512}, "joint4": {"file": 275381, "tensor": 248864},
    "joint8": {"file": 429365, "tensor": 402720}, "ce": {"file": 342451, "tensor": 319680},
    "kd": {"file": 342451, "tensor": 319680}, "kd4": {"file": 106229, "tensor": 91680}}
summary["original_file_byte_limit"] = "Original run report and artifact-container receipts audited; original run binary weights not downloaded or loaded. Current random-model save sizes are deliberately not treated as original file bytes."
binary_receipts = []
for wrapper, tail, identity, export in [
    (deployment, "joint.pt", "joint", exports["joint"]),
    (deployment, "joint-int4.pt", "joint4", exports["joint-int4"]),
    (deployment, "joint-int8.pt", "joint8", exports["joint-int8"]),
    (student_report, "ce/model.pt", "ce", student_report["results"]["branches"]["ce"]["export"]),
    (student_report, "kd/model.pt", "kd", student_report["results"]["branches"]["kd"]["export"]),
    (student_report, "model-int4.pt", "kd4", student_report["results"]["ptq"]),
]:
    receipt = next(item for item in wrapper["artifacts"] if item["path"] == tail)
    assert receipt["bytes"] == summary["original_container_and_payload_bytes"][identity]["file"]
    assert receipt["sha256"] == export["sha256"]
    binary_receipts.append({"identity": identity, **receipt, "export_metadata_matches_artifact_receipt": True})
summary["original_binary_container_receipts_compared"] = binary_receipts

record_sources = ["docs/course-experiments/results/" + name + ".json" for name in
                  ["capstone_deployment", "capstone_student", "capstone_joint", "capstone_preference"]]
record_sources += ["docs/course-experiments/capstone-evidence/deployment/data.json"]
record_sources += [value["path"] for value in totals.values()]
record_sources += ["docs/course-experiments/capstone-evidence/student/" + branch + "/train-report.json" for branch in ("ce", "kd")]
summary["inspected_original_record_fullfile_shas"] = {path: sha(path) for path in record_sources}
summary["status"] = "All bounded CPU assertions and original raw-record recomputations completed; normalization description requires correction."
OUT.joinpath("audit-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
