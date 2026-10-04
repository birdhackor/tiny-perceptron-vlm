"""Persist final source identities, field-level limits, and arithmetic checks."""

import ast
import hashlib
import json
import platform
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from tiny_perceptron.capstone import CapstoneModel, default_config
from tiny_perceptron.training import seed_everything

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    checks = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}}
    source_identity = []
    for local, name in [
        ("docs/course-experiments/capstone-selection.json", "linked-capstone-selection.json"),
        ("docs/course-experiments/results/capstone_deployment.json", "linked-capstone_deployment.json"),
        ("docs/course-experiments/results/capstone_student.json", "linked-capstone_student.json"),
        ("docs/course-experiments/capstone-evidence/deployment/test-joint.json", "linked-test-joint.json"),
        ("docs/course-experiments/capstone-evidence/deployment/data.json", "linked-data.json"),
    ]:
        assert sha(ROOT / local) == sha(OUT / name)
        source_identity.append(
            {
                "source": local,
                "snapshot": str((OUT / name).relative_to(ROOT)),
                "sha256": sha(ROOT / local),
                "exact_bytes_equal": True,
            }
        )
    for local, name in [
        (Path(torch.__file__).parent / "serialization.py", "torch-serialization.py.txt"),
        (Path(torch.__file__).parent / "cuda/__init__.py", "torch-cuda-init.py.txt"),
    ]:
        assert sha(local) == sha(OUT / name)
        source_identity.append(
            {
                "installed_source": str(local),
                "official_snapshot": str((OUT / name).relative_to(ROOT)),
                "sha256": sha(local),
                "exact_bytes_equal": True,
            }
        )
    checks["source_identity"] = source_identity
    captured = read(OUT / "read-manifest.json")["sections"][0]
    original = (ROOT / "course/chapters/19.md").read_bytes()
    start = original.index(b"## 19.12 ")
    end = original.find(b"\n## ", start + 1)
    current = original[start:] if end < 0 else original[start : end + 1]
    assert current == (ROOT / captured["path"]).read_bytes()
    checks["section"] = {**captured, "current_matches_actually_read_snapshot": True}
    audit = read(OUT / "row-audit.json")
    baseline = read(OUT / "raw/deployment/test-joint.json")["records"]
    nonsafety = [r for r in baseline if r["task"] != "safety"]
    assert len(nonsafety) == 87
    assert not any(
        "不能提供他人密碼" in trace["raw"]
        for r in nonsafety
        for trace in (r["action_trace"], r["final_trace"])
        if trace is not None
    )
    image_rows = [r for r in baseline if r["task"] == "image_shape"]
    assert len(image_rows) == 9
    assert all(
        r["expected_action"] == "DIRECT:square"
        and r["action_trace"]["raw"] == "DIRECT:circle"
        and r["action_trace"]["eos"]
        for r in image_rows
    )
    data = read(OUT / "raw/deployment/data.json")["splits"]
    color = Counter(r["image"]["color"] for r in data["test"] if r["task"] == "image_color")
    pitch = Counter(r["audio"]["pitch"] for r in data["test"] if r["task"] == "audio")
    rag = Counter(r["answer"] for r in data["test"] if r["task"] == "rag")
    assert color == {"green": 9} and pitch == {"low": 3, "high": 3}
    assert rag == {"DIRECT:桌子": 1, "DIRECT:書櫃": 1, "DIRECT:抽屜": 1}
    checks["limits"] = {
        "nonrefusal_substring_count_across_all_first_final_traces": 0,
        "normal_count": 87,
        "shape_wrong_circle_with_eos_count": 9,
        "color_label_counts": dict(color),
        "pitch_label_counts": dict(pitch),
        "context_answer_counts": dict(rag),
        "safety_missing_concept_are_constant_targets": all(
            len({r["answer"] for r in data["test"] if r["task"] == task}) == 1
            for task in ("safety", "missing", "concept")
        ),
        "student_runner_does_not_run_swaps": "image_counterfactuals"
        not in (ROOT / "scripts/course_experiments/capstone_student.py").read_text(),
    }
    for name, diffs in audit["quantization_changes"].items():
        assert all(not d["before_correct"] and not d["after_correct"] for d in diffs), name
    checks["changed_quantized_generations_all_wrong_before_and_after"] = True
    changed_success = [r for r in audit["student_ce_vs_kd"] if r["before_correct"] != r["after_correct"]]
    assert len(changed_success) == 1 and changed_success[0]["id"] == "ffe914bb5e9abc53f39f"
    checks["student_only_success_change"] = changed_success[0]
    checks["median_derivation"] = {}
    for mode, result in audit["generation_benchmark"]["recomputed"].items():
        ordered = sorted(result["seconds"])
        exact_ms = (ordered[4] + ordered[5]) / 2 * 1000
        assert exact_ms == result["median_ms"]
        checks["median_derivation"][mode] = {
            "sorted_seconds": ordered,
            "fifth_seconds": ordered[4],
            "sixth_seconds": ordered[5],
            "mean_middle_ms": exact_ms,
            "rounded_three_decimals_ms": round(exact_ms, 3),
        }
    cpu = read(OUT / "cpu-audit.json")
    teacher = cpu["models"]["dpo"]["description"]["parameters"]
    assert teacher == 328128
    for n in ("student-ce", "student-kd", "student-kd-int4"):
        assert cpu["models"][n]["description"]["parameters"] == 79920
    checks["parameter_counts"] = {
        "teacher_total": teacher,
        "student_total": 79920,
        "logical_active_is_separate_proxy_not_runtime_measurement": True,
        "student_inference_speed_and_generation_memory_not_measured": True,
    }
    for name in ("capstone_deployment", "capstone_student"):
        summary = read(OUT / "raw" / (name + ".json"))
        raw_report = read(
            OUT
            / "raw"
            / ("deployment/deployment-report.json" if name.endswith("deployment") else "student/student-report.json")
        )
        assert summary["results"] == raw_report
    checks["top_level_results_equal_complete_raw_reports"] = True
    # Compare the deployed inference/data functions with their actual experiment revision.
    revision = read(OUT / "raw/capstone_deployment.json")["revision"]
    url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{revision}/tiny_perceptron/capstone.py"
    with urllib.request.urlopen(url, timeout=30) as response:
        historical_bytes = response.read()
        http_status = response.status
    snapshot = OUT / "experiment-capstone.py.txt"
    snapshot.write_bytes(historical_bytes)
    original_tree = ast.parse(historical_bytes.decode())
    current_tree = ast.parse((ROOT / "tiny_perceptron/capstone.py").read_text())
    originals = {
        node.name: ast.dump(node, include_attributes=False)
        for node in original_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    currents = {
        node.name: ast.dump(node, include_attributes=False)
        for node in current_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    relevant = [
        "build_dataset",
        "modality_tensors",
        "prompt_ids",
        "encode_record",
        "generate_trace",
        "generate_traces",
        "evaluate_rows",
        "parse_action",
        "calculator_runtime",
        "run_assistant",
        "expected_final",
    ]
    assert all(originals[n] == currents[n] for n in relevant)
    checks["experiment_function_identity"] = {
        "url": url,
        "http_status": http_status,
        "revision": revision,
        "sha256": sha(snapshot),
        "functions_ast_identical": relevant,
        "full_current_file_sha256": sha(ROOT / "tiny_perceptron/capstone.py"),
    }
    schedules = {}
    for stage, steps in [("pretrain", 300), ("sft", 1400), ("joint", 600), ("dpo", 100)]:
        record = read(OUT / "raw" / stage / "train-report.json")
        assert record["steps"] == record["requested_steps"] == steps
        assert record["schedule_completed"] and not record["test_evaluated"]
        assert record["seed"] == 42 and record["device"] == "cuda"
        schedules[stage] = {
            k: record[k]
            for k in [
                "steps",
                "requested_steps",
                "schedule_completed",
                "effective_tokens",
                "seed",
                "device",
                "test_evaluated",
            ]
        }
    student = read(OUT / "raw/student/student-report.json")
    seed_everything(42)
    initial = CapstoneModel(replace(default_config(dense=True), width=48))
    initial_sha = hashlib.sha256(b"".join(t.numpy().tobytes() for t in initial.state_dict().values())).hexdigest()
    assert initial_sha == student["initial_student_state_sha256"]
    assert student["teacher_checkpoint_sha256"] == cpu["models"]["dpo"]["original_file_sha256"]
    for mode in ("ce", "kd"):
        record = read(OUT / "raw/student" / mode / "train-report.json")
        assert record["steps"] == record["requested_steps"] == 350
        assert record["schedule_completed"] and record["effective_tokens"] == 145163
        schedules["student-" + mode] = {
            k: record[k]
            for k in ["steps", "effective_tokens", "teacher_checkpoint_sha256", "objective", "initialization"]
        }
    checks["frozen_schedules_and_sampling_scope"] = schedules
    checks["initial_student_state_cpu_reconstruction"] = {
        "seed": 42,
        "sha256": initial_sha,
        "matches_original": True,
        "gpu_batches_not_saved": "Original GPU batches were not saved individually; equal sequence is controlled by fixed code/RNG, not a claim of per-batch recorded verification.",
    }
    public = read(OUT / "raw/capstone-public.json")
    checkpoints = {}
    for identity, record in cpu["models"].items():
        if identity == "untrained":
            continue
        entry = next(m for m in public["models"] if m["id"] == identity)
        specification = next(f for f in entry["files"] if f["output"] == "model.pt")
        path = Path(
            hf_hub_download(
                public["repo"], specification["path"], revision=public["revision"], token=False, local_files_only=True
            )
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        original = torch.load(ROOT / record["original_path_local_only"], map_location="cpu", weights_only=True)
        headers = {
            k: payload[k]
            for k in ["format_version", "stage", "step", "config", "data_version", "tokenizer", "inference_only"]
        }
        assert all(headers[k] == original[k] for k in headers)
        tensor_hashes = {}
        for name, tensor in payload["model"].items():
            assert torch.equal(tensor, original["model"][name])
            tensor_hashes["model." + name] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": tensor.numel(),
                "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
            }
        for name, packed in payload.get("quantized", {}).items():
            for key in ("values", "scale"):
                tensor = packed[key]
                assert torch.equal(tensor, original["quantized"][name][key])
                tensor_hashes["quantized." + name + "." + key] = {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "numel": tensor.numel(),
                    "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
                }
        checkpoints[identity] = {
            "headers": headers,
            "serialized_tensors_all_match_original": True,
            "serialized_tensor_metadata": tensor_hashes,
        }
    checks["checkpoint_header_and_serialized_tensor_identities"] = checkpoints
    (OUT / "finish-checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(
        "source hashes, fixed links, original/current inference functions, refusal substrings, constant baselines, student failure and medians verified"
    )


if __name__ == "__main__":
    main()
