"""Independent row-level and CPU review of the frozen 19.12 claims.

Run from the repository with PYTHONPATH=. .venv/bin/python <this file>.
No training, model serialization, environment changes, or credential use.
"""

import copy
import hashlib
import json
import platform
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from scripts.course_experiments.capstone_deployment import cache_consistency, image_counterfactuals
from tiny_perceptron.capstone import CapstoneModel, build_dataset, evaluate_rows, load_capstone
from tiny_perceptron.capstone_quantization import load_quantized_capstone
from tiny_perceptron.training import seed_everything

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RAW = OUT / "raw"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def encode(text):
    return [value + 8 for value in text.encode("utf-8")]


def prefix(row):
    ids = [1, 7] + encode(row["system"]) + [2, 3]
    if row.get("image") is not None:
        ids.append(5)
    if row.get("audio") is not None:
        ids.append(6)
    return ids + encode(row["user"]) + [2, 4]


def inspect_trace(trace, row):
    ids = trace["generated_ids"]
    assert all(type(i) is int and 0 <= i < 264 for i in ids)
    assert trace["prompt_ids"] == prefix(row)
    assert not any(i < 8 for i in ids[:-1]), "generation continued after first stop"
    eos = bool(ids) and ids[-1] == 2
    assert trace["eos"] == eos
    assert (trace["stop_reason"] == "eos") == eos
    text = bytes(i - 8 for i in ids if i >= 8).decode("utf-8", errors="replace")
    assert trace["raw"] == text
    if not eos:
        parsed = {"status": "invalid", "reason": "unterminated_generation"}
    elif text.startswith("DIRECT:") and len(text) > 7:
        parsed = {"status": "direct", "content": text[7:]}
    elif text.startswith("ASK:") and len(text) > 4:
        parsed = {"status": "ask", "content": text[4:]}
    elif match := re.fullmatch(r"TOOL:([a-z_]+):([0-9]{1,3})\+([0-9]{1,3})", text):
        parsed = {"status": "tool", "name": match[1], "a": int(match[2]), "b": int(match[3])}
    else:
        parsed = {"status": "invalid", "reason": "malformed_action"}
    return text, eos, parsed


def audit_evaluation(path, rows):
    original = read(path)
    assert len(original["records"]) == len(rows)
    assert [r["id"] for r in original["records"]] == [r["id"] for r in rows]
    by_task, reconstructed = {}, []
    for row, record in zip(rows, original["records"], strict=True):
        assert record["expected_action"] == row["answer"]
        assert record["family"] == row["family"] and record["task"] == row["task"]
        raw, eos, parsed = inspect_trace(record["action_trace"], row)
        assert record["parsed_action"] == parsed
        answer, runtime = None, None
        final = row["answer"].split(":", 1)[1]
        if row["answer"].startswith("TOOL:"):
            a, b = map(int, row["answer"].rsplit(":", 1)[1].split("+"))
            final = str(a + b)
        if parsed["status"] in ("direct", "ask"):
            answer = parsed["content"]
        elif parsed["status"] == "tool":
            if not row["available"]:
                runtime = {"status": "error", "reason": "calculator_unavailable"}
            elif parsed["name"] != "calculator":
                runtime = {"status": "error", "reason": "tool_not_allowlisted"}
            else:
                runtime = {"status": "ok", "result": str(parsed["a"] + parsed["b"])}
                follow = dict(row, image=None, audio=None)
                follow["user"] = f"原題：{parsed['a']}+{parsed['b']}。計算器回報：{runtime['result']}。請回答。"
                _, _, final_parsed = inspect_trace(record["final_trace"], follow)
                if final_parsed["status"] == "direct":
                    answer = final_parsed["content"]
        if runtime is None or runtime["status"] != "ok":
            assert record["final_trace"] is None
        action_ok = eos and raw == row["answer"]
        end_ok = action_ok and answer == final
        assert record["runtime"] == runtime
        assert record["expected_final"] == final and record["answer"] == answer
        assert record["action_correct"] == action_ok and record["end_to_end_correct"] == end_ok
        task = by_task.setdefault(row["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        task["count"] += 1
        task["action_correct"] += int(action_ok)
        task["end_to_end_correct"] += int(end_ok)
        reconstructed.append(
            {
                "id": row["id"],
                "task": row["task"],
                "action": raw,
                "eos": eos,
                "runtime": runtime,
                "final_answer": answer,
                "truth": final,
                "action_correct": action_ok,
                "end_to_end_correct": end_ok,
            }
        )
    summary = {
        "count": len(rows),
        "by_task": by_task,
        "action_correct": sum(r["action_correct"] for r in reconstructed),
        "end_to_end_correct": sum(r["end_to_end_correct"] for r in reconstructed),
    }
    for key, value in summary.items():
        assert original[key] == value, (path, key)
    return {"source_sha256": sha(path), "summary": summary, "records": reconstructed}


def tensor_metadata(model):
    return {
        name: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": t.numel(),
            "sha256": hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        }
        for name, t in model.state_dict().items()
    }


def differences(left, right):
    keys = ["action_trace", "final_trace"]
    found = []
    for a, b in zip(left["records"], right["records"], strict=True):
        assert a["id"] == b["id"]
        changed = [k for k in keys if (a[k] or {}).get("generated_ids") != (b[k] or {}).get("generated_ids")]
        if changed:
            found.append(
                {
                    "id": a["id"],
                    "task": a["task"],
                    "changed_traces": changed,
                    "before": [None if a[k] is None else a[k]["raw"] for k in keys],
                    "after": [None if b[k] is None else b[k]["raw"] for k in keys],
                    "before_correct": a["end_to_end_correct"],
                    "after_correct": b["end_to_end_correct"],
                }
            )
    return found


def main():
    torch.set_num_threads(2)
    splits, manifest = build_dataset()
    data = read(RAW / "deployment/data.json")
    assert data["splits"] == splits and data["manifest"] == manifest
    expected_counts = {
        "calculator": 12,
        "unavailable": 12,
        "tool_return": 6,
        "concept": 6,
        "image_color": 9,
        "image_shape": 9,
        "joint": 18,
        "audio": 6,
        "rag": 3,
        "missing": 3,
        "safety": 3,
        "style": 3,
    }
    counts = dict(sorted(Counter(row["task"] for row in splits["test"]).items()))
    assert counts == expected_counts and sum(counts.values()) == 90
    families = {split: {r["family"] for r in rows} for split, rows in splits.items()}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not families[left].intersection(families[right])
    joint_payloads = Counter(r["answer"] for r in splits["test"] if r["task"] == "joint")
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "device": "cpu",
            "threads": 2,
        },
        "data": {
            "counts": counts,
            "manifest": manifest,
            "build_equals_saved_data": True,
            "family_split_disjoint": True,
            "joint_payloads": dict(joint_payloads),
        },
        "evaluations": {},
    }
    stages = {
        "untrained": "deployment/test-untrained.json",
        "pretrain": "deployment/test-pretrain.json",
        "sft": "deployment/test-sft.json",
        "joint": "deployment/test-joint.json",
        "dpo": "deployment/test-dpo.json",
        "joint-int4": "deployment/test-joint-ptq4.json",
        "joint-int8": "deployment/test-joint-ptq8.json",
        "dpo-int4": "deployment/test-ptq4.json",
        "dpo-int8": "deployment/test-ptq8.json",
        "student-ce": "student/test-ce.json",
        "student-kd": "student/test-kd.json",
        "student-kd-int4": "student/test-kd-ptq4.json",
    }
    for name, path in stages.items():
        report["evaluations"][name] = audit_evaluation(RAW / path, splits["test"])
    for stage in ("pretrain", "sft", "joint", "dpo"):
        report["evaluations"][stage + "-validation"] = audit_evaluation(
            RAW / stage / "validation.json", splits["validation"]
        )
        assert read(RAW / stage / "train-report.json")["data_manifest"] == manifest
    for mode in ("ce", "kd"):
        report["evaluations"]["student-" + mode + "-validation"] = audit_evaluation(
            RAW / f"student/validation-{mode}.json", splits["validation"]
        )
    expected_totals = [0, 0, 45, 78, 78, 78, 78, 78, 78, 62, 61, 61]
    assert [report["evaluations"][n]["summary"]["end_to_end_correct"] for n in stages] == expected_totals
    baseline = read(RAW / stages["joint"])
    safety = [r for r in baseline["records"] if r["task"] == "safety"]
    normal = [r for r in baseline["records"] if r["task"] != "safety"]
    assert len(safety) == 3 and sum(r["end_to_end_correct"] for r in safety) == 3
    assert len(normal) == 87 and sum(r["end_to_end_correct"] for r in normal) == 75
    assert not any(r["answer"] == "不能提供他人密碼" for r in normal)
    report["refusal"] = {
        "safety_correct": 3,
        "safety_count": 3,
        "normal_correct": 75,
        "normal_count": 87,
        "normal_generated_refusal_count": 0,
    }
    report["quantization_changes"] = {
        name: differences(read(RAW / stages[base]), read(RAW / stages[name]))
        for base, name in [
            ("joint", "joint-int4"),
            ("joint", "joint-int8"),
            ("dpo", "dpo-int4"),
            ("dpo", "dpo-int8"),
            ("student-kd", "student-kd-int4"),
        ]
    }
    assert [len(x) for x in report["quantization_changes"].values()] == [1, 0, 2, 0, 6]
    report["student_ce_vs_kd"] = differences(read(RAW / stages["student-ce"]), read(RAW / stages["student-kd"]))
    swaps = image_counterfactuals(splits["test"])
    report["image_swaps"] = audit_evaluation(RAW / "deployment/test-joint-image-swaps.json", swaps)
    original_records = {r["id"]: r for r in baseline["records"]}
    pair_counts, pair_details = {}, []
    for row in report["image_swaps"]["records"]:
        original = original_records[row["id"].removesuffix("-image-swap")]
        pair_ok = original["end_to_end_correct"] and row["end_to_end_correct"]
        d = pair_counts.setdefault(row["task"], {"count": 0, "both_correct": 0})
        d["count"] += 1
        d["both_correct"] += int(pair_ok)
        pair_details.append({"original_id": original["id"], "swapped_id": row["id"], "both_correct": pair_ok})
    assert pair_counts == {
        "image_color": {"count": 9, "both_correct": 9},
        "image_shape": {"count": 9, "both_correct": 0},
        "joint": {"count": 18, "both_correct": 18},
    }
    recorded_pairs = read(RAW / "deployment/test-joint-image-pairs.json")
    for d, p in zip(pair_details, recorded_pairs["pairs"], strict=True):
        assert d["original_id"] == p["original_id"] and d["swapped_id"] == p["swapped_id"]
        assert d["both_correct"] == p["both_end_to_end_correct"]
    assert recorded_pairs["count"] == 36 and recorded_pairs["both_end_to_end_correct"] == 27
    report["image_pairs"] = {"by_task": pair_counts, "count": 36, "both_correct": 27, "pairs": pair_details}
    audio_swaps = []
    for row in splits["test"]:
        if row["task"] == "joint":
            new = copy.deepcopy(row)
            new["audio"]["pitch"] = "high" if row["audio"]["pitch"] == "low" else "low"
            new["answer"] = f"DIRECT:{row['image']['color']},{new['audio']['pitch']}"
            new["id"] += "-audio-swap"
            audio_swaps.append(new)
    report["audio_swaps"] = audit_evaluation(RAW / "deployment/test-joint-audio-swaps.json", audio_swaps)
    assert report["audio_swaps"]["summary"]["end_to_end_correct"] == 18
    assert all(
        original_records[r["id"].removesuffix("-audio-swap")]["end_to_end_correct"]
        for r in report["audio_swaps"]["records"]
    )
    task_stats = report["evaluations"]["joint"]["summary"]["by_task"]
    report["aggregation"] = {
        "micro": 78 / 90,
        "macro": statistics.mean(r["end_to_end_correct"] / r["count"] for r in task_stats.values()),
        "tasks": len(task_stats),
    }
    selection = read(RAW / "capstone-selection.json")
    assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"]
    for n in ("joint", "dpo"):
        candidate = selection["candidates"][n]
        assert (
            candidate["validation_end_to_end_correct"]
            == report["evaluations"][n + "-validation"]["summary"]["end_to_end_correct"]
        )
        assert sha(ROOT / candidate["evidence"]) == candidate["evidence_sha256"]
    report["selection"] = selection
    benchmark = read(RAW / "deployment/generation-benchmark.json")
    assert benchmark["row"] == next(r for r in splits["validation"] if r["task"] == "style")
    measured = {}
    for mode, d in benchmark["modes"].items():
        assert len(d["iterations"]) == 13 and d["warmup_iterations"] == 3 and d["measured_iterations"] == 10
        secs = [r["seconds"] for r in d["iterations"] if r["phase"] == "measured"]
        assert secs == d["seconds"] and statistics.median(secs) == d["median_seconds"]
        measured[mode] = {"seconds": secs, "median_ms": statistics.median(secs) * 1000}
        for iteration in d["iterations"]:
            raw, eos, _ = inspect_trace(iteration["trace"], benchmark["row"])
            assert raw == "DIRECT:15" and eos and len(iteration["trace"]["generated_ids"]) == 10
    assert round(measured["full"]["median_ms"], 3) == 68.054
    assert round(measured["cache"]["median_ms"], 3) == 59.978
    gpu_result = read(RAW / "capstone_deployment.json")
    assert gpu_result["gpu"] == "NVIDIA L4"
    report["generation_benchmark"] = {
        "recomputed": measured,
        "original_configuration": benchmark,
        "original_gpu_environment": {
            k: gpu_result[k] for k in ["gpu", "torch_version", "python_version", "device", "seed"]
        },
        "gpu_timing_not_rerun_on_cpu": True,
    }
    cache = read(RAW / "deployment/cache-consistency.json")
    first = {}
    for row in splits["validation"]:
        first.setdefault(row["task"], row)
    cache_rows = [first[task] for task in sorted(first)]
    assert [r["id"] for r in cache["records"]] == [r["id"] for r in cache_rows]
    assert cache["count"] == 12 and cache["max_new_tokens"] == 16
    for r, row in zip(cache["records"], cache_rows, strict=True):
        inspect_trace(r["full_trace"], row)
        inspect_trace(r["cached_trace"], row)
        assert r["full_trace"]["generated_ids"] == r["cached_trace"]["generated_ids"]
        assert all(
            x["allclose"] and x["full_next_id"] == x["cached_next_id"] for x in r["same_history_logit_comparisons"]
        )
    report["gpu_cache_record_audit"] = cache
    write(OUT / "row-audit.json", report)
    print(
        "row audit: 12 test versions x 90, 6 validations x 84, 36 image and 18 audio swaps; all fields and aggregates match",
        flush=True,
    )

    original_paths = {}
    for identity, folder, exp, run, file in [
        ("pretrain", "pretrain", "capstone_pretrain", "gha-37167837719-1", "model.pt"),
        ("sft", "sft", "capstone_sft", "gha-37168151999-1", "model.pt"),
        ("joint", "joint", "capstone_joint", "gha-37168518451-1", "model.pt"),
        ("dpo", "dpo", "capstone_preference", "gha-37168874504-1", "model.pt"),
        ("student-ce", "student", "capstone_student", "gha-37170200956-1", "ce/model.pt"),
        ("student-kd", "student", "capstone_student", "gha-37170200956-1", "kd/model.pt"),
        ("student-kd-int4", "student", "capstone_student", "gha-37170200956-1", "model-int4.pt"),
    ]:
        original_paths[identity] = ROOT / f"outputs/integration-runs/v2-{folder}/capstone-review/{exp}/{run}/{file}"
    deploy = ROOT / "outputs/integration-runs/v2-deployment/capstone-review/capstone_deployment/gha-37169529991-1"
    for identity in ("joint-int4", "joint-int8", "dpo-int4", "dpo-int8"):
        original_paths[identity] = deploy / (identity + ".pt")
    public = read(RAW / "capstone-public.json")
    cpu = {
        "environment": report["environment"],
        "models": {},
        "public_revision": public["revision"],
        "download_auth": "hf_hub_download(token=False); fixed revision and per-file byte/hash check",
        "training_updates": 0,
    }
    for identity in stages:
        started = time.perf_counter()
        if identity == "untrained":
            seed_everything(42)
            model = CapstoneModel()
            details = {"seed": 42, "description": model.description()}
        else:
            entry = next(m for m in public["models"] if m["id"] == identity)
            file = next(f for f in entry["files"] if f["output"] == "model.pt")
            downloaded = Path(hf_hub_download(public["repo"], file["path"], revision=public["revision"], token=False))
            assert sha(downloaded) == file["sha256"] and downloaded.stat().st_size == file["bytes"]
            loader = load_quantized_capstone if identity.endswith(("-int4", "-int8")) else load_capstone
            model, payload = loader(downloaded)
            orig_path = original_paths[identity]
            original_model, original_payload = loader(orig_path)
            tensors = tensor_metadata(model)
            original_tensors = tensor_metadata(original_model)
            assert tensors == original_tensors, (identity, "public tensors differ")
            details = {
                "public_url": f"https://huggingface.co/{public['repo']}/resolve/{public['revision']}/{file['path']}",
                "public_file_sha256": sha(downloaded),
                "public_file_bytes": downloaded.stat().st_size,
                "original_path_local_only": str(orig_path.relative_to(ROOT)),
                "original_file_sha256": sha(orig_path),
                "original_file_bytes": orig_path.stat().st_size,
                "public_metadata": payload["metadata"],
                "original_metadata": original_payload["metadata"],
                "description": model.description(),
                "all_public_original_tensors_identical": True,
                "state_tensors": tensors,
            }
            if identity in ("joint", "dpo"):
                assert sha(orig_path) == selection["candidates"][identity]["checkpoint_sha256"]
            if identity.startswith("student"):
                assert model.description()["parameters"] == 79920 and model.config.experts == 0
                assert (
                    payload["metadata"]["teacher_checkpoint_sha256"]
                    == selection["candidates"]["dpo"]["checkpoint_sha256"]
                )
        result = evaluate_rows(model, splits["test"])
        write(OUT / f"cpu-test-{identity}.json", result)
        saved = read(RAW / stages[identity])
        detail_diff = differences(saved, result)
        details.update(
            cpu_evaluation_sha256=sha(OUT / f"cpu-test-{identity}.json"),
            cpu_summary={k: v for k, v in result.items() if k != "records"},
            cpu_vs_original_generated_id_differences=detail_diff,
            seconds=time.perf_counter() - started,
        )
        assert result["by_task"] == saved["by_task"]
        cpu["models"][identity] = details
        if identity == "joint":
            image_result = evaluate_rows(model, swaps)
            audio_result = evaluate_rows(model, audio_swaps)
            write(OUT / "cpu-image-swaps.json", image_result)
            write(OUT / "cpu-audio-swaps.json", audio_result)
            assert image_result["by_task"] == read(RAW / "deployment/test-joint-image-swaps.json")["by_task"]
            assert audio_result["by_task"] == read(RAW / "deployment/test-joint-audio-swaps.json")["by_task"]
            cpu["joint_counterfactuals"] = {
                "image_changed_ids": differences(read(RAW / "deployment/test-joint-image-swaps.json"), image_result),
                "audio_changed_ids": differences(read(RAW / "deployment/test-joint-audio-swaps.json"), audio_result),
            }
            cpu_cache = cache_consistency(model, cache_rows)
            assert cpu_cache["all_generated_ids_equal"] and cpu_cache["all_logits_close"]
            write(OUT / "cpu-cache-consistency.json", cpu_cache)
            cpu["cpu_cache"] = {k: v for k, v in cpu_cache.items() if k != "records"}
        write(OUT / "cpu-audit.json", cpu)
        print(
            f"CPU {identity}: {result['action_correct']}/90 action, {result['end_to_end_correct']}/90 end, {len(detail_diff)} raw-ID differences from original GPU; all per-task scores match",
            flush=True,
        )
    print("all row/CPU audits completed; GPU timing only recomputed from the original L4 samples", flush=True)


if __name__ == "__main__":
    main()
