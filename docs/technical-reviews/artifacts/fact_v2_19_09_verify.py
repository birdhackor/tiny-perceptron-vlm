"""Reproduce deterministic 19.9 claims and audit every retained GPU record on CPU.

Local ignored weight binaries are deliberately not copied into review artifacts.
The default paths can be restored via the repository's completed-run review
transport; hashes are checked before use. No training or remote compute occurs.
"""

import argparse
import contextlib
import hashlib
import io
import json
import platform
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.check_technical_reviews import sections  # noqa: E402
from scripts.course_experiments.capstone_deployment import (  # noqa: E402
    benchmark_generation,
    cache_consistency,
)
from tiny_perceptron.capstone import (  # noqa: E402
    CapstoneModel,
    build_dataset,
    digest,
    generate_trace,
    load_capstone,
    prompt_ids,
)

OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_09_"
EXPECTED_WEIGHT = "8f7e85820bd70bf1f16e7ef789b11f2d10873cf26a2688d651e23136bd7c3b47"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / f"{PREFIX}{name}.json").read_text())


def write(name, value):
    (OUT / f"{PREFIX}{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight",
        type=Path,
        default=ROOT / "outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=ROOT
        / "outputs/integration-runs/v2-deployment/capstone-review/capstone_deployment/gha-37169529991-1/joint.pt",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    body = next(body for lesson, body in sections(ROOT / "course/chapters/19.md") if lesson == "19.9")
    snapshot = OUT / f"{PREFIX}section.md"
    if snapshot.exists():
        assert snapshot.read_bytes() == body.encode(), "Section changed after the inspected snapshot"
    else:
        snapshot.write_bytes(body.encode())
    published_cache = read("cache-consistency")
    published_generation = read("generation-benchmark")
    deployment = read("deployment-result")
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "dtype": "torch.float32",
        "threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
    }
    functional_sha256 = sha(Path(torch.nn.functional.__file__))
    official_receipt = next(
        receipt for receipt in read("source-receipts")["receipts"] if receipt["url"].endswith("/torch/nn/functional.py")
    )
    assert functional_sha256 == official_receipt["sha256"]
    snippet = re.search(r"```python\n(.*?)```", body, re.S)[1]
    snippet_output = io.StringIO()
    snippet_globals = {}
    with contextlib.redirect_stdout(snippet_output):
        exec(compile(snippet, "course/chapters/19.md#19.9", "exec"), snippet_globals)
    assert torch.allclose(snippet_globals["full"], snippet_globals["cached"], atol=1e-4, rtol=1e-4)
    assert snippet_globals["full"].shape == (1, 264)
    code_hashes = {}
    for path in [
        "tiny_perceptron/attention.py",
        "tiny_perceptron/capstone.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/modern.py",
        "scripts/course_experiments/capstone_deployment.py",
    ]:
        code_hashes[path] = sha(ROOT / path)
        assert code_hashes[path] == deployment["code_sha256"][path]
    assert sha(args.weight) == EXPECTED_WEIGHT
    assert EXPECTED_WEIGHT == published_cache["checkpoint_sha256"] == published_generation["checkpoint_sha256"]
    export_record = next(item for item in deployment["artifacts"] if item["path"] == "joint.pt")
    assert sha(args.export) == export_record["sha256"]
    model, payload = load_capstone(args.weight)
    exported, export_payload = load_capstone(args.export)
    assert payload["stage"] == export_payload["stage"] == "joint"
    tensor_inspection = []
    for name, tensor in model.state_dict().items():
        other = exported.state_dict()[name]
        assert torch.equal(tensor, other), name
        assert bool(tensor.isfinite().all()), name
        tensor_inspection.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
                "sha256_contiguous_numpy_bytes": hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest(),
                "finite": True,
                "equal_deployment_export_tensor": True,
            }
        )
    write(
        "weight-inspection",
        {
            "original_path": str(args.weight.relative_to(ROOT)),
            "original_sha256": sha(args.weight),
            "original_bytes": args.weight.stat().st_size,
            "export_path": str(args.export.relative_to(ROOT)),
            "export_sha256": sha(args.export),
            "export_bytes": args.export.stat().st_size,
            "format_version": payload["format_version"],
            "stage": payload["stage"],
            "step": payload["step"],
            "config": payload["config"],
            "tokenizer": payload["tokenizer"],
            "data_version": payload["data_version"],
            "metadata": payload["metadata"],
            "tensor_count": len(tensor_inspection),
            "total_parameters": sum(item["numel"] for item in tensor_inspection),
            "tensor_bytes": sum(item["bytes"] for item in tensor_inspection),
            "tensors": tensor_inspection,
            "binary_storage": "Existing local binary retained; no weight bytes placed in review artifacts",
        },
    )
    splits, manifest = build_dataset(seed=42)
    assert digest(manifest) == published_generation["dataset_manifest_sha256"]
    first = {}
    for row in splits["validation"]:
        first.setdefault(row["task"], row)
    rows = [first[task] for task in sorted(first)]
    assert len(rows) == 12
    assert [(row["task"], row["id"]) for row in rows] == [
        (record["task"], record["id"]) for record in published_cache["records"]
    ]
    assert first["style"] == published_generation["row"]
    gpu_records = []
    for row, record in zip(rows, published_cache["records"], strict=True):
        full, cached = record["full_trace"], record["cached_trace"]
        assert full["prompt_ids"] == cached["prompt_ids"] == prompt_ids(row)
        assert full["generated_ids"] == cached["generated_ids"]
        assert full["stop_reason"] == cached["stop_reason"]
        assert full["eos"] == cached["eos"]
        assert record["generated_ids_equal"]
        comparisons = record["same_history_logit_comparisons"]
        assert len(comparisons) == len(full["generated_ids"])
        for index, step in enumerate(comparisons):
            assert step["step"] == index
            assert step["full_next_id"] == step["cached_next_id"] == full["generated_ids"][index]
            assert step["allclose"]
            assert 0 <= step["max_abs_logit_difference"] < published_cache["atol"]
        gpu_records.append(
            {
                "task": row["task"],
                "id": row["id"],
                "prompt_tokens": len(full["prompt_ids"]),
                "generated_tokens_including_eos": len(full["generated_ids"]),
                "stop_reason": full["stop_reason"],
                "generated_ids": full["generated_ids"],
                "max_logit_difference": max(step["max_abs_logit_difference"] for step in comparisons),
                "image_present": row["image"] is not None,
                "audio_present": row["audio"] is not None,
            }
        )
    assert published_cache["atol"] == published_cache["rtol"] == 1e-4
    modes = {}
    for mode, data in published_generation["modes"].items():
        runs = data["iterations"]
        assert len(runs) == 13
        assert [run["phase"] for run in runs] == ["warmup"] * 3 + ["measured"] * 10
        assert [run["iteration"] for run in runs] == list(range(13))
        seconds = [run["seconds"] for run in runs[3:]]
        assert seconds == data["seconds"]
        assert statistics.median(seconds) == data["median_seconds"]
        assert statistics.mean(seconds) == data["mean_seconds"]
        for run in runs:
            assert run["generated_token_count_including_eos"] == 10
            assert run["trace"]["generated_ids"] == [76, 81, 90, 77, 75, 92, 66, 57, 61, 2]
            assert run["trace"]["raw"] == "DIRECT:15"
            assert run["eos"] and run["stop_reason"] == run["trace"]["stop_reason"] == "eos"
        modes[mode] = {
            "warmup_count": 3,
            "measured_count": 10,
            "seconds": seconds,
            "recomputed_median_milliseconds": statistics.median(seconds) * 1000,
            "rounded_milliseconds": f"{statistics.median(seconds) * 1000:.3f}",
            "all_13_outputs_equal": True,
            "new_tokens_per_iteration_including_eos": 10,
        }
    assert modes["full"]["rounded_milliseconds"] == "68.054"
    assert modes["cache"]["rounded_milliseconds"] == "59.978"
    assert deployment["gpu"] == "NVIDIA L4" and deployment["seed"] == 42
    assert published_generation == deployment["results"]["generation_benchmark"]
    with torch.no_grad():
        torch.manual_seed(42)
        core = CapstoneModel().language.eval()
        ids = torch.tensor([[1, 21, 22, 23, 24]])
        prefix = core(ids[:, :3])["cache"]
        cases = []
        for last in (24, 25):
            changed = ids.clone()
            changed[0, -1] = last
            full = core(changed)["logits"][:, -1]
            cached = core(changed[:, 3:], cache=prefix)["logits"][:, -1]
            close = bool(torch.allclose(full, cached, atol=1e-4, rtol=1e-4))
            assert close and full.shape == cached.shape == (1, 264)
            cases.append(
                {
                    "ids": changed.tolist(),
                    "shape": list(full.shape),
                    "max_abs_difference": float((full - cached).abs().max()),
                    "allclose_atol_rtol_1e_4": close,
                }
            )
        changed = ids.clone()
        changed[0, 0] = 2
        full = core(changed)["logits"][:, -1]
        stale = core(changed[:, 3:], cache=prefix)["logits"][:, -1]
        rebuilt_prefix = core(changed[:, :3])["cache"]
        rebuilt = core(changed[:, 3:], cache=rebuilt_prefix)["logits"][:, -1]
        assert torch.allclose(full, rebuilt, atol=1e-4, rtol=1e-4)
        assert not torch.allclose(full, stale, atol=1e-4, rtol=1e-4)
        numerical = {
            "seed": 42,
            "atol": 1e-4,
            "rtol": 1e-4,
            "cases": cases,
            "prefix_cache_shapes": [[list(t.shape) for t in pair] for pair in prefix],
            "prefix_cache_bytes": sum(t.numel() * t.element_size() for pair in prefix for t in pair),
            "changed_first_id": {
                "ids": changed.tolist(),
                "stale_max_difference": float((full - stale).abs().max()),
                "rebuilt_max_difference": float((full - rebuilt).abs().max()),
                "rebuilt_allclose": True,
            },
        }
        cpu_cache = cache_consistency(model, rows)
        assert cpu_cache["all_generated_ids_equal"] and cpu_cache["all_logits_close"]
        for cpu, gpu in zip(cpu_cache["records"], published_cache["records"], strict=True):
            for mode in ("full_trace", "cached_trace"):
                assert cpu[mode] == gpu[mode], (cpu["task"], mode)
        write("cpu-cache-consistency", cpu_cache)
        cpu_benchmark = benchmark_generation(model, first["style"])
        assert cpu_benchmark["all_generated_ids_equal"] and cpu_benchmark["weights_unchanged"]
        write("cpu-generation-benchmark", cpu_benchmark)
        instrumented = []
        for task in ("image_color", "audio", "joint"):
            for use_cache in (False, True):
                calls = []
                counts = Counter()

                def prehook(module, args, kwargs):
                    cache = kwargs.get("cache")
                    calls.append(
                        {
                            "input_positions": args[0].shape[1],
                            "cache_positions": 0 if cache is None else cache[0][0].shape[2],
                            "images_supplied": kwargs.get("images") is not None,
                            "audio_supplied": kwargs.get("audio_features") is not None,
                        }
                    )

                def image_hook(module, args, output):
                    counts["image_projector"] += 1

                def audio_hook(module, args, output):
                    counts["audio_projector"] += 1

                hooks = [
                    model.register_forward_pre_hook(prehook, with_kwargs=True),
                    model.image_projector.register_forward_hook(image_hook),
                    model.audio_projector.register_forward_hook(audio_hook),
                ]
                trace = generate_trace(model, first[task], 16, use_cache=use_cache)
                for hook in hooks:
                    hook.remove()
                if use_cache:
                    assert calls[0]["input_positions"] == len(trace["prompt_ids"])
                    assert all(call["input_positions"] == 1 for call in calls[1:])
                    assert counts["image_projector"] == int(first[task]["image"] is not None)
                    assert counts["audio_projector"] == int(first[task]["audio"] is not None)
                    assert all(not call["images_supplied"] and not call["audio_supplied"] for call in calls[1:])
                instrumented.append({"task": task, "use_cache": use_cache, "calls": calls, "counts": dict(counts)})
    result = {
        "environment": environment,
        "installed_functional_sha256_equal_official_v2_14_1": functional_sha256,
        "verbatim_section_snippet_stdout": snippet_output.getvalue(),
        "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "code_sha256": code_hashes,
        "weight_sha256": EXPECTED_WEIGHT,
        "data_manifest_sha256": digest(manifest),
        "dataset_split_counts": {key: len(value) for key, value in splits.items()},
        "gpu_record_audit": {
            "gpu": deployment["gpu"],
            "python": deployment["python_version"],
            "torch": deployment["torch_version"],
            "dtype": published_generation["dtype"],
            "seed": deployment["seed"],
            "records": gpu_records,
            "sample_count": len(gpu_records),
            "same_history_steps": sum(len(row["generated_ids"]) for row in gpu_records),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in gpu_records)),
            "max_new_tokens": 16,
            "all_raw_ids_equal": True,
            "all_stop_reasons_equal": True,
            "all_logit_differences_below_absolute_tolerance": True,
            "max_abs_difference": max(row["max_logit_difference"] for row in gpu_records),
        },
        "gpu_generation_benchmark_audit": {
            "modes": modes,
            "prompt_tokens": len(prompt_ids(first["style"])),
            "timing_scope": published_generation["timing_scope"],
            "limitation": published_generation["limitation"],
            "statement": "GPU times recalculated from retained original records; CPU times do not remeasure L4",
        },
        "random_weight_numerical_examples": numerical,
        "multimodal_prefill_instrumentation": instrumented,
        "cpu_reproduction": {
            "sample_count": 12,
            "same_history_steps": sum(len(row["same_history_logit_comparisons"]) for row in cpu_cache["records"]),
            "all_gpu_trace_ids_and_stop_reasons_reproduced": True,
            "all_logits_close": True,
            "max_abs_difference": max(
                step["max_abs_logit_difference"]
                for row in cpu_cache["records"]
                for step in row["same_history_logit_comparisons"]
            ),
            "benchmark_warmup_and_measured_iterations_per_mode": [3, 10],
            "weights_unchanged": True,
            "scope": "CPU FP32 correctness and software behavior, not verification of GPU latency or kernel choice",
        },
    }
    write("audit", result)
    print(
        json.dumps({"environment": environment, "audit_passed": True, "cpu_reproduction": result["cpu_reproduction"]})
    )


if __name__ == "__main__":
    main()
