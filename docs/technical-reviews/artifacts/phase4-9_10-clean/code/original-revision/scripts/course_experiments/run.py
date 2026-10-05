"""Run exactly one course experiment, or list its assets without importing PyTorch."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "docs/course-experiments/plan.json"


def experiment_spec(experiment_id):
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    matches = [item for item in plan["sequence"] if item["id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    return matches[0]


def list_assets(experiment_id):
    spec = experiment_spec(experiment_id)
    manifest = json.loads((ROOT / "assets/training/manifest.json").read_text(encoding="utf-8"))
    lookup = {item["id"]: item for item in manifest["assets"]}
    return [lookup[name]["archive"] for name in spec["assets"]]


def _revision():
    if os.environ.get("COURSE_REVISION"):
        return os.environ["COURSE_REVISION"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def unfinished_schedules(value, path="results"):
    """執行成功與跑完原定步數是兩件事；提前停下的權重只算部分結果。"""
    schedules = []
    if isinstance(value, dict):
        if value.get("budget_exhausted") is True:
            schedules.append(
                {"path": path, "requested_steps": value.get("requested_steps"), "steps": value.get("steps")}
            )
        for key, item in value.items():
            schedules.extend(unfinished_schedules(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            schedules.extend(unfinished_schedules(item, f"{path}[{index}]"))
    return schedules


def execute(experiment_id, device, output, dependencies, assets, revision=None, step_scale=1.0):
    import torch

    from scripts.course_experiments.common import Context, seed, write_json

    spec = experiment_spec(experiment_id)
    if not 0 < step_scale <= 1:
        raise ValueError("step_scale must be between zero and one")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A GPU experiment must not silently fall back to CPU")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    context = Context(str(device), output, Path(dependencies), Path(assets), seed=42)
    context.step_scale = step_scale
    seed(context.seed)
    torch.set_num_threads(2)
    module = importlib.import_module(f"scripts.course_experiments.{spec['module']}")
    experiment = getattr(module, spec["function"])
    code_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (ROOT / "tiny_perceptron", ROOT / "scripts/course_experiments")
        for path in sorted(directory.rglob("*.py"))
    }
    source_manifest = json.loads((Path(assets) / "manifest.json").read_text(encoding="utf-8"))
    source_assets = [item for item in source_manifest["assets"] if item["id"] in spec["assets"]]
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        results = experiment(context)
        # Ensure all returned data can cross machines without unpickling PyTorch objects.
        json.dumps(results, ensure_ascii=False, allow_nan=False)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "experiment_id": experiment_id,
                "revision": revision or _revision(),
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "code_sha256": code_hashes,
                "step_scale": step_scale,
            },
        )
        raise
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    unfinished = unfinished_schedules(results)
    artifacts = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in ("result.json", "failure.json")
    ]
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "title": spec["title"],
        "revision": revision or _revision(),
        "device": str(device),
        "seed": context.seed,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "elapsed_seconds": elapsed,
        "timing_scope": "experiment training, evaluation and local checkpoint saves; excludes image build, startup and HF uploads",
        "step_scale": step_scale,
        "evidence_status": (
            "interface_smoke_only"
            if step_scale < 1 or (spec["module"] == "modalities" and device == "cpu")
            else "incomplete_run"
            if unfinished
            else "complete_run"
        ),
        "unfinished_schedules": unfinished,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_memory_scope": (
            "Main-process PyTorch CUDA allocator allocated-byte peak since its most recent reset. "
            "Branch measurements can reset this counter; this is not necessarily the full-experiment peak. "
            "It excludes subprocesses, reserved-but-unused allocator memory and CUDA driver memory. "
            "Use branch-specific memory measurements and their scopes for comparisons."
            if device == "cuda"
            else "No CUDA allocator measurement on this device."
        ),
        "results": results,
        "artifacts": artifacts,
        "assets": source_assets,
        "code_sha256": code_hashes,
        "public_exports": [],
    }
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-assets", metavar="ID", help="One repo-relative archive path per line; no torch needed")
    parser.add_argument("--experiment", metavar="ID")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/course-experiments/course-v1")
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--step-scale", type=float, default=1.0, help="Less than one is explicitly a smoke check")
    args = parser.parse_args()
    if args.list_assets:
        print("\n".join(list_assets(args.list_assets)), end="\n" if list_assets(args.list_assets) else "")
        return
    if not args.experiment:
        parser.error("Choose --experiment or --list-assets")
    result = execute(
        args.experiment,
        args.device,
        args.output / args.experiment,
        args.dependencies or args.output,
        ROOT / "assets/training",
        step_scale=args.step_scale,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "elapsed_seconds": result["elapsed_seconds"],
                "evidence_status": result["evidence_status"],
                "artifacts": len(result["artifacts"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
