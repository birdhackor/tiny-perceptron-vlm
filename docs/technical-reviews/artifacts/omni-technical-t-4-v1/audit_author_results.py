"""Read historical author evidence and pinned code; no training or remote calls."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path("/workspace/work/reference/omni-technical-t-4-v1/original-code")
CACHE.mkdir(parents=True, exist_ok=True)


def trim_training(value):
    return {key: value.get(key) for key in ("steps", "seconds", "effective_tokens", "records", "parameters", "checkpoint")}


def scan(value, path="results"):
    found = {}
    if isinstance(value, dict):
        if "steps" in value and "checkpoint" in value:
            found[path] = trim_training(value)
        if "samples" in value:
            fields = ("nll", "effective_tokens", "examples", "records", "matches", "exact_match", "eos_rate", "generation_evaluated_records", "metric_denominators")
            found[path] = {key: value[key] for key in fields if key in value}
            found[path]["saved_generation_samples"] = len(value["samples"])
        if all(key in value for key in ("train", "validation", "test")) and isinstance(value["train"], dict) and "records" in value["train"]:
            found[path] = {split: {key: value[split].get(key) for key in ("records", "families", "sha256")} for split in ("train", "validation", "test")}
        for key, nested in value.items():
            found.update(scan(nested, path + "." + key))
    return found


audit = {"created_at": datetime.now(UTC).isoformat(), "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-4-v1/audit_author_results.py", "environment": {"python": platform.python_version(), "device": "CPU read-only JSON/hash and Git object inspection"}, "results": []}
for name, shown_time in (("text_foundation", 16.0), ("real_text", 23.9), ("tokenizer", 16.2), ("sft", 31.7), ("sft_ablation", 23.5)):
    path = ROOT / "docs/course-experiments/results" / (name + ".json")
    report = json.loads(path.read_text())
    row = {key: report.get(key) for key in ("experiment_id", "revision", "seed", "device", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status", "modal")}
    row.update(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), code=[])
    assert report["device"] == "cuda" and report["gpu"] == "NVIDIA L4"
    assert round(report["elapsed_seconds"], 1) == shown_time
    assert report["step_scale"] == 1 and report["evidence_status"] == "complete_run"
    assert report["seed"] == 42
    for relative in ("scripts/course_experiments/run.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/training.py", "tiny_perceptron/model.py"):
        code = subprocess.run(["git", "show", report["revision"] + ":" + relative], cwd=ROOT, check=True, capture_output=True).stdout
        digest = hashlib.sha256(code).hexdigest()
        expected = report["code_sha256"][relative]
        assert digest == expected, (name, relative, digest, expected)
        cached = CACHE / (digest + "-" + Path(relative).name)
        if not cached.exists():
            cached.write_bytes(code)
        row["code"].append({"path": relative, "report_sha256": expected, "git_sha256": digest, "matched": True, "current_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), "local_original": str(cached)})
    row["observations"] = scan(report["results"])
    row["artifact_records"] = len(report["artifacts"])
    row["raw_source_assets"] = [{key: asset.get(key) for key in ("id", "archive_sha256", "training_records", "source_metadata")} for asset in report["assets"]]
    if name == "tokenizer":
        row["matched_tokenizer_budget"] = {kind: {key: run.get(key) for key in ("vocab_size_including_8_specials", "parameters", "training_raw_utf8_bytes_exposed", "tokenizer_file")} for kind, run in report["results"]["runs"].items()}
    audit["results"].append(row)
audit["limitations"] = ["Historical elapsed_seconds is the full recipe's training, evaluation and local saves, not a rerun or isolated training-only time.", "The source reports explicitly exclude image build, startup and HF uploads; modal.function_seconds is a different scope.", "Code hashes at recorded Git revisions match; current run.py has changed and text_foundation used older text.py/common.py. The historical versions were separately inspected.", "No historical weights were downloaded or GPU executions repeated. The reported seed-42 synthetic/limited-data outcomes cannot establish broad conversation ability or CPU feasibility for the fixed recipes."]
(OUT / "author-results-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "all 5 time roundings and 30 pinned source hashes matched", "experiments": [row["experiment_id"] for row in audit["results"]]}, ensure_ascii=False))
