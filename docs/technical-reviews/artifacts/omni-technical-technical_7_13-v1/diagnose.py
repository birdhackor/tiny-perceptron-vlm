"""7.13 的短 CPU 診斷；不下載資料、不做任何訓練或 GPU 呼叫。"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import run as runner
from scripts.course_experiments.common import new_lm, records_sha256, split_records
from scripts.course_experiments.text import _steps
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(label, value):
    print(label + " " + json.dumps(value, ensure_ascii=False, sort_keys=True))


emit("started_at", datetime.now(UTC).isoformat())
emit("environment", {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()})
freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads(freeze_path.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "7.13")
raw = (ROOT / page["snapshot"]).read_bytes()
current = re.search(rb"^## 7\.13 .+?(?=^## |\Z)", (ROOT / page["source"]).read_bytes(), re.M | re.S).group()
assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
assert current == raw
assert page["figures_sha256"] == {} and not re.findall(rb"!\[[^\]]*\]\([^)]*\)", raw)
emit("source", {"sha256": page["source_sha256"], "current_equals_snapshot": True, "figures": {}})
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(blocks) == 1
code = blocks[0]
code_sha = hashlib.sha256(code.encode()).hexdigest()


def execute_snippet(text):
    namespace, output = {}, io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(text, "7.13-exact-inline", "exec"), namespace)
    except AssertionError:
        error = "AssertionError"
    return {"reply": namespace["reply"], "parsed": namespace["parsed"], "scores": namespace["scores"], "stdout": output.getvalue(), "error": error}


original = execute_snippet(code)
assert original["scores"] == {"格式符合": True, "1+1內容正確": False} and original["error"] is None
emit("inline_original", {"code_sha256": code_sha, **original})
two_code = code.replace("reply = '{\"answer\": 3}'", "reply = '{\"answer\": 2}'")
two = execute_snippet(two_code)
assert two["scores"] == {"格式符合": True, "1+1內容正確": True} and two["error"] == "AssertionError"
emit("exercise_two_original_assert", two)
fixed_code = two_code.replace('assert scores["格式符合"] and not scores["1+1內容正確"]', 'assert scores["格式符合"] and scores["1+1內容正確"]')
fixed = execute_snippet(fixed_code)
assert fixed["error"] is None
emit("exercise_two_fixed_assert", fixed)
renamed = execute_snippet(fixed_code.replace("reply = '{\"answer\": 2}'", "reply = '{\"result\": 2}'"))
assert renamed["scores"] == {"格式符合": False, "1+1內容正確": False}
emit("exercise_renamed_field", renamed)

boundaries = []
for reply in ['{"answer": true}', '{"answer": false}', '{"answer": 2.0}', '{"answer": "2"}', '{"answer": 2, "extra": 0}', '{}']:
    parsed = json.loads(reply)
    format_ok = isinstance(parsed, dict) and set(parsed) == {"answer"} and type(parsed["answer"]) is int
    boundaries.append({"reply": reply, "type": type(parsed.get("answer")).__name__, "format": format_ok, "content": parsed.get("answer") == 2})
assert not any(row["format"] for row in boundaries)
assert isinstance(True, int) and type(True) is not int and True == 1
assert json.loads('{"answer": true}')["answer"] is True
assert json.loads('{"answer": false}')["answer"] is False
assert {}.get("answer") is None
emit("format_boundaries", boundaries)
emit("bool_int", {"isinstance_True_int": isinstance(True, int), "type_True_is_int": type(True) is int, "True_equals_1": True == 1, "missing_get": {}.get("answer")})
vectors = {"before": [0, 1], "after": [1, 0]}
assert sum(vectors["before"]) == sum(vectors["after"]) == 1
emit("aggregation_counterexample", {"vectors": vectors, "sums": [1, 1], "scope": "transparent arithmetic example, not measured model quality"})

result_path = ROOT / "docs/course-experiments/results/sft.json"
published = json.loads(result_path.read_bytes())
r = published["results"]
tok = ByteTokenizer()
samples = []
for i, sample in enumerate(r["after"]["test"]["samples"]):
    ids = sample["generated_ids"]
    ended = tok.eos_id in ids
    raw_ids = ids[:ids.index(tok.eos_id)] if ended else ids
    exact = raw_ids == tok.encode(sample["expected"])
    assert sample["exact"] == exact and sample["eos"] == ended
    assert tok.decode(raw_ids) == sample["generated"]
    assert {"messages", "expected", "generated_ids", "exact", "eos"} <= sample.keys()
    samples.append({"index": i, "expected": sample["expected"], "generated": sample["generated"], "exact_recomputed": exact, "eos_recomputed": ended})
matches = sum(s["exact_recomputed"] for s in samples)
assert len(samples) == r["after"]["test"]["records"] == 10
assert matches == r["after"]["test"]["matches"] == 5
assert matches / len(samples) == r["after"]["test"]["exact_match"] == 0.5
assert sum(s["eos_recomputed"] for s in samples) == 10
assert r["training"]["steps"] == 900 and r["training"]["records"] == 45
assert r["checkpoint"] == r["training"]["checkpoint"] == "model.pt"
assert r["pretrain_then_sft"]["after_sft"]["test"]["matches"] == 7
assert r["pretrain_then_sft"]["after_sft"]["test"]["records"] == 10
emit("recorded_sft_test_recount", {"source_sha256": sha(result_path), "samples": samples, "matches": matches, "records": len(samples), "exact_match": matches / len(samples), "eos_rate": 1.0, "pretrain_sft_matches": 7, "pretrain_sft_records": 10, "training_steps": 900, "training_effective_tokens": r["training"]["effective_tokens"], "training_records": 45, "seed": published["seed"], "scope": "recount of existing raw ID records; no inference or retraining"})

parts = split_records(generate_records("attributes-sft"), seed=42)
split_summary = {}
for name, rows in parts.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(serialized).hexdigest()
    assert digest == r["data"][name]["sha256"]
    split_summary[name] = {"records": len(rows), "families": len({row["family"] for row in rows}), "sha256": digest}
assert [len(parts[k]) for k in ["train", "validation", "test"]] == [45, 5, 10]
assert records_sha256(parts["train"]) == r["training"]["records_sha256"]
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not ({row["family"] for row in parts[a]} & {row["family"] for row in parts[b]})
emit("recreated_fixed_data", split_summary)
torch.set_num_threads(1)
ctx = SimpleNamespace(seed=42, device="cpu", step_scale=1.0)
model1 = new_lm(ctx, width=64, layers=2)
model2 = new_lm(ctx, width=64, layers=2)
assert all(torch.equal(a, b) for a, b in zip(model1.parameters(), model2.parameters()))
assert sum(p.numel() for p in model1.parameters()) == r["training"]["parameters"]
assert _steps(ctx, 900) == 900
emit("untrained_initialization", {"same_seed_same_parameters": True, "parameters": sum(p.numel() for p in model1.parameters()), "seed": 42, "steps_at_scale_1": _steps(ctx, 900), "optimizer_updates_executed": 0})

ablation_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
ablation = json.loads(ablation_path.read_bytes())
ablation_before = ablation["results"]["before"]["A_attributes"]["test"]
assert ablation_before["samples"] == r["after"]["test"]["samples"]
emit("subsequent_base_record", {"source_sha256": sha(ablation_path), "before_A_test_matches": ablation_before["matches"], "before_A_test_records": ablation_before["records"], "all_baseline_samples_identical": True})
assert round(published["elapsed_seconds"], 1) == 31.7
assert published["gpu"] == "NVIDIA L4" and published["device"] == "cuda" and published["step_scale"] == 1.0
emit("recorded_timing", {k: published[k] for k in ["revision", "device", "gpu", "elapsed_seconds", "timing_scope", "python_version", "torch_version", "step_scale"]})

relevant_paths = ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py"]
code_hashes = {p: {"current_sha256": sha(ROOT / p), "recorded_sha256": published["code_sha256"][p]} for p in relevant_paths}
assert all(v["current_sha256"] == v["recorded_sha256"] for v in code_hashes.values())
historical_runner = subprocess.run(["git", "show", published["revision"] + ":scripts/course_experiments/run.py"], cwd=ROOT, capture_output=True, check=True).stdout
historical_runner_sha = hashlib.sha256(historical_runner).hexdigest()
assert historical_runner_sha == published["code_sha256"]["scripts/course_experiments/run.py"]
current_runner = (ROOT / "scripts/course_experiments/run.py").read_bytes()
timing_pattern = rb"    started = time.perf_counter\(\).*?    elapsed = time.perf_counter\(\) - started"
assert re.search(timing_pattern, historical_runner, re.S).group() == re.search(timing_pattern, current_runner, re.S).group()
emit("implementation_fingerprints", {"matching_core": code_hashes, "historical_runner_sha256": historical_runner_sha, "current_runner_sha256": sha(ROOT / "scripts/course_experiments/run.py"), "runner_timing_region_identical": True})

commands = []
for argv in [[sys.executable, "scripts/fetch_training_assets.py", "--list"], [sys.executable, "scripts/fetch_training_assets.py", "--help"], [sys.executable, "-m", "scripts.course_experiments.run", "--help"], [sys.executable, "-m", "scripts.course_experiments.run", "--list-assets", "sft"]]:
    proc = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=True)
    commands.append({"argv": argv, "exit_code": proc.returncode, "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest()})
    if argv[-1] == "--list":
        asset = next(json.loads(line) for line in proc.stdout.splitlines() if json.loads(line)["id"] == "ultrachat-sft")
        assert asset["training_records"] == 100
    if argv[-1] == "sft":
        assert proc.stdout.strip() == "assets/training/ultrachat-sft-v1.tar.gz"
        commands[-1]["stdout"] = proc.stdout
emit("read_only_cli_checks", commands)
spec = runner.experiment_spec("sft")
assert spec["module"] == "text" and spec["function"] == "run_sft" and spec["assets"] == ["ultrachat-sft"]
emit("runtime_dispatch_spec", {k: spec[k] for k in ["id", "module", "function", "assets"]})
fake_result = {"experiment_id": "sft", "elapsed_seconds": "MOCK_ONLY", "evidence_status": "MOCK_ONLY", "artifacts": []}
with patch.object(sys, "argv", ["scripts.course_experiments.run", "--experiment", "sft", "--device", "cuda"]), patch.object(runner, "execute", return_value=fake_result) as called, contextlib.redirect_stdout(io.StringIO()):
    runner.main()
args, kwargs = called.call_args
assert args[:2] == ("sft", "cuda") and args[2] == ROOT / "outputs/course-experiments/course-v1/sft"
assert kwargs == {"step_scale": 1.0}
emit("cli_dispatch_without_execution", {"experiment": args[0], "device_argument_only": args[1], "output": str(args[2].relative_to(ROOT)), "dependencies": str(args[3].relative_to(ROOT)), "assets": str(args[4].relative_to(ROOT)), "step_scale": kwargs["step_scale"], "execute_mocked": True, "training_or_GPU_invoked": False})
emit("limitations", "No new training, checkpoint inference, GPU run, or asset download was performed. 5/10 and 31.7 seconds are audited historical records, not fresh measurements. The fixed JSON example is not a general parser for malformed or non-object replies.")
emit("result", "PASS: every assertion above completed")
emit("completed_at", datetime.now(UTC).isoformat())
