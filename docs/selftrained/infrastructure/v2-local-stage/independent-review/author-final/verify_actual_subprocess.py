"""Synthetic CPU contract verification; never reads production records/weights."""

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import torch

ROOT = Path("/workspace/selftrained-v2")
BASE = Path(__file__).parent
SCRIPT = BASE / "scripts/selftrained/train_local_stage.py"
PYTHON = "/workspace/tiny-perceptron-vlm/.venv/bin/python"
FIXTURE = Path("/tmp/p5-native-voice-integration-candidate/synthetic-cpu/pytest-tmp/test_native_actual_main_fresh_0")
DATA = BASE / "synthetic-data"
RUNS = BASE / "actual-cpu-subprocesses"
CHECKS = []
DATA.mkdir()
RUNS.mkdir()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, value):
    assert value, name
    CHECKS.append(name)


for name in ("records.jsonl", "pixels.png", "wave.wav"):
    shutil.copyfile(FIXTURE / name, DATA / name)
config = json.loads((FIXTURE / "config.json").read_text()) | {"max_length": 512, "backend": "sdpa"}


def entry(name):
    return {"path": name, "bytes": (DATA / name).stat().st_size, "sha256": sha(DATA / name)}


manifest = {
    "schema_version": 1,
    "initialization": "random",
    "scope": "Synthetic CPU local wrapper contract only; no original heldout records or private checkpoints",
    "model_config": config,
    "records": [entry("records.jsonl")],
    "assets": [entry("pixels.png"), entry("wave.wav")],
}
MANIFEST = BASE / "synthetic-manifest.json"
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
source_paths = [Path(x) for x in subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", "HEAD", "tiny_perceptron", "scripts/selftrained"], cwd=ROOT, text=True
).splitlines()]
before = {str(path): sha(ROOT / path) for path in source_paths}
common = [
    PYTHON, str(SCRIPT), "--repo-root", str(ROOT), "--manifest", str(MANIFEST),
    "--manifest-sha256", sha(MANIFEST), "--data-root", str(DATA), "--",
    "--stage", "joint", "--architecture", "moe", "--batch-size", "16", "--context", "512",
    "--learning-rate", "0.0002", "--seed", "20261006", "--eval-every", "1", "--save-every", "1",
    "--freeze-perception-backbones", "--sampling-mode", "task-family", "--threads", "1", "--device", "cpu",
]
commands = []


def run(name, extra, expected=0, interrupt=False):
    output = RUNS / name
    command = common + ["--output-dir", str(output), *extra]
    commands.append({"name": name, "command": command})
    if interrupt:
        child = subprocess.Popen(command, cwd="/tmp", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = []
        sent = False
        for line in child.stdout:
            lines.append(line)
            if not sent and '"event": "validation", "step": 1' in line:
                os.kill(child.pid, signal.SIGTERM)
                sent = True
        result_code = child.wait(timeout=120)
        check(name + " real SIGTERM sent after genuine positive validation step", sent)
        raw = "".join(lines)
    else:
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, timeout=120)
        raw, result_code = result.stdout + result.stderr, result.returncode
    (BASE / (name + "-subprocess.log")).write_text(raw)
    check(name + " actual subprocess exit", result_code == expected)
    return output


baseline = run("baseline-joint", ["--steps", "1"])
weighted = run("weighted44-joint", ["--steps", "1", "--tool-loss-weight", "4", "--numeric-run-loss-weight", "4",
                                    "--init-checkpoint", str(baseline / "best.pt")])
native = run("native414-joint", ["--steps", "1", "--tool-loss-weight", "4", "--numeric-run-loss-weight", "1",
                                "--native-voice-loss-weight", "4", "--init-checkpoint", str(weighted / "best.pt")])
spec = importlib.util.spec_from_file_location("candidate_local", SCRIPT)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
for directory in (baseline, weighted, native):
    receipt = json.loads((directory / "receipt.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    training = json.loads((directory / "train-receipt.json").read_text())
    check(directory.name + " actual completed exit/provenance", execution["status"] == "completed" and execution["returncode"] == 0)
    check(directory.name + " exact shared outer identity", all(receipt[key] == value for key, value in execution.items()))
    for item in receipt["files"]:
        wrapper.verified_file(directory, item)
    check(directory.name + " all actual output bytes/SHA verified", True)
    check(directory.name + " positive actual CPU step + batch16/context512", training["steps"] == 1 and
          execution["job"]["batch_size"] == 16 and execution["job"]["context"] == 512)
    check(directory.name + " unweighted validation selector", training["test_used_for_selection"] is False and
          json.loads((directory / "inference-manifest.json").read_text())["selection"] == "validation_loss")
    checkpoint = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
    check(directory.name + " genuine full optimizer/RNG/sampler checkpoint", all(k in checkpoint for k in ("optimizer", "rng", "sampler", "model")))
native_checkpoint = torch.load(native / "best.pt", map_location="cpu", weights_only=False)
initialization = native_checkpoint["origin"]["new_joint_initialization"]
check("native actual core parent selected SHA binding", initialization["source_checkpoint_sha256"] == sha(weighted / "best.pt"))
check("native actual completed parent source binding", initialization["source_integrity"]["source_completed_steps"] == 1)
check("native true fresh reset before first CPU step", initialization["reset_state"] == {
    "optimizer": "new", "rng": "fresh_stage_seed_not_source_rng", "sampler_draws": 0, "family_draws": 0,
    "stage_step": 0, "tokens": 0, "target_tokens": 0,
})
check("native actual V2 policy", native_checkpoint["language_objective_policy"]["version"] == "selftrained-language-objective-v2")
failed = run("invalid-zero-step", ["--steps", "0"], expected=1)
failed_execution = json.loads((failed / "execution.json").read_text())
check("actual child failure preserved", failed_execution["status"] == "failed" and failed_execution["returncode"] != 0)
interrupted = run("weighted44-interrupted", ["--steps", "8", "--tool-loss-weight", "4", "--numeric-run-loss-weight", "4",
                                            "--init-checkpoint", str(baseline / "best.pt")], expected=1, interrupt=True)
interrupted_training = json.loads((interrupted / "train-receipt.json").read_text())
interrupted_execution = json.loads((interrupted / "execution.json").read_text())
check("genuine graceful child exit0 remains failed when incomplete", interrupted_execution["returncode"] == 0 and
      interrupted_execution["status"] == "failed" and interrupted_training["interrupted"] is True and
      0 < interrupted_training["steps"] < 8)
for name, source in (("reject-interrupted-init", interrupted), ("reject-failed-init", failed)):
    rejected = run(name, ["--steps", "1", "--init-checkpoint", str(source / "best.pt")], expected=1)
    check(name + " no next attempt launch/output", not rejected.exists())
resume = run("weighted44-exact-resume", ["--steps", "8", "--tool-loss-weight", "4", "--numeric-run-loss-weight", "4",
                                       "--resume", str(interrupted / "latest.pt")])
resumed_training = json.loads((resume / "train-receipt.json").read_text())
check("actual incomplete exact resume reaches original target", resumed_training["steps"] == 8 and resumed_training["completed_requested_steps"] is True)
check("source bytes unchanged", before == {str(path): sha(ROOT / path) for path in source_paths})
proof = {
    "scope": "Preparation only: genuine synthetic CPU subprocesses, tiny width16; batch16/context512; no production performance/lineage rerun claim",
    "script_sha256": sha(SCRIPT), "manifest_sha256": sha(MANIFEST),
    "production_source_before": before,
    "production_source_after": {str(path): sha(ROOT / path) for path in source_paths},
    "checks": CHECKS, "check_count": len(CHECKS), "commands": commands,
    "interrupted_actual_steps": interrupted_training["steps"],
    "actual_new_optimizer_steps_by_attempt": {
        directory.name: sum(json.loads(row).get("event") == "train" for row in (directory / "metrics.jsonl").read_text().splitlines())
        for directory in RUNS.iterdir() if (directory / "metrics.jsonl").is_file()
    },
    "generation_calls": 0, "paid_dispatches": 0, "account_api_calls": 0,
    "original_test_records_read": 0, "production_private_checkpoint_loads": 0,
    "production_repo_edits": 0,
}
(BASE / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"checks": len(CHECKS), "script_sha256": sha(SCRIPT), "proof_sha256": sha(BASE / "proof.json")}))
