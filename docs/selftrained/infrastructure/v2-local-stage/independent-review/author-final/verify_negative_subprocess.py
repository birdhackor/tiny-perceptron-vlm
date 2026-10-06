"""Real CLI rejection checks against the frozen wrapper, no training launched."""

import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
ROOT = Path("/workspace/selftrained-v2")
SCRIPT = BASE / "scripts/selftrained/train_local_stage.py"
PYTHON = "/workspace/tiny-perceptron-vlm/.venv/bin/python"
manifest = BASE / "synthetic-manifest.json"
missing = BASE / "synthetic-manifest-undeclared-assets.json"
omitted = json.loads(manifest.read_text())
omitted["assets"] = []
missing.write_text(json.dumps(omitted, ensure_ascii=False, indent=2) + "\n")
foreign = BASE / "foreign-records-never-read.jsonl"
foreign.write_text("not JSON; must never reach read_records through an abbreviation\n")
foreign_config = BASE / "foreign-config-never-read.json"
foreign_config.write_text("not JSON; must never replace frozen model config\n")
prefix = [PYTHON, str(SCRIPT), "--repo-root", str(ROOT), "--manifest", str(manifest), "--data-root", str(BASE / "synthetic-data")]
common = ["--stage", "joint", "--architecture", "moe", "--steps", "2", "--batch-size", "16", "--context", "512"]
cases = [
    ("abbreviated-record", [], ["--record", str(foreign)], 2),
    ("abbreviated-config", [], ["--conf", str(foreign_config)], 2),
    ("abbreviated-asset-dir", [], ["--asset-d", str(BASE)], 2),
    ("abbreviated-wrapper-pin", ["--manifest-sha", "0" * 64], [], 2),
    ("selected-best-resume", [], ["--resume", "/tmp/p5-local-stage-candidate-v2/actual-cpu-subprocesses/weighted44-joint/best.pt"], 1),
    ("omitted-referenced-assets", ["--manifest", str(missing)], [], 1),
    ("wrong-manifest-sha", ["--manifest-sha256", "0" * 64], [], 1),
]


def run(case):
    name, wrapper_args, trainer_args, wanted = case
    output = BASE / ("negative-output-" + name)
    command = prefix + wrapper_args + ["--", *common, "--output-dir", str(output), *trainer_args]
    result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, timeout=90)
    log = BASE / (name + "-subprocess.log")
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == wanted, (name, result.returncode, log.read_text())
    assert not output.exists(), name
    return {"name": name, "command": command, "returncode": result.returncode, "no_attempt_output": True,
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run, cases))
report = {"scope": "Real frozen file-mode CLI prelaunch failures; no child training process created",
          "wrapper_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), "cases": results,
          "paid_dispatches": 0, "account_api_calls": 0, "generation_calls": 0,
          "original_test_records_read": 0, "production_private_checkpoint_loads": 0}
proof = BASE / "negative-subprocess-proof.json"
proof.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"case_count": len(results), "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest()}))
