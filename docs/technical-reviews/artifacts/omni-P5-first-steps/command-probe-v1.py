"""Read-only command probes; dry run uses only the repository's toy default."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
external = Path("/workspace/work/omni-P5-first-steps-technical-v1-external")
env = {**os.environ, "UV_PROJECT_ENVIRONMENT": "/workspace/tiny-perceptron-vlm/.venv", "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
commands = [
    ["git", "--version"],
    ["uv", "--version"],
    ["uv", "sync", "--help"],
    ["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run", "--offline"],
    [sys.executable, "scripts/check_env.py"],
    [sys.executable, "-m", "ipykernel", "install", "--help"],
    [sys.executable, "-m", "jupyterlab", "--help"],
    [sys.executable, "scripts/train.py", "--help"],
    [sys.executable, "scripts/train.py", "--device", "cpu", "--output", str(external / "dry-run-no-write.pt")],
]
before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["uv.lock", "pyproject.toml", "scripts/train.py", "scripts/check_env.py", "notebooks/01/1.1.ipynb"]}
rows = []
for command in commands:
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=80)
    assert result.returncode == 0, (command, result.returncode, result.stderr)
    rows.append({"command": command, "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
dry = json.loads(rows[-1]["stdout"].strip().splitlines()[-1])
assert dry["mode"] == "dry-run-no-weight-update"
assert dry["device"] == "cpu" and len(dry["history"]) == 1
assert not (external / "dry-run-no-write.pt").exists()
assert not (external / "dry-run-no-write.json").exists()
after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in before}
assert before == after
print(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "environment": {"python": platform.python_version(), "device": "CPU", "python_executable": sys.executable}, "commands": rows, "source_digests_unchanged": before, "dry_run_checks": {"mode": dry["mode"], "steps": len(dry["history"]), "checkpoint_written": False}, "limits": "No fresh installation, git clone, kernel registration, Windows/macOS, interactive Jupyter UI, or Colab execution. --help and dry-run are not full platform execution."}, ensure_ascii=False, indent=2))
