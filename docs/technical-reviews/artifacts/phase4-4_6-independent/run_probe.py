"""Save actual independent CPU probe command, exit status and streams."""
import json
import os
import subprocess
import time
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
command = [str(ROOT / ".venv/bin/python"), str(ART / "probe.py")]
environment = dict(os.environ)
environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
start = time.monotonic()
result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=30, check=False)
(ART / "probe.stdout.txt").write_bytes(result.stdout)
(ART / "probe.stderr.txt").write_bytes(result.stderr)
(ART / "probe-receipt.json").write_text(json.dumps({"command_argv": command, "cwd": str(ROOT),
    "exit_code": result.returncode, "timeout_seconds": 30, "elapsed_seconds": time.monotonic() - start,
    "offline_environment": {key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    "input_provenance": "Literal synthetic tensors in probe.py; repository model/attention/modern files preserved in original/; no checkpoint/data files."},
    ensure_ascii=False, indent=2) + "\n")
print("Probe exit:", result.returncode)
print(result.stdout.decode(), end="")
print(result.stderr.decode(), end="")
raise SystemExit(result.returncode)
