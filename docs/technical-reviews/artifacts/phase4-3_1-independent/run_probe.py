"""Record the real short CPU process, including its return code and output."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import os
import shlex
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
command = [str(ROOT / ".venv/bin/python"), str(BASE / "probe.py")]
env = os.environ.copy()
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
env.update(offline)
started = datetime.now(UTC).isoformat()
with (BASE / "probe-stdout.txt").open("wb") as stdout, (BASE / "probe-stderr.txt").open("wb") as stderr:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=30, check=False)
record = {"command": shlex.join(command), "command_argv": command, "cwd": str(ROOT),
          "started_utc": started, "exit_code": result.returncode, "timeout_seconds": 30,
          "offline_environment": offline,
          "artifacts": [{"path": str(BASE / name), "sha256": hashlib.sha256((BASE / name).read_bytes()).hexdigest()}
                        for name in ["probe.py", "probe-stdout.txt", "probe-stderr.txt", "probe-result.json", "swapped-fence.py"]
                        if (BASE / name).exists()]}
(BASE / "probe-execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
raise SystemExit(result.returncode)
