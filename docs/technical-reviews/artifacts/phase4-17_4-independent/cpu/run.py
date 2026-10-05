"""Capture the actual bounded CPU invocation and its complete outputs."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
argv = [str(ROOT / ".venv/bin/python"), str(HERE / "variants.py")]
start = time.perf_counter()
with (HERE / "stdout.json").open("wb") as stdout, (HERE / "stderr.txt").open("wb") as stderr:
    completed = subprocess.run(argv, cwd=ROOT, env=os.environ | offline, stdout=stdout,
                               stderr=stderr, timeout=20, check=False)
record = {"argv": argv, "command": shlex.join(["env"] + [k+"="+v for k,v in offline.items()] + argv),
          "cwd": str(ROOT), "offline_overrides": offline, "timeout_seconds": 20,
          "elapsed_seconds": time.perf_counter()-start, "exit_code": completed.returncode,
          "artifacts": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                        for name in ("variants.py", "run.py", "stdout.json", "stderr.txt")}}
(HERE / "execution.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"exit_code": completed.returncode, "execution_record": str(HERE / "execution.json")}))
sys.exit(completed.returncode)
