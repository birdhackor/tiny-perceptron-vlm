"""Record the real CPU command, exit status, output and hashes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
command = [str(REPO / ".venv/bin/python"), str(HERE / "bounded_checks.py")]
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
started = time.perf_counter()
with (HERE / "bounded-stdout.txt").open("wb") as stdout, (HERE / "bounded-stderr.txt").open("wb") as stderr:
    result = subprocess.run(command,cwd=REPO,env=env,stdout=stdout,stderr=stderr,timeout=30,check=False)
receipt = {"command":shlex.join(command), "cwd":str(REPO), "exit_code":result.returncode,
           "elapsed_seconds":time.perf_counter()-started, "timeout_seconds":30,
           "environment":{"python":sys.version,"python_executable":sys.executable,"device":"CPU",
                          "network":"Offline environment flags; no network API is invoked by bounded_checks.py"},
           "artifacts": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                         for name in ("bounded_checks.py","bounded-results.json","bounded-stdout.txt","bounded-stderr.txt")}}
(HERE/"bounded-execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
raise SystemExit(result.returncode)
