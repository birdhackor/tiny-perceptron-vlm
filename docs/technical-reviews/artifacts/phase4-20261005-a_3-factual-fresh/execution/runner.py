"""Record exact original-fence and independent bounded verification commands."""
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[3]
assert ROOT.joinpath("tiny_perceptron/retrieval.py").is_file()
ENV = os.environ.copy()
ENV.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
           HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
environment = {"python": sys.version, "executable": sys.executable,
               "device": "cpu", "platform": platform.platform(), "cwd": str(ROOT),
               "packages": "Python standard library; torch not imported by either target",
               "offline_settings": {k: ENV[k] for k in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
                   "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"]}}
(OUT / "execution/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
records = []
for name, script in [("original-fence", OUT / "code/fence-1.py"), ("verification", OUT / "execution/verify.py")]:
    command = [sys.executable, str(script)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, timeout=30)
    (OUT / f"execution/{name}.stdout.txt").write_bytes(completed.stdout)
    (OUT / f"execution/{name}.stderr.txt").write_bytes(completed.stderr)
    records.append({"name": name, "command": shlex.join(command), "argv": command,
                    "code_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                    "exit_code": completed.returncode, "timeout_seconds": 30,
                    "elapsed_seconds": time.perf_counter() - started,
                    "stdout": f"execution/{name}.stdout.txt", "stderr": f"execution/{name}.stderr.txt"})
    (OUT / "execution/commands.json").write_text(json.dumps(records, indent=2) + "\n")
    print(name, "exit", completed.returncode)
    if completed.returncode:
        print(completed.stderr.decode())
        raise SystemExit(completed.returncode)
