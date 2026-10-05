"""Record commands and outputs for the exact fence and bounded CPU checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
shutil.copyfile(ROOT / "tiny_perceptron/data.py", BASE / "code/tiny_perceptron--data.py")
fence = (BASE / "fence-1.py").read_bytes()
assert hashlib.sha256(fence).hexdigest() == "95c859a9cb7e437b21b87d212887d0228b1357e6027ac4910638f416fa677109"
assert fence.count(b"bits=4") == 1
(BASE / "fence-1-bits8.py").write_bytes(fence.replace(b"bits=4", b"bits=8"))
env = os.environ.copy()
env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           PYTHONDONTWRITEBYTECODE="1")
jobs = [("original-fence", "fence-1.py"), ("bits8-variant", "fence-1-bits8.py"),
        ("bounded-cpu-checks", "check_structures.py")]
records = []
for name, script in jobs:
    argv = [str(ROOT / ".venv/bin/python"), str(BASE / script)]
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=45, check=False)
    (BASE / f"{name}.stdout.txt").write_bytes(result.stdout)
    (BASE / f"{name}.stderr.txt").write_bytes(result.stderr)
    records.append({"name": name, "command_argv": argv, "cwd": str(ROOT), "timeout_seconds": 45,
                    "exit_code": result.returncode, "script_sha256": hashlib.sha256((BASE / script).read_bytes()).hexdigest(),
                    "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()})
    print(name, "exit", result.returncode)
    print(result.stdout.decode(), end="")
    if result.stderr:
        print(result.stderr.decode(), end="", file=sys.stderr)
    if result.returncode:
        (BASE / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        raise SystemExit(result.returncode)
    if name == "original-fence":
        assert result.stdout.decode() == "教師結構 bytes 67840\n學生結構 bytes 24416\n量化學生結構 bytes 15680\n量化學生/教師比例 0.2311\n"
    if name == "bits8-variant":
        assert result.stdout.decode() == "教師結構 bytes 67840\n學生結構 bytes 24416\n量化學生結構 bytes 17120\n量化學生/教師比例 0.2524\n"
(BASE / "commands.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
