"""Persist real bounded subprocess commands, output, device/version and exit status."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ARTIFACT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
version_cmd = [str(ROOT / ".venv/bin/python"), "-c",
               "import sys,torch,json; print(json.dumps(dict(python=sys.version,torch=str(torch.__version__),torch_git_version=str(torch.version.git_version),device='cpu',cuda_build=str(torch.version.cuda),cuda_available=str(torch.cuda.is_available()))))"]
version = subprocess.run(version_cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
assert version.returncode == 0
versions = json.loads(version.stdout)
receipts = []
for script in ["prepare-selected-inputs.py", "original-fence.py", "bounded-variants.py", "replay-encoders-logits.py"]:
    path = ARTIFACT / "code" / script
    cmd = [str(ROOT / ".venv/bin/python"), str(path)]
    started = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=45)
    name = path.stem
    stdout = ARTIFACT / "execution" / (name + ".stdout.txt")
    stderr = ARTIFACT / "execution" / (name + ".stderr.txt")
    stdout.write_bytes(r.stdout)
    stderr.write_bytes(r.stderr)
    receipts.append({"name": name, "argv": cmd, "cwd": str(ROOT), "timeout_seconds": 45,
                     "elapsed_seconds": time.perf_counter() - started, "exit_code": r.returncode,
                     "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "environment": versions, "offline_overrides": {k: env[k] for k in
                         ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
                     "stdout": str(stdout.relative_to(ROOT)), "stdout_sha256": hashlib.sha256(r.stdout).hexdigest(),
                     "stderr": str(stderr.relative_to(ROOT)), "stderr_sha256": hashlib.sha256(r.stderr).hexdigest()})
    (ARTIFACT / "execution/run-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
    print(script, "exit", r.returncode)
    print(r.stdout.decode())
    if r.returncode:
        print(r.stderr.decode())
        raise SystemExit(r.returncode)
