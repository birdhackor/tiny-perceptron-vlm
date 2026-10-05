"""Record actual exit status, command, logs and CPU environment for bounded checks."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parents[1]
env = dict(os.environ)
restrictions = {
    "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
env.update(restrictions)
records = []
for identifier, script in (("standalone", "original-fence.py"), ("variants", "check_variants.py")):
    argv = [str(ROOT / ".venv/bin/python"), str(ART / "code" / script)]
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=60, check=False)
    (ART / (identifier + ".stdout.txt")).write_bytes(proc.stdout)
    (ART / (identifier + ".stderr.txt")).write_bytes(proc.stderr)
    records.append({"id": identifier, "command_argv": argv, "command": shlex.join(argv),
                    "cwd": str(ROOT), "timeout_seconds": 60, "exit_code": proc.returncode,
                    "elapsed_seconds": time.monotonic()-start,
                    "stdout_sha256": hashlib.sha256(proc.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(proc.stderr).hexdigest(),
                    "source_sha256": hashlib.sha256((ART/"code"/script).read_bytes()).hexdigest()})
result = {"actual_commands": records, "cpu_offline_environment": restrictions,
          "scope": "Original fence without course bootstrap, then four local forward pairs, three missing-marker cases and one derivative pair. No model/data download or optimizer step."}
(ART / "execution-receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
if any(r["exit_code"] != 0 for r in records):
    raise SystemExit(1)
