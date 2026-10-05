"""Run original-preserving exercise variations and a bounded independent probe."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH") if key in os.environ}
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
original = (BASE / "original-fence-1.py").read_text()
bootstrap = (BASE / "original-bootstrap.py").read_text()
missing = original.replace("+ [tok.eos_id, tok.assistant_id]", "+ [tok.eos_id]")
assert missing != original and missing.endswith("assert prompt[-1] == tok.assistant_id\n")
for name, text in [("exercise-missing", missing), ("exercise-restored", original)]:
    (BASE / f"{name}-fence.py").write_text(text)
    (BASE / f"{name}-runner.py").write_text(bootstrap + "\n" + text)
results = {}
for name, script, expected in [("exercise-missing", "exercise-missing-runner.py", 1),
                               ("exercise-restored", "exercise-restored-runner.py", 0),
                               ("contract", "contract_probe.py", 0)]:
    argv = [str(ROOT / ".venv/bin/python"), "-I", str(BASE / script)]
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=20)
    (BASE / f"{name}.stdout.txt").write_bytes(proc.stdout)
    (BASE / f"{name}.stderr.txt").write_bytes(proc.stderr)
    results[name] = {"command_argv": argv, "cwd": str(ROOT), "timeout_seconds": 20,
                     "exit_code": proc.returncode, "expected_exit_code": expected,
                     "expected_outcome_observed": proc.returncode == expected,
                     "code_sha256": hashlib.sha256((BASE / script).read_bytes()).hexdigest(),
                     "stdout_sha256": hashlib.sha256(proc.stdout).hexdigest(),
                     "stderr_sha256": hashlib.sha256(proc.stderr).hexdigest()}
    assert proc.returncode == expected, proc.stderr.decode()
assert "AssertionError" in (BASE / "exercise-missing.stderr.txt").read_text()
assert (BASE / "exercise-missing.stdout.txt").read_text().strip().endswith("最後邊界 2")
assert (BASE / "exercise-restored-fence.py").read_bytes() == (BASE / "original-fence-1.py").read_bytes()
(BASE / "cpu-checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({name: item["exit_code"] for name, item in results.items()}))
