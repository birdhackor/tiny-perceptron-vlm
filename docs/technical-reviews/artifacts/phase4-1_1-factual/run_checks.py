"""Capture actual commands, stdout, stderr, environment, return codes and hashes."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
records = []
commands = {
    "original": [str(ROOT / ".venv/bin/python"), str(OUT / "original.py")],
    "probe": [str(ROOT / ".venv/bin/python"), str(OUT / "probe.py")],
    "figure-render": ["/usr/bin/chromium", "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--user-data-dir=" + str(OUT / "chromium-profile"), "--screenshot=" + str(OUT / "figure.png"), "--window-size=640,710", "file://" + str(OUT / "figure.svg")],
}
for name, argv in commands.items():
    run = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=30)
    (OUT / f"{name}-stdout.txt").write_bytes(run.stdout)
    (OUT / f"{name}-stderr.txt").write_bytes(run.stderr)
    record = {"command":shlex.join(argv),"command_argv":argv,"cwd":str(ROOT),"exit_code":run.returncode,"environment":{"python":sys.version,"python_executable":str(ROOT / ".venv/bin/python"),"device":"CPU; CUDA_VISIBLE_DEVICES empty; Chromium --disable-gpu","platform":platform.platform(),"offline_python":"HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1","chromium":subprocess.run(["/usr/bin/chromium","--version"],capture_output=True,text=True,check=True).stdout.strip()},"stdout_sha256":hashlib.sha256(run.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(run.stderr).hexdigest()}
    records.append(record)
    print(json.dumps({"name":name,"command":record["command"],"exit_code":run.returncode,"stdout":run.stdout.decode("utf-8")},ensure_ascii=False))
    if run.returncode:
        print(run.stderr.decode("utf-8"),file=sys.stderr)
        raise SystemExit(run.returncode)
(OUT / "execution.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
