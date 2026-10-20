import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
record = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
          "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU"}, "steps": []}


def run(command, env, timeout):
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        timed_out = True
    result = {"command": [str(c) for c in command], "exit_code": proc.returncode, "timed_out": timed_out,
              "stdout": out, "stderr": err, "elapsed_seconds": time.monotonic() - started}
    record["steps"].append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return proc.returncode == 0


with tempfile.TemporaryDirectory(prefix="rebuild-", dir=ART) as temp:
    work = Path(temp)
    env = dict(os.environ, TMPDIR=str(work), PYTHONDONTWRITEBYTECODE="1", PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    py = work / "env/bin/python"
    if run([sys.executable, "-m", "venv", work / "env"], env, 45):
        if run([py, "-m", "pip", "install", "--no-cache-dir", "--no-compile", "Pillow==12.3.0", "h5py==3.16.0", "numpy==2.5.3", "soundfile==0.13.1"], env, 100):
            if run([py, ROOT / "scripts/build_natural_v4_assets.py", "--reconstruct", "--data-root", work / "data",
                    "--assets", work / "archives", "--manifest", work / "manifest.json", "--verify-manifest", ROOT / "docs/natural-assistant/v4/manifest.json"], env, 240):
                raw = (work / "manifest.json").read_bytes()
                record["reconstructed_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
                record["manifest_byte_equal"] = raw == (ROOT / "docs/natural-assistant/v4/manifest.json").read_bytes()
    record["temporary_files_removed_after_probe"] = True
output = ART / "rebuild-result.json"
output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
