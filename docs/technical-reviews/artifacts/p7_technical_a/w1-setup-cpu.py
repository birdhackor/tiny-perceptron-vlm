"""Execute the unlocked W.1 local setup recipe in an isolated temporary clone."""
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

evidence = Path(__file__).parent
work = Path(tempfile.mkdtemp(prefix="p7-technical-a-w1-", dir="/tmp"))
steps = []
for command, cwd in [
    (["git", "clone", "https://github.com/birdhackor/tiny-perceptron-vlm.git"], work),
    (["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook"], work / "tiny-perceptron-vlm"),
]:
    print("Running", command, "in", cwd, flush=True)
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=240)
        steps.append({"command": command, "cwd": str(cwd), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    except subprocess.TimeoutExpired as error:
        steps.append({"command": command, "cwd": str(cwd), "timeout": 240, "stdout": str(error.stdout), "stderr": str(error.stderr)})
        break
    print("Returned", result.returncode, flush=True)
    if result.returncode:
        break
record = {"runner_command": ".venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/w1-setup-cpu.py", "environment": {"python": sys.version.split()[0], "device": platform.machine() + " CPU"}, "steps": steps}
path = evidence / "w1-setup-cpu-result.json"
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print("Artifact", path, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
sys.exit(0 if len(steps) == 2 and all(step.get("returncode") == 0 for step in steps) else 1)
