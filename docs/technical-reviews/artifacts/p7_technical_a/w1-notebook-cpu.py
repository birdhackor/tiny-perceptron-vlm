"""Check unlocked W.1 environment/kernel/JupyterLab in the isolated clone."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

base = Path(__file__).parent
repo = Path("/tmp/p7-technical-a-w1-g4z24kt8/tiny-perceptron-vlm")
python = str(repo / ".venv/bin/python")
steps = []
for command in [[python, "scripts/check_env.py"], [python, "-m", "ipykernel", "install", "--sys-prefix", "--name", "tiny-perceptron", "--display-name", "Tiny Perceptron"]]:
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=60)
    steps.append({"command": command, "cwd": str(repo), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(command, "returned", result.returncode, result.stdout, result.stderr, flush=True)
command = [python, "-m", "jupyterlab", "notebooks", "--no-browser", "--ip=127.0.0.1", "--port=8778", "--allow-root", "--IdentityProvider.token=", "--ServerApp.password="]
log_path = base / "w1-jupyterlab-server.log"
with log_path.open("w") as log:
    server = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    status = None
    try:
        for _ in range(40):
            if server.poll() is not None:
                break
            try:
                response = urllib.request.urlopen("http://127.0.0.1:8778/lab", timeout=2)
                raw = response.read()
                status = {"http_status": response.status, "body_sha256": hashlib.sha256(raw).hexdigest(), "jupyterlab_markup": b"JupyterLab" in raw}
                break
            except OSError:
                time.sleep(0.2)
    finally:
        server.terminate()
        server.wait(timeout=10)
steps.append({"command": command, "cwd": str(repo), "functional_request": status, "terminated_own_server": True})
record = {"runner_command": ".venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/w1-notebook-cpu.py", "steps": steps, "scope": "Linux CPU environment/kernel registration and local JupyterLab HTTP only; no Windows, Google login, or unread notebook execution"}
p = base / "w1-notebook-cpu-result.json"
p.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print("Artifact", hashlib.sha256(p.read_bytes()).hexdigest(), status)
