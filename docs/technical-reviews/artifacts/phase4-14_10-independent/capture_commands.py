"""Capture exact local CPU and SVG render commands and their results."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[4]
proof = Path(__file__).resolve().parent
math_command = [str(repo / ".venv/bin/python"), str(proof / "check_math.py")]
figure_command = [
    "/usr/bin/inkscape", str(proof / "frozen-figure.svg"), "--export-type=png",
    "--export-filename=" + str(proof / "figure-render.png"),
]
records = []
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
for name, command in [("math", math_command), ("figure-render", figure_command)]:
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, timeout=30, check=False)
    stdout_name = "math.stdout.json" if name == "math" else "figure-render.stdout.txt"
    stderr_name = name + ".stderr.txt"
    (proof / stdout_name).write_bytes(result.stdout)
    (proof / stderr_name).write_bytes(result.stderr)
    records.append({
        "name": name, "command_argv": command, "cwd": str(repo),
        "returncode": result.returncode, "timeout_seconds": 30,
        "stdout": stdout_name, "stderr": stderr_name,
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
    })
    if result.returncode:
        (proof / "command-records.json").write_text(json.dumps(records, indent=2)+"\n")
        raise SystemExit(result.returncode)
inkscape = subprocess.run(["/usr/bin/inkscape", "--version"], capture_output=True, text=True, check=True).stdout.strip()
environment = json.loads((proof / "math.stdout.json").read_text())["environment"]
environment["inkscape"] = inkscape
(proof / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
(proof / "command-records.json").write_text(json.dumps(records, indent=2)+"\n")
print(json.dumps({"runs": [{"name": r["name"], "returncode": r["returncode"]} for r in records], "environment": environment}, indent=2))
