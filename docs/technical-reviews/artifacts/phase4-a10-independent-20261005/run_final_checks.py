"""Capture bounded checks after image-tool limitations were diagnosed."""

import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
commands = [
    ("local-rules", [str(ROOT / ".venv/bin/python"), str(HERE / "check_local_rules.py")]),
    ("paper-counts", [str(ROOT / ".venv/bin/python"), str(HERE / "check_paper_counts.py")]),
    ("figure-render", [str(ROOT / ".venv/bin/python"), str(HERE / "render_figure.py")]),
    ("font", ["fc-match", "Noto Sans CJK TC"]),
]
records = []
for label, command in commands:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=20, check=False)
    for stream in ("stdout", "stderr"):
        (HERE / (label + "." + stream + ".txt")).write_bytes(getattr(result, stream))
    records.append({"label": label, "command_argv": command, "exit_code": result.returncode, "stdout": label + ".stdout.txt", "stderr": label + ".stderr.txt"})
    print(label, "exit_code=" + str(result.returncode))
    if result.returncode:
        print(result.stderr.decode("utf-8", errors="replace"))
        break
record = {"environment": {"python": sys.version, "python_executable": sys.executable, "device": "CPU", "cwd": str(ROOT)}, "commands": records}
(HERE / "bounded-execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
raise SystemExit(any(x["exit_code"] for x in records))
