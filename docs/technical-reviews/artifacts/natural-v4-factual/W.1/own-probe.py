"""Bounded W.1 review: parse commands, check CPU, execute actual small cells.

No installation, activation, kernelspec registration, Git mutation, or training.
"""
from pathlib import Path
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys

from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
RESEARCH = ROOT / "outputs/natural-v4/factual-research/W.1"
os.chdir(ROOT)
environment = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "os": platform.platform(),
    "device": "cpu",
}
for package in ["torch", "ipykernel", "IPython", "jupyterlab", "jupyter-client"]:
    environment[package] = importlib.metadata.version(package)
commands = [
    ["git", "--version"],
    ["git", "clone", "-h"],
    ["uv", "--version"],
    ["uv", "sync", "--help"],
    [sys.executable, "-m", "ipykernel", "install", "--help"],
    [sys.executable, "-m", "jupyterlab", "--help"],
    [sys.executable, "scripts/check_env.py"],
]
results = []
for index, command in enumerate(commands):
    run = subprocess.run(command, text=True, capture_output=True, timeout=30)
    results.append({"command": command, "exit_code": run.returncode,
                    "stdout": run.stdout, "stderr": run.stderr})
    print("COMMAND", index, command, "EXIT", run.returncode)
    if index in [0, 2, 6]:
        print(run.stdout)
    expected_status = 129 if index == 1 else 0
    assert run.returncode == expected_status, results[-1]
assert all(flag in results[3]["stdout"] for flag in ["--frozen", "--extra", "--group"])
assert all(flag in results[4]["stdout"] for flag in ["--sys-prefix", "--name", "--display-name"])
assert "OK（cpu）" in results[6]["stdout"]
assert "未安裝" not in results[6]["stdout"]

notebook = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_text())
code = [(i, "".join(c["source"])) for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"]
assert [i for i, source in code] == [2, 3, 5]
assert notebook["metadata"]["kernelspec"]["name"] == "tiny-perceptron"

# Modify only this in-memory KernelSpec; do not register or write it.
manager = KernelManager(kernel_name="python3", connection_file=str(RESEARCH / "own-kernel-connection.json"))
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
manager.start_kernel(cwd=str(ROOT), env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
client = manager.blocking_client()
client.start_channels()
client.wait_for_ready(timeout=30)
messages = []

def execute(label, source):
    cell_messages = []
    def hook(message):
        kind = message["header"]["msg_type"]
        content = message["content"]
        if kind == "stream":
            cell_messages.append({"type": kind, "name": content["name"], "text": content["text"]})
        elif kind == "error":
            cell_messages.append({"type": kind, "ename": content["ename"], "evalue": content["evalue"], "traceback": content["traceback"]})
        elif kind in ["display_data", "execute_result"]:
            cell_messages.append({"type": kind, "mime_types": list(content["data"])})
    reply = client.execute_interactive(source, output_hook=hook, timeout=30)
    record = {"label": label, "source": source, "reply_status": reply["content"]["status"], "messages": cell_messages}
    messages.append(record)
    assert record["reply_status"] == "ok", record
    return "".join(m["text"] for m in cell_messages if m["type"] == "stream")

try:
    baseline = []
    for index, source in code:
        baseline.append(execute(f"baseline-cell-{index}", source))
    assert baseline[1].splitlines()[-1] == "貓看狗，狗看貓。"
    stale = execute("state-before-restart", 'print(restored)\nprint(__import__("sys").executable)\nprint(__import__("torch").__version__)')
    assert stale.splitlines()[0] == "貓看狗，狗看貓。"
    assert stale.splitlines()[2] == "2.14.1+cpu"
    manager.restart_kernel(now=True)
    client.wait_for_ready(timeout=30)
    fresh = execute("fresh-state", 'print("restored" in globals())')
    assert fresh.strip() == "False"
    changed = []
    for index, source in code:
        edited = source.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"')
        changed.append(execute(f"changed-after-restart-cell-{index}", edited))
    assert changed[1].splitlines()[-1] == "鳥看狗，狗看鳥。"
    print("BASELINE", baseline[1])
    print("FRESH STATE", fresh.strip())
    print("CHANGED", changed[1])
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)

summary = {
    "environment": environment,
    "commands": results,
    "kernel_cells": messages,
    "denominators": {"notebooks": 1, "code_cells_per_full_run": 3,
                     "full_small_notebook_runs": 2, "kernel_restarts": 1,
                     "input_characters_per_sentence": 8, "unique_characters_per_sentence": 5,
                     "seed_in_bootstrap": 42, "training_updates": 0, "device": "cpu"},
    "scope": "Linux .venv kernel only; Windows commands checked by original documentation and portable argv; Colab not authenticated or executed."
}
(OUT / "own-probe-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print("ALL BOUNDED CHECKS COMPLETED")
