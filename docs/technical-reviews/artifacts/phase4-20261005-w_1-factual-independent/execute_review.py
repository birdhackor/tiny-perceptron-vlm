"""Bounded CPU checks written and executed by the fresh W.1 reviewer."""

import ast
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import nbformat
import torch
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def command(argv, name, env=None):
    completed = subprocess.run(
        argv, cwd=ROOT, env=env, capture_output=True, timeout=40, check=False
    )
    (BASE / (name + ".stdout.txt")).write_bytes(completed.stdout)
    (BASE / (name + ".stderr.txt")).write_bytes(completed.stderr)
    result = {
        "argv": argv,
        "cwd": str(ROOT),
        "exit_code": completed.returncode,
        "stdout": name + ".stdout.txt",
        "stdout_sha256": sha(BASE / (name + ".stdout.txt")),
        "stderr": name + ".stderr.txt",
        "stderr_sha256": sha(BASE / (name + ".stderr.txt")),
    }
    assert completed.returncode == 0, result
    print(name, "exit=0")
    return result


assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "device": "cpu",
    "torch_cuda_build": str(torch.version.cuda),
    "torch_cuda_available": str(torch.cuda.is_available()),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ["torch", "ipykernel", "jupyterlab", "jupyter_client", "nbclient", "nbformat"]
    },
}
save("environment.json", environment)
proof = {"environment": environment, "commands": [], "scope": {
    "full_installation": "Not executed; frozen/offline dry-run and pinned configuration inspected.",
    "clone": "Not executed; git version and read-only remote HEAD check replace network checkout.",
    "windows": "Not executed on Windows; Python venv and Microsoft execution-policy contracts inspected.",
    "colab": "Cloud runtime not launched; current notebook bootstrap and official FAQ inspected.",
    "gpu_training_new_data_or_models": "None.",
}}
section = (BASE / "section.md").read_bytes()
fences = []
for i, match in enumerate(re.finditer(rb"```([^\n]+)\n(.*?)```", section, re.S), 1):
    path = BASE / f"original-fence-{i}.{match[1].decode()}"
    path.write_bytes(match[2])
    fences.append({"number": i, "language": match[1].decode(), "path": path.name,
                   "sha256": sha(path), "execution": "contract + bounded alternatives"})
proof["fences"] = fences

for argv, name in [
    (["uv", "--version"], "uv-version"),
    (["git", "--version"], "git-version"),
    (["git", "ls-remote", "https://github.com/birdhackor/tiny-perceptron-vlm.git", "HEAD"], "git-remote"),
    (["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run", "--offline"], "uv-dry-run"),
    (["bash", "--noprofile", "--norc", "-c", "source .venv/bin/activate\npython -c 'import sys; print(sys.executable)'\npython scripts/check_env.py"], "activation-check-env"),
    ([str(PYTHON), "scripts/check_env.py"], "direct-check-env"),
]:
    proof["commands"].append(command(argv, name))
assert str(PYTHON) in (BASE / "activation-check-env.stdout.txt").read_text()
assert "OK（cpu）" in (BASE / "activation-check-env.stdout.txt").read_text()
assert "未安裝" not in (BASE / "activation-check-env.stdout.txt").read_text()

prefix = BASE / "kernel-prefix"
env = dict(os.environ)
env.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "JUPYTER_RUNTIME_DIR": str(BASE / "runtime"),
    "JUPYTER_CONFIG_DIR": str(BASE / "jupyter-config"),
    "IPYTHONDIR": str(BASE / "ipython"),
    "MPLCONFIGDIR": str(BASE / "mpl-cache"),
})
proof["commands"].append(command([
    str(PYTHON), "-m", "ipykernel", "install", "--prefix", str(prefix),
    "--name", "tiny-perceptron", "--display-name", "Tiny Perceptron",
], "isolated-kernel-install", env))
ksdir = prefix / "share/jupyter/kernels"
ksm = KernelSpecManager(kernel_dirs=[str(ksdir)])
ks = ksm.get_kernel_spec("tiny-perceptron")
assert ks.display_name == "Tiny Perceptron" and ks.argv[0] == str(PYTHON)
proof["kernelspec"] = {"path": str(Path(ks.resource_dir).relative_to(BASE)),
                       "display_name": ks.display_name, "argv": ks.argv}

def fresh_notebook(name, changed):
    nb = nbformat.read(BASE / "frozen/notebooks/01/1.1.ipynb", as_version=4)
    if changed:
        nb.cells[5].source = nb.cells[5].source.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"')
    km = KernelManager(kernel_name="tiny-perceptron", kernel_spec_manager=ksm)
    client = NotebookClient(nb, km=km, timeout=40, resources={"metadata": {"path": str(ROOT)}})
    client.execute(cwd=str(ROOT), env=env)
    nbformat.write(nb, BASE / f"{name}.executed.ipynb")
    stdout = "".join(o.get("text", "") for o in nb.cells[5].outputs if o.output_type == "stream")
    (BASE / f"{name}.stdout.txt").write_text(stdout)
    expected = "鳥看狗，狗看鳥。" if changed else "貓看狗，狗看貓。"
    assert stdout.strip().splitlines()[-1] == expected
    result = {"changed_input_only": changed, "executed_code_cells": [i for i,c in enumerate(nb.cells) if c.cell_type == "code"],
              "last_text_output": expected, "stdout_path": f"{name}.stdout.txt",
              "executed_notebook": f"{name}.executed.ipynb", "fresh_kernel": True}
    print(name, expected)
    return result

proof["baseline"] = fresh_notebook("baseline", False)
proof["changed"] = fresh_notebook("changed", True)

km = KernelManager(kernel_name="tiny-perceptron", kernel_spec_manager=ksm)
km.start_kernel(cwd=str(ROOT), env=env)
kc = km.client()
kc.start_channels()
try:
    kc.wait_for_ready(timeout=20)
    def execute(code):
        mid = kc.execute(code)
        outputs = []
        while True:
            message = kc.get_iopub_msg(timeout=20)
            if message["parent_header"].get("msg_id") != mid:
                continue
            if message["msg_type"] == "error":
                raise RuntimeError(message["content"])
            if message["msg_type"] == "stream":
                outputs.append(message["content"]["text"])
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                return "".join(outputs)
    original = (BASE / "notebook-cell-5.py").read_text()
    initial = execute(original)
    change_only = execute('text = "鳥看狗，狗看鳥。"\nprint(text)')
    old_derived = execute("print(restored)")
    recomputed = execute(original.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"'))
    assert old_derived.strip() == "貓看狗，狗看貓。"
    assert recomputed.strip().splitlines()[-1] == "鳥看狗，狗看鳥。"
    proof["state_probe"] = {"initial": initial, "input_changed_without_recomputing": change_only,
                            "old_derived": old_derived, "after_recomputing": recomputed}
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)

try:
    compile("for item in [1]:\nprint(item)\n", "indentation-probe", "exec")
except IndentationError as error:
    proof["indentation_probe"] = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("Expected missing-indentation failure")

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
argv = [str(ROOT / ".venv/bin/jupyter"), "lab", "notebooks", "--no-browser", "--ip=127.0.0.1",
        f"--port={port}", "--ServerApp.port_retries=0", "--ServerApp.allow_root=True", "--IdentityProvider.token="]
log = BASE / "jupyter-lab.stdout-stderr.txt"
with log.open("wb") as output:
    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("JupyterLab exited: " + log.read_text())
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/lab", timeout=1) as response:
                    raw = response.read()
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise TimeoutError("JupyterLab did not respond")
        assert b"JupyterLab" in raw
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/contents/01/1.1.ipynb?content=0", timeout=5) as response:
            content = json.load(response)
        assert content["type"] == "notebook" and content["path"] == "01/1.1.ipynb"
        proof["jupyter_lab"] = {"argv": argv, "root_directory": str(ROOT / "notebooks"),
                                "lab_http_status": 200, "api_notebook": {k: content[k] for k in ["name", "path", "type"]},
                                "stdout_stderr": log.name, "external_access": False}
        print("JupyterLab served /lab and 01/1.1.ipynb: HTTP 200")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
save("cpu-proof.json", proof)
print("All bounded CPU assertions passed")
