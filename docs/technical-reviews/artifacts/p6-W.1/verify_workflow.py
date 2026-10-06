"""Bounded W.1 verification, using the existing interpreter without installing packages."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from ipykernel.kernelspec import get_kernel_dict

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
results = {"python": sys.version, "executable": sys.executable,
           "platform": sys.platform, "packages": {name: importlib.metadata.version(name)
           for name in ("torch", "jupyterlab", "ipykernel", "nbclient", "safetensors")}}

# Create an isolated evidence-only kernel descriptor; never register into the active environment.
kernel_root = OUT / "kernel-descriptors"
kernel_dir = kernel_root / "tiny-perceptron"
kernel_dir.mkdir(parents=True, exist_ok=True)
spec = get_kernel_dict()
spec["display_name"] = "Tiny Perceptron"
(kernel_dir / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n")
ksm = KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False)
results["kernel_descriptor"] = spec

def run_notebook(path, label, changed=False):
    nb = nbformat.read(path, as_version=4)
    if changed:
        replaced = 0
        for cell in nb.cells:
            if cell.cell_type == "code" and 'text = "貓看狗，狗看貓。"' in cell.source:
                cell.source = cell.source.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"')
                replaced += 1
        assert replaced == 1, replaced
    # Every NotebookClient owns a new independent kernel.
    km = AsyncKernelManager(kernel_name="tiny-perceptron", kernel_spec_manager=ksm)
    nb.cells.append(nbformat.v4.new_code_cell(
        "assert 'p6_namespace_sentinel' not in globals()\n"
        "p6_namespace_sentinel = True\n"
        "assert torch.cuda.is_available() is False\n"
        "print('p6_fresh_namespace_verified', torch.__version__, torch.tensor([1]).device.type)"))
    client = NotebookClient(nb, km=km, timeout=60, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            from jupyter_core.utils import run_sync
            run_sync(km.shutdown_kernel)(now=True)
    target = OUT / (label + ".ipynb")
    nbformat.write(nb, target)
    streams = [{"cell": i, "text": ''.join(o.get("text", "") for o in c.get("outputs", [])
                if o.get("output_type") == "stream")} for i, c in enumerate(nb.cells)
                if c.cell_type == "code"]
    expected = "鳥看狗，狗看鳥。" if changed else "貓看狗，狗看貓。"
    assert any(x["text"].splitlines()[-1:] == [expected] for x in streams), streams
    return {"input": str(path.relative_to(ROOT)), "changed": changed, "output": str(target.relative_to(ROOT)),
            "executed_code_cells": len(streams), "streams": streams, "expected_final_text": expected,
            "fresh_kernel": True, "device": "cpu"}

results["notebooks"] = [run_notebook(ROOT / "notebooks/01/1.1.ipynb", "current-original"),
                          run_notebook(ROOT / "notebooks/01/1.1.ipynb", "current-modified", True)]
published = OUT / "official/published-download.ipynb"
if published.exists():
    results["notebooks"].extend([run_notebook(published, "published-original"),
                                 run_notebook(published, "published-modified", True)])

# Directly verify the state dependency, rather than assuming a changed input updates derived data.
body = next(c.source for c in nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4).cells
            if c.cell_type == "code" and c.source.startswith('text = '))
namespace = {}
exec(body, namespace)
namespace["text"] = "鳥看狗，狗看鳥。"
assert namespace["restored"] == "貓看狗，狗看貓。"
exec(body.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"'), namespace)
assert namespace["restored"] == namespace["text"]
results["state_dependency"] = {"after_input_only": "貓看狗，狗看貓。",
                               "after_recompute": namespace["restored"]}

# Verify the module entry point and notebook serving behavior; stop only our own temporary server.
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
token = "p6-w1-local-functional-check"
server_log = OUT / "jupyter-server-log.txt"
command = [sys.executable, "-m", "jupyterlab", "notebooks", "--no-browser",
           "--ServerApp.allow_root=True", f"--ServerApp.port={port}", "--ServerApp.port_retries=0",
           "--ServerApp.ip=127.0.0.1", f"--IdentityProvider.token={token}"]
with server_log.open("w") as log:
    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 35
        while True:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/contents/01/1.1.ipynb",
                    headers={"Authorization": "token " + token})
                with urllib.request.urlopen(req, timeout=2) as r:
                    response = json.loads(r.read()); status = r.status
                break
            except Exception:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        assert status == 200 and response["type"] == "notebook"
        assert response["content"]["metadata"]["kernelspec"]["name"] == "tiny-perceptron"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/lab",
                    headers={"Authorization": "token " + token})
        with urllib.request.urlopen(req, timeout=5) as r:
            html = r.read(); lab_status = r.status
        assert lab_status == 200 and b"jupyter" in html.lower()
        results["jupyter_server"] = {"command": [x.replace(token, "REDACTED") for x in command],
            "notebook_api_status": status, "lab_status": lab_status, "served_notebook": response["path"],
            "root_directory": str(ROOT / "notebooks"), "terminated_after_check": True}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)
server_log.write_text(server_log.read_text().replace(token, "REDACTED"))

(OUT / "workflow-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
