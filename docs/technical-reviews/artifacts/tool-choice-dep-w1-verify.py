"""Bounded independent CPU checks for W.1; never launches a browser or Colab."""

import copy
import hashlib
import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
ENV = Path("/tmp/tool-choice-dep-w1-venv")
ARTIFACT = ROOT / "docs/technical-reviews/artifacts/tool-choice-dep-w1-execution.json"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
env["PATH"] = str(ENV / "bin") + os.pathsep + env["PATH"]
commands = []


def run(args, *, additions=None):
    proc = subprocess.run(args, cwd=ROOT, env={**env, **(additions or {})}, text=True,
                          capture_output=True, timeout=90)
    commands.append({"command": args, "exit_code": proc.returncode,
                     "stdout": proc.stdout, "stderr": proc.stderr})
    assert proc.returncode == 0, commands[-1]
    return proc.stdout


result = {
    "scope": "Actual Linux CPU installation, kernel registration, authenticated JupyterLab HTTP, fresh kernels and current local site. No Colab runtime, browser UI, native Windows or macOS execution, GPU or training.",
    "initial_install": {
        "command": "UV_PROJECT_ENVIRONMENT=/tmp/tool-choice-dep-w1-venv uv sync --frozen --extra cpu --group notebook",
        "observed": "129 packages installed; torch 2.14.1+cpu, JupyterLab 4.6.4 and ipykernel 7.4.0. The preceding extra --offline attempt failed because jinja2 3.1.6 was absent from the cache; the documented online command succeeded.",
    },
    "environment": {"python": platform.python_version(), "python_executable": sys.executable,
                    "platform": platform.platform(), "torch": version("torch"), "device": "cpu",
                    "jupyterlab": version("jupyterlab"), "ipykernel": version("ipykernel"),
                    "nbclient": version("nbclient"), "uv": run(["uv", "--version"]).strip()},
}
run(["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook"],
    additions={"UV_PROJECT_ENVIRONMENT": str(ENV)})
run(["bash", "-c", "source /tmp/tool-choice-dep-w1-venv/bin/activate\npython scripts/check_env.py"])
run([str(ENV / "bin/python"), "-m", "ipykernel", "install", "--sys-prefix", "--name", "tiny-perceptron",
     "--display-name", "Tiny Perceptron"])
run([str(ENV / "bin/python"), "-m", "jupyterlab", "--version"])
run(["git", "--version"])
run(["git", "ls-remote", "https://github.com/birdhackor/tiny-perceptron-vlm.git", "HEAD"])
kernel = json.loads((ENV / "share/jupyter/kernels/tiny-perceptron/kernel.json").read_text())
assert kernel["display_name"] == "Tiny Perceptron" and kernel["argv"][0] == str(ENV / "bin/python")
result["registered_kernel"] = kernel

with socket.socket() as available:
    available.bind(("127.0.0.1", 0))
    port = available.getsockname()[1]
token = secrets.token_hex(24)
args = [str(ENV / "bin/jupyter"), "lab", "notebooks", "--no-browser", "--allow-root",
        f"--port={port}", "--ServerApp.port_retries=0", "--ServerApp.ip=127.0.0.1",
        f"--IdentityProvider.token={token}"]
with tempfile.TemporaryFile(mode="w+") as logfile:
    process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=logfile, stderr=logfile, text=True)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{port}/api/contents/01/1.1.ipynb",
                                                 headers={"Authorization": f"token {token}"})
                with urllib.request.urlopen(request, timeout=2) as response:
                    notebook = json.load(response)
                    status = response.status
                break
            except (urllib.error.URLError, TimeoutError):
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.1)
        assert status == 200 and notebook["type"] == "notebook"
        assert notebook["content"]["metadata"]["lesson_id"] == "1.1"
        routes = {}
        for route in ["/lab", "/files/01/1.1.ipynb"]:
            request = urllib.request.Request(f"http://127.0.0.1:{port}{route}",
                                             headers={"Authorization": f"token {token}"})
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read()
                routes[route] = {"status": response.status, "bytes": len(body)}
            if route.startswith("/files"):
                assert hashlib.sha256(body).hexdigest() == hashlib.sha256((ROOT / "notebooks/01/1.1.ipynb").read_bytes()).hexdigest()
        result["jupyterlab_http"] = {"command": [a.replace(token, "<ephemeral token not saved>") for a in args],
                                     "api_status": status, "routes": routes,
                                     "note": "Extra flags only make startup bounded and appropriate for this headless root container. No browser UI was exercised."}
    finally:
        process.terminate()
        process.wait(timeout=15)
        logfile.seek(0)
        result["jupyterlab_http"]["server_log"] = logfile.read().replace(token, "<ephemeral token not saved>")

original = nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4)
result["fresh_notebook_runs"] = []
for text in ["貓看狗，狗看貓。", "鳥看狗，狗看鳥。"]:
    notebook = copy.deepcopy(original)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace('text = "貓看狗，狗看貓。"', 'text = ' + repr(text))
    NotebookClient(notebook, kernel_name="tiny-perceptron", timeout=60,
                   resources={"metadata": {"path": str(ROOT / "notebooks/01")}}).execute()
    outputs = [{"cell": i, "execution_count": cell.execution_count,
                "stdout": "".join(output.get("text", "") for output in cell.outputs if output.output_type == "stream"),
                "output_types": [output.output_type for output in cell.outputs]}
               for i, cell in enumerate(notebook.cells) if cell.cell_type == "code"]
    assert all(x["execution_count"] is not None for x in outputs)
    assert outputs[1]["stdout"].splitlines()[-1] == text
    result["fresh_notebook_runs"].append({"input": text, "all_code_cells": outputs, "restored_exactly": True})

index = json.loads((ROOT / "course/lesson-index.json").read_text())
from scripts.export_course import checked_notebook

checked = 0
for item in index:
    src = json.loads((ROOT / item["notebook"]).read_text())
    path = ROOT / "outputs/tool-choice-site-kernels" / Path(item["notebook"]).relative_to("notebooks")
    checked_notebook(src, path)
    checked += 1
site = ROOT / "outputs/site"
home = (site / "index.html").read_text()
assert f"全部 {len(index)} 個小節" in home
downloads = []
for item in index:
    page = (site / f"{item['id']}.html").read_text()
    local = site / item["notebook"]
    assert local.read_bytes() == (ROOT / item["notebook"]).read_bytes()
    assert f'notebooks/{Path(item["notebook"]).relative_to("notebooks").as_posix()}' in page
    assert "https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/main/" + item["notebook"] in page
    downloads.append(item["id"])
result["current_export"] = {"executed_input": "outputs/tool-choice-site-kernels", "destination": "outputs/site",
                            "verified_current_executed_notebooks": checked, "homepage_count": len(index),
                            "colab_links_and_exact_notebook_downloads": len(downloads),
                            "note": "Static local site inspection; no rebuild, public-site deployment or Google sign-in test."}
result["commands"] = commands
ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"artifact": str(ARTIFACT), "fresh_runs": len(result["fresh_notebook_runs"]),
                  "current_executed_notebooks": checked, "site_sections": len(index)}, ensure_ascii=False))
