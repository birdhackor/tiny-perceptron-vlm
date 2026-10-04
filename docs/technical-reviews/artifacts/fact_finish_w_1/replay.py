"""Read-only CPU checks; use existing dependencies and existing kernelspec."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PYTHON = ROOT / ".venv/bin/python"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_snapshot():
    return {
        "protected_files": {name: digest(ROOT / name) for name in ["pyproject.toml", "uv.lock", ".python-version"]},
        "installed_distributions": sorted(
            (package.metadata["Name"], package.version) for package in importlib.metadata.distributions()
        ),
        "kernel_json": digest(ROOT / ".venv/share/jupyter/kernels/tiny-perceptron/kernel.json"),
    }


def run(command, timeout=35):
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
    record = {
        "argv": [str(value) for value in command],
        "cwd": str(ROOT),
        "started_at": started,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def execute_notebook(bird=False):
    notebook = nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4)
    if bird:
        count = 0
        for cell in notebook.cells:
            if cell.cell_type == "code" and 'text = "貓看狗，狗看貓。"' in cell.source:
                cell.source = cell.source.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"')
                count += 1
        assert count == 1
    client = NotebookClient(
        notebook,
        kernel_name="tiny-perceptron",
        timeout=40,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    target = OUT / ("bird-executed.ipynb" if bird else "original-executed.ipynb")
    nbformat.write(notebook, target)
    streams = [
        output.text
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream"
    ]
    expected = "鳥看狗，狗看鳥。" if bird else "貓看狗，狗看貓。"
    assert streams[-1].strip().splitlines()[-1] == expected, streams
    return {
        "input": "notebooks/01/1.1.ipynb",
        "fresh_kernel": True,
        "kernel": "tiny-perceptron",
        "bird_edit": bird,
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "execution_counts": [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"],
        "streams": streams,
        "expected_last_line": expected,
        "artifact": str(target.relative_to(ROOT)),
    }


def state_probe():
    manager = KernelManager(kernel_name="tiny-perceptron")
    manager.start_kernel(cwd=str(ROOT))
    client = manager.blocking_client()
    client.start_channels()
    client.wait_for_ready(timeout=30)

    def execute(code):
        message_id = client.execute(code)
        outputs = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            message = client.get_iopub_msg(timeout=30)
            if message["parent_header"].get("msg_id") != message_id:
                continue
            kind = message["header"]["msg_type"]
            if kind == "stream":
                outputs.append(message["content"]["text"])
            elif kind == "error":
                raise AssertionError(message["content"])
            elif kind == "status" and message["content"]["execution_state"] == "idle":
                break
        return {"code": code, "stdout": "".join(outputs)}

    try:
        notebook = nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4)
        encoding = next(cell.source for cell in notebook.cells if cell.cell_type == "code" and "text =" in cell.source)
        original = execute(encoding)
        stale = execute('text = "鳥看狗，狗看鳥。"\nprint(restored)')
        assert stale["stdout"].strip() == "貓看狗，狗看貓。"
        manager.restart_kernel(now=True)
        client.wait_for_ready(timeout=30)
        cleared = execute('print("restored" in globals())')
        assert cleared["stdout"].strip() == "False"
        rerun = execute(encoding.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"'))
        assert rerun["stdout"].strip().splitlines()[-1] == "鳥看狗，狗看鳥。"
        return {"original": original, "stale": stale, "after_restart": cleared, "rerun": rerun}
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


def bootstrap_dispatch_probe():
    notebook = nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4)
    bootstrap = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    google.colab = colab
    original_path = sys.path.copy()
    with (
        patch.dict(sys.modules, {"google": google, "google.colab": colab}),
        patch("subprocess.run") as dispatched,
        patch("os.chdir") as changed_directory,
    ):
        exec(compile(bootstrap, "notebooks/01/1.1.ipynb:bootstrap", "exec"), {})
        calls = [
            {
                "argv": call.args[0],
                "check": call.kwargs.get("check"),
                "lfs_skip_smudge": call.kwargs.get("env", {}).get("GIT_LFS_SKIP_SMUDGE"),
            }
            for call in dispatched.call_args_list
        ]
        directories = [str(call.args[0]) for call in changed_directory.call_args_list]
    sys.path[:] = original_path
    assert calls[0]["argv"] == [
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/birdhackor/tiny-perceptron-vlm.git",
        "/content/tiny-perceptron-vlm",
    ]
    assert calls[1]["argv"] == [sys.executable, "-m", "pip", "install", "--quiet", "-e", "/content/tiny-perceptron-vlm"]
    assert all(call["check"] for call in calls)
    return {"mocked_dispatch_only": True, "cloud_or_install_execution": False, "calls": calls, "chdir": directories}


def main():
    before = environment_snapshot()
    commands = [
        ["git", "--version"],
        ["uv", "--version"],
        [str(PYTHON), "--version"],
        ["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run", "--offline"],
        [str(PYTHON), "scripts/check_env.py"],
        [str(PYTHON), "-m", "ipykernel", "install", "--help"],
        [str(PYTHON), "-m", "jupyter", "kernelspec", "list", "--json"],
        [str(PYTHON), "-m", "jupyterlab", "notebooks", "--show-config-json"],
        ["bash", "-c", "source .venv/bin/activate; python -c 'import sys; print(sys.executable); print(sys.prefix)'"],
        ["git", "ls-remote", "https://github.com/birdhackor/tiny-perceptron-vlm.git", "HEAD"],
    ]
    results = [run(command) for command in commands]
    notebooks = [execute_notebook(), execute_notebook(bird=True)]
    state = state_probe()
    dispatch = bootstrap_dispatch_probe()
    warmups = []
    for lesson in ["W.2", "W.3"]:
        source = (OUT / f"{lesson}-source.txt").read_text()
        code = re.search(r"```python\n(.*?)```", source, re.S)[1]
        result = run([str(PYTHON), "-c", code])
        assert result["exit_code"] == 0
        warmups.append({"lesson": lesson, **result})
    after = environment_snapshot()
    record = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_w_1/replay.py",
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "device": "CPU; no training",
            "package_versions": {
                name: importlib.metadata.version(name) for name in ["torch", "ipykernel", "jupyterlab", "nbclient"]
            },
        },
        "commands": results,
        "notebooks": notebooks,
        "state_probe": state,
        "colab_dispatch_probe": dispatch,
        "warmups": warmups,
        "shared_environment_unchanged": before == after,
        "protected_files": before["protected_files"],
        "colab_runtime_executed": False,
        "installation_executed": False,
        "windows_runtime_executed": False,
    }
    (OUT / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    assert all(result["exit_code"] == 0 for result in results)
    assert before == after
    print(
        json.dumps(
            {"notebooks": notebooks, "state_probe": state, "environment_unchanged": before == after}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
