"""有界 CPU 命令核對；這裡沒有組裝或執行本輪正式全 manifest 閘門。"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import nbformat
import torch

ROOT = Path.cwd()
SCRATCH = Path("/workspace/work/tokenizer-dynamic-20261027/validation-callback")
OUT = ROOT / "docs/technical-reviews/artifacts/tokenizer-20261027/validation-callback/command-checks.json"


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    records = []

    def run(argv, env=None, expected=0):
        p = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT)
        records.append({"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
        assert p.returncode == expected, records[-1]
        return p

    run([sys.executable, "scripts/build_course.py", "--check"])
    help_result = run([sys.executable, "-m", "ipykernel", "install", "--help"])
    assert "--sys-prefix" in help_result.stdout and "--name NAME" in help_result.stdout
    prefix = SCRATCH / "kernel-prefix"
    run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(prefix),
            "--name",
            "tiny-perceptron",
            "--display-name",
            "Tiny Perceptron",
        ]
    )
    env = {
        **os.environ,
        "JUPYTER_PATH": str(prefix / "share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", ""),
    }
    real_output = SCRATCH / "real-notebook-output"
    run(
        [
            sys.executable,
            "scripts/check_notebooks.py",
            "--mode",
            "kernel",
            "--workers",
            "1",
            "--lesson",
            "1.1",
            "--output",
            str(real_output),
        ],
        env=env,
    )
    real_summary = json.loads((real_output / "validation-kernel.json").read_bytes())
    assert real_summary["total"] == real_summary["passed"] == 1 and real_summary["failed"] == 0

    fake_root = SCRATCH / "synthetic-kernels"
    (fake_root / "notebooks").mkdir(parents=True, exist_ok=True)
    cells = {
        "a": "review_sentinel = 1847\nprint('first kernel: sentinel set')",
        "b": "assert 'review_sentinel' not in globals()\nimport torch\nprint('second kernel: clean namespace,', torch.tensor([1]).device)",
        "c": "raise RuntimeError('controlled validation failure')",
    }
    for name, source in cells.items():
        nbformat.write(
            nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]),
            fake_root / "notebooks" / (name + ".ipynb"),
        )
    synthetic_output = SCRATCH / "synthetic-output"
    driver = (
        "import importlib.util,sys;from pathlib import Path;"
        "s=importlib.util.spec_from_file_location('checked_notebook_runner','scripts/check_notebooks.py');"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"m.ROOT=Path({str(fake_root)!r});"
        f"sys.argv=['check_notebooks.py','--mode','kernel','--workers','1','--output',{str(synthetic_output)!r}];m.main()"
    )
    run([sys.executable, "-c", driver], env=env, expected=1)
    synthetic = json.loads((synthetic_output / "validation-kernel.json").read_bytes())
    assert (synthetic["total"], synthetic["passed"], synthetic["failed"]) == (3, 2, 1)
    assert len(synthetic["failures"]) == 1 and synthetic["failures"][0]["notebook"] == "notebooks/c.ipynb"
    assert "controlled validation failure" in synthetic["failures"][0]["error"]
    run(["git", "--version"])
    run(
        [
            "git",
            "-c",
            "user.name=Validation Reviewer",
            "-c",
            "user.email=reviewer@example.invalid",
            "var",
            "GIT_AUTHOR_IDENT",
        ]
    )
    fixture = SCRATCH / "controlled_pytest.py"
    fixture.write_text(
        "import pytest\n\ndef test_pass():\n    assert 2 + 2 == 4\n"
        "\ndef test_skip():\n    pytest.skip('controlled skip reason')\n"
        "\ndef test_fail():\n    assert False, 'controlled failure reason'\n"
    )
    pytest_output = run([sys.executable, "-m", "pytest", "-ra", str(fixture)], expected=1)
    assert all(
        term in pytest_output.stdout
        for term in ("1 failed", "1 passed", "1 skipped", "controlled skip reason", "controlled failure reason")
    )
    result = {
        "command": "uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/validation-callback/verify_commands.py",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "CPU",
            "platform": platform.platform(),
        },
        "records": records,
        "real_notebook_summary": real_summary,
        "synthetic_notebook_summary": synthetic,
        "result": "Build synchronization passes; real lesson 1.1 executes in a new kernel; controlled 3-notebook fixture confirms independent kernels, persisted outputs, failed/failures JSON and exit 1 on error; pytest -ra controlled fixture shows pass/fail/skip counts and reasons.",
        "scope": "The review kernel is installed only under scratch with --prefix, preserving the shared environment. Documented --sys-prefix was inspected through actual CLI help. No full current round gate, full Notebook suite, model training, public model download, Git add/commit/config write, Windows/Colab/MPS/CUDA execution, or publication was performed.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": result["result"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
