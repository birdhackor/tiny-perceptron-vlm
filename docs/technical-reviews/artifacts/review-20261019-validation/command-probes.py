"""以小型 Notebook 與 pytest 例子驗證輸出契約，全部暫存寫在自己的目錄。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
scratch = HERE / "probe-work"
scratch.mkdir()
environment = dict(os.environ)
environment.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   IPYTHONDIR=str(scratch / "ipython"), JUPYTER_RUNTIME_DIR=str(scratch / "runtime"),
                   TMPDIR=str(scratch / "tmp"))
(scratch / "tmp").mkdir()
os.environ.update(environment)
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-validation/command-probes.py",
          "environment": {"python": platform.python_version(), "device": "cpu"}}


def run(argv):
    process = subprocess.run(argv, cwd=ROOT, env=environment, text=True, capture_output=True)
    return {"argv": argv, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}


try:
    if importlib.util.find_spec("nbformat") is None:
        result["notebook_execution"] = {
            "status": "unavailable", "reason": "本機 .venv 未安裝 notebook dependency group；不變更專案環境。",
            "observed": "ModuleNotFoundError: No module named 'nbformat'"}
        test = scratch / "test_probe.py"
        test.write_text("import pytest\n\ndef test_pass():\n    assert True\n\ndef test_skip():\n    pytest.skip('探針跳過')\n\ndef test_fail():\n    raise RuntimeError('探針失敗')\n")
        result["pytest_probe_source"] = test.read_text()
        result["pytest"] = run([sys.executable, "-m", "pytest", "-ra", "-q", "-p", "no:cacheprovider", str(test)])
        assert result["pytest"]["exit_code"] == 1
        assert "探針跳過" in result["pytest"]["stdout"] and "探針失敗" in result["pytest"]["stdout"]
        result["scope"] = "已執行 pytest 摘要探針；未執行 Jupyter kernel 探針。刻意失敗用於檢查回報。"
        (HERE / "command-probes-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0)
    import nbformat
    import torch
    result["environment"]["torch"] = torch.__version__
    result["environment"]["nbformat"] = nbformat.__version__
    prefix = scratch / "registration"
    result["kernel_install"] = run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(prefix),
                                     "--name", "tiny-perceptron", "--display-name", "Tiny Perceptron"])
    assert result["kernel_install"]["exit_code"] == 0
    environment["JUPYTER_PATH"] = str(prefix / "share/jupyter")
    os.environ["JUPYTER_PATH"] = environment["JUPYTER_PATH"]
    result["kernel_spec"] = json.loads((prefix / "share/jupyter/kernels/tiny-perceptron/kernel.json").read_text())
    notebooks = scratch / "notebooks"
    notebooks.mkdir()
    for name in ["one", "two"]:
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "import os, torch\ncount = globals().get('count', 0) + 1\nprint(count, os.getpid(), torch.tensor(1).device)")])
        nbformat.write(notebook, notebooks / (name + ".ipynb"))
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise RuntimeError('探針失敗')")])
    nbformat.write(notebook, notebooks / "failure.ipynb")
    path = ROOT / "scripts/check_notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = scratch
    sys.argv = [str(path), "--mode", "kernel", "--workers", "1", "--output", str(scratch / "executed")]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            module.main()
            exit_code = 0
        except SystemExit as error:
            exit_code = error.code
    summary = json.loads((scratch / "executed/validation-kernel.json").read_text())
    result["notebook_execution"] = {"invocation": sys.argv, "exit_code": exit_code,
                                     "stdout": captured.getvalue(), "summary": summary, "outputs": {}}
    for name in ["one", "two"]:
        executed = nbformat.read(scratch / "executed" / (name + ".ipynb"), as_version=4)
        result["notebook_execution"]["outputs"][name] = [dict(o) for o in executed.cells[0].outputs]
    assert summary["total"] == 3 and summary["passed"] == 2 and summary["failed"] == 1 and exit_code == 1
    texts = [v[0]["text"].strip().split() for v in result["notebook_execution"]["outputs"].values()]
    assert all(t[0] == "1" and t[2] == "cpu" for t in texts)
    assert texts[0][1] != texts[1][1]
    test = scratch / "test_probe.py"
    test.write_text("import pytest\n\ndef test_pass():\n    assert True\n\ndef test_skip():\n    pytest.skip('探針跳過')\n\ndef test_fail():\n    raise RuntimeError('探針失敗')\n")
    result["pytest_probe_source"] = test.read_text()
    result["pytest"] = run([sys.executable, "-m", "pytest", "-ra", "-q", "-p", "no:cacheprovider", str(test)])
    assert result["pytest"]["exit_code"] == 1
    assert "探針跳過" in result["pytest"]["stdout"] and "探針失敗" in result["pytest"]["stdout"]
    result["scope"] = "自己的三份短 Notebook 與三個 pytest 範例；刻意失敗用於檢查回報。未跑全套教材。kernel 註冊使用自己目錄的 --prefix，沒有執行會寫入 .venv 的 --sys-prefix。"
    result["checker_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE / "command-probes-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"notebook": summary, "pytest_exit_code": result["pytest"]["exit_code"]}, ensure_ascii=False))
finally:
    shutil.rmtree(scratch)
