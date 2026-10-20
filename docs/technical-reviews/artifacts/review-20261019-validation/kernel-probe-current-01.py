"""用現成 Notebook 環境核對獨立 kernel、CPU、保存與失敗輸出。"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULT = HERE / "kernel-results-current-01.json"


def dump_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    import nbformat
    import nbclient
    import ipykernel
    import torch
    from jupyter_client.kernelspec import KernelSpecManager
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. " + sys.executable + " " + str(Path(__file__).relative_to(ROOT)),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "nbclient": nbclient.__version__,
                        "nbformat": nbformat.__version__, "ipykernel": ipykernel.__version__, "device": "cpu"},
        "kernel_spec": KernelSpecManager().get_kernel_spec("tiny-perceptron").to_dict(),
    }
    with tempfile.TemporaryDirectory(prefix="kernel-work-current-", dir=HERE) as scratch_name:
        scratch = Path(scratch_name)
        runtime = scratch / "runtime"
        runtime.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
                   TMPDIR=str(runtime), IPYTHONDIR=str(scratch / "ipython"),
                   JUPYTER_RUNTIME_DIR=str(runtime))
        os.environ.update(env)
        target = HERE / "notebooks-current-01"
        if target.exists():
            raise FileExistsError("Notebook 執行證據目錄已存在")
        argv = [sys.executable, "scripts/check_notebooks.py", "--mode", "kernel", "--workers", "1",
                "--lesson", "5.7", "--output", str(target)]
        process = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
        result["real_notebook_command"] = {"argv": argv, "exit_code": process.returncode,
                                           "stdout": process.stdout, "stderr": process.stderr}
        result["real_notebook_summary"] = json.loads((target / "validation-kernel.json").read_bytes())
        executed = target / "05/5.7.ipynb"
        if executed.exists():
            notebook = nbformat.read(executed, as_version=4)
            result["real_notebook"] = {
                "path": str(executed.relative_to(ROOT)),
                "sha256": hashlib.sha256(executed.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256((ROOT / "notebooks/05/5.7.ipynb").read_bytes()).hexdigest(),
                "outputs": [dict(output) for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs],
            }
        spec = importlib.util.spec_from_file_location("isolated_notebook_checker", ROOT / "scripts/check_notebooks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = scratch
        inputs = scratch / "notebooks"
        inputs.mkdir()
        samples = {
            "one": "import os, torch\nassert 'isolation_marker' not in globals()\nisolation_marker = 1\nprint(os.getpid(), isolation_marker, torch.tensor(1).device)",
            "two": "import os, torch\nassert 'isolation_marker' not in globals()\nisolation_marker = 1\nprint(os.getpid(), isolation_marker, torch.tensor(1).device)",
            "failure": "raise ValueError('刻意驗證失敗回報')",
        }
        for name, source in samples.items():
            notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
            dump_new(inputs / (name + ".ipynb"), notebook)
        sys.argv = ["scripts/check_notebooks.py", "--mode", "kernel", "--workers", "1", "--output", str(scratch / "executed")]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            try:
                module.main()
                exit_code = 0
            except SystemExit as error:
                exit_code = error.code
        result["isolated_probe"] = {
            "command": sys.argv, "exit_code": exit_code, "stdout": output.getvalue(),
            "summary": json.loads((scratch / "executed/validation-kernel.json").read_bytes()),
            "sources": samples, "successful_notebooks": {},
        }
        for name in ["one", "two"]:
            value = nbformat.read(scratch / "executed" / (name + ".ipynb"), as_version=4)
            result["isolated_probe"]["successful_notebooks"][name] = value
        texts = [value["cells"][0]["outputs"][0]["text"].strip().split()
                 for value in result["isolated_probe"]["successful_notebooks"].values()]
        summary = result["isolated_probe"]["summary"]
        result["checks"] = {
            "real_notebook_passed": process.returncode == 0 and result["real_notebook_summary"]["passed"] == 1,
            "different_kernel_pids": texts[0][0] != texts[1][0],
            "independent_namespaces": all(v[1] == "1" for v in texts),
            "cpu_tensor_outputs": all(v[2] == "cpu" for v in texts),
            "failure_summary_and_exit": exit_code == 1 and summary["total"] == 3 and summary["passed"] == 2 and summary["failed"] == 1,
            "failure_details": "刻意驗證失敗回報" in summary["failures"][0]["error"],
        }
        result["scope"] = "本次實際執行一份課程 Notebook、兩份 kernel 隔離探針及一份刻意失敗探針。大型全書結果另只讀指定的摘要，不宣稱本次重跑全書。暫存探針的來源與完整輸出嵌入本檔。"
        dump_new(RESULT, result)
        print(json.dumps({"checks": result["checks"], "environment": result["environment"]}, ensure_ascii=False))
        if not all(result["checks"].values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
