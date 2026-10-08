"""逐份從乾淨 namespace／新 Jupyter kernel 執行，保留失败與成功數。"""

import argparse
import contextlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_python(path):
    # 快速檢查使用新namespace；真正獨立kernel另用--mode kernel。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{path}:cell{number}", "exec"), namespace)
    plt.close("all")
    return output.getvalue()


def run_kernel(path, output_root, timeout):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="tiny-perceptron",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    target = output_root / path.relative_to(ROOT / "notebooks")
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    return str(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["python", "kernel"], default="python")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/notebooks")
    p.add_argument("--lesson", help="只檢查指定小節，例如3.6；省略則全數執行")
    args = p.parse_args()
    if args.workers < 1:
        raise ValueError("workers須為正")
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    if args.lesson:
        notebooks = [path for path in notebooks if path.stem == args.lesson]
    if not notebooks:
        raise ValueError("没有符合的Notebook，不能視為驗證成功")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    started = time.perf_counter()
    passed = 0
    if args.mode == "python":
        for path in notebooks:
            try:
                run_python(path)
                passed += 1
            except Exception as error:
                failures.append({"notebook": str(path.relative_to(ROOT)), "error": repr(error)})
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_kernel, path, args.output, args.timeout): path for path in notebooks}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                    passed += 1
                except Exception as error:
                    failures.append({"notebook": str(path.relative_to(ROOT)), "error": str(error)})
                if (passed + len(failures)) % 20 == 0:
                    print(f"已完成 {passed + len(failures)}/{len(notebooks)}，失败 {len(failures)}", flush=True)
    result = {
        "mode": args.mode,
        "total": len(notebooks),
        "passed": passed,
        "failed": len(failures),
        "seconds": time.perf_counter() - started,
        "failures": failures,
    }
    (args.output / f"validation-{args.mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
