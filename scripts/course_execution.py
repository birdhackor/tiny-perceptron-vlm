"""以程式與執行輸入為鍵，重用整套已成功執行的 CPU Notebook。"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_signature(notebook):
    return {
        "kernel": notebook.get("metadata", {}).get("kernelspec", {}),
        "cells": [
            {"source": "".join(cell["source"]), "metadata": cell.get("metadata", {})}
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ],
    }


def execution_key(root=ROOT):
    digest = hashlib.sha256()

    def include(name, raw):
        digest.update(name.encode() + b"\0" + hashlib.sha256(raw).digest())

    for path in sorted((root / "notebooks").rglob("*.ipynb")):
        include(
            path.relative_to(root).as_posix(),
            json.dumps(code_signature(json.loads(path.read_bytes())), sort_keys=True).encode(),
        )
    # 粗粒度保守重用：共用程式、資料、圖與環境任何一項改動就重跑全套。
    paths = {root / "pyproject.toml", root / "uv.lock"}
    for directory in ("tiny_perceptron", "scripts"):
        paths.update((root / directory).rglob("*.py"))
    # 可選的原始研究明細不供教材短程式使用；是否下載不能改變 CPU 執行鍵。
    catalog_path = root / "assets/research/catalog.json"
    research_inputs = set()
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_bytes())
        research_inputs = {name for spec in catalog["groups"].values() for name in spec["files"]}
        for spec in catalog["groups"].values():
            for name, item in spec["files"].items():
                path = root / name
                if path.is_file():
                    raw = path.read_bytes()
                    if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                        raise ValueError(f"已下載的固定研究輸入已變動：{name}")
    for directory in ("docs", "assets", "course/figures", "checkpoints"):
        paths.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.suffix not in {".md", ".pyc"}
            and "__pycache__" not in path.parts
            and path.relative_to(root).as_posix() not in research_inputs
        )
    for path in sorted(paths):
        if path.is_file():
            include(path.relative_to(root).as_posix(), path.read_bytes())
    include(
        "runtime",
        json.dumps(
            {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "torch": version("torch"),
                "nbclient": version("nbclient"),
                "threads": {name: os.environ.get(name, "") for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            },
            sort_keys=True,
        ).encode(),
    )
    return digest.hexdigest()


def validate_bundle(root, output, key):
    if __package__:
        from .export_course import checked_notebook
    else:
        from export_course import checked_notebook

    receipt = json.loads((output / "execution.json").read_bytes())
    if receipt.get("key") != key:
        raise ValueError("執行輸入已改變")
    paths = sorted((root / "notebooks").rglob("*.ipynb"))
    result = json.loads((output / "validation-kernel.json").read_bytes())
    if (
        not paths
        or result.get("total") != len(paths)
        or result.get("passed") != len(paths)
        or result.get("failed") != 0
    ):
        raise ValueError("缺少全套成功的 CPU 執行結果")
    for path in paths:
        checked_notebook(json.loads(path.read_bytes()), output / path.relative_to(root / "notebooks"))
    return receipt


def ensure(root=ROOT, output=None, revision="unknown"):
    output = output or root / "outputs/notebooks"
    if (
        not output.resolve().is_relative_to((root / "outputs").resolve())
        or output.resolve() == (root / "outputs").resolve()
    ):
        raise ValueError("執行快取須位於 outputs/ 的子目錄")
    key = execution_key(root)
    try:
        receipt = validate_bundle(root, output, key)
    except (OSError, ValueError, KeyError, TypeError):
        if output.exists():
            shutil.rmtree(output)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/check_notebooks.py"),
                "--mode",
                "kernel",
                "--workers",
                "1",
                "--output",
                str(output),
            ],
            cwd=root,
            check=True,
        )
        receipt = {"key": key, "executed_revision": revision, "executed_at": datetime.now(UTC).isoformat()}
        (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        validate_bundle(root, output, key)
        print("已重新執行全部 CPU Notebook。")
    else:
        print(f"重用相同執行輸入的成功結果：{receipt['executed_revision']}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["key", "ensure"])
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--revision", default="unknown")
    args = parser.parse_args()
    if args.command == "key":
        key = execution_key()
        print(key)
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(f"key={key}\n")
    else:
        ensure(revision=args.revision)


if __name__ == "__main__":
    main()
