"""Run the exact chapter-20 Python fences, without loading mature models.

This checks only the introduction's summary of those examples. It is not a
technical re-review of each section or a mature-model capability evaluation.
"""

import ast
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


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "course/chapters/20.md"
OUTPUT = Path(__file__).with_name("inline-results.json")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit("Refuse to overwrite an existing execution receipt")
    raw = SOURCE.read_bytes()
    text = raw.decode("utf-8")
    results = []
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
    )
    for index, match in enumerate(re.finditer(r"^```python\n(.*?)^```", text, re.M | re.S), 1):
        code = match.group(1)
        section = list(re.finditer(r"^## (.*)$", text[: match.start()], re.M))[-1].group(1)
        imports = []
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        if any(name.split(".")[0] not in {"torch", "tiny_perceptron", "collections", "hashlib", "copy", "math"} for name in imports):
            raise SystemExit(f"Unexpected import in fence {index}: {imports}")
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=20, check=False,
        )
        results.append({
            "index": index,
            "section": section,
            "fence_start_line": text[: match.start()].count("\n") + 1,
            "code_sha256": digest(code.encode()),
            "imports": sorted(imports),
            "command": "same Python interpreter -c <exact UTF-8 Python fence, SHA above>",
            "elapsed_seconds_including_interpreter_startup": time.perf_counter() - started,
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        })
    receipt = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-20-v1/run_inline.py",
        "environment": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "platform": platform.platform(),
            "device": "CPU; CUDA_VISIBLE_DEVICES empty; HF/Transformers offline",
            "threads": "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1",
        },
        "chapter_path": str(SOURCE.relative_to(ROOT)),
        "chapter_sha256": digest(raw),
        "scope": "Exact current chapter Python fences only; no Bash startup/download/training commands executed; no Qwen/Whisper/Kev/Omni loaded; not per-section factual review.",
        "results": results,
        "summary": {
            "python_fence_count": len(results),
            "all_exit_zero": all(row["exit_code"] == 0 for row in results),
            "maximum_elapsed_seconds": max(row["elapsed_seconds_including_interpreter_startup"] for row in results),
        },
    }
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
