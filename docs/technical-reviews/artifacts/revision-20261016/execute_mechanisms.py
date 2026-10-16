"""執行凍結新增正文的短片段；不產生審閱判定或成品訓練結果。"""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_raw = args.freeze.read_bytes()
    freeze = json.loads(freeze_raw)
    # 只挑本輪新增節號；不執行操作頁上的正式訓練命令。
    selected = {f"7.{n}" for n in range(23, 26)} | {f"9.{n}" for n in range(11, 17)}
    selected |= {f"14.{n}" for n in range(11, 19)} | {f"19.{n}" for n in range(13, 21)}
    results = []
    for page in freeze["pages"]:
        if page["page_id"] not in selected:
            continue
        raw = (ROOT / page["snapshot"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != page["source_sha256"]:
            raise ValueError("凍結正文已改變")
        for i, code in enumerate(re.findall(r"```python\n(.*?)```", raw.decode(), re.S)):
            run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=45, cwd=ROOT)
            results.append(
                {
                    "lesson": page["page_id"],
                    "source_sha256": page["source_sha256"],
                    "block": i,
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "exit_code": run.returncode,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                }
            )
    import torch

    report = {
        "schema_version": 1,
        "kind": "author_mechanism_execution",
        "command": "uv run --frozen --extra cpu python "
        + str(Path(__file__).relative_to(ROOT))
        + " --freeze "
        + str(args.freeze)
        + " --output "
        + str(args.output),
        "freeze_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "scope": "Only authored offline snippets; no formal model training or independent review.",
        "results": results,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"blocks": len(results), "failed": sum(r["exit_code"] != 0 for r in results)}))
    raise SystemExit(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    main()
