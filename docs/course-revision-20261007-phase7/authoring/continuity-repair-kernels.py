"""Fresh source-matched CPU outputs needed to preview six wording repairs."""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.check_notebooks import run_kernel
from scripts.export_course import checked_notebook
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ids = {"1.9", "11.1", "11.13", "12.11", "18.4", "19.9"}
index = json.loads((ROOT / "course/lesson-index.json").read_text())
selected = [item for item in index if item["id"] in ids]
assert len(selected) == 6
rows = []
for item in selected:
    source = ROOT / item["notebook"]
    started = time.perf_counter()
    executed = Path(run_kernel(source, ROOT / "outputs/notebooks", 120))
    checked_notebook(json.loads(source.read_text()), executed)
    notebook = json.loads(executed.read_text())
    row = {
        "page_id": item["id"],
        "source_notebook_sha256": digest(source),
        "executed_path": executed.relative_to(ROOT).as_posix(),
        "executed_sha256": digest(executed),
        "python_from_kernel": notebook["metadata"]["language_info"]["version"],
        "torch_in_venv": torch.__version__,
        "seconds": round(time.perf_counter() - started, 3),
        "status": "executed_without_errors_and_source_matched",
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

record = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "kind": "author six fresh CPU kernels, not reviewer judgments",
    "command": ".venv/bin/python docs/course-revision-20261007-phase7/authoring/continuity-repair-kernels.py",
    "device": "CPU",
    "scope": "Refresh source-matched preview outputs after Markdown-only repairs. Code fences unchanged; no capability training or GPU work.",
    "records": rows,
    "completed": len(rows),
}
output = ROOT / "docs/course-revision-20261007-phase7/authoring/continuity-repair-kernel-checks.json"
with output.open("x") as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
