"""R.1 reviewer-owned bounded CPU verification; no model training or network calls."""

from contextlib import redirect_stdout
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("execution.json")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def section(path, lesson):
    return re.search(
        r"^## " + re.escape(lesson) + r"\s.*?(?=^## |\Z)",
        path.read_text(),
        re.M | re.S,
    ).group()


def run_block(code):
    namespace = {}
    stream = StringIO()
    with redirect_stdout(stream):
        exec(compile(code, "course/chapters/01.md#1.1", "exec"), namespace)
    return namespace, stream.getvalue()


started = datetime.now(UTC).isoformat()
clock = time.monotonic()
freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v2.json"
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "R.1")
snapshot = ROOT / page["snapshot"]
assert digest(snapshot) == page["source_sha256"]
assert section(ROOT / "course/README.md", "R.1").encode() == snapshot.read_bytes()
assert page["figures_sha256"] == {}

prerequisite = section(ROOT / "course/chapters/01.md", "1.1")
blocks = re.findall(r"```python\n(.*?)```", prerequisite, re.S)
assert len(blocks) == 1
embedded = blocks[0].strip()
baseline, baseline_stdout = run_block(embedded)
assert baseline["text"] == "貓看狗，狗看貓。"
assert baseline["chars"] == ["。", "狗", "看", "貓", "，"]
assert baseline["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
assert baseline["text"].count("貓") == 2
assert [baseline["ids"][i] for i, c in enumerate(baseline["text"]) if c == "貓"] == [3, 3]
assert baseline["restored"] == baseline["text"]

reverse, reverse_stdout = run_block(
    embedded.replace("sorted(set(text))", "sorted(set(text), reverse=True)")
)
assert reverse["chars"] == ["，", "貓", "看", "狗", "。"]
assert reverse["ids"] == [1, 2, 3, 0, 3, 2, 1, 4]
assert reverse["restored"] == baseline["text"]

notebook_path = ROOT / "notebooks/01/1.1.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
teaching_cells = [
    (i, c) for i, c in enumerate(notebook.cells)
    if c.cell_type == "code" and c.source.strip() == embedded
]
assert len(teaching_cells) == 1
assert any(c.cell_type == "markdown" for c in notebook.cells)
client = NotebookClient(
    notebook,
    timeout=25,
    startup_timeout=25,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": str(notebook_path.parent)}},
)
executed = client.execute()
cell_results = []
for i, cell in enumerate(executed.cells):
    if cell.cell_type != "code":
        continue
    assert cell.execution_count is not None
    assert all(o.output_type != "error" for o in cell.get("outputs", []))
    stdout = "".join(o.text for o in cell.get("outputs", []) if o.output_type == "stream")
    cell_results.append({
        "index": i,
        "execution_count": cell.execution_count,
        "source_sha256": sha256(cell.source.encode()).hexdigest(),
        "stdout": stdout,
        "rich_output_mimetypes": [
            sorted(o.data) for o in cell.get("outputs", []) if "data" in o
        ],
    })
assert cell_results[-1]["stdout"] == baseline_stdout

result = {
    "started_at": started,
    "completed_at": datetime.now(UTC).isoformat(),
    "elapsed_seconds": round(time.monotonic() - clock, 3),
    "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/verify_r1.py",
    "environment": {
        "python": sys.version,
        "torch": importlib.metadata.version("torch"),
        "nbclient": importlib.metadata.version("nbclient"),
        "nbformat": importlib.metadata.version("nbformat"),
        "ipykernel": importlib.metadata.version("ipykernel"),
        "platform": platform.platform(),
        "device": "CPU; CUDA_VISIBLE_DEVICES is empty; no accelerator or training invoked",
    },
    "freeze": {
        "path": str(freeze_path.relative_to(ROOT)),
        "sha256": digest(freeze_path),
        "snapshot": page["snapshot"],
        "source_sha256": digest(snapshot),
        "live_section_identical": True,
        "figures_sha256": {},
    },
    "inputs": {
        "source": "course/chapters/01.md#1.1",
        "source_file_sha256": digest(ROOT / "course/chapters/01.md"),
        "section_sha256": sha256(prerequisite.encode()).hexdigest(),
        "embedded_code_sha256": sha256(embedded.encode()).hexdigest(),
        "notebook": str(notebook_path.relative_to(ROOT)),
        "notebook_sha256": digest(notebook_path),
        "notebook_teaching_cell_matches_embedded": True,
    },
    "baseline": {
        "input": baseline["text"],
        "chars": baseline["chars"],
        "ids": baseline["ids"],
        "cat_count": baseline["text"].count("貓"),
        "cat_ids": [3, 3],
        "restored": baseline["restored"],
        "stdout": baseline_stdout,
    },
    "reverse_order_exercise": {
        "chars": reverse["chars"],
        "ids": reverse["ids"],
        "restored": reverse["restored"],
        "stdout": reverse_stdout,
    },
    "local_notebook_execution": {"code_cells": cell_results, "all_passed": True},
    "scope": "One exact linked 1.1 example, one reverse-order exercise, and three code cells in a fresh local CPU kernel; no Colab cloud session, training, performance claim, or visual figure review.",
}
with OUT.open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
