"""Bounded CPU check of source15's new5.7 Notebook and prescribed exercise."""

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
OLD = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
page = next(p for p in json.loads(FREEZE.read_text())["pages"] if p["page_id"] == "5.7")
old_page = next(p for p in json.loads(OLD.read_text())["pages"] if p["page_id"] == "5.7")
raw = (ROOT / page["snapshot"]).read_bytes()
assert sha256(raw).hexdigest() == page["source_sha256"]
blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", raw.decode(), re.S)]
old_blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", (ROOT / old_page["snapshot"]).read_text(), re.S)]
assert blocks == old_blocks and len(blocks) == 1
baseline = blocks[0]
exercise = baseline.replace('    restored_optimizer.load_state_dict(payload["optimizer"])', '    # restored_optimizer.load_state_dict(payload["optimizer"])')
assert exercise != baseline
first_assert = '    assert torch.equal(model(x)["logits"], loaded(x)["logits"])\n'
exercise = exercise.replace(first_assert, first_assert + '    print("FIRST_ASSERT_PASSED")\n', 1)
results = []
for name, body, expected_exit in [("exact_current_embedded_code", baseline, 0), ("prescribed_omit_optimizer_state_exercise", exercise, 1)]:
    program = "import torch\ntorch.set_num_threads(1)\n" + body
    if expected_exit == 0:
        program += '\nassert payload["step"] == 1\nassert payload["metadata"]["chars"] == ["。", "狗", "看", "貓", "，"]\nassert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())\n'
    start = time.monotonic()
    process = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=25)
    if expected_exit == 0:
        assert process.returncode == 0 and "保存步數 1 字表" in process.stdout and "檢查點欄位" in process.stdout
    else:
        assert process.returncode == 1 and "FIRST_ASSERT_PASSED" in process.stdout and process.stderr.rstrip().endswith("AssertionError")
    results.append({"name": name, "program_sha256": sha256(program.encode()).hexdigest(), "seconds": round(time.monotonic()-start, 3), "expected_exit_code": expected_exit, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "scope": "Tiny one-layer width8/vocab5/length3 toy checkpoint mechanism; three AdamW step calls per case, no formal/long training."})

out = Path("/workspace/work/omni-technical-r-1-callback-v5/kernel")
out.mkdir(parents=True, exist_ok=False)
notebook_path = ROOT / page["notebook"]
assert sha256(notebook_path.read_bytes()).hexdigest() == page["notebook_sha256"]
nb = nbformat.read(notebook_path, as_version=4)
start = time.monotonic()
NotebookClient(nb, timeout=25, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
seconds = round(time.monotonic()-start, 3)
assert all(c.get("execution_count") is not None and not any(o["output_type"] == "error" for o in c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
stdout = "".join(o["text"] for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o["output_type"] == "stream")
assert stdout == results[0]["stdout"]
executed = out / "5.7.ipynb"
nbformat.write(nb, executed)
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/execute_course_samples_callback_v5.py",
    "environment": {"python": sys.version, "torch": importlib.metadata.version("torch"), "nbclient": importlib.metadata.version("nbclient"), "nbformat": importlib.metadata.version("nbformat"), "device": "CPU; one thread; 25-second bound per subprocess/cell"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha256(FREEZE.read_bytes()).hexdigest()},
    "page_id": "5.7", "source_sha256": page["source_sha256"], "notebook_sha256": page["notebook_sha256"], "python_blocks_sha256": [sha256(b.encode()).hexdigest() for b in blocks],
    "source14_and15_python_fences_identical": True, "results": results,
    "own_fresh_local_notebook": {"kernel": "python3", "seconds": seconds, "executed_code_cells": sum(c.cell_type == "code" for c in nb.cells), "path_outside_git": str(executed), "sha256": sha256(executed.read_bytes()).hexdigest(), "stdout_matches_exact_source": True},
    "all_expected_results_matched": True,
    "scope": "R.1 dependency/inline-output check of5.7's exact example, prescribed optimizer-omission exercise and actual current Notebook. The expected assertion failure is the documented exercise result, not an unresolved defect. Does not establish general deserialization security, GPU reproducibility, a formal training result, full331 reviewer kernels, or tool approval. No Colab or publication."}
with Path(__file__).with_name("sample-executions-callback-v5.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
