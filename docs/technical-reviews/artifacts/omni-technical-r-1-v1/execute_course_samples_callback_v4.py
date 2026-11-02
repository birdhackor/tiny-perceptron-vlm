"""Execute exact current12.2, its changed-label exercise, and a fresh local kernel."""

import ast
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
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
pages = {p["page_id"]: p for p in json.loads(FREEZE.read_text())["pages"]}
page = pages["12.2"]
raw = (ROOT / page["snapshot"]).read_bytes()
assert sha256(raw).hexdigest() == page["source_sha256"]
blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", raw.decode(), re.S)]
previous = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json").read_text())
old_page = next(p for p in previous["pages"] if p["page_id"] == "12.2")
assert blocks == [b.strip() for b in re.findall(r"```python\n(.*?)```", (ROOT / old_page["snapshot"]).read_text(), re.S)]
precursor = (ROOT / pages["12.1"]["snapshot"]).read_text()
inline_tone = re.findall(r"```python\n(.*?)```", precursor, re.S)[1]
implementation = (ROOT / "tiny_perceptron/multimodal.py").read_text()
actual_tone = next(n for n in ast.parse(implementation).body if isinstance(n, ast.FunctionDef) and n.name == "tone")
assert ast.dump(ast.parse(inline_tone).body[0], include_attributes=False) == ast.dump(actual_tone, include_attributes=False)

results = []
baseline = "\n\n".join(blocks)
exercise = baseline.replace("9000.0", "9500.0").replace("7000.0", "6500.0").replace("9000Hz", "9500Hz").replace("7000Hz", "6500Hz")
self_contained = baseline.replace("from tiny_perceptron.multimodal import tone", "import math\n" + inline_tone)
for name, body in [("exact_current_embedded_code", baseline), ("prescribed_exercise_with_updated_labels", exercise), ("same_inline_precursor_tone_instead_of_module_import", self_contained)]:
    program = "import torch\ntorch.set_num_threads(1)\ntorch.manual_seed(42)\n" + body + "\nassert len(high) == len(low) == 16\nassert torch.allclose(high, -low, atol=1e-5)\n"
    start = time.monotonic()
    process = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=25)
    assert process.returncode == 0 and "每一點都只差正負號 True" in process.stdout
    assert "取樣率 16000 時長 0.1" in process.stdout and "取樣率 8000 時長 0.2" in process.stdout
    assert ("9500Hz前四點" if name == "prescribed_exercise_with_updated_labels" else "9000Hz前四點") in process.stdout
    results.append({"name": name, "program_sha256": sha256(program.encode()).hexdigest(), "seconds": round(time.monotonic()-start, 3), "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
assert results[0]["stdout"] == results[2]["stdout"]
out = Path("/workspace/work/omni-technical-r-1-callback-v4/kernel")
out.mkdir(parents=True, exist_ok=False)
notebook = ROOT / page["notebook"]
nb = nbformat.read(notebook, as_version=4)
start = time.monotonic()
NotebookClient(nb, timeout=25, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
duration = round(time.monotonic()-start, 3)
assert all(c.get("execution_count") is not None and not any(o["output_type"] == "error" for o in c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
kernel_stdout = "".join(o["text"] for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o["output_type"] == "stream")
assert kernel_stdout == results[0]["stdout"]
executed_path = out / "12.2.ipynb"
nbformat.write(nb, executed_path)
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/execute_course_samples_callback_v4.py",
    "environment": {"python": sys.version, "torch": importlib.metadata.version("torch"), "nbclient": importlib.metadata.version("nbclient"), "nbformat": importlib.metadata.version("nbformat"), "device": "CPU; one thread; 25-second bound per subprocess/cell"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha256(FREEZE.read_bytes()).hexdigest()},
    "page_id": "12.2", "source_sha256": page["source_sha256"], "notebook_sha256": page["notebook_sha256"],
    "python_blocks_sha256": [sha256(b.encode()).hexdigest() for b in blocks], "source13_and14_python_fences_identical": True,
    "precursor_inline_tone": {"source": pages["12.1"]["snapshot"], "source_sha256": pages["12.1"]["source_sha256"], "locator": "second Python fence", "matches_repository_AST": True, "repository_path": "tiny_perceptron/multimodal.py", "repository_sha256": sha256(implementation.encode()).hexdigest()},
    "results": results, "own_fresh_local_notebook": {"kernel": "python3", "seconds": duration, "executed_code_cells": sum(c.cell_type == "code" for c in nb.cells), "path_outside_git": str(executed_path), "sha256": sha256(executed_path.read_bytes()).hexdigest(), "outputs_match_exact_source": True},
    "all_passed": True, "scope": "Only12.2 baseline/exercise, the explicitly linked12.1 inline tone, and a fresh local12.2 kernel. Verifies instructions/output completeness for R.1, not a wholesale review of12.2's sampling theory or real-recording evidence. No training, GPU, cloud, paid execution, or whole-book rerun."}
with Path(__file__).with_name("sample-executions-callback-v4.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
