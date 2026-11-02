"""Run only the eight specifically inspected source examples on bounded CPU."""

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
freeze = json.loads(FREEZE.read_text())
selected = ["1.11", "3.7", "6.6", "12.13", "14.1", "19.23", "20.13", "20.16"]
results = []
for pid in selected:
    page = next(p for p in freeze["pages"] if p["page_id"] == pid)
    snapshot = ROOT / page["snapshot"]
    raw = snapshot.read_bytes()
    assert sha256(raw).hexdigest() == page["source_sha256"]
    blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", raw.decode(), re.S)]
    program = "import torch\ntorch.set_num_threads(1)\ntorch.manual_seed(42)\n" + "\n\n".join(blocks)
    started = datetime.now(UTC).isoformat()
    clock = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        timeout=25,
    )
    results.append({
        "page_id": pid,
        "snapshot": page["snapshot"],
        "source_sha256": page["source_sha256"],
        "python_block_sha256s": [sha256(b.encode()).hexdigest() for b in blocks],
        "started_at": started,
        "elapsed_seconds": round(time.monotonic() - clock, 3),
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    })

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/execute_course_samples_callback_v1.py",
    "environment": {"python": sys.version, "torch": importlib.metadata.version("torch"), "platform": platform.platform(), "device": "CPU; one thread; one subprocess per example; timeout 25 seconds each"},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha256(FREEZE.read_bytes()).hexdigest()},
    "results": results,
    "all_passed": all(r["exit_code"] == 0 for r in results),
    "scope": "Eight exact source examples already inspected by this reviewer; no whole-course rerun, GPU, paid job, or model training. The two-parameter backward example computes a gradient only. This execution checks code/output correspondence, not reader comprehension or learned model quality.",
}
with Path(__file__).with_name("sample-executions-callback-v1.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
