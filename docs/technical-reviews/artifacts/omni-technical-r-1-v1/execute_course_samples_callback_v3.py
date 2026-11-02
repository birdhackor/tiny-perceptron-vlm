"""R.1 current-v3: bounded CPU execution of the two newly changed lesson sources."""

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


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
freeze = json.loads(FREEZE.read_text())
results = []
for pid in ["9.12", "9.15"]:
    page = next(p for p in freeze["pages"] if p["page_id"] == pid)
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert sha256(raw).hexdigest() == page["source_sha256"]
    blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", raw.decode(), re.S)]
    program = "import torch\ntorch.set_num_threads(1)\ntorch.manual_seed(42)\n" + "\n\n".join(blocks)
    start = time.monotonic()
    process = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=25)
    results.append({"page_id": pid, "snapshot": page["snapshot"],
        "source_sha256": page["source_sha256"],
        "python_blocks_sha256": [sha256(b.encode()).hexdigest() for b in blocks],
        "seconds": round(time.monotonic() - start, 3),
        "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr})

assert results[0]["exit_code"] == results[1]["exit_code"] == 0
assert results[0]["stdout"].count("選擇 查卡片進度") == 2
assert "最大機率 0.6 集中度 0.4" in results[1]["stdout"]
result = {"created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/execute_course_samples_callback_v3.py",
    "environment": {"python": sys.version, "torch": importlib.metadata.version("torch"), "device": "CPU; one thread; each exact-source subprocess timeout 25 seconds"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha256(FREEZE.read_bytes()).hexdigest()},
    "results": results, "all_passed": True,
    "scope": "Only the two currently changed lesson examples, exact embedded source. No training, GPU, paid execution, cloud runtime, or whole-book rerun; the training guide commands were read only."}
with Path(__file__).with_name("sample-executions-callback-v3.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
