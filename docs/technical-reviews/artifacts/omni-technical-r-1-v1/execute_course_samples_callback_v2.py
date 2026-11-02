"""R.1 current-v2: bounded CPU execution of the two changed lesson sources."""

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
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads(FREEZE.read_text())
results = []
for pid in ["9.18", "12.17"]:
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
assert "固定tau 0.9" in results[0]["stdout"]
assert "全部final 接受 4 / 6 錯誤 2 coverage 0.6667 risk 0.5" in results[0]["stdout"]
assert "移時兩題 ['圓', '方'] 整對正確 True" in results[1]["stdout"]
assert "短塊補零後的支持／錨點 1000.0 1006.25 1003.125" in results[1]["stdout"]
result = {"created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/execute_course_samples_callback_v2.py",
    "environment": {"python": sys.version, "torch": importlib.metadata.version("torch"), "device": "CPU; one thread; each exact-source subprocess timeout 25 seconds"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha256(FREEZE.read_bytes()).hexdigest()},
    "results": results, "all_passed": True,
    "scope": "Only the two currently changed lesson examples, exact embedded source. No training, GPU, paid execution, cloud runtime, or whole-book rerun; the training guide commands were read only."}
with Path(__file__).with_name("sample-executions-callback-v2.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
