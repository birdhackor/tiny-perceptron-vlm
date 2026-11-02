"""Bounded local Git identity gate for the actual source15 Notebook set."""
from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.export_course import verify_colab_notebooks

index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
revision = json.loads((ROOT / "outputs/omni-review-site-technical-fix15/build-info.json").read_bytes())["revision"]
sample = [row for row in index if row["id"] == "1.1"]
assert len(sample) == 1
verify_colab_notebooks(sample, revision)
try:
    verify_colab_notebooks(index, revision)
    raise AssertionError("Current331 unexpectedly matched historical preview revision")
except ValueError as exc:
    message = str(exc)
    assert "Colab入口尚未對上本次Notebook" in message
    assert "notebooks/05/5.7.ipynb" in message
    different = sum("固定版本與目前下載內容不同" in line for line in message.splitlines())
    missing = sum("固定版本沒有此檔案" in line for line in message.splitlines())
    assert (different, missing) == (19, 18)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_first_steps",
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/source15_formal_gate_v1.py",
        "environment": {"python": platform.python_version(), "device": "CPU/localGitonly", "git": subprocess.run(["git", "--version"], capture_output=True, text=True, check=True).stdout.strip()},
        "exporter_sha256": hashlib.sha256((ROOT / "scripts/export_course.py").read_bytes()).hexdigest(),
        "actual_revision": revision, "indexed_notebooks": len(index),
        "real1_1_matching_blob": "accepted", "current_full331": "rejected_as_expected",
        "different": different, "missing": missing, "matching": len(index) - different - missing,
        "actual_error": message,
        "scope": "Current source15 local Git content identity only. Existing real commit/blob read with unmodified current662355 function. No new objects, build, commit, cloud access, publication or successful full331formal release.",
    }
    target = OUT / "source15-formal-gate-v1.json"
    with target.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"receipt": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "matching": result["matching"], "different": different, "missing": missing}, ensure_ascii=False))
