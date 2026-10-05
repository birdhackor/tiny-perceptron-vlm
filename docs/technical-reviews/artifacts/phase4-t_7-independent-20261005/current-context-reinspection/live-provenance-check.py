"""Original raw input/provenance fingerprints only; no result content printed."""
import hashlib
import json
import platform
import sys
from datetime import datetime, UTC
from pathlib import Path
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
initial = json.loads((BASE / "repository-input-hashes.json").read_text())
checks = []
for path in ["docs/course-experiments/results/dpo.json", "assets/training/manifest.json",
             "assets/training/sources/ultrafeedback-dpo.json", "docs/course-experiments/plan.json"]:
    previous = next(x["sha256"] for x in initial if x["path"] == path)
    current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert previous == current, path
    checks.append({"path":path,"initial_and_current_sha256":current,"matched":True})
proof = {"checked_at_utc":datetime.now(UTC).isoformat(),
    "command":".venv/bin/python docs/technical-reviews/artifacts/phase4-t_7-independent-20261005/current-context-reinspection/live-provenance-check.py",
    "environment":{"python":platform.python_version(),"executable":sys.executable,"cwd":str(ROOT),
        "device_scope":"bytes/provenance only; no model execution"}, "checks":checks}
(OUT / "live-provenance-hash-check.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(proof,ensure_ascii=False,indent=2))
