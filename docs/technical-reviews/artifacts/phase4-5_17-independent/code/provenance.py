"""Record exact input origins and installed/official source identity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
origins = {
    "inputs/factual-reviewer-instructions.md": "docs/review-tools/factual-reviewer-instructions.md",
    "inputs/check_technical_reviews.py": "scripts/check_technical_reviews.py",
    "inputs/section_facts.py": "docs/review-tools/section_facts.py",
    "inputs/review-protocol.md": ".agents/skills/clear-tutorial/references/review-protocol.md",
    "inputs/build_course.py": "scripts/build_course.py",
    "original-run/section.md": "course/chapters/05.md#5.17 raw UTF-8 bytes, no normalization",
    "original-run/fence-1.py": "course/chapters/05.md#5.17 first original fence, source line 630",
    "original-run/fence-2.py": "course/chapters/05.md#5.17 second original fence, source line 654",
    "original-run/bootstrap.py": "scripts/build_course.py BOOTSTRAP extracted by section_facts.py",
}
rows = [{"saved_path": path, "origin": origin, "sha256": sha(BASE / path), "bytes": (BASE / path).stat().st_size} for path, origin in origins.items()]
snapshot = json.loads((BASE / "original-run/extraction.json").read_text())
provenance = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_5_17",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": snapshot["source_sha256"], "source_first_line": snapshot["section_first_line"],
    "raw_byte_policy": "original UTF-8; no newline normalization or strip",
    "original_helper_command": ".venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.17 --output /tmp/phase4-5_17-fresh --execute --timeout 60",
    "copy_policy": "Only named regular files copied from /tmp; no symlinks, cache, weights or ignored outputs registered as formal artifacts.",
    "inputs": rows,
    "no_prior_reviews_read": True, "no_training_or_model_data_download": True,
}
(BASE / "input-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
installed_root = Path(torch.__file__).parent.parent
receipts = json.loads((BASE / "sources/fetch-receipts.json").read_text())
comparison = []
for receipt in receipts:
    path = receipt["url"].split("/" + torch.version.git_version + "/")[1]
    if not path.startswith("torch/"):
        continue
    installed = installed_root / path
    comparison.append({"path": path, "installed_path": str(installed),
        "installed_sha256": sha(installed), "official_sha256": receipt["sha256"],
        "identical": sha(installed) == receipt["sha256"]})
assert all(item["identical"] for item in comparison)
(BASE / "installed-official-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps({"input_files": len(rows), "installed_official_identical": len(comparison), "source_sha256": snapshot["source_sha256"]}))
