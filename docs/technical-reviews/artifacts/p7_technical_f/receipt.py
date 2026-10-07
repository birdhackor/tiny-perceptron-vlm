"""Store owner's observation only after real image tool returns."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = json.load(sys.stdin)
if "page_id" in spec:
    manifest = json.loads((ROOT / "docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json").read_text())
    page_id = spec.pop("page_id")
    spec["source_sha256"] = next(p["source_sha256"] for p in manifest["pages"] if p["page_id"] == page_id)
target = ROOT / "docs/technical-reviews/artifacts/p7_technical_f" / (spec.pop("id") + ".json")
if target.exists():
    raise SystemExit("Refusing to overwrite receipt")
spec.update(tool="tools.view_image", reviewer_task="/root/p7_technical_f", received_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
for artifact in spec["artifacts"]:
    artifact["sha256"] = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
target.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), **spec}, ensure_ascii=False))
