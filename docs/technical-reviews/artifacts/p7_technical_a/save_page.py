"""Save the owner's supplied page assessment; fill only file metadata, never judgments."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
record = json.load(sys.stdin)
manifest_path = ROOT / "docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json"
manifest = json.loads(manifest_path.read_text())
meta = next(p for p in manifest["pages"] if p["page_id"] == record["page_id"])
record["source_sha256"] = meta["source_sha256"]
record["figures_sha256"] = meta["figures_sha256"]
record["trace_file"] = "docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/a/86ddcaf7b50f4afeafb3f5b041a7162c.jsonl"
record["question_refs"] = []
for artifact in record.get("artifacts", []):
    artifact["sha256"] = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
dest = Path(__file__).parent / "pages" / f"{record['page_id']}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    raise SystemExit("Refusing to overwrite assessment")
dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(f"Saved own assessment: {dest.relative_to(ROOT)}")
