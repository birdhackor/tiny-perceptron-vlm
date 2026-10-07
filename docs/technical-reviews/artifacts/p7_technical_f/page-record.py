"""Save supplied page judgment with mechanically bound metadata and receipts."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FOLDER = ROOT / "docs/technical-reviews/artifacts/p7_technical_f"
MANIFEST = ROOT / "docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json"
TRACE = "docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/f/6f2c72d53cc64e9d8396b432fb621e7d.jsonl"
record = json.load(sys.stdin)
registry = json.loads((FOLDER / "source-registry.json").read_text())
record.setdefault("sources", [])
record["sources"].extend(registry[key] for key in record.pop("source_refs", []))
meta = next(p for p in json.loads(MANIFEST.read_text())["pages"] if p["page_id"] == record["page_id"])
record.update(source_sha256=meta["source_sha256"], figures_sha256=meta["figures_sha256"], trace_file=TRACE, question_refs=[])
record.setdefault("issues", [])
record.setdefault("artifacts", [])
for artifact in record["artifacts"]:
    path = ROOT / artifact["path"]
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifact["kind"] == "execution":
        execution = json.loads(path.read_text())
        artifact["command"] = json.dumps(execution["command"], ensure_ascii=False)
        artifact["environment"] = execution["environment"]
for source in record["sources"]:
    if source["kind"] == "repository_code":
        source["sha256"] = hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
        source.setdefault("version", "SHA256:" + source["sha256"])
if "page_receipt" in record:
    path = record.pop("page_receipt")
    receipt = json.loads((ROOT / path).read_text())
    record["page_visual_check"] = {"status": "verified", "required": True, "source_sha256": meta["source_sha256"], "details": record.pop("page_view_details"), "observation": receipt["observation"], "receipt": {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}, "artifacts": receipt["artifacts"]}
record["visual_checks"] = []
for line in (ROOT / TRACE).read_text().splitlines():
    event = json.loads(line)
    if event.get("event") == "checkpoint" and event.get("page_id") == record["page_id"]:
        record["visual_checks"].extend(event.get("visual_checks", []))
suffix = record.pop("record_suffix", "")
target = FOLDER / ("page-" + record["page_id"] + suffix + ".json")
if target.exists():
    raise SystemExit("Refusing to overwrite page judgment")
target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(target.relative_to(ROOT))
