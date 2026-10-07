"""Check one owner's existing report metadata; never collect, fill, or approve content."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/review-tools"))
import phase7_review as review

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
manifest_path = args.manifest.resolve()
report_path = args.report.resolve()
manifest = review.load_manifest(ROOT, manifest_path)
report = json.loads(report_path.read_text(encoding="utf-8"))
group = next(item for item in manifest["groups"] if item["group"] == report["group"])
unverified = []
report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
# The check API accepts a collection-shaped input. This transient mapping is
# explicitly preflight-only: it is never written to collections or a FINAL receipt.
preflight = {
    "path": str(report_path.relative_to(ROOT)),
    "sha256": report_sha,
    "recorded_at": review.now(),
    "preflight_only": True,
}
owner, _ = review.group_report(
    ROOT, manifest_path, manifest, group, report["stage"], preflight, unverified
)
print(json.dumps({
    "status": "passed_one_group_metadata_preflight",
    "not_collected": True,
    "not_full_stage_gate": True,
    "reviewer_task": owner,
    "report_sha256": report_sha,
    "unverified": unverified,
    "scope_note": review.SCOPE_NOTE,
}, ensure_ascii=False))
