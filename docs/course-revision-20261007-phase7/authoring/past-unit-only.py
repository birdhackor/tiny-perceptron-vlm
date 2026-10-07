"""Redisplay one actually checkpointed unit; no advance and no unread page text."""

import argparse
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("trace")
parser.add_argument("event", type=int)
parser.add_argument("--reviewer", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("phase7_past", root / "docs/review-tools/phase7_review.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
rows = tool.raw_trace(root, args.trace)
header = rows[0]
tool.require(header["reviewer_task"] == args.reviewer, "requested trace is not this reviewer")
tool.require(0 < args.event < len(rows), "event has not actually been recorded")
row = rows[args.event]
tool.require(row["event"] == "checkpoint", "event is not an actual reading checkpoint")
manifest_path = tool.path_in(root, header["manifest"])
tool.require(tool.sha(manifest_path.read_bytes()) == header["manifest_sha256"], "manifest changed")
manifest = tool.load_manifest(root, manifest_path)
unit = next(
    item
    for item in tool.session_units(root, manifest, [row["page_id"]])
    if item["unit_index"] == row["unit_index"]
)
tool.require(unit["unit_sha256"] == row["unit_sha256"], "unit bytes changed")
tool.require(unit["source_sha256"] == row["source_sha256"], "source bytes changed")
tool.require(unit["figures_sha256"] == row["figures_sha256"], "figure bytes changed")
print(
    json.dumps(
        {
            "reviewer_task": args.reviewer,
            "trace_file": args.trace,
            "event_index": args.event,
            "page_id": row["page_id"],
            "unit_index": row["unit_index"],
            "unit_sha256": row["unit_sha256"],
            "source_sha256": row["source_sha256"],
            "recorded_at": row["recorded_at"],
            "text": unit["text"],
            "current_figures": unit["visible_figures"],
            "figure_snapshots": unit["figure_snapshots"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
