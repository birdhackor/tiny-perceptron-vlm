"""Record actual reader dispatch and report status, without changing reader judgments."""

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "docs/course-revision-20261005/review-progress.json"
spec = importlib.util.spec_from_file_location("incremental_reader", ROOT / "docs/review-tools/incremental_reader.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("dispatch", "collect", "pending", "summary"))
parser.add_argument("ids", nargs="*")
args = parser.parse_args()
p = json.loads(PATH.read_text())
now = datetime.now(UTC).isoformat()
inventory = reader.inventory()
if args.action == "dispatch":
    for lesson in args.ids:
        r = p["records"][lesson]
        task = p["coordinator_task"] + "/reader_" + lesson.replace(".", "_").lower()
        r["dispatches"].append(
            {
                "reviewer_task": task,
                "dispatched_at": now,
                "frozen_source_sha256": reader.sha(inventory[lesson][1].encode()),
                "fork_turns": "none",
                "status": "reading",
            }
        )
        r["status"] = "reading"
if args.action in ("collect", "dispatch"):
    for lesson, r in p["records"].items():
        if not r["dispatches"]:
            continue
        path = ROOT / "docs/reader-reviews" / f"{lesson}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        task = report.get("reviewer_task")
        if task not in [d["reviewer_task"] for d in r["dispatches"]]:
            continue
        source, body, intro = inventory[lesson]
        current = reader.sha(body.encode())
        trace = report.get("trace_file", "")
        latest_dispatch = r["dispatches"][-1]
        if latest_dispatch.get("status") == "rechecking" and "previous_trace_file" not in latest_dispatch:
            latest_dispatch["previous_trace_file"] = r.get("trace_file")
        previous = r.get("report_sha256")
        digest = reader.sha(path.read_bytes())
        r.update(
            {
                "current_source_sha256": current,
                "report_sha256": digest,
                "report_path": path.relative_to(ROOT).as_posix(),
                "reader_task": task,
                "verdict": report.get("verdict"),
                "trace_file": trace,
                "issues": report.get("issues", []),
            }
        )
        validation = []
        if report.get("source_sha256") != current:
            validation.append("source hash mismatch")
        if intro and (report.get("intro_sha256") != reader.sha(intro.encode()) or not report.get("intro_summary")):
            validation.append("introduction missing or stale")
        figure_hashes = report.get("figure_sha256", {})
        required = reader.figure_records(body, source)
        required.update({k: reader.sha((ROOT / k).read_bytes()) for k in figure_hashes})
        if any(figure_hashes.get(k) != v for k, v in required.items()):
            validation.append("figure hash missing or stale")
        if not trace or not (ROOT / trace).is_file():
            validation.append("trace missing")
        else:
            notes = [json.loads(line) for line in (ROOT / trace).read_text().splitlines()]
            header = notes[0]
            state_path = reader.RUNS / header["session"] / "state.json"
            state = json.loads(state_path.read_text()) if state_path.is_file() else None
            expected_units = (
                len(reader.units(intro + body)) if header["source_sha256"] == current else r.get("unit_count")
            )
            r["checkpoint_count"] = len(notes) - 1
            r["unit_count"] = expected_units
            if (
                header["reviewer_task"] != task
                or header["source_sha256"] != report.get("source_sha256")
                or expected_units is None
                or len(notes) - 1 != expected_units
                or (state and state["next_index"] != len(state["units"]))
            ):
                validation.append("trace incomplete or identity mismatch")
        if not report.get("variation") or not report.get("missing_visuals") or not report.get("visual_checks"):
            validation.append("understanding or visual assessment missing")
        r["format_validation_issues"] = validation
        if latest_dispatch.get("status") == "rechecking" and latest_dispatch.get("previous_trace_file") == trace:
            validation.append("original reader followup trace not yet replaced")
        rechecking = latest_dispatch.get("status") == "rechecking" and bool(validation)
        r["status"] = (
            report.get("verdict", "invalid")
            if not validation
            else ("rechecking" if rechecking else "stale_or_incomplete")
        )
        if digest != previous:
            r["last_report_at"] = now
        for d in r["dispatches"]:
            if (
                not validation
                and d["status"] != "superseded_during_recheck"
                and d["reviewer_task"] == task
                and d["frozen_source_sha256"] == report.get("source_sha256")
            ):
                d["status"] = report.get("verdict", "invalid")
        if r["status"] == "pass":
            for handling in r["issues_handled"]:
                if handling.get("status") == "awaiting_original_reader_recheck":
                    handling["status"] = "original_reader_rechecked_pass"
                    handling["recheck_trace"] = trace
            for edit in r.get("author_predispatch_edits", []):
                if edit.get("status") == "awaiting_fresh_reader":
                    edit["status"] = "fresh_reader_reviewed_pass"
                    edit["reader_trace"] = trace
            for todo in p.get("author_visual_todos", []):
                if todo["lesson_id"] == lesson:
                    todo["reader_verdict"] = "pass"
                    todo["reader_trace"] = trace
if args.action in ("dispatch", "collect"):
    p["updated_at"] = now
    PATH.write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n")
counts = {}
for r in p["records"].values():
    counts[r["status"]] = counts.get(r["status"], 0) + 1
if args.action == "pending":
    print(
        json.dumps(
            [
                {"id": k, "source": p["records"][k]["source"]}
                for k in p.get("readability_dispatch_order", p["records"])
                if p["records"][k]["status"] == "pending"
            ][:10],
            ensure_ascii=False,
        )
    )
else:
    print(
        json.dumps(
            {
                "counts": counts,
                "attention": [
                    {"id": k, "status": r["status"], "issues": r.get("issues", [])}
                    for k, r in p["records"].items()
                    if r["status"] in ("revise", "stale_or_incomplete")
                ],
            },
            ensure_ascii=False,
        )
    )
