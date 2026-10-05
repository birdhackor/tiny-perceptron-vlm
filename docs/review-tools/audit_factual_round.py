"""Audit factual dispatch declarations and current evidence; never write reviewer judgments."""

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = Path("docs/course-revision-20261005/factual-review-progress.json")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dispatch_errors(record, report_raw, coordinator_task):
    report = json.loads(report_raw)
    task = report.get("reviewer_task", "")
    errors = []
    matches = [d for d in record.get("dispatches", []) if d.get("reviewer_task") == task]
    if (
        not task.startswith(coordinator_task + "/")
        or not matches
        or any(d.get("fork_turns") != "none" for d in matches)
    ):
        errors.append("missing current round fork-none dispatch declaration")
    if any((d.get("status") or "").startswith("void") for d in matches):
        errors.append("current report is attributed to a void factual dispatch")
    if record.get("reviewer_task") != task or record.get("report_sha256") != sha(report_raw):
        errors.append("collected identity or report digest is stale")
    if record.get("status") != "pass" or record.get("current_source_sha256") != report.get("source_sha256"):
        errors.append("progress does not describe the passed report version")
    return errors


def audit(root=ROOT, current_passes_only=False):
    progress_raw = (root / PROGRESS).read_bytes()
    progress = json.loads(progress_raw)
    spec = importlib.util.spec_from_file_location("technical_checker", root / "scripts/check_technical_reviews.py")
    technical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(technical)
    inventory = {}
    sources = sorted((root / "course/chapters").glob("*.md"))
    sources += [root / "course" / name for name in technical.FRONT_MATTER]
    for source in sources:
        raw = source.read_bytes()
        intro = raw[: re.search(rb"(?m)^## ", raw).start()]
        for index, (lid, _) in enumerate(technical.sections(source)):
            inventory[lid] = intro if index == 0 else None
    errors, records, tasks = [], [], set()
    if set(progress["records"]) != set(inventory):
        errors.append("progress scope differs from the current course inventory")
    selected = [lid for lid in inventory if progress["records"].get(lid, {}).get("status") == "pass"]
    if not current_passes_only and set(selected) != set(inventory):
        errors.append("not all factual sections are currently passed")
    checked = technical.check(root, selected)
    errors.extend(checked["failures"])
    declarations = [d for record in progress["records"].values() for d in record.get("dispatches", [])]
    declared_tasks = {d.get("reviewer_task") for d in declarations if d.get("reviewer_task")}
    voided_tasks = {
        d["reviewer_task"]
        for d in declarations
        if d.get("reviewer_task") and (d.get("status") or "").startswith("void")
    }
    for lid in selected:
        record = progress["records"][lid]
        report_raw = (root / "docs/technical-reviews" / (lid + ".json")).read_bytes()
        report = json.loads(report_raw)
        errors.extend(f"{lid}: {error}" for error in dispatch_errors(record, report_raw, progress["coordinator_task"]))
        task = report.get("reviewer_task")
        if task in tasks:
            errors.append(f"{lid}: repeated factual reviewer")
        tasks.add(task)
        if inventory[lid] is not None and (
            report.get("intro_sha256") != sha(inventory[lid]) or not report.get("intro_summary", "").strip()
        ):
            errors.append(f"{lid}: current introduction digest or actual summary is missing")
        legacy = record.get("legacy_backup")
        if legacy and sha((root / legacy["path"]).read_bytes()) != legacy["sha256"]:
            errors.append(f"{lid}: preserved legacy bytes changed")
        records.append({"lesson_id": lid, "reviewer_task": task, "report_sha256": sha(report_raw)})
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "failed" if errors else ("passed_partial" if current_passes_only else "passed"),
        "scope": "Current report/evidence schema and hashes, introduction versions, collected report digests, unique current-round fork-none dispatch declarations, and preserved legacy bytes. This metadata audit does not prove actual dispatch, source inspection or scientific correctness; those require the real orchestration and reviewers.",
        "course_sections": len(inventory),
        "checked_current_pass_sections": len(records),
        "distinct_factual_tasks": len(tasks),
        "recorded_initial_dispatch_tasks": len(declared_tasks),
        "recorded_void_dispatch_tasks": len(voided_tasks),
        "progress_sha256": sha(progress_raw),
        "verification_program_sha256": sha(Path(__file__).read_bytes()),
        "records": records,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-passes-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(current_passes_only=args.current_passes_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    keys = (
        "status",
        "checked_current_pass_sections",
        "distinct_factual_tasks",
        "recorded_initial_dispatch_tasks",
        "recorded_void_dispatch_tasks",
        "errors",
    )
    print(json.dumps({k: result[k] for k in keys}))
    raise SystemExit(bool(result["errors"]))


if __name__ == "__main__":
    main()
