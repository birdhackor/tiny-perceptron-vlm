"""Intake a genuinely saved report after real R completion and T opening.

Administrative helper only: calls existing native validators, then creates a new
actual-time parse/hash/structure receipt. Does not create reports, dispatches,
judgments, completed seals, candidate manifests, or publication changes.
"""

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANONICAL = {
    "path": "docs/course-revision-20261101-c-v3/source-freeze-v21.json",
    "sha256": "d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea",
}
NUMBERED = {
    "path": "docs/course-revision-20261101-c-v3/numbered-source-freeze-v21.json",
    "sha256": "0db79be8a75904d464b0dd8245eefe93b0dd617b85583e79cd631d116fb598fd",
}


def local(name):
    path = Path(name)
    if path.is_absolute():
        raise ValueError("repository-relative paths required")
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("path escapes repository")
    return path


def reference(path):
    path = local(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def document(ref):
    raw = local(ref["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("artifact changed: " + ref["path"])
    return json.loads(raw)


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, local(relative))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timezone required")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("original", "current"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--assignment", required=True,
                        help="new file containing the actual single technical assignment")
    parser.add_argument("--reader-seal", required=True)
    parser.add_argument("--reader-seal-sha256", required=True)
    parser.add_argument("--technical-started-at", required=True)
    parser.add_argument("--parent-page", required=True)
    parser.add_argument("--supplemental-freeze")
    parser.add_argument("--supplemental-freeze-sha256")
    parser.add_argument("--original-receipt")
    parser.add_argument("--current-dispatch")
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    # Check the phase boundary before opening any textbook body or report.
    seal_ref = {"path": args.reader_seal, "sha256": args.reader_seal_sha256}
    seal = document(seal_ref)
    now = datetime.now(UTC)
    t_start = instant(args.technical_started_at)
    if not (
        seal.get("role") == "reader"
        and seal.get("status") == "completed"
        and instant(seal["started_at"]) <= instant(seal["completed_at"])
        <= t_start <= now
    ):
        raise ValueError("a genuine completed R seal and actual subsequent T opening are required")

    assignment_ref = reference(args.assignment)
    assignment = document(assignment_ref)
    task, lesson = assignment["task"], assignment["lesson_id"]
    window = assignment["dispatch_window"]
    if not (
        task.startswith("/root/c_v3_technical_coord/")
        and assignment["role"] == "technical"
        and assignment["context"] == "fresh"
        and t_start <= instant(window["not_before"])
        <= instant(window["not_after"]) <= now
    ):
        raise ValueError("actual technical assignment/dispatch window required")

    current_dispatch_ref = None
    if args.current_dispatch:
        if args.kind != "current":
            raise ValueError("a subsequent current dispatch applies only to current intake")
        current_dispatch_ref = reference(args.current_dispatch)
        callback = document(current_dispatch_ref)
        if (callback.get("task") != task or callback.get("lesson_id") != lesson
                or callback.get("wave") != assignment["wave"]):
            raise ValueError("current callback does not match actual owner/page/wave")
        window = callback["current_dispatch_window"]
        if not (t_start <= instant(window["not_before"])
                <= instant(window["not_after"]) <= now):
            raise ValueError("current dispatch lies outside actual open T interval")

    api = module("c_v3_v21_intake_api", "docs/review-tools/incremental_revision.py")
    setup = module("c_v3_v21_intake_setup", "scripts/omni_revision/review_setup.py")
    validator = module("c_v3_v21_intake_validator", "scripts/check_technical_reviews.py")
    if lesson == "__current_data__":
        if not args.supplemental_freeze or not args.supplemental_freeze_sha256:
            raise ValueError("the non-canonical current-data contract requires its genuine supplemental freeze")
        freeze_ref = {"path": args.supplemental_freeze,
                      "sha256": args.supplemental_freeze_sha256}
    else:
        if args.supplemental_freeze or args.supplemental_freeze_sha256:
            raise ValueError("canonical/numbered reports must use their fixed source21 freezes")
        freeze_ref = CANONICAL if lesson == args.parent_page else NUMBERED
    freeze = document(freeze_ref)
    page = next(row for row in freeze["pages"] if row["page_id"] == lesson)
    report_ref = reference(args.report)
    report = document(report_ref)
    if not (
        report.get("reviewer_task") == task
        and report.get("lesson_id") == lesson
        and instant(window["not_before"]) <= instant(report["created_at"]) <= now
    ):
        raise ValueError("report identity or actual creation time does not match assignment")

    original_ref = None
    current_pass_errors = []
    if args.kind == "original":
        if args.original_receipt:
            raise ValueError("original intake does not take an earlier receipt")
        setup.check_original(report, freeze_ref, lesson, args.parent_page,
                             task, validator, api)
    else:
        if not args.original_receipt:
            raise ValueError("current requires the genuinely saved original intake receipt")
        original_receipt_ref = reference(args.original_receipt)
        initial = document(original_receipt_ref)
        if not (
            initial.get("intake_kind") == "original"
            and initial.get("reviewer_task") == task
            and initial.get("lesson_id") == lesson
            and initial.get("wave") == assignment["wave"]
            and instant(initial["received_at"]) <= instant(report["created_at"])
        ):
            raise ValueError("current must follow actual original receipt by the same owner")
        original_ref = initial["report"]
        original = document(original_ref)
        setup.check_original(original, freeze_ref, lesson, args.parent_page,
                             task, validator, api)
        expected = set(api.reported_issues(original))
        closures = report.get("original_issue_resolutions", [])
        if (len(closures) != len(expected)
                or {row.get("json_pointer") for row in closures} != expected
                or any(not isinstance(row.get("resolution"), str)
                       or not row["resolution"].strip() for row in closures)):
            raise ValueError("current lacks the same owner's issue-by-issue original closure")
        errors = validator._validate(
            ROOT, local(page["source"]), lesson, setup.frozen_body(page), report,
            {task: [lesson]}, set(),
        )
        if report.get("figure_sha256") != page["figures_sha256"]:
            raise ValueError("current figure identity mismatch")
        if report.get("verdict") == "pass":
            if errors:
                raise ValueError("native current structure: " + repr(errors))
        elif report.get("verdict") == "revise":
            # Receive an honest, structurally complete revise without relabeling
            # it as a pass. Native original validation accepts unresolved claims.
            setup.check_original(report, freeze_ref, lesson, args.parent_page,
                                 task, validator, api)
            current_pass_errors = errors
        else:
            raise ValueError("current verdict must be pass or revise")

    # This time is obtained only after successful JSON/hash/native inspection.
    received = datetime.now(UTC)
    if instant(report["created_at"]) > received:
        raise ValueError("report creation is in the future")
    receipt = {
        "schema_version": 1,
        "kind": "actual_technical_report_intake",
        "collector_task": "/root/c_v3_technical_coord",
        "intake_kind": args.kind,
        "report": report_ref,
        "reviewer_task": task,
        "lesson_id": lesson,
        "created_at": report["created_at"],
        "received_at": received.isoformat(),
        "collection_role": "technical",
        "wave": assignment["wave"],
        "assignment": assignment_ref,
        "dispatch_window": window,
        "current_dispatch": current_dispatch_ref,
        "reader_seal": seal_ref,
        "technical_started_at": args.technical_started_at,
        "source_freeze": freeze_ref,
        "original_report": original_ref,
        "report_verdict": report["verdict"],
        "native_current_pass_errors": current_pass_errors,
        "native_validation": "check_original" if args.kind == "original" else "_validate + original closure",
        "validation_scope": "actual JSON parse, full report hash and native structure; not semantic approval or phase completion",
    }
    output = local(args.receipt)
    if not output.is_relative_to(local("docs/course-revision-20261101-c-v3/technical-coordination-v21")):
        raise ValueError("receipt must stay within this new coordination directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"receipt": reference(args.receipt), "report": report_ref,
                      "received_at": receipt["received_at"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
