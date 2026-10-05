"""核對本輪逐段閱讀的版本與紀錄；不產生讀者判定或理解摘要。"""

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def trace_errors(header, notes, *, task, source, body, intro, figures, units, fields):
    """比對當時揭露的內容，不能只用整節指紋與 checkpoint 數量。"""
    errors = []
    if header.get("reviewer_task") != task or header.get("source") != source:
        errors.append("trace identity/source mismatch")
    if header.get("source_sha256") != digest(body):
        errors.append("trace section missing or stale")
    if intro and header.get("intro_sha256") != digest(intro):
        errors.append("trace introduction missing or stale")
    if header.get("figure_sha256") != figures:
        errors.append("trace primary figures missing or stale")
    if [note.get("unit_index") for note in notes] != list(range(len(units))):
        errors.append("trace incomplete or checkpoint order mismatch")
    if [note.get("unit_sha256") for note in notes] != [digest(unit.encode()) for unit in units]:
        errors.append("trace revealed unit hashes mismatch")
    if any(not isinstance(note.get(field), str) or not note[field].strip() for note in notes for field in fields):
        errors.append("actual checkpoint notes missing")
    return errors


def audit(current_passes_only=False):
    reader = module("round_reader", ROOT / "docs/review-tools/incremental_reader.py")
    identities = module("round_identities", ROOT / "docs/review-tools/check_review_round.py")
    progress_path = ROOT / "docs/course-revision-20261005/review-progress.json"
    progress_raw = progress_path.read_bytes()
    progress = json.loads(progress_raw)
    baseline = json.loads(identities.BASELINE.read_bytes())
    additions = json.loads(identities.ADDITIONS.read_bytes())
    expected = identities.expected_inventory(baseline, additions)
    inventory = reader.inventory()
    failures, records, tasks = [], [], set()
    if set(inventory) != expected or set(progress["records"]) != expected:
        failures.append("source/progress inventory differs from the retained baseline and explicit additions")
    for lesson, (path, body, intro) in inventory.items():
        entry = progress["records"].get(lesson, {})
        if entry.get("status") != "pass":
            if not current_passes_only:
                failures.append(f"{lesson}: current version has not passed")
            continue
        report_path = ROOT / "docs/reader-reviews" / f"{lesson}.json"
        try:
            report_raw = report_path.read_bytes()
            report = json.loads(report_raw)
            task = report.get("reviewer_task", "")
            source = path.relative_to(ROOT).as_posix() + "#" + lesson
            if not task.startswith("/root/") or task in tasks:
                failures.append(f"{lesson}: missing or reused actual reader identity declaration")
            tasks.add(task)
            if not any(
                item.get("reviewer_task") == task and item.get("fork_turns") == "none"
                for item in entry.get("dispatches", [])
            ):
                failures.append(f"{lesson}: no corresponding fresh dispatch record")
            if report.get("verdict") != "pass" or report.get("source_sha256") != digest(body.encode()):
                failures.append(f"{lesson}: report verdict/body mismatch")
            if intro and (report.get("intro_sha256") != digest(intro.encode()) or not report.get("intro_summary")):
                failures.append(f"{lesson}: report introduction missing or stale")
            figures = reader.figure_records(body, path)
            for name, hash_value in figures.items():
                if report.get("figure_sha256", {}).get(name) != hash_value:
                    failures.append(f"{lesson}: report figure missing or stale: {name}")
            trace_path = (ROOT / report["trace_file"]).resolve()
            if not trace_path.is_relative_to(ROOT / "docs/reader-reviews/traces"):
                raise ValueError("trace is outside the durable review trace directory")
            trace_raw = trace_path.read_bytes()
            rows = [json.loads(line) for line in trace_raw.splitlines()]
            if not rows or not all(isinstance(row, dict) for row in rows):
                raise ValueError("trace must contain a session header and actual note objects")
            errors = trace_errors(
                rows[0],
                rows[1:],
                task=task,
                source=source,
                body=body.encode(),
                intro=intro.encode(),
                figures=figures,
                units=reader.units(intro + body),
                fields=reader.FIELDS,
            )
            failures.extend(f"{lesson}: {error}" for error in errors)
            if not report.get("variation") or not report.get("missing_visuals") or not report.get("visual_checks"):
                failures.append(f"{lesson}: understanding variation or visual assessment missing")
            records.append(
                {
                    "lesson_id": lesson,
                    "reader_task": task,
                    "source_sha256": report.get("source_sha256"),
                    "report_sha256": digest(report_raw),
                    "trace_path": trace_path.relative_to(ROOT).as_posix(),
                    "trace_sha256": digest(trace_raw),
                    "actual_checkpoint_count": len(rows) - 1,
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{lesson}: invalid or missing review evidence: {error}")
    if not records:
        failures.append("no current-pass records; an empty selection cannot pass")
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "failed" if failures else ("passed_partial" if current_passes_only else "passed"),
        "scope": "Fresh dispatch declarations, current report versions, frozen introductions and primary figures, and the actual revealed-unit SHA sequence with five nonempty reader notes. This does not prove comprehension, that an agent actually looked at an image, or the independence of an agent beyond recorded declarations; actual orchestration and review judgments remain separate evidence.",
        "expected_sections": len(expected),
        "checked_current_pass_sections": len(records),
        "distinct_reader_tasks": len(tasks),
        "actual_checkpoints": sum(record["actual_checkpoint_count"] for record in records),
        "progress_sha256": digest(progress_raw),
        "verification_program_sha256": digest(Path(__file__).read_bytes()),
        "records": records,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-passes-only", action="store_true", help="中途核對；不代表整輪完成")
    parser.add_argument("--output", type=Path, help="只有通過才保存完整核對結果")
    args = parser.parse_args()
    result = audit(args.current_passes_only)
    if result["failures"]:
        print("\n".join(result["failures"]))
        raise SystemExit(1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "checked_current_pass_sections", "distinct_reader_tasks", "actual_checkpoints")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
