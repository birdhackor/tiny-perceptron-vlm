"""Rebuild an evidence inventory; this never creates or approves review reports."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from check_course_reviews import sections

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def review_state(directory, lesson_id, body):
    report = read_json(ROOT / directory / f"{lesson_id}.json")
    if report is None:
        return "pending"
    if report.get("source_sha256") != hashlib.sha256(body.encode()).hexdigest():
        return "outdated"
    for path, digest in (report.get("figure_sha256") or {}).items():
        file = ROOT / path
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            return "outdated"
    return report.get("verdict", "pending")


def main():
    plan = read_json(ROOT / "docs/course-experiments/plan.json")
    experiments = []
    for spec in plan["sequence"]:
        path = ROOT / "docs/course-experiments/results" / f"{spec['id']}.json"
        result = read_json(path)
        experiments.append(
            {
                "id": spec["id"],
                "status": result.get("evidence_status", "pending") if result else "pending",
                "lessons": spec["lessons"],
                "evidence": path.relative_to(ROOT).as_posix() if result else None,
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if result else None,
                "training_revision": result.get("revision") if result else None,
                "hf_private_backup": result.get("hf") if result else None,
            }
        )
    sources = [ROOT / "course" / name for name in ("README.md", "first-steps.md")]
    sources += sorted((ROOT / "course/chapters").glob("*.md"))
    sources += [ROOT / "course" / name for name in ("training.md", "glossary.md")]
    lessons = []
    for source in sources:
        for lesson_id, body in sections(source):
            lessons.append(
                {
                    "lesson_id": lesson_id,
                    "source": source.relative_to(ROOT).as_posix(),
                    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "experiments": [e["id"] for e in experiments if lesson_id in e["lessons"]],
                    "reader_review": review_state("docs/reader-reviews", lesson_id, body),
                    "technical_review": review_state("docs/technical-reviews", lesson_id, body),
                }
            )
    attempts = []
    for evidence in sorted((ROOT / "outputs/course-control").glob("*/result.json")):
        raw = read_json(evidence)
        billing = raw.get("billing", {})
        entry = billing.get("entry", {})
        if not entry:
            continue
        attempts.append(
            {key: entry.get(key) for key in ("run_id", "batch_id", "experiment_id", "mode", "reserved_usd", "status")}
            | {"observed_cumulative_reserved_usd": billing.get("reserved_total_usd")}
        )
    budget = {
        "ceiling_usd": str(plan["budget"]["maximum_compute_cost"]),
        "observed_reserved_total_usd": str(
            max((Decimal(a["observed_cumulative_reserved_usd"]) for a in attempts), default=Decimal(0))
        ),
        "scope": "Conservative project reservations from completed attempt reports; failures and reruns retain reservations. Active jobs can reserve additional funds before their report arrives. These values are not measured invoice charges or account-wide billing deltas.",
        "attempts": attempts,
    }
    report = {
        "schema_version": 1,
        "budget_ceiling_usd": plan["budget"]["maximum_compute_cost"],
        "counts": {
            "experiments": len(experiments),
            "complete_runs": sum(e["status"] == "complete_run" for e in experiments),
            "sections": len(lessons),
        },
        "note": "Review status reflects existing reports and hashes only; final strict gates also validate report content, figures, and unique reviewers. Experiments without a section mapping do not imply the section was omitted: offline notebooks verify mechanisms separately.",
        "experiments": experiments,
        "sections": lessons,
    }
    path = ROOT / "docs/course-experiments/progress.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path.parent / "budget-reservations.json").write_text(
        json.dumps(budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
