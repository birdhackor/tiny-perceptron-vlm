"""Rebuild an evidence inventory; this never creates or approves review reports."""

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

from check_course_reviews import sections

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODEL_REPOSITORY = "birdhackor/tiny-perceptron-course-models"
ATTEMPT_FIELDS = ("run_id", "batch_id", "experiment_id", "mode", "reserved_usd", "status")


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


def _usd(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite nonnegative USD amount")
    try:
        amount = Decimal(str(value))
    except ArithmeticError as error:
        raise ValueError(f"{field} must be a finite nonnegative USD amount") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field} must be a finite nonnegative USD amount")
    return amount


def merge_budget_reservations(existing, receipts, ceiling_usd):
    """Keep durable history and merge only project-reservation receipt fields."""
    existing = existing or {}
    attempts = {}
    observed = [_usd(existing.get("observed_reserved_total_usd", "0"), "existing total")]

    def include(entry):
        safe = {key: entry.get(key) for key in ATTEMPT_FIELDS}
        for key in ("run_id", "batch_id", "experiment_id", "mode", "status"):
            if not isinstance(safe[key], str) or not safe[key]:
                raise ValueError(f"Reservation attempt requires a nonempty {key}")
        reservation = _usd(safe["reserved_usd"], "reserved_usd")
        safe["reserved_usd"] = str(reservation)
        cumulative = entry.get("observed_cumulative_reserved_usd")
        if cumulative is not None:
            cumulative = _usd(cumulative, "observed_cumulative_reserved_usd")
            observed.append(cumulative)
        safe["observed_cumulative_reserved_usd"] = str(cumulative) if cumulative is not None else None
        previous = attempts.get(safe["run_id"])
        if previous is not None:
            identity = ("batch_id", "experiment_id", "mode")
            if any(previous[key] != safe[key] for key in identity):
                raise ValueError(f"Conflicting reservation identity for run {safe['run_id']}")
            if _usd(previous["reserved_usd"], "reserved_usd") != reservation:
                raise ValueError(f"Conflicting reservation amount for run {safe['run_id']}")
            # Pending entries may acquire a final receipt; terminal failures cannot disappear.
            if previous["status"] != safe["status"]:
                pending = {"reserved", "running", "pending"}
                if previous["status"] in pending:
                    previous["status"] = safe["status"]
                elif safe["status"] not in pending:
                    raise ValueError(f"Conflicting reservation terminal status for run {safe['run_id']}")
            totals = [
                value
                for value in (previous["observed_cumulative_reserved_usd"], safe["observed_cumulative_reserved_usd"])
                if value is not None
            ]
            previous["observed_cumulative_reserved_usd"] = str(max(map(Decimal, totals))) if totals else None
        else:
            attempts[safe["run_id"]] = safe

    for entry in existing.get("attempts", []):
        include(entry)
    for raw in receipts:
        billing = raw.get("billing", {})
        entry = billing.get("entry", {})
        if entry:
            # Do not retain billing_before/after, prices, account observations, or credentials.
            include(
                {key: entry.get(key) for key in ATTEMPT_FIELDS}
                | {"observed_cumulative_reserved_usd": billing.get("reserved_total_usd")}
            )
    observed.append(sum((Decimal(entry["reserved_usd"]) for entry in attempts.values()), Decimal(0)))
    return {
        "ceiling_usd": str(_usd(ceiling_usd, "ceiling_usd")),
        "observed_reserved_total_usd": str(max(observed)),
        "scope": "Conservative project reservations from durable history and attempt receipts; failures and reruns retain reservations. The total is the maximum of prior totals, receipt cumulative totals and unique attempt reservations. Active jobs can reserve additional funds before their report arrives. These values are not measured invoice charges or account-wide billing deltas.",
        "attempts": list(attempts.values()),
    }


def student_release_evidence(spec):
    """A planned release flag is not evidence of an anonymously verified public model."""
    relative = spec.get("release_evidence")
    if relative is None:
        return {"published": False, "evidence": None}
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("release_evidence must be a repository-relative JSON path")
    path = ROOT / relative
    if path.suffix != ".json" or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("release_evidence must resolve inside the repository")
    receipt = read_json(path)
    result = {"published": False, "evidence": relative}
    if receipt is None:
        return result
    result["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if receipt.get("experiment_id") != spec["id"]:
        raise ValueError("release_evidence experiment identity mismatch")
    if spec.get("batch_id") and receipt.get("batch_id") != spec["batch_id"]:
        raise ValueError("release_evidence batch identity mismatch")
    nested = receipt.get("release")
    release = nested if isinstance(nested, dict) else receipt
    if release.get("anonymous_download_verified") is not True:
        return result
    if nested is not None:
        if receipt.get("mode") != "release" or release.get("experiment_id") != spec["id"]:
            raise ValueError("Verified release receipt has inconsistent release identity")
    elif (
        receipt.get("release_status") != "completed"
        or not isinstance(receipt.get("source_run_id"), str)
        or not receipt["source_run_id"]
    ):
        raise ValueError("Verified release receipt requires completed status and source run")
    manifest = release.get("public_manifest", {})
    repo, revision = manifest.get("repo"), manifest.get("revision")
    if repo != PUBLIC_MODEL_REPOSITORY or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Verified release requires the course public repository and pinned revision")
    if manifest.get("id", spec["id"]) != spec["id"]:
        raise ValueError("Verified public manifest experiment identity mismatch")
    if nested is not None and (release.get("repo") != repo or release.get("revision") != revision):
        raise ValueError("Release and public manifest disagree on the pinned source")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Verified public release requires a nonempty file manifest")
    outputs = set()
    checkpoints = []
    for file in files:
        remote, output, digest, size = (file.get(key) for key in ("path", "output", "sha256", "bytes"))
        if any(
            not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts
            for value in (remote, output)
        ):
            raise ValueError("Verified public files require safe relative paths")
        if output in outputs or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Verified public files require unique outputs and SHA-256 hashes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Verified public files require positive byte counts")
        outputs.add(output)
        if (
            Path(output).suffix in (".pt", ".safetensors")
            and Path(output).stem != "fixture"
            and file.get("kind") != "input_fixture"
        ):
            checkpoints.append(output)
    if not checkpoints:
        raise ValueError("A public input fixture or manifest alone is not a student model release")
    return {**result, "published": True, "repo": repo, "revision": revision, "verified_checkpoints": checkpoints}


def main():
    plan = read_json(ROOT / "docs/course-experiments/plan.json")
    all_specs = plan["sequence"] + plan.get("supporting_experiments", [])
    ids = [spec["id"] for spec in all_specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Formal and supporting experiment IDs must be unique")
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
    supporting = []
    for spec in plan.get("supporting_experiments", []):
        path = ROOT / "docs/course-experiments/results" / f"{spec['id']}.json"
        result = read_json(path)
        release = student_release_evidence(spec)
        supporting.append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "scope": spec["scope"],
                "status": result.get("evidence_status", "pending") if result else "pending",
                "lessons": spec["lessons"],
                "evidence": path.relative_to(ROOT).as_posix() if result else None,
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if result else None,
                "code_revision": result.get("revision") if result else None,
                "probe_outcome": result.get("results", {}).get("status") or result.get("evidence_status")
                if result
                else None,
                "supported_routes": result.get("results", {}).get("supported_routes", []) if result else [],
                "hf_private_backup": result.get("hf") if result else None,
                "student_model_release": release["published"],
                "release_evidence": release,
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
                    "supporting_experiments": [e["id"] for e in supporting if lesson_id in e["lessons"]],
                    "reader_review": review_state("docs/reader-reviews", lesson_id, body),
                    "technical_review": review_state("docs/technical-reviews", lesson_id, body),
                }
            )
    receipt_paths = sorted(
        {
            path
            for directory in ("outputs/course-control", "outputs/integration-runs")
            for path in (ROOT / directory).rglob("result.json")
        }
    )
    existing_budget = read_json(ROOT / "docs/course-experiments/budget-reservations.json", {})
    budget = merge_budget_reservations(
        existing_budget, (read_json(path) for path in receipt_paths), plan["budget"]["maximum_compute_cost"]
    )
    report = {
        "schema_version": 1,
        "budget_ceiling_usd": plan["budget"]["maximum_compute_cost"],
        "counts": {
            "experiments": len(experiments),
            "complete_runs": sum(e["status"] == "complete_run" for e in experiments),
            "sections": len(lessons),
            "supporting_experiments": len(supporting),
            "complete_supporting_runs": sum(e["status"] == "complete_run" for e in supporting),
        },
        "note": "Review status reflects existing reports and hashes only and may include legacy readers; it does not count completion of the current fresh review round. Actual current dispatches are recorded in review-dispatch/, and check_review_round.py enforces new identities and reviewed introductions alongside the reader/technical content gates. Experiments without a section mapping do not imply the section was omitted: offline notebooks verify mechanisms separately.",
        "experiments": experiments,
        "supporting_evidence": supporting,
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
