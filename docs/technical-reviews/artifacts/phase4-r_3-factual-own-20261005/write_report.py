"""Write only this reviewer's new R.3 report; never read the old report."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = "/root/phase4_factual_coordinator/factual_r_3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = json.loads((OUT / "navigation-results.json").read_bytes())
    env = json.loads((OUT / "environment.json").read_bytes())
    command = ".venv/bin/python docs/technical-reviews/artifacts/phase4-r_3-factual-own-20261005/audit_navigation.py > docs/technical-reviews/artifacts/phase4-r_3-factual-own-20261005/audit.stdout.txt 2> docs/technical-reviews/artifacts/phase4-r_3-factual-own-20261005/audit.stderr.txt"
    artifacts = []

    def artifact(identifier, path, kind, description):
        record = {"id": identifier, "path": path.relative_to(ROOT).as_posix(),
                  "sha256": sha(path), "kind": kind, "description": description}
        if kind == "execution":
            record.update(command=command, environment=env,
                          result="Actual audit exit0; 23 chapter rows, 26 valid local links, 283 chapter lesson IDs and Notebook IDs; original build --check exit0; assertions pass.")
        artifacts.append(record)

    for identifier, filename, kind, description in [
        ("r3-section", "section.md", "source_snapshot", "R.3 original UTF-8 bytes; no newline normalization."),
        ("r3-inspection", "inspection.md", "derivation", "Own inspected scope, row-by-row original heading locators, classification and limitations."),
        ("r3-input-manifest", "frozen-input-manifest.json", "source_snapshot", "Full frozen-input versions; original/copy SHA comparisons, not a current whole-book assertion."),
        ("r3-chapter-inventory", "chapter-heading-inventory.json", "source_snapshot", "Own extraction of 23 chapter names, all numbered headings and exact line locators."),
        ("r3-notebook-inventory", "notebook-id-inventory.json", "source_snapshot", "Own /metadata/lesson_id inspection and version fingerprints of generated notebooks."),
        ("r3-audit-code", "audit_navigation.py", "code", "Actual bounded independent navigation audit; no training or model inference."),
        ("r3-report-code", "write_report.py", "code", "This reviewer's new-report writer; does not read previous report."),
        ("r3-execution", "navigation-results.json", "execution", "Actual navigation audit results and original generator command/exit0."),
        ("r3-stdout", "audit.stdout.txt", "execution", "Actual audit stdout."),
        ("r3-stderr", "audit.stderr.txt", "source_snapshot", "Actual audit stderr; empty on exit0."),
        ("r3-build-stdout", "build-check.stdout.txt", "execution", "Original build_course.py --check stdout from the audited run."),
        ("r3-build-stderr", "build-check.stderr.txt", "source_snapshot", "Original generator stderr; empty on exit0."),
        ("r3-environment", "environment.json", "source_snapshot", "Actual Python version, executable, CPU platform and execution scope."),
    ]:
        artifact(identifier, OUT / filename, kind, description)
    manifest = json.loads((OUT / "frozen-input-manifest.json").read_bytes())
    for i, (original, record) in enumerate(manifest.items(), 1):
        artifact(f"r3-frozen-{i}", ROOT / record["frozen_snapshot"], "source_snapshot",
                 f"Actual frozen input copy of {original}; full source SHA compared with original. Read scope is in r3-inspection.")
    generator = OUT / "frozen-inputs/scripts/build_course.py"
    report = {
        "schema_version": 1, "review_stage": "technical", "lesson_id": "R.3",
        "source": "course/README.md#R.3", "source_sha256": results["source_sha256"],
        "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": [],
        "verdict": "pass", "figure_sha256": {},
        "applicability": {"substantive_claims": False,
                          "reason": "R.3 is a chapter navigation table and local reading links. Method names occur only in learning questions; it contains no conceptual explanation, quantitative example, software fence, empirical score or completed-capability claim. Actual chapter/question mapping, link inventory, original generation contract and design/extension status were independently inspected; see r3-inspection and r3-execution."},
        "claims": [], "issues": [], "artifacts": artifacts,
        "sources": [
            {"id": "r3-generator", "kind": "repository_code", "title": "Original course notebook/index generator",
             "path": generator.relative_to(ROOT).as_posix(), "sha256": sha(generator), "verified": True,
             "version": "Actual frozen input SHA-256 " + sha(generator),
             "inspection_note": "AST function locations then original lines64–236 inspected: notebook transforms chapter prose; build enumerates chapter headings, enforces lesson-contract IDs, constructs lessons/index, and --check byte-compares every generated notebook and table. Executed --check exit0; this validates navigation generation, not textbook methods or model ability."},
            {"id": "r3-navigation-execution", "kind": "execution", "title": "Own R.3 bounded navigation/inventory audit",
             "verified": True, "artifact_id": "r3-execution"},
        ],
        "checks": {
            "factual_accuracy": {"status": "pass", "claim_ids": [], "details": "All 23 chapter labels/questions match actual original chapter/section headings; own row-by-row locators in r3-inspection and r3-chapter-inventory. Opening chapters1–7 matches original foundation-to-dialogue outline. All26 local R.3 links exist; scope is chapter navigation, no mature-method mechanism or model score asserted."},
            "numeric_verification": {"status": "not_applicable", "claim_ids": [], "details": "R.3 has chapter ordinals and navigation questions only; no measured quantity, arithmetic, denominator, tolerance or formula. Audit counts23/26/283 are independently observed inventory facts rather than added textbook quantitative claims."},
            "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "Raw R.3 inspection and extraction find0 Markdown/HTML image references or inline SVG, 0 SVG references. Chapter labels and question table directly state the navigation mapping; no visual input, spatial relation, data shape or arrow requires rendering/viewing."},
            "source_verification": {"status": "pass", "claim_ids": [], "details": "Read actual chapter sources, original outline/rewrite-contract and generator functions, not old reviews. Lesson contract /schema_version,/purpose,/lesson_ids, index id/title/source/notebook and notebook /metadata/lesson_id were inspected. Original --check exit0 confirms283 chapter sections/Notebooks/table entries. Frozen inputs are retained with original/copy SHA agreement. No original paper or official external source is applicable to these local navigation claims."},
            "limitations": {"status": "pass", "claim_ids": [], "details": "19 points to the self-trained integration design whose current intro/original contract state new training/acceptance is pending;20 points to the mature-model extension. Neither row asserts achieved new capabilities. 'All sections' is read under R.3's 'each chapter' scope:283 chapter lessons; W/R/T/G support/navigation pages are outside this notebook generator. No training, model/data download, inference score, reading-time, engineering or transition acceptance performed."},
        },
        "reviewed_scope_artifact": "r3-inspection",
        "frozen_input_whole_file": {"path": manifest["course/README.md"]["frozen_snapshot"],
                                    "sha256": manifest["course/README.md"]["sha256"],
                                    "meaning": "Full Markdown frozen input at own audit; formal source_sha256 is raw R.3 section only."},
    }
    assert report["reviewer_task"] == TASK
    assert report["source_sha256"] == sha(OUT / "section.md")
    target = ROOT / "docs/technical-reviews/R.3.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    own = json.loads(target.read_bytes())
    assert own["reviewer_task"] == TASK
    assert own["source_sha256"] == sha(OUT / "section.md")
    print(json.dumps({"report": str(target.relative_to(ROOT)), "reviewer_task": own["reviewer_task"],
                      "source_sha256": own["source_sha256"], "report_sha256": sha(target), "verdict": own["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
