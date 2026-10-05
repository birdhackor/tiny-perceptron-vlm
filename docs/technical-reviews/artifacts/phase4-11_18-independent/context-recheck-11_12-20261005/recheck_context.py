"""Append the original reviewer's contextual recheck without altering prior evidence."""

import ast
import difflib
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_11_18"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, number):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    i = next(i for i, h in enumerate(headings) if h[0].startswith(("## " + number + " ").encode()))
    return raw[headings[i].start():headings[i+1].start() if i+1 < len(headings) else len(raw)]


def relative(path):
    return path.relative_to(ROOT).as_posix()


def artifact(identifier, name, description, kind="source_snapshot"):
    path = OUT / name
    return {"id": identifier, "path": relative(path), "sha256": sha(path.read_bytes()), "kind": kind, "description": description}


prior_path = OUT / "prior-11.18-report.opaque.json"
prior_bytes = prior_path.read_bytes()
assert sha(prior_bytes) == "d317114d70308bf97faaa16b4592dcc28f5942a6aa711a280ed0de029183fea9"
prior = json.loads(prior_bytes)
assert prior["reviewer_task"] == TASK
assert prior["verdict"] == "pass"
canonical = ROOT / "docs/technical-reviews/11.18.json"
assert canonical.read_bytes() == prior_bytes
current_chapter = (ROOT / "course/chapters/11.md").read_bytes()
new_context = section(current_chapter, "11.12")
live_body = section(current_chapter, "11.18")
assert live_body == (BASE / "extraction/section.md").read_bytes()
assert new_context == (OUT / "current-11.12-raw.md").read_bytes()
assert live_body == (OUT / "current-11.18-raw.md").read_bytes()
frozen_path = BASE / "inputs/course/chapters/11.md"
old_context = section(frozen_path.read_bytes(), "11.12")
assert sha(frozen_path.read_bytes()) == prior["read_scope"]["frozen_chapter_input"]["sha256"]
diff = "".join(difflib.unified_diff(old_context.decode().splitlines(keepends=True), new_context.decode().splitlines(keepends=True), fromfile="first-read-frozen-11.12", tofile="current-11.12"))
(OUT / "context-11.12-raw.diff").write_text(diff, encoding="utf-8")

proof = json.loads((OUT / "proof-sha-verification.json").read_text())
assert proof["checked_by"] == TASK
for check in proof["hash_checks"]:
    assert sha((ROOT / check["path"]).read_bytes()) == check["expected_sha256"] == check["observed_sha256"]
tree = ast.parse((ROOT / "tiny_perceptron/natural_concepts.py").read_text())
locators = [{"name": n.name, "start_line": n.lineno, "end_line": n.end_lineno} for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"edit_distance", "text_error_report"}]
assert locators == proof["current_ast_locators"]
assert sha((ROOT / "course/figures/new-11.18-two-regions.svg").read_bytes()) == prior["figure_sha256"]["course/figures/new-11.18-two-regions.svg"]

impacts = [
    {"claim_id": "region_and_recognition", "impact": "none", "reason": "Selected-region authored targets and the prospective Chinese alphabet do not depend on how the historical digit data split is described."},
    {"claim_id": "crop_and_region_conditioning", "impact": "none", "reason": "Coordinate crop geometry and optional image/region conditioning contracts are unchanged; the new prerequisite scope changes neither source pixels nor these APIs."},
    {"claim_id": "paired_supervision", "impact": "none", "reason": "Image/box/ordered-target correspondence remains the supervised data contract. Class coverage versus source-family grouping does not change the intended target or establish convergence."},
    {"claim_id": "counterfactual_targets", "impact": "none", "reason": "Changing the selected box or swapping region contents still changes the intended answer. Position-shortcut risk and its support are unchanged."},
    {"claim_id": "ordered_scope_evaluation", "impact": "none; prerequisite distinction explicitly retained", "reason": "11.18 asks for known-character new strings and held-out font/layout conditions, rather than claiming recognition of untrained character classes. The new 11.12 distinguishes familiar-character composition from new independently sourced images within familiar classes. Group independence should be applied with training coverage of the intended known classes; holding out the only source of a class changes the evaluation task. My existing claim does not assert that arbitrary family holding-out preserves class coverage, cite the 2/30 historical score, or infer natural-sign capability."},
    {"claim_id": "figure_and_font_license", "impact": "none", "reason": "The selected-region figure/font/license do not depend on digit-example split wording. Original figure bytes, viewed renders and official font evidence retain their original hashes."},
]
receipt = {
    "schema_version": 1,
    "kind": "independent_context_support_recheck",
    "lesson_id": "11.18",
    "reviewer_task": TASK,
    "checked_at_utc": datetime.now(UTC).isoformat(),
    "trigger": "11.12 prerequisite source-family/class-coverage and historical fixed-string split paragraphs changed; own 11.18 input unchanged.",
    "actual_reading": ["Current 11.12 full raw subsection, with substantive recheck focused on the two changed paragraphs", "Own complete current 11.18 raw subsection including figure reference and font details", "Own original six claims and their precise evidence/support scopes", "Latest factual-reviewer-instructions.md; no other reviewers' reports, author correction summaries, or extra result explanations read"],
    "changed_context": {
        "source": "course/chapters/11.md#11.12",
        "first_read_frozen_section_sha256": sha(old_context),
        "current_section_sha256": sha(new_context),
        "current_raw_snapshot": relative(OUT / "current-11.12-raw.md"),
        "raw_diff": relative(OUT / "context-11.12-raw.diff"),
        "interpretation": "The first changed paragraph notes that this toy has one prototype per class, so holding out a class's complete source family would remove that class from training; familiar-class new-source evaluation needs multiple independent originals per class. The second describes the old 0–99 experiment as complete-string grouping with all digit classes in training and familiar-character novel strings in held-out data. These are read as current prerequisite scope, not accepted as fresh independent verification of that experiment.",
    },
    "own_current_source_sha256": sha(live_body),
    "prior_canonical": {"path": relative(prior_path), "sha256": sha(prior_bytes), "verdict": prior["verdict"], "meaning": "Opaque byte-preserving backup of my own prior canonical before this recheck; no other report read."},
    "first_read_chapter_frozen_input": {"path": relative(frozen_path), "sha256": sha(frozen_path.read_bytes()), "meaning": "Original 2026-10-05 first-read snapshot, retained unchanged. It is historical input and does not claim that the present entire chapter has this hash."},
    "claim_support_reassessment": impacts,
    "proof_sha_verification": {"path": relative(OUT / "proof-sha-verification.json"), "sha256": sha((OUT / "proof-sha-verification.json").read_bytes()), "checked_entries": len(proof["hash_checks"]), "all_original_hashes_match": True},
    "honest_reuse": {
        "original_bounded_cpu": "Reused unchanged original pixel, target-table, alphabet-membership and Unicode comparison evidence; not rerun.",
        "actual_result_pointers": ["bounded-checks-result.json#/coordinate_check", "bounded-checks-result.json#/target_contract_cases", "bounded-checks-result.json#/prospective_alphabet_count", "bounded-checks-result.json#/unicode_comparisons", "execution-record.json#/runs"],
        "implementation": "Current natural_concepts.py hash verified; AST confirms edit_distance lines 47–55 and text_error_report lines 58–67. Existing original inspected computation/denominator evidence reused.",
        "authority_sources": "All previously inspected original paper/official source snapshots and hashes verified unchanged; original precise support ranges reused.",
        "visuals": "Original personally viewed Inkscape and Chromium desktop/mobile evidence reused because current 11.18 raw and figure bytes are unchanged; no new render claimed.",
        "not_rechecked": "No independent new adjudication of 11.12 empirical results or other sections. My 11.18 claims never rely on its historical 2/30 result.",
    },
    "substantive_uncertainty": [],
    "claim_modifications": "none",
    "verdict": "pass",
    "work_performed": "Read, compare raw bytes, inspect own claim support, verify SHA/AST metadata, save receipt and update only own technical report.",
    "model_training_or_inference": False,
    "new_bounded_cpu_or_visual_execution": False,
    "source_edits": False,
}
(OUT / "context-support-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
manifest = {"reviewer_task": TASK, "scope": "New context recheck files before checker; original context snapshots and original manifest preserved.", "files": [{"path": relative(p), "sha256": sha(p.read_bytes())} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"context-recheck-manifest.json", "checker-record.json", "checker-stdout.txt", "checker-stderr.txt"}]}
(OUT / "context-recheck-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

report = json.loads(prior_bytes)
report["artifacts"].extend([
    artifact("context_scope_recheck_11_12_20261005", "context-support-receipt.json", "My actual current-prerequisite reading, six-claim support reassessment and explicit unchanged evidence reuse; original historical context retained."),
    artifact("context_proof_sha_11_12_20261005", "proof-sha-verification.json", "Personally checked 34 original proof/source/figure hashes and live Unicode-helper AST locators; no CPU/model rerun."),
    artifact("context_current_11_12_20261005", "current-11.12-raw.md", "Current prerequisite raw bytes read by this reviewer, dated context recheck rather than replacement of initial snapshot."),
    artifact("context_current_11_18_20261005", "current-11.18-raw.md", "Complete current own subsection reread; raw bytes identical to originally reviewed 11.18."),
    artifact("context_diff_11_12_20261005", "context-11.12-raw.diff", "Raw first-read/current prerequisite paragraph diff; no author correction summary."),
    artifact("context_prior_canonical_11_12_20261005", "prior-11.18-report.opaque.json", "My own PASS canonical preserved byte-for-byte before recheck, with original hash and history."),
    artifact("context_manifest_11_12_20261005", "context-recheck-manifest.json", "SHA manifest of new context receipt/input/proof files."),
    artifact("context_recheck_code_11_12_20261005", "recheck_context.py", "Actual report update/hash verification script; reads only own review and explicitly bounded source context.", "code"),
])
report.setdefault("context_rechecks", []).append({"checked_at_utc": receipt["checked_at_utc"], "trigger_source": receipt["changed_context"]["source"], "trigger_source_sha256": receipt["changed_context"]["current_section_sha256"], "receipt_artifact_id": "context_scope_recheck_11_12_20261005", "claim_impact": "none; six original claims/support scopes preserved", "honest_reuse": True, "original_context_retained": True, "verdict": "pass"})
report.setdefault("review_history", []).append({"event": "current-prerequisite-context-recheck", "reviewer_task": TASK, "prior_report_sha256": sha(prior_bytes), "prior_report_backup": relative(prior_path), "receipt_artifact_id": "context_scope_recheck_11_12_20261005", "claims_unchanged": True, "source_sha256_unchanged": True})
assert report["claims"] == prior["claims"]
assert report["read_scope"] == prior["read_scope"]
assert report["source_sha256"] == sha(live_body)
temp = OUT / "new-canonical.tmp.json"
temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
temp.replace(canonical)
written = json.loads(canonical.read_text())
assert written["reviewer_task"] == TASK
assert written["claims"] == prior["claims"]
assert written["source_sha256"] == sha(live_body)
print(json.dumps({"report_sha256": sha(canonical.read_bytes()), "receipt_artifact_id": "context_scope_recheck_11_12_20261005", "receipt_path": relative(OUT / "context-support-receipt.json"), "receipt_sha256": sha((OUT / "context-support-receipt.json").read_bytes()), "verdict": written["verdict"], "reviewer_task": written["reviewer_task"]}))
