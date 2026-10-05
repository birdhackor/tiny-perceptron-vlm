"""Same original owner's bounded source/context reinspection; no training or new technical experiment."""
from pathlib import Path
from datetime import datetime, UTC
import ast
import difflib
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-13_16-independent"
REL = A.relative_to(ROOT).as_posix()
OWNER = "/root/phase4_factual_coordinator/factual_13_16"
TARGET = ROOT / "docs/technical-reviews/13.16.json"
def digest(raw): return hashlib.sha256(raw).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

# Preserve prior canonical verbatim before reading this owner's own report to extend its history.
prior_raw = TARGET.read_bytes()
assert digest(prior_raw) == "108c7a31fe4ba5980ab1f7b67f79bc220b852a1b9aeed1a88e44d4b49cefa207"
history = A / "own-prior-report-108c7a31.json"
assert not history.exists()
history.write_bytes(prior_raw)
assert history.read_bytes() == prior_raw
report = json.loads(prior_raw)
assert report["reviewer_task"] == OWNER

spec = importlib.util.spec_from_file_location("sf", ROOT / "docs/review-tools/section_facts.py")
sf = importlib.util.module_from_spec(spec); spec.loader.exec_module(sf)
current_file = ROOT / "course/chapters/13.md"
slices = {}
diffs = []
for lesson, read_this_round in [("13.16", True), ("13.12", True), ("13.14", True), ("13.13", False), ("13.15", False)]:
    body, whole, first = sf.original_section(current_file, lesson)
    old_path = OLD / ("section.md" if lesson == "13.16" else lesson + "-prerequisite.md")
    old = old_path.read_bytes()
    snapshot = A / (lesson + "-current.md")
    snapshot.write_bytes(body)
    change = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), body.decode().splitlines(keepends=True), fromfile=old_path.relative_to(ROOT).as_posix(), tofile=snapshot.relative_to(ROOT).as_posix()))
    if change: diffs.append(change)
    old_fences = sf.fences(old, first)
    new_fences = sf.fences(body, first)
    assert len(old_fences) == len(new_fences)
    fence_checks = []
    for index, (before, after) in enumerate(zip(old_fences, new_fences, strict=True), 1):
        same_ast = ast.dump(ast.parse(before["raw"]), include_attributes=False) == ast.dump(ast.parse(after["raw"]), include_attributes=False)
        assert same_ast
        fence_checks.append({"fence": index, "original_sha256": digest(before["raw"]), "current_sha256": digest(after["raw"]), "python_ast_identical": same_ast})
    slices[lesson] = {"current_sha256": digest(body), "current_locator": f"course/chapters/13.md#{lesson}; lines {first}–{first + body.count(bytes([10])) - 1}", "current_first_line": first, "current_snapshot": snapshot.relative_to(ROOT).as_posix(), "prior_snapshot": old_path.relative_to(ROOT).as_posix(), "prior_sha256": digest(old), "raw_bytes_identical": body == old, "personally_read_current_section_this_round": read_this_round, "read_scope": "entire section incl details" if read_this_round else "current bytes/hash comparison only; original entire section was read in prior review", "fences": fence_checks}
    if lesson in {"13.16", "13.13", "13.15"}: assert body == old
    if lesson == "13.12": assert body == old.replace("收集這批迴答".encode(), "收集這批回答".encode(), 1)
    if lesson == "13.14": assert body == old.replace(b'4))\n\n```', b'4))\n```', 1)

(A / "current-chapter13-frozen-input.md").write_bytes(whole)
(A / "own-input-diffs.patch").write_text("\n".join(diffs))
method = ROOT / "docs/review-tools/factual-reviewer-instructions.md"
(A / "reviewer-method-current.md").write_bytes(method.read_bytes())

fingerprints = []
for item in report["artifacts"]:
    path = ROOT / item["path"]
    actual = digest(path.read_bytes())
    assert actual == item["sha256"], item["path"]
    fingerprints.append({"id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "actual_sha256": actual, "unchanged": True})
code_fingerprints = []
for source in report["sources"]:
    if source["kind"] != "repository_code": continue
    actual = digest((ROOT / source["path"]).read_bytes())
    assert actual == source["sha256"]
    code_fingerprints.append({"source_id": source["id"], "path": source["path"], "recorded_sha256": source["sha256"], "actual_sha256": actual, "unchanged": True})

receipt = {"schema_version": 1, "kind": "same_owner_context_reinspection", "reviewer_task": OWNER, "recorded_on": datetime.now(UTC).isoformat(), "verdict": "pass", "source": "course/chapters/13.md#13.16", "source_sha256": slices["13.16"]["current_sha256"], "own_prior_history": {"path": history.relative_to(ROOT).as_posix(), "sha256": digest(prior_raw), "preservation": "verbatim opaque copy before parsing only this owner's own prior report; original artifacts unchanged"}, "source_and_context": slices, "frozen_current_full_input": {"path": (A / "current-chapter13-frozen-input.md").relative_to(ROOT).as_posix(), "sha256": digest(whole), "meaning": "frozen bytes for this reinspection; does not mean full chapter content read this round or future live version"}, "method_instructions": {"path": method.relative_to(ROOT).as_posix(), "sha256": digest(method.read_bytes()), "snapshot": (A / "reviewer-method-current.md").relative_to(ROOT).as_posix(), "personally_read": True}, "own_comparison": {"13.12": "Only 迴答→回答 in the old-policy definition; ratio, saved old probability, reference distinction, formulas, fence and authority link are unchanged.", "13.14": "Only one empty line before the first closing Python fence removed. The Python AST, reward/critic/old/reference roles, KL formula and reference objective remain unchanged.", "13.16": "Own current complete reread; bytes unchanged and all original ten substantive claims retain their original support scope.", "decision": "These two context changes introduce no new scientific claim or changed computation. The fixed reference/clipping statement in13.16 remains supported by original PPO/DPO/InstructGPT equations and the original bounded counterexample. Existing task/measurement/scope evidence remains valid; no unresolved question."}, "unchanged_original_method_sources": code_fingerprints, "unchanged_prior_evidence": fingerprints, "reused_evidence_scope": "Original unmodified fence/exercise and bounded KL/PPO CPU checks; raw-sample measurement audit; personally inspected exact-version original papers/official docs; original desktop/mobile isolated section renders. All registered prior artifact hashes verified, no CPU repetition or paper retrieval this round.", "figure_scope": "13.16 has no figure references and remains unchanged.13.14 image reference itself is unchanged; its visual content is not needed for the13.16 reward-criterion claim and was not newly inspected this round.", "visual_scope": "The supplied live preview URL was not opened during this narrow context check. Prior genuine isolated-section render/view evidence retained unchanged; no new published-site visual claim.", "limitations": ["Same original owner context reinspection, not a new fresh reviewer.", "Only13.16,13.12,13.14 personally reread this round;13.13/13.15 hash comparison only.", "No full chapter reread, new independent reader review, new CPU experiment, original paper fetch, GPU/train/model data/weight action.", "Existing no-observed-hacking claim remains bounded to saved finite rows."], "unresolved_questions": [], "execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-13_16-context-reinspection-20261005/reinspect.py", "cwd": str(ROOT), "python": sys.version, "device": "cpu, bytes/hash/AST comparisons only", "code_sha256": digest(Path(__file__).read_bytes())}}
receipt_path = A / "context-reinspection-receipt.json"
dump(receipt_path, receipt)
artifact_id = "context-reinspection-20261005"
for path in sorted(A.iterdir()):
    identifier = artifact_id if path == receipt_path else "context-reinspection-" + path.name.replace(".", "-")
    kind = "derivation" if path.name in {"context-reinspection-receipt.json", "own-input-diffs.patch"} else "code" if path.suffix == ".py" else "source_snapshot"
    report["artifacts"].append({"id": identifier, "kind": kind, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes()), "description": "Same original13.16 owner's actual context reinspection: " + path.name})
report["read_scope"]["course/chapters/13.md"] = ["13.16 complete incl details personally reread; current source bytes unchanged, locator578–619", "13.12 current complete personally reread against own frozen prerequisite; only迴答→回答; locator373–404", "13.14 current complete personally reread against own frozen prerequisite; only blank fence line removed; locator443–492", "13.13 and13.15 compared by section hash only this round and unchanged; original whole-section reads retained in prior frozen inputs"]
report["reinspection_context"] = "same_original_owner"
report.setdefault("reinspections", []).append({"artifact_id": artifact_id, "receipt_path": receipt_path.relative_to(ROOT).as_posix(), "receipt_sha256": digest(receipt_path.read_bytes()), "verdict": "pass", "context_sections": ["13.12", "13.14"], "history_path": history.relative_to(ROOT).as_posix(), "history_sha256": digest(prior_raw)})
for claim in report["claims"]:
    if claim["id"] in {"constraints-do-not-correct-reward", "finite-task-contract", "independent-task-evaluation"}:
        claim["artifact_ids"].append(artifact_id)
for name in ["factual_accuracy", "source_verification", "limitations"]:
    report["checks"][name]["details"] += " Same original owner current context reinspection:13.12 wording correction and13.14 removed empty fence line personally compared against own frozen inputs; effective method/claim scope unchanged; prior artifact/code hashes verified and genuine evidence reused. Canonical context-reinspection-20261005 artifact records actual scope and original history."
assert report["source_sha256"] == slices["13.16"]["current_sha256"]
dump(TARGET, report)
actual = json.loads(TARGET.read_bytes())
assert actual["reviewer_task"] == OWNER and actual["reinspections"][-1]["artifact_id"] == artifact_id
print(json.dumps({"verdict": report["verdict"], "report_path": TARGET.relative_to(ROOT).as_posix(), "new_report_sha256": digest(TARGET.read_bytes()), "source_sha256": report["source_sha256"], "canonical_confirmed": True, "own_prior_history_path": history.relative_to(ROOT).as_posix(), "own_prior_history_sha256": digest(prior_raw), "canonical_artifact_id": artifact_id, "receipt_path": receipt_path.relative_to(ROOT).as_posix(), "receipt_sha256": digest(receipt_path.read_bytes()), "scope": "Personal complete current13.16/13.12/13.14 read, own frozen-byte comparison, unchanged prior evidence/code hash validation;13.13/13.15 hash-only", "unresolved_questions": []}, ensure_ascii=False))
