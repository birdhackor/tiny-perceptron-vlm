"""Same-owner 9.8 callback: preserve originals, verify fingerprints, report narrow read."""

import datetime
import difflib
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-9_8-independent"
REL = BASE.relative_to(ROOT).as_posix()
COMMAND = ".venv/bin/python docs/technical-reviews/artifacts/phase4-9_8-callback-20261006/callback.py"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, lesson):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    position = next(i for i, h in enumerate(headers) if h[0].startswith(("## " + lesson + " ").encode()))
    end = headers[position + 1].start() if position + 1 < len(headers) else len(raw)
    return raw[headers[position].start():end]


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


prior_path = BASE / "prior-report.opaque.json"
prior_raw = prior_path.read_bytes()
preservation = json.loads((BASE / "initial-preservation.json").read_bytes())
assert digest(prior_raw) == preservation["prior_report_sha256"]
prior = json.loads(prior_raw)  # Own full original report only; no other reviewer reports.
assert prior["reviewer_task"] == "/root/phase4_factual_coordinator/factual_9_8"
current = section((ROOT / "course/chapters/09.md").read_bytes(), "9.8")
assert current == (BASE / "current-section.md").read_bytes()
assert digest(current) == "ced668b386f76ddf1047b0b3131ade93a786958e9db174b0bf97d1c58651fe28"
frozen = (OLD / "execution/section.md").read_bytes()
assert digest(frozen) == prior["source_sha256"]
actual_diff = "".join(difflib.unified_diff(frozen.decode().splitlines(True), current.decode().splitlines(True),
                                        fromfile="own original raw section", tofile="current raw 9.8"))
(BASE / "section.diff").write_text(actual_diff, encoding="utf-8")
print(actual_diff)
current_fences = re.findall(rb"(?ms)^```python\n(.*?)^```", current)
frozen_fences = re.findall(rb"(?ms)^```python\n(.*?)^```", frozen)
assert current_fences == frozen_fences
assert len(current_fences) == 1
assert digest(current_fences[0]) == "c138997bd2249a90bd193c7496321be1d4008bfcb901e7a0391ce344960fbf26"
assert re.findall(rb"\d+", current) == re.findall(rb"\d+", frozen)
assert re.findall(rb"https://[^)]+", current) == re.findall(rb"https://[^)]+", frozen)
assert re.findall(rb"!\[[^\]]*\]\([^)]+\)", current) == []

fingerprints = []
for artifact in prior["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], artifact["id"]
    fingerprints.append({"artifact_id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        assert digest((ROOT / source["path"]).read_bytes()) == source["sha256"], source["id"]
    elif source["kind"] == "official_docs":
        assert digest((ROOT / source["snapshot_path"]).read_bytes()) == source["snapshot_sha256"], source["id"]

live_fingerprints = []
for filename in ["scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py",
                 "scripts/course_experiments/text.py", "tiny_perceptron/data.py",
                 "docs/course-experiments/results/safety.json", "docs/course-experiments/results/style.json"]:
    live = (ROOT / filename).read_bytes()
    saved = (OLD / "inputs" / filename).read_bytes()
    assert live == saved, filename
    live_fingerprints.append({"path": filename, "sha256": digest(live), "comparison": "exact match own frozen input"})
for filename in ["tiny_perceptron/model.py", "tiny_perceptron/training.py"]:
    live = (ROOT / filename).read_bytes()
    saved = (OLD / "original-code" / filename).read_bytes()
    assert live == saved, filename
    live_fingerprints.append({"path": filename, "sha256": digest(live), "comparison": "exact match original-run code already personally inspected"})

contexts = []
for filename, lesson in [("course/chapters/08.md", "8.3"), ("course/chapters/09.md", "9.2"), ("course/chapters/09.md", "9.6")]:
    new = section((ROOT / filename).read_bytes(), lesson)
    old = section((OLD / "inputs" / filename).read_bytes(), lesson)
    assert new == (BASE / ("current-context-" + lesson + ".md")).read_bytes()
    diff = "".join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                                     fromfile="own frozen " + lesson, tofile="current " + lesson))
    (BASE / ("context-" + lesson + ".diff")).write_text(diff, encoding="utf-8")
    contexts.append({"source": filename + "#" + lesson, "current_sha256": digest(new),
                     "own_frozen_sha256": digest(old), "inspection": "Personally read complete necessary context and actual diff: 8.3 unchanged; 9.2 and 9.6 only orthographic glyph corrections. Relevant baseline 0/7, unknown-count example and separate behavior denominators unchanged."})

environment = {"python": sys.version, "python_executable": sys.executable,
               "torch_distribution": importlib.metadata.version("torch"),
               "device": "CPU file/hash comparison; no torch/model execution", "cwd": str(ROOT)}
json_write(BASE / "environment.json", environment)
inspection = {
    "reviewer_task": prior["reviewer_task"], "reviewed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "current_source_sha256": digest(current), "own_original_source_sha256": digest(frozen),
    "prior_report_opaque": {"path": preservation["prior_report_path"], "sha256": digest(prior_raw),
                            "preservation_scope": "Complete raw prior report, including all original issues/history fields, retained before new report; none omitted or regenerated."},
    "actual_read": ["Latest factual-reviewer-instructions.md in full", "Exact clear-tutorial/references/review-protocol.md in full",
                    "Current full 9.8 and actual diff from own frozen raw input", "Complete necessary current 8.3, 9.2, 9.6 and actual diffs"],
    "changed_claims": [],
    "manual_change_assessment": "Only glyph corrections: 几颗→幾顆, 记→記, 扰→擾, 辞→辭, 颖→穎, 幹擾→干擾. The hypothetical guess3, family design, code/fence, exercise, original data/recipe, all numeric/table results and finite-template/multi-turn limitations retain the same meaning. Relevant context changes are also orthographic. The raw diff above, rather than an author correction summary, was the basis of this assessment.",
    "automated_checks": {"fence_raw_bytes_identical": True, "numeric_literal_sequence_identical": True,
                         "external_source_urls_identical": True, "no_image_references": True},
    "prior_artifact_fingerprints": fingerprints, "live_inputs": live_fingerprints, "necessary_context": contexts,
    "reuse_scope_by_claim": {
        "heldout-method": "Prior personally inspected CPython/sklearn sources support unchanged family/test independence reasoning; opening remains hypothetical rather than saved model generation3.",
        "python-fence": "Identical raw fence and identical exercise; reuse original actual CPU fence execution and test-only variation, not a new execution claim.",
        "data-split": "Unchanged original generators/result bytes and previous reconstructed JSONL SHA; reuse168→136,102/17/17,6/1/1 and no-overlap proof.",
        "training-recipe": "Unchanged code and saved result bytes support same-base copies,102/151 records,900 steps and seed42. Reuse saved original training record inspection, not rerun training.",
        "effective-targets": "Unchanged render_chat/sampler inputs; reuse previous bounded900x16 sampling check and394888/275389 effective targets.",
        "original-metrics": "Unchanged original result IDs and previous recomputation support15/17,16/17,14/17 and separate arithmetic/base0/7.",
        "wording-metrics": "Unchanged saved prompts/IDs and previous independent reaggregation support0/6 plus6/6EOS and green→red contrast only for two suffix changes.",
        "exposure-and-scope": "Same templates/colors and wording limits; original six cases remain finite evidence and proposed multi-turn trace remains unmeasured.",
    },
    "not_redone": "No original-paper/document reread/refetch, prior numeric audit rerun, model inference, training, GPU, download or full pipeline. File fingerprints and current claim scope were actually checked now. No visual claim/image exists, so no render/view claimed.",
    "verdict": "pass", "unresolved_substantive_issues": [],
}
json_write(BASE / "actual-inspection.json", inspection)
print("UNCHANGED_PRIOR_ARTIFACTS", len(fingerprints))
print("CURRENT_FENCE_SHA256", digest(current_fences[0]))
print("UNCHANGED_LIVE_IMPLEMENTATION_AND_MEASUREMENTS", len(live_fingerprints))
print("ACTUAL_CHANGED_CLAIMS", len(inspection["changed_claims"]))


def artifact(identifier, filename, kind, description, **extra):
    return {"id": identifier, "path": REL + "/" + filename, "sha256": digest((BASE / filename).read_bytes()),
            "kind": kind, "description": description, **extra}


new = json.loads(prior_raw)
new["source_sha256"] = digest(current)
new["reviewed_on"] = "2026-10-06"
new["verdict"] = inspection["verdict"]
new["figure_sha256"] = {}
for artifact_item in new["artifacts"]:
    if artifact_item["id"] == "current-section":
        artifact_item["description"] = "Historical own initial raw UTF-8 9.8 frozen input, SHA266c17...; preserved unchanged and not the current glyph-corrected section. Current bytes are callback-current-section."
    if artifact_item["id"] == "extraction":
        artifact_item["description"] = "Historical own initial raw section/fence extraction; full source-file hash denotes its original frozen input only. Current section is recorded separately."
new["artifacts"] += [
    artifact("callback-prior-report-opaque", "prior-report.opaque.json", "source_snapshot", "Complete opaque prior owner report preserved byte-for-byte including issues/history; canonical permanent artifact."),
    artifact("callback-current-section", "current-section.md", "source_snapshot", "Personally read complete current raw UTF-8 9.8, ced668...; no normalization."),
    artifact("callback-code", "callback.py", "code", "Actually executed same-owner narrow callback: raw diff, evidence fingerprint and support-scope record; no numeric-model rerun."),
    artifact("callback-diff", "section.diff", "source_snapshot", "Real diff of own frozen raw section versus current raw section; only glyph edits."),
    artifact("callback-environment", "environment.json", "source_snapshot", "Actual callback Python process and installed CPU distribution version; no model execution."),
    artifact("callback-inspection", "actual-inspection.json", "execution", "Actual current read/diff, exact evidence fingerprints and per-claim reuse limits.",
             command=COMMAND, result="Completed current/full-context inspection, all22 prior artifact fingerprints and8 live input fingerprints exact, unchanged fence, no changed claims; narrow callback pass.",
             environment=environment),
    artifact("callback-initial-preservation", "initial-preservation.json", "source_snapshot", "Actual first-step preservation timestamp and full original report/current raw hashes."),
]
for lesson in ["8.3", "9.2", "9.6"]:
    new["artifacts"].append(artifact("callback-context-" + lesson, "current-context-" + lesson + ".md", "source_snapshot", "Personally read current necessary context " + lesson + "; full raw section bytes."))
new["artifacts"] += [
    artifact("callback-method", "method-inputs/docs/review-tools/factual-reviewer-instructions.md", "source_snapshot", "Latest factual review instructions actually read for this callback."),
    artifact("callback-protocol", "method-inputs/.agents/skills/clear-tutorial/references/review-protocol.md", "source_snapshot", "Exact clear-tutorial review protocol actually read for this callback."),
]
new["read_scope"] = {
    "current": "Personally read complete current9.8 plus necessary current8.3/9.2/9.6 and compared actual diffs to my own frozen input. Nonfirst section, no intro. All changes are orthographic, no substantive claim changes.",
    "original_evidence": "Reuse only own prior personally inspected official originals, named original code/measurement and actual CPU proof, after exact current fingerprint checks recorded by callback-inspection. No external fetch/reread or numeric rerun is claimed on2026-10-06.",
    "figures": "Current raw section has no image and figure_sha256={}. No visual claim changed; no render/view was needed or claimed.",
    "independence": "Same original owner callback, not a fresh replacement task. Only own prior report preserved; no other review reports or author repair summaries read.",
}
new["callback_review"] = {
    "kind": "same_original_owner_narrow_callback", "reviewer_task": prior["reviewer_task"],
    "prior_report_artifact_id": "callback-prior-report-opaque", "prior_report_sha256": digest(prior_raw),
    "current_source_sha256": digest(current), "changed_claim_ids": [],
    "actual_inspection_artifact_id": "callback-inspection",
    "reused_evidence": "All previous8 claim scopes retain exact supporting source/code/measurement/CPUproof fingerprints. Prior executed numeric/empirical verification remains historical execution, not a new run.",
}
for claim in new["claims"]:
    claim["callback_review"] = {"status": "scope_reconfirmed", "details": inspection["reuse_scope_by_claim"][claim["id"]],
                                "inspection_artifact_id": "callback-inspection", "no_new_execution_claim": True}
    claim["artifact_ids"].append("callback-inspection")
for check in new["checks"].values():
    check["details"] += " Same-owner2026-10-06 callback: current glyph-only edits manually inspected; exact supporting fingerprints and scopes reconfirmed; historical executed proof reused without new execution. See callback-inspection."
json_write(ROOT / "docs/technical-reviews/9.8.json", new)
print("NEW_REPORT_SHA256", digest((ROOT / "docs/technical-reviews/9.8.json").read_bytes()))
print("CALLBACK_COMPLETE")
