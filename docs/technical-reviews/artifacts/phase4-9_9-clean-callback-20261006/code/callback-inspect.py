"""Owner's narrow current-section callback: actual diffs and exact evidence fingerprints."""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
A = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-9_9-clean"
TASK = "/root/phase4_factual_coordinator/factual_9_9_clean"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def section(path, lesson):
    raw = path.read_bytes()
    hs = list(re.finditer(rb"(?m)^## ([0-9]+\.[0-9]+) [^\r\n]+", raw))
    i = next(i for i, h in enumerate(hs) if h[1].decode() == lesson)
    end = hs[i + 1].start() if i + 1 < len(hs) else len(raw)
    return raw[hs[i].start():end], raw[:hs[i].start()].count(b"\n") + 1

current, line = section(ROOT / "course/chapters/09.md", "9.9")
frozen = (OLD / "inputs/section-9.9.md").read_bytes()
assert sha(current) == "e662bd705ef772ece0df9e894d576de49a3df826f621776969f75fa03b3f44db"
assert sha(frozen) == "d4d2a9db5fad831772da5ad7f7bd61bded3450deb561ffec5d2694f5ce6deb1d"
assert current == (A / "inputs/current-section-9.9.md").read_bytes()
replacements = {
    "颜色": "顏色", "绿编": "綠編", "们转": "們轉", "比较": "比較", "標准": "標準",
    "四舍": "四捨", "调整": "調整", "用于": "用於", "校准": "校準", "概率": "機率", "合适": "合適",
}
translated = frozen.decode()
applied = []
for before, after in replacements.items():
    count = translated.count(before)
    assert count > 0
    translated = translated.replace(before, after)
    applied.append({"before": before, "after": after, "occurrences": count,
                    "classification": "same probability term" if before == "概率" else "traditional character spelling"})
assert translated.encode() == current
old_fence = re.search(rb"```python\n(.*?)```", frozen, re.S)[1]
new_fence = re.search(rb"```python\n(.*?)```", current, re.S)[1]
assert old_fence == new_fence
context, _ = section(ROOT / "course/chapters/12.md", "12.8")
assert context == (OLD / "inputs/prerequisite-12.8.md").read_bytes()
assert context == (A / "inputs/current-needed-context-12.8.md").read_bytes()
archive = A / "prior/9.9-prior-complete-report.opaque.json"
prior_raw = archive.read_bytes()
assert sha(prior_raw) == "fbfdd78723ac9099d1b77b9718994c8a571e06bd5659e9388c3be762834cdd83"
# Parse only this same owner's preserved report to enumerate the exact existing proof dependencies.
# No peer reports, verdict text or scientific answers are printed or used as a new proof.
prior = json.loads(prior_raw)
assert prior["reviewer_task"] == TASK
checks = []
for artifact in prior["artifacts"]:
    p = ROOT / artifact["path"]
    actual = sha(p.read_bytes())
    checks.append({"id": artifact["id"], "kind": "prior_owned_artifact", "path": artifact["path"],
                   "expected_sha256": artifact["sha256"], "actual_sha256": actual, "matches": actual == artifact["sha256"]})
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        actual = sha((ROOT / source["path"]).read_bytes())
        checks.append({"id": source["id"], "kind": "current_primary_source", "path": source["path"],
                       "expected_sha256": source["sha256"], "actual_sha256": actual, "matches": actual == source["sha256"]})
for path, expected in prior.get("prerequisite_figure_sha256", {}).items():
    actual = sha((ROOT / path).read_bytes())
    checks.append({"kind": "necessary_context_figure", "path": path, "expected_sha256": expected,
                   "actual_sha256": actual, "matches": actual == expected})
assert all(x["matches"] for x in checks)
claim_scope = {
    "softmax-definition": "Current candidate-score/normalization/truth distinction unchanged; traditional spellings only. Existing original softmax formula and calibration definition support exactly the same meaning.",
    "toy-probabilities": "Current scores, candidate order, target1, scales1/10, values0.6652/1.0 and below1 rounding explanation unchanged; 四舍→四捨 only. Original code and numeric proof identical.",
    "code-api-coverage": "Current shape/axis/max/item/basicPython explanation and original fence bytes identical; sameAPI scope and CPU proof retained.",
    "temperature-scope": "调整/用于/校准/合适 traditionalized and 概率→機率 expresses the same probability concept. Positive temperature, same ranking, calibration and finite-candidate limitations unchanged; original Eq9 support remains exact.",
    "relabel-exercise": "Current correct_index0 exercise is byte-identical; previously executed unchanged-probability/changed-correctness variant retained.",
    "calibration-reliability": "Many independent labeled examples, matching confidence with actual correctness and no whole-answer/safety guarantee remain unchanged; original calibration scope retained.",
    "historical-audio-task": "Current supplement is byte-identical. Necessary12.8 context is byte-identical; original result/config/instrument SHA all match. Same synthetic low/high task, training/report denominators and heldout sample scope retained.",
    "historical-audio-confidence": "Current320Hz/.5/.12s,T.5,.7490→.8990,wronglow and testargmax invariance wording is byte-identical. Same original result/input/pointers, instrument, saved-logit proof and denominator scope retained.",
}
assert {c["id"] for c in prior["claims"]} == set(claim_scope)
record = {
    "reviewer_task": TASK, "callback_kind": "same original independent owner; narrow current-version callback",
    "performed_on": "2026-10-06", "current_source": "course/chapters/09.md#9.9",
    "current_source_sha256": sha(current), "current_section_first_line": line,
    "prior_frozen_section_sha256": sha(frozen), "fence_sha256": sha(new_fence),
    "actual_read_scopes": [
        "Complete current9.9 body, including all prose, original fence and supplement; complete actual diff to this owner's frozen raw input.",
        "Complete current necessary12.8 context and its actual raw diff (identical) to this owner's frozen context.",
        "Latest factual-reviewer-instructions.md, exact clear-tutorial/references/review-protocol.md and current check_technical_reviews.py schema.",
        "Actual bytes and hashes of this owner's37 unchanged artifacts/primary sources/necessary figure; source bodies and external papers not reread because only orthography/term synonym changed.",
    ],
    "actual_diff_path": str((A / "inputs/actual-raw-section.diff").relative_to(ROOT)),
    "orthography_and_same_term_changes": applied,
    "substantively_changed_claim_ids": [], "current_claim_scope_checks": claim_scope,
    "necessary_context_sha256": sha(context), "exact_evidence_fingerprint_checks": checks,
    "reuse_decision": "Same owner explicitly reuses previously performed original source verification, exact originalCPU fence/variants/saved-logit replay, and unchanged necessarySVG render/view. Every retained scientific claim keeps exactly its prior supported scope; this callback does not claim any of those operations were newly rerun.",
    "reuse_limits": "HistoricalCUDA training is not rerun; CPU numeric proof remains2.14.1+cpu, originalAPI sources remain pinnedPyTorchv2.9.0, paper remainsGuoarXiv1706.04599v2. No new generalization, reliability, complete-answer, speech or safety inference. No current page render: no directfigure, visualclaim or necessarycontext changed; prior visual proof is explicitly historical and fingerprint-identical.",
    "original_raw_json_policy": "Originalencoders.json fullSHA is fingerprint-identical; no current JSON values, author result notes or review/scope_correction annotations were read. Existing exact measurement-pointer receipt remains the true prior inspection scope.",
    "independence": "No repair expectations, peer report answers, oldVOID identities or peer proofs read. Prior own complete report preserved opaque; exact hashes/dependency metadata used only for this same-owner callback.",
    "opaque_prior": {"path": str(archive.relative_to(ROOT)), "sha256": sha(prior_raw), "bytes": len(prior_raw)},
    "source_file_fingerprint_policy": "Prior whole-fileSHA remains historical; canonical binding is this callback's current sectionSHA. Capture-receipt explicitly labels callback whole-file capture as a frozen observation.",
    "visual_claim_applicability": "No9.9 diagram or visualclaim changed; no render required in this narrowfactual callback.",
    "unresolved_substantive_questions": [], "verdict": "pass",
    "environment": {"python": sys.version, "python_executable": sys.executable, "device": "CPU hash/diff inspection; no tensor/model execution"},
}
(A / "execution/actual-inspection.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"verdict":record["verdict"], "section_sha256":sha(current), "identical_fence_sha256":sha(new_fence),
                  "only_orthography_or_same_probability_term_changes":True, "unchanged_necessary_context":True,
                  "fingerprint_checks":len(checks), "matched":sum(x["matches"] for x in checks),
                  "substantively_changed_claims":[], "reused_existing_proof":True,
                  "new_model_or_tensor_execution":False},ensure_ascii=False,indent=2))
