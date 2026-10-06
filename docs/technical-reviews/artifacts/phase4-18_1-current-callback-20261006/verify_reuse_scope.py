"""Verify pinned own prior evidence and the exact current SFT support slice; no model run."""
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / "docs/technical-reviews/artifacts/phase4-18_1-factual-fresh"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


# Read only the evidence ID/path/hash projection of my opaquely preserved own
# report; do not emit or use its prior verdict/conclusions as validation.
prior = json.loads((HERE / "history/prior-18.1.canonical.json").read_bytes())
matches = []
for item in prior["artifacts"]:
    raw = (ROOT / item["path"]).read_bytes()
    assert sha(raw) == item["sha256"]
    matches.append({"artifact_id": item["id"], "path": item["path"], "expected_original_sha256": item["sha256"], "current_sha256": sha(raw), "original_pinned_bytes_match": True})
for source in prior["sources"]:
    if source.get("kind") == "repository_code":
        assert sha((ROOT / source["path"]).read_bytes()) == source["sha256"]

old = (PRIOR / "inputs/context-7.11.md").read_bytes()
new = (HERE / "inputs/current-context7_11.md").read_bytes()
# SFT mechanism needed by 18.1 is the opening through the gradient-only example.
marker = "固定題目與生成規則".encode()
old_support = old[:old.index(marker)]
new_support = new[:new.index(marker)]
assert old_support == new_support
fence = re.compile(rb"(?ms)^```python\n(.*?)^```[ \t]*$", re.M)
old_fences, new_fences = fence.findall(old), fence.findall(new)
assert old_fences == new_fences and len(new_fences) == 2
fence_hashes = [sha(code) for code in new_fences]
diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True), fromfile="prior-personally-read-7.11", tofile="current-personally-read-7.11"))
(HERE / "context7_11.raw-diff.txt").write_text(diff)
print(diff, end="")

result = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_18_1",
    "original_pinned_formal_artifact_matches": matches,
    "unchanged_prior_formal_artifact_count": len(matches),
    "prior_context_sha256": sha(old),
    "current_context_sha256": sha(new),
    "current_context_source": "course/chapters/07.md#7.11",
    "support_current_lines": [374, 396],
    "support_scope": "SFT answer-token CE mechanism used by 18.1 line9: role/question supplied as context, answer bytes/EOS label targets, ignore_index masking, distinction between gradient demonstration and actual optimizer updates.",
    "support_prefix_old_sha256": sha(old_support),
    "support_prefix_current_sha256": sha(new_support),
    "support_prefix_exact_bytes_unchanged": True,
    "context_python_fences_exact_bytes_unchanged": True,
    "context_python_fence_sha256": fence_hashes,
    "actual_context_change": "Only the supplemental final-results link changes from7.12 to7.17 and explicitly says7.12 first explains family-split rules. The empirical budgets stated there are unchanged; no new SFT mechanism or18.1 capability claim is introduced.",
    "own_current_intro_summary": "章首以較小學生和教師文字或候選比例提出蒸餾路線，要求分别判斷教師可靠性、訊號對齊及學生是否改善；後續用共同真值比較任務品質與成本，不能用像教師或較小檔案代替完成任務。固定數字和短程式只協助理解機制。",
    "prior_evidence_reuse_scope": [
        "Prior original18.1 fence and print-only exercise outputs, exact candidate normalization/argmax/information-loss calculation.",
        "Prior original-paper and official-source inspections for taxonomy, softmax/CE/KL, tokenizer and column identity; unchanged original snapshots and original formal hashes checked. Original accessed dates remain2026-10-05; none re-fetched/re-read as a new paper verification.",
        "Prior exact original bounded CPU cache/forward/backward/one-step CE and CE+KL evidence, only shapes/masking/denominators/gradient/update/frozen-teacher contracts. Current required helper bytes and selected distillation method ASTs verified unchanged; no model run repeated.",
        "Prior original raw teacher provenance/artifact/source-step/token cross-links, only original identity and recorded implementation/training contracts. Raw result full bytes remain identical; no current result-content values or author notes re-read, no full-model score/cost reevaluation."
    ],
    "scope_not_extended": ["No review of7.17 performance results or7.12 split claims; changed link is context navigation only for18.1.", "No original historical weights rehash/re-download/forward; original export-identity limitation is retained.", "No new referenced figure or visual claim.18.1 fig={} and explicit candidate/probability lists remain sufficient; render not applicable, desktop/mobile page layout unverified.", "No GPU, training, download, full recipe, new subagent or source-text edit."],
    "new_model_execution": False,
    "new_authoritative_source_fetch": False,
    "render_applicability": "not_applicable",
    "current_callback_verdict": "pass",
    "verdict_basis": "Personally reread entire own current18.1+intro+necessary current7.11; changed context mechanism/support slice is identical and unchanged original proof/code/source identities match exact pinned prior hashes. No unresolved substantive18.1 issue introduced by this context change."
}
(HERE / "support-and-reuse-facts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k:v for k,v in result.items() if k!='original_pinned_formal_artifact_matches'}, ensure_ascii=False, indent=2))
print("CURRENT NECESSARY CONTEXT SUPPORT AND PINNED PRIOR REUSE SCOPE VERIFIED")
