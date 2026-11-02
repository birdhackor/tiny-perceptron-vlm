"""Same W.7 owner, source12 bounded callback. Only stdlib static checks."""

import ast
import difflib
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = "docs/course-revision-20261029-omni/reviews/P5-actual/"
ART = "docs/technical-reviews/artifacts/omni-P5-W.7/"


def raw(path):
    return (ROOT / path).read_bytes()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def ref(path):
    return {"path": path, "sha256": sha(raw(path)), "bytes": len(raw(path))}


def load(path):
    return json.loads(raw(path))


def page(freeze, sid):
    return next(p for p in freeze["pages"] if p["page_id"] == sid)


def section(path, sid):
    match = re.search(r"^## " + re.escape(sid) + r" [^\n]*\n.*?(?=^## |\Z)", raw(path).decode(), re.M | re.S)
    assert match, (path, sid)
    return match[0].encode()


previous_path = BASE + "technical-W.7/technical-current-v1.json"
previous = load(previous_path)
original_path = previous["original_report"]["path"]
assert sha(raw(original_path)) == previous["original_report"]["sha256"]
old_canonical_path = previous["canonical_source_freeze"]["path"]
old_numbered_path = previous["source_freeze"]["path"]
new_canonical_path = BASE + "source-freeze-v12.json"
new_numbered_path = BASE + "technical-numbered-source-freeze-v7.json"
repairs_path = BASE + "coherence-three-author-repairs-source12-v1.json"
assert sha(raw(new_canonical_path)) == "1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
assert sha(raw(new_numbered_path)) == "f7ba7cc4df7edfb693f2024f819e8886a0fabba56bdfb0b6b94f18c364551423"
assert sha(raw(repairs_path)) == "76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"
assert sha(raw(old_canonical_path)) == previous["canonical_source_freeze"]["sha256"]
assert sha(raw(old_numbered_path)) == previous["source_freeze"]["sha256"]
old_canonical, new_canonical = load(old_canonical_path), load(new_canonical_path)
old_numbered, new_numbered = load(old_numbered_path), load(new_numbered_path)
assert new_numbered["canonical_source_freeze"] == {"path": new_canonical_path, "sha256": sha(raw(new_canonical_path))}
old_w7, new_w7 = page(old_numbered, "W.7"), page(new_numbered, "W.7")
assert raw(old_w7["snapshot"]) == raw(new_w7["snapshot"]) == section("course/first-steps.md", "W.7")
assert sha(raw(new_w7["snapshot"])) == previous["source_sha256"] == new_w7["source_sha256"]
assert old_w7["figures_sha256"] == new_w7["figures_sha256"] == {}
assert raw(page(old_canonical, "first-steps")["snapshot"]) == raw(page(new_canonical, "first-steps")["snapshot"]) == raw("course/first-steps.md")

repo_sources = [s for s in previous["sources"] if s["kind"] == "repository_code"]
drift = []
for s in repo_sources:
    digest = sha(raw(s["path"]))
    if digest != s["sha256"]:
        drift.append({"source_id": s["id"], "path": s["path"], "previous_sha256": s["sha256"], "current_sha256": digest})
assert {s["source_id"] for s in drift} == {"s_repo_16", "s_repo_21", "s_repo_28"}
for a in previous["artifacts"]:
    assert sha(raw(a["path"])) == a["sha256"]

prior_mapping_path = ART + "callback-source-check-v1.json"
prior_mapping = load(prior_mapping_path)
mapping_drift = []
for item in prior_mapping["mapped_section_rechecks"]:
    digest = sha(section(item["path"], item["section"]))
    if digest != item["current_sha256"]:
        mapping_drift.append({"package": item["package"], "section": item["section"], "path": item["path"],
                              "previous_sha256": item["current_sha256"], "current_sha256": digest})
assert len(mapping_drift) == 1 and mapping_drift[0]["section"] == "19.23"

repairs = load(repairs_path)
assert {x["page_id"] for x in repairs["changes"]} == {"9.17", "12.17", "19.23"}
changes = []
for sid in ["9.17", "12.17", "19.23"]:
    old, new = page(old_canonical, sid), page(new_canonical, sid)
    before, after = raw(old["snapshot"]), raw(new["snapshot"])
    assert sha(before) == old["source_sha256"] and sha(after) == new["source_sha256"]
    declared = next(x for x in repairs["changes"] if x["page_id"] == sid)
    assert declared["before"] == {"path": old["snapshot"], "sha256": old["source_sha256"]}
    assert declared["after"] == {"path": new["snapshot"], "sha256": new["source_sha256"]}
    assert after == section(new["source"], sid)
    current_chapter = raw(new["source"])
    assert current_chapter.count(after) == 1
    restored_sha = sha(current_chapter.replace(after, before, 1))
    registered = next(s for s in drift if s["path"] == new["source"])
    assert restored_sha == registered["previous_sha256"]
    old_blocks = re.findall(r"```python\n(.*?)```", before.decode(), re.S)
    new_blocks = re.findall(r"```python\n(.*?)```", after.decode(), re.S)
    for block in new_blocks:
        ast.parse(block)
    if sid == "9.17":
        old_line = 'chosen = min(range(len(ids)), key=lambda i: (-float(p[i]), ids[i]))'
        new_line = 'chosen = min(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))'
        assert before.count(old_line.encode()) == 1 and before.replace(old_line.encode(), new_line.encode(), 1) == after
    else:
        assert old_blocks == new_blocks
    changes.append({"section": sid, "source_id": registered["source_id"], "source": new["source"],
                    "before": ref(old["snapshot"]), "after": ref(new["snapshot"]),
                    "diff": "".join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                                          fromfile="source10/" + sid, tofile="source12/" + sid)),
                    "restored_whole_chapter_sha256": restored_sha, "matches_previous_whole_chapter": True,
                    "python_block_count": len(new_blocks), "python_blocks_identical": old_blocks == new_blocks,
                    "python_block_sha256_before": [sha(b.encode()) for b in old_blocks],
                    "python_block_sha256_after": [sha(b.encode()) for b in new_blocks],
                    "figure_list_unchanged": old["figures_sha256"] == new["figures_sha256"]})

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": previous["reviewer_task"],
    "kind": "source12 same-owner bounded callback; static source recheck, not new blind lesson reviews",
    "original_report": ref(original_path), "previous_current_report": ref(previous_path),
    "previous_canonical_freeze": ref(old_canonical_path), "current_canonical_freeze": ref(new_canonical_path),
    "previous_numbered_freeze": ref(old_numbered_path), "current_numbered_freeze": ref(new_numbered_path),
    "author_diff_input": {**ref(repairs_path), "use": "Only source locators/diffs; independently checked against actual snapshots and whole-chapter SHA restoration, not accepted as a verdict."},
    "w7": {"previous_snapshot": ref(old_w7["snapshot"]), "current_snapshot": ref(new_w7["snapshot"]),
           "assigned_unit_identical": True, "whole_first_steps_identical": True, "figure_sha256": {}},
    "repository_source_count_rechecked": len(repo_sources), "repository_source_drift": drift,
    "unchanged_repository_source_count": len(repo_sources) - len(drift),
    "previous_artifact_references_rechecked": [{"id": a["id"], "path": a["path"], "sha256": a["sha256"]} for a in previous["artifacts"]],
    "all_previous_artifact_bytes_unchanged": True,
    "mapping_basis_reused_by_reference": ref(prior_mapping_path),
    "mapped_reference_count_rehashed": len(prior_mapping["mapped_section_rechecks"]),
    "unchanged_mapped_reference_count": len(prior_mapping["mapped_section_rechecks"]) - len(mapping_drift),
    "changed_mapped_targets": mapping_drift, "actual_changed_units": changes,
    "reviewer_assessment": {
        "9.17": "Only the selected-key switches p[i] to raw scores[i]. W.7 points to 9.10/reliability_bins in chapter09, not 9.17. Restoring old9.17 reproduces the prior wholechapter09 SHA, proving that mapped9.10 is unchanged. No new W.7 numerical or selection claim is introduced.",
        "12.17": "The added paragraph identifies mean-square amplitude and matches the unchanged raw.square().mean() code. W.7 chapter12 targets12.1/12.6/12.8/12.14/12.15 are unchanged; restoring old12.17 reproduces prior wholechapter12 SHA. This is not a fresh review of all12.17 conclusions.",
        "19.23": "Rubric criteria may reside jointly in question and level descriptions; the visible rubric example and formal model computation remain unchanged. This does not change the W.7 promise that new candidate judgment computation is embedded in19.23. All Python blocks are byte-identical; heading and target remain valid.",
        "verdict": "W.7 pass remains supported; no new necessary or optional W.7 issue.",
    },
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; stdlib hashes/byte comparison/AST parsing only"},
    "limits": ["No NN execution, backward, optimizer update, training, tuning, final-family inference, Modal, workflow, installation or publishing.",
               "No rerun or copy of unchanged original/current1 evidence; those files are referenced in place.",
               "No new27/50-lesson or whole-page blind review.69 table references were rehashed, not reread as independent full lesson reviews.",
               "No sidecar dependency or bundle binding is needed to support this source/lookup callback; no collection or seal authored."],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
