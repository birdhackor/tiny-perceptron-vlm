"""Same-owner, narrow 9.6 callback: inspect unchanged claims by bytes, not rerun models."""
import difflib
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OWNER = "/root/phase4_factual_coordinator/factual_9_6"
PRIOR_SHA = "e7d173ff3169f13dabc7106aadd072b66f74a23b3da978b9d2628abab389d9e2"
OLD_SECTION_SHA = "3838991ed08b43593c81d5b686966156a83e6822978c2063662d9b3ecf24eac6"
CURRENT_SHA = "4d872c3402ee5ea1f715237fa167d7e81b6dabe6babe240bea138eaa36e08213"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


prior_raw = (HERE / "prior-report.opaque.json").read_bytes()
assert digest(prior_raw) == PRIOR_SHA
# Only this same owner's exact prior report is parsed for artifact inventories and preservation.
# Its verdict is not used to decide whether current prose changed.
prior = json.loads(prior_raw)
assert prior["reviewer_task"] == OWNER
old = (ROOT / "docs/technical-reviews/artifacts/phase4-9_6-independent/extraction/section.md").read_bytes()
assert digest(old) == OLD_SECTION_SHA
chapter_raw = (ROOT / "course/chapters/09.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter_raw))
indices = [i for i, h in enumerate(headings) if h[0].startswith(b"## 9.6 ")]
assert len(indices) == 1
i = indices[0]
end = headings[i + 1].start() if i + 1 < len(headings) else len(chapter_raw)
current = chapter_raw[headings[i].start():end]
first_line = chapter_raw[:headings[i].start()].count(b"\n") + 1
assert current == (HERE / "current-section.md").read_bytes()
assert digest(current) == CURRENT_SHA
changes = [("满", "滿"), ("等于", "等於"), ("须", "須")]
expected = old.decode("utf-8")
counts = []
for before, after in changes:
    count = expected.count(before)
    assert count == 1
    expected = expected.replace(before, after)
    counts.append({"old": before, "new": after, "occurrences": count})
assert expected.encode("utf-8") == current
old_fences = re.findall(rb"(?ms)^```python\r?\n(.*?)^```", old)
new_fences = re.findall(rb"(?ms)^```python\r?\n(.*?)^```", current)
assert len(old_fences) == len(new_fences) == 1
assert old_fences == new_fences
assert digest(new_fences[0]) == "51b9cfaaaf0077ee468373f56a8f006ac1eae2969ddaa862948f1cd13281021e"
assert new_fences[0] == (ROOT / "docs/technical-reviews/artifacts/phase4-9_6-independent/extraction/fence-1.py").read_bytes()
assert not re.search(rb"!\[[^\]]*\]\([^)]*\)", current)
assert not re.search(rb"<(?:img|svg|picture)\b", current, re.I)
assert prior["figure_sha256"] == {}

hash_checks = []
for group, entries in [("artifacts", prior["artifacts"]), ("sources", prior["sources"])]:
    for entry in entries:
        if "path" in entry and "sha256" in entry:
            actual = digest((ROOT / entry["path"]).read_bytes())
            assert actual == entry["sha256"], entry["path"]
            hash_checks.append({"group": group, "id": entry["id"], "path": entry["path"],
                                "expected_sha256": entry["sha256"], "actual_sha256": actual})
assert len(hash_checks) == 28
# Verify the byte-identical schema previously read, with no old-report checker invocation.
checker = ROOT / "scripts/check_technical_reviews.py"
assert digest(checker.read_bytes()) == "e0cc6772faebd51e081f31c09b678f88e1ea748d72be406e6e310b9db2d94ad4"

environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "device": "cpu (hashing, raw extraction and comparison only)",
    "torch_distribution_version": importlib.metadata.version("torch"),
    "tensor_execution_this_callback": "none; original CPU execution is reused with unchanged SHA",
    "network_this_callback": "none",
    "training_or_model_inference": "none",
}
assert environment["torch_distribution_version"] == "2.14.1+cpu"
dump("environment.json", environment)
claim_supports = {
    "c1": "Refusal-vs-completion distinction and handwritten counterexample unchanged; reuse original pinned XSTest §1/§2/§4.2 and original CPU evidence within their prior bounds.",
    "c2": "Raw fence and both requested variations described in prose unchanged; reuse actual original CPU run and matching PyTorch API/source snapshots, not a new tensor execution.",
    "c3": "17=3+14 labels, clarification role, split/holdout language and separate arithmetic denominator unchanged; reuse original record/split code and result provenance checks.",
    "c4": "All table cells and EOS statements unchanged; reuse original saved-generation recomputation, historical run version and per-group denominators, not new model measurements.",
    "c5": "Substring detection, raw target IDs before EOS and independent EOS check unchanged; reuse original evaluator/tokenizer/generation contracts with fixed-template limits.",
}
assert set(claim_supports) == {c["id"] for c in prior["claims"]}
receipt = {
    "reviewer_task": OWNER,
    "callback_date": "2026-10-06",
    "prior_report_opaque_path": (HERE / "prior-report.opaque.json").relative_to(ROOT).as_posix(),
    "prior_report_opaque_sha256": digest(prior_raw),
    "prior_frozen_section_sha256": digest(old),
    "current_source_sha256": digest(current),
    "current_section_first_line": first_line,
    "orthographic_changes": counts,
    "substantive_changed_claims": [],
    "fence_sha256": digest(new_fences[0]),
    "figure_sha256": {},
    "intro": None,
    "evidence_hash_checks": hash_checks,
    "per_claim_support_reuse": claim_supports,
    "scope": "Same original independent owner personally read full current 9.6 and exact diff; three orthographic substitutions only. Original raw histories remain untouched. No new paper reading, source retrieval, numeric/model execution, visual inspection, or full-pipeline verification is claimed.",
}
dump("verification-receipt.json", receipt)
diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), current.decode().splitlines(keepends=True),
                                  fromfile="original-owner-frozen-9.6", tofile="current-9.6"))
(HERE / "section.diff").write_text(diff, encoding="utf-8")
print("OWNER", OWNER)
print("PRIOR_OPAQUE_SHA256", digest(prior_raw))
print("CURRENT_SOURCE_SHA256", digest(current))
print("CHANGES", json.dumps(counts, ensure_ascii=False))
print("FENCE_IDENTICAL", digest(new_fences[0]))
print("REUSED_EVIDENCE_HASHES", len(hash_checks), "all exact")
print("SUBSTANTIVE_CHANGED_CLAIMS", "none; c1,c2,c3,c4,c5 support scopes unchanged")
print("FIGURES", "none; no visual claim added; rendering not applicable")
print("PASS same-owner narrow current-section verification; original CPU/source proof reused explicitly")
