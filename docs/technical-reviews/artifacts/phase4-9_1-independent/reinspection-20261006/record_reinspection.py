"""Record this owner's actual bounded 9.1 reinspection; no model execution."""
from pathlib import Path
from datetime import UTC, datetime
import ast
import difflib
import hashlib
import json
import platform
import re
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = OUT.parents[4]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def file_sha(path):
    return sha(Path(path).read_bytes())

source = ROOT / "course/chapters/09.md"
raw = source.read_bytes()
raw.decode("utf-8")
heads = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
assert heads[0][0].startswith(b"## 9.1 ")
intro = raw[:heads[0].start()]
section = raw[heads[0].start():heads[1].start()]
assert sha(section) == "63cc0d9a971bad00cad59ba5ef43a127ecb146e1cb5b87627c9fa2859f68c72a"
old = (BASE / "original-fence/section.md").read_bytes()
old_intro = (BASE / "intro.md").read_bytes()
fence = re.search(rb"```python\n(.*?)\n```", section, re.S)[1] + b"\n"
assert fence == (BASE / "original-fence/fence-1.py").read_bytes()
assert intro == old_intro
for name, data in [("current-intro.md", intro), ("current-section.md", section), ("current-fence.py", fence)]:
    (OUT / name).write_bytes(data)
diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), section.decode().splitlines(keepends=True), fromfile="own-frozen-original", tofile="current"))
(OUT / "section-change.diff").write_text(diff, encoding="utf-8")
context_lines = raw.splitlines(keepends=True)[40:60]
(OUT / "current-necessary-context.md").write_bytes(b"".join(context_lines))

frozen_hashes = json.loads((BASE / "file-sha256.json").read_bytes())
for item in frozen_hashes:
    assert file_sha(ROOT / item["path"]) == item["sha256"], item["path"]
unchanged_contracts = []
for name in ["scripts/course_experiments/behavior.py", "assets/training/sources/pku-safe-rlhf.json", "docs/course-experiments/release-licenses.md"]:
    now = (ROOT / name).read_bytes()
    then = (BASE / "inputs" / name).read_bytes()
    assert now == then
    unchanged_contracts.append({"path": name, "sha256": sha(now), "matches_own_frozen_bytes": True})

code_path = ROOT / "scripts/course_experiments/behavior.py"
code = code_path.read_text(encoding="utf-8")
node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "_pku_pilot")
lines = code.splitlines(keepends=True)
stop = next(i for i in range(node.lineno - 1, len(lines)) if lines[i].startswith("    parts = split_records"))
selection = "".join(lines[node.lineno - 1:stop]).encode("utf-8")
(OUT / "current-pku-selection-contract.py").write_bytes(selection)

prior = OUT / "prior-report-opaque.json"
assert file_sha(prior) == "25385eabd4f343e9f96365b196b21cd71bd4d714f99845da11019c4206bf30f9"
verification = json.loads((BASE / "verification.json").read_bytes())
pointers = ["/fence_sha256", "/original_stdout", "/exercise_stdout", "/lookup_cases", "/independent_record", "/fence_imports", "/fence_named_calls", "/existing_archive_sha256", "/existing_raw_rows_sha256", "/existing_label_counts"]
retained_values = {p: verification[p[1:]] for p in pointers}
archive = ROOT / "assets/training/pku-safe-rlhf-v1.tar.gz"
assert file_sha(archive) == retained_values["/existing_archive_sha256"]
api = json.loads((BASE / "sources/archive-source-api.json").read_bytes())
assert api["sha"] == "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
assert api["cardData"]["license"] == "cc-by-nc-4.0"
assert (BASE / "sources/pku-card.md").read_bytes() == (BASE / "sources/archive-source-README.md").read_bytes()

receipt = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_9_1",
    "checked_at": datetime.now(UTC).isoformat(),
    "actual_command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-9_1-independent/reinspection-20261006/record_reinspection.py",
    "environment": {"python": platform.python_version(), "executable": sys.executable, "cwd": str(Path.cwd()), "device": "CPU metadata/fingerprint check; no tensors or model execution"},
    "read_scope": "Personally reread complete current raw UTF-8 introduction lines 1-4 and section 9.1 lines 5-40 including details; current necessary 9.2 opening lines 41-60; exact _pku_pilot selection body lines 467-485; retained original authority paragraphs named below. No repair answers, other reports or author result commentary read. Prior own report copied opaquely without reading contents.",
    "current_source_sha256": sha(section),
    "current_intro_sha256": sha(intro),
    "own_intro_summary": "本章用球數、加法前提及授權規則，分開檢查內容是否有依據、回答是否遵守邊界，以及是否仍提供幫助；再換問法查規則，最後用候選分數和四題信心區分自信與實際答對。",
    "prior_history": {"path": prior.relative_to(ROOT).as_posix(), "sha256": file_sha(prior), "handling": "Opaque byte copy only; contents not used to decide verdict."},
    "changed_claims_review": [
        {"location": "current lines 7-9", "actual_change": "Traditional-character normalization and 編3 -> 編出3.", "finding": "Meaning remains absence of observable count, human-labelled task criteria; original HHH evidence still supports this scope."},
        {"location": "current line 26 versus original lines 26-28", "actual_change": "Duplicate PKU/context paragraph removed from main text; relative-vs-absolute statement consolidated.", "finding": "No new equivalence or safety guarantee. Core distinction is preserved; original PKU ranking/absolute labels support it."},
        {"location": "current details lines 33-35", "actual_change": "Original external PKU/license/context statements retained once in details.", "finding": "Fixed card, original licence and context-dependent HHH discussion continue to support each statement; no claim removed from this review."},
        {"location": "current lines 11-24,28", "actual_change": "Exact Python fence and exercise unchanged.", "finding": "Own original and variant CPU runs apply to identical bytes, not a new run or model capability."},
    ],
    "current_fence_sha256": sha(fence),
    "fence_bytes_unchanged": True,
    "figure_sha256": {},
    "figure_check": "Current raw section contains no image/SVG reference or spatial/numeric figure claim; no rendering/viewing result is asserted.",
    "authority_reinspection": [
        {"path": (BASE / "sources/hhh-paper.pdf").relative_to(ROOT).as_posix(), "sha256": file_sha(BASE / "sources/hhh-paper.pdf"), "version": "arXiv:2112.00861v3", "locator": "PDF pp.4-5, §1.1 HHH; retained text lines 165-177 and 205-236", "action": "Personally reread original retained paragraphs; no network refetch needed."},
        {"path": (BASE / "sources/pku-card.md").relative_to(ROOT).as_posix(), "sha256": file_sha(BASE / "sources/pku-card.md"), "version": api["sha"], "locator": "Dataset Summary lines 85-101 and Human-Preference/Ranking lines 135-153; YAML license", "action": "Personally reread publisher original retained card; fixed revision/provenance and bytes verified."},
        {"path": (BASE / "sources/cc-by-nc-4.0.html").relative_to(ROOT).as_posix(), "sha256": file_sha(BASE / "sources/cc-by-nc-4.0.html"), "version": "CC BY-NC 4.0 English legal code", "locator": "§1(i), §2(a)(1), following exception; retained text lines 203-250", "action": "Personally reread original text; supports dataset license, not private-weight mandate."},
        {"path": (BASE / "sources/python3135-lexical-analysis.rst").relative_to(ROOT).as_posix(), "sha256": file_sha(BASE / "sources/python3135-lexical-analysis.rst"), "version": "CPython v3.13.5", "locator": "lines 709-756; original quote-version boundary retained at 836-844", "action": "Personally reread original f-string contract; exact code remains valid."},
    ],
    "unchanged_repository_contracts": unchanged_contracts,
    "own_permanent_evidence_hashes_validated": len(frozen_hashes),
    "original_json_pointers_actually_used": {"verification.json": pointers, "archive-source-api.json": ["/id", "/sha", "/cardData/license"]},
    "retained_raw_verification_values": retained_values,
    "execution_reuse": "Own actual original CPU execution, exercise variant and original label recount retained and fingerprint-checked. No fence rerun needed: source code, source-data archive and relevant contracts are byte-identical. No training, inference, GPU or new model measurements.",
    "verdict": "pass",
    "unresolved_issues": [],
}
(OUT / "reinspection-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: receipt[k] for k in ["current_source_sha256", "current_intro_sha256", "current_fence_sha256", "own_permanent_evidence_hashes_validated", "verdict", "unresolved_issues"]}, ensure_ascii=False, indent=2))
