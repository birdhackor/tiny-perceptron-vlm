"""Same original technical owner: reinspection, evidence reuse, then one checker."""
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
PRIOR_SHA = "06cf6edd0a0409373704842295b1193804177682e85a524798e8d2f77917ee7b"
EXPECTED = "1a157c5f8ccc33018d80515cd3beb151b68978adf44e7511525b29ea7a02e8be"
HISTORY = ROOT / "docs/technical-reviews/history" / f"phase4-6_9-own-before-current-reinspection-{PRIOR_SHA}.json"
ORIGINAL = OUT.parent / "original-run/section.md"
OWNER = "/root/phase4_factual_coordinator/factual_6_9"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def save(name, value):
    target = OUT / name
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return target
def relative(path):
    return path.relative_to(ROOT).as_posix()

prior_raw = HISTORY.read_bytes()
assert sha(prior_raw) == PRIOR_SHA
prior = json.loads(prior_raw)  # This owner's own report only; no other review is read here.
assert prior["reviewer_task"] == OWNER
spec = importlib.util.spec_from_file_location("facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
body, _, line = facts.original_section(ROOT / "course/chapters/06.md", "6.9")
old = ORIGINAL.read_bytes()
assert sha(body) == EXPECTED
assert sha(old) == prior["source_sha256"]
assert old.count("錶示".encode()) == 1
assert body == old.replace("錶示".encode(), "表示".encode())
(OUT / "current-section.md").write_bytes(body)
(OUT / "section-diff.txt").write_text("".join(difflib.unified_diff(old.decode().splitlines(keepends=True), body.decode().splitlines(keepends=True), fromfile="own-original-section", tofile="personally-read-current-section")))
old_fences = facts.fences(old, line)
new_fences = facts.fences(body, line)
assert len(old_fences) == len(new_fences) == 1
assert old_fences[0]["raw"] == new_fences[0]["raw"]
assert new_fences[0]["language"] == "python"
fence_hash = sha(new_fences[0]["raw"])
(OUT / "current-fence-1.py").write_bytes(new_fences[0]["raw"])
refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body.decode()) + re.findall(r'(?:src|href)=["\']([^"\']+)["\']', body.decode())
assert refs == [] and prior["figure_sha256"] == {}
pre63, _, _ = facts.original_section(ROOT / "course/chapters/06.md", "6.3")
(OUT / "current-prerequisite-6.3.md").write_bytes(pre63)
assert pre63 == (OUT.parent / "inputs/prerequisite-6.3.md").read_bytes()

# Current methods are evidence of procedure, not technical answers or repair notes.
for source, destination in (
    (ROOT / "docs/review-tools/factual-reviewer-instructions.md", OUT / "current-factual-reviewer-instructions.md"),
    (ROOT / "scripts/check_technical_reviews.py", OUT / "current-check_technical_reviews.py"),
):
    destination.write_bytes(source.read_bytes())

validated = []
for artifact in prior["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], artifact["path"]
    validated.append({"id": artifact["id"], "path": artifact["path"], "prior_sha256": artifact["sha256"], "current_sha256": actual})
current_support = []
for relative_path in (
    "tiny_perceptron/tokenization.py",
    "tiny_perceptron/data.py",
    "scripts/course_experiments/text.py",
    "scripts/course_experiments/common.py",
    "docs/course-experiments/results/tokenizer.json",
):
    current_path = ROOT / relative_path
    frozen_path = OUT.parent / "inputs/current" / relative_path
    same = current_path.read_bytes() == frozen_path.read_bytes()
    assert same, relative_path
    current_support.append({"path": relative_path, "sha256": sha(current_path.read_bytes()), "equal_to_own_frozen_input": same})

import tokenizers
import torch
assert tokenizers.__version__ == "0.23.2"
assert str(torch.__version__) == "2.14.1+cpu" and torch.version.cuda is None
environment = {"python": sys.version, "tokenizers": tokenizers.__version__, "torch": str(torch.__version__), "device": "CPU; no original fence or bounded tokenizer tests rerun", "shell": "bash login:false", "cwd": str(ROOT)}

support_review = [
    {"claim_ids": ["C1-chunk-is-not-boundary", "C5-stream-and-record-contract"], "current_read": "Entire current section, particularly paragraph 24; only typo replacement 錶示→表示. This continues to distinguish storage chunks from tokenizer/document boundaries.", "primary_reinspection": "Personally reread HF v0.23.2 tokenizer.rs encode_single_sequence 761–792; each call preprocesses the supplied input independently. Verified own full source snapshot hash. Whitespace and ByteLevel pretokenizer rule snapshots also reread and hash-checked.", "decision": "No substantive claim or support-scope change. Earlier exact fence/interior-cut/safe-boundary evidence remains applicable."},
    {"claim_ids": ["C2-example-and-api", "C3-numeric-example", "C4-whitespace-exercise"], "current_read": "Original fence and explanation lines 5–22, exercise line 28; exact original fence bytes and claimed ID-count comparison unchanged.", "primary_reinspection": "Personally reread HF Whitespace.pre_tokenize lines 20–28; literal whitespace rule and no universal length claim checked. Original execution and bounded-result SHA validated; installed versions unchanged.", "decision": "Reuse original actual CPU execution: [8] vs [6,7], exercise [8,8] on both paths, and original vocab/ID count evidence. No new execution claimed."},
    {"claim_ids": ["C6-space-rules-and-roundtrip", "C7-utf8-separate-state"], "current_read": "Current paragraph 26 and full necessary 6.3; both unchanged, including distinct Whitespace/byte-level and UTF8 tail contracts.", "primary_reinspection": "Personally reread HF ByteLevel regex lines 40–47 and CPython 3.13.5 IncrementalDecoder.decode/getstate lines 641–698. Existing official snapshot hashes and prior bounded decoder/ID/roundtrip result hashes verified.", "decision": "Reuse direct official-source and original bounded checks. Correct UTF8 decoding still does not imply regex/BPE ID equality; text read and byte read scopes remain explicit."},
    {"claim_ids": ["C8-recipe-scope"], "current_read": "Current supplemental paragraph line 33 and relevant 6.3; no changed empirical claim, new measurement, or required long shell recipe.", "primary_reinspection": "Current tokenizer helper, data, text recipe, common helper and result JSON whole-file SHA equal own snapshots. Historical code, original saved tokenizer/data snapshots and original audit hashes are among 50 personally revalidated artifacts.", "decision": "Prior original source-record/tokenizer input and recipe audit remains valid and separate from the short 6.9 demonstration. No model scores or interpretations reread, no weights, inference, training, or data downloads."},
]
receipt = {
    "reviewer_task": OWNER,
    "stage": "same_original_owner_current_reinspection",
    "reviewed_at": datetime.now(timezone.utc).isoformat(),
    "source": "course/chapters/06.md#6.9",
    "prior_history_path": relative(HISTORY),
    "prior_history_sha256": PRIOR_SHA,
    "own_original_section_path": relative(ORIGINAL),
    "own_original_section_sha256": sha(old),
    "current_source_sha256": sha(body),
    "current_section_first_line": line,
    "actual_change": "Only 錶示→表示 in section line 24; a lexical correction, with zero changed substantive claims.",
    "changed_substantive_claim_ids": [],
    "current_fence_sha256": fence_hash,
    "fence_identical_to_original": True,
    "figure_references": refs,
    "figure_check": "No image/diagram/SVG reference in current section or own original section. No factual visual claim or changed layout requires render/view; figure-consistency remains not_applicable.",
    "prerequisite_6_3_sha256": sha(pre63),
    "prerequisite_6_3_identical_to_own_original": True,
    "personally_revalidated_artifacts": validated,
    "current_repository_support_fingerprints": current_support,
    "environment": environment,
    "support_scope_reinspection": support_review,
    "commands": [{"command": ".venv/bin/python " + relative(Path(__file__)), "cwd": str(ROOT), "scope": "Read-only original/current byte comparison, evidence fingerprints and installed-version check; then own report rewrite and single-section checker."}],
    "prior_execution_reuse": "Original real CPU original fence and bounded variation results explicitly retained after unchanged-code/inputs/versions/hash and semantic support checks; not reported as a fresh tokenizer execution.",
    "old_whole_chapter_metadata": "The original extraction.json source_file_sha256 remains historical extraction metadata. It is not a current whole-chapter version claim; the exact retained original section/fence bytes support the old section fingerprint.",
    "others_reviews_or_author_repair_answers_read": False,
    "unresolved_questions": [],
    "verdict": "pass",
}
receipt_path = save("reinspection-receipt.json", receipt)

report = prior
report["source_sha256"] = EXPECTED
report["verdict"] = "pass"
report["issues"] = []
reinspection_id = "A-6_9-current-reinspection-20261006"
for path, identifier, kind, description in (
    (Path(__file__), "A-6_9-reinspection-code-20261006", "code", "Actual current reinspection byte/hash/semantic-support registration and checker capture source."),
    (OUT / "current-section.md", "A-6_9-current-section-20261006", "source_snapshot", "Personally read current full 6.9 UTF8 bytes; one typo correction from own original."),
    (OUT / "section-diff.txt", "A-6_9-own-original-current-diff-20261006", "source_snapshot", "Actual unified diff against own original section; no author repair explanation consulted."),
    (OUT / "current-fence-1.py", "A-6_9-current-fence-20261006", "code", "Current original fence copied verbatim, fingerprint identical to earlier actually executed bytes."),
    (OUT / "current-prerequisite-6.3.md", "A-6_9-current-prerequisite-20261006", "source_snapshot", "Personally reread necessary 6.3; identical to own retained prerequisite bytes."),
    (OUT / "current-factual-reviewer-instructions.md", "A-6_9-current-method-20261006", "source_snapshot", "Current reinspection procedure personally read."),
    (OUT / "current-check_technical_reviews.py", "A-6_9-current-checker-source-20261006", "code", "Current single-section checker actually invoked after own canonical report rewrite."),
    (receipt_path, reinspection_id, "execution", "Actual same-original-owner reinspection receipt: semantic comparison, all prior evidence fingerprints, environment and precise reuse scopes."),
):
    artifact = {"id": identifier, "path": relative(path), "sha256": sha(path.read_bytes()), "kind": kind, "description": description}
    if kind == "execution":
        artifact.update(command=".venv/bin/python " + relative(Path(__file__)), result="Current reinspection checks passed; only lexical typo changed, unchanged fence and 50/50 original artifacts verified; original CPU tokenizer results reused explicitly.", environment=environment)
    report["artifacts"].append(artifact)
report["reinspections"] = [{"reviewer_task": OWNER, "source_sha256": EXPECTED, "prior_history_path": relative(HISTORY), "prior_history_sha256": PRIOR_SHA, "artifact_id": reinspection_id, "receipt_path": relative(receipt_path), "receipt_sha256": sha(receipt_path.read_bytes()), "verdict": "pass", "actual_scope": "Personally reread current full 6.9 and necessary 6.3; only typo correction, no changed claims/code/figures; primary boundary contracts and all evidence fingerprints rechecked."}]
for claim in report["claims"]:
    claim["artifact_ids"].append(reinspection_id)
    claim["reinspection"] = {"artifact_id": reinspection_id, "claim_changed": False, "evidence_reuse": "Original owner has personally rechecked current wording, unchanged fence/source/input fingerprints and same support scope; original actual CPU executions retained, not rerun."}
report["checks"]["factual_accuracy"]["details"] += " 2026-10-06 原技術 owner 真回查目前全節，唯一錶示→表示錯字修正；所有8組實質主張未變，逐組原證據支持範圍重新確認。"
report["checks"]["source_verification"]["details"] += " 本次 reinspection 親核原50份 artifact SHA及current helper/result原檔，全部相符；官方snapshot沿用原2026-10-05存取版本，不宣稱新HTTP存取。"
report["checks"]["limitations"]["details"] += " 未變code/原JSON/資料/official snapshot核指紋後沿用本人真實舊執行；本次沒有重跑原fence或新模型實測，沒有讀作者repair答案或他人review。"
report["read_scope"]["current_reinspection"] = "2026-10-06: full current 6.9, full necessary 6.3, selected original official boundary source locators; prior own frozen input and artifact identities compared."
report_path = ROOT / "docs/technical-reviews/6.9.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

checker_argv = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "6.9"]
completed = subprocess.run(checker_argv, cwd=ROOT, capture_output=True, timeout=30)
(OUT / "checker-stdout.txt").write_bytes(completed.stdout)
(OUT / "checker-stderr.txt").write_bytes(completed.stderr)
checker_receipt = {"command_argv": checker_argv, "cwd": str(ROOT), "exit_code": completed.returncode, "stdout_sha256": sha(completed.stdout), "stderr_sha256": sha(completed.stderr), "checker_sha256": sha((ROOT / "scripts/check_technical_reviews.py").read_bytes()), "report_path": relative(report_path), "report_sha256": sha(report_path.read_bytes()), "source_sha256": EXPECTED, "prior_history_path": relative(HISTORY), "prior_history_sha256": PRIOR_SHA, "canonical_reinspection_artifact_id": reinspection_id, "reinspection_receipt_path": relative(receipt_path), "reinspection_receipt_sha256": sha(receipt_path.read_bytes()), "verdict": report["verdict"], "scope": "6.9 only; real same-original-owner semantic and fingerprint reinspection followed by one section checker."}
save("checker-receipt.json", checker_receipt)
print(json.dumps(checker_receipt, ensure_ascii=False, indent=2))
raise SystemExit(completed.returncode)
