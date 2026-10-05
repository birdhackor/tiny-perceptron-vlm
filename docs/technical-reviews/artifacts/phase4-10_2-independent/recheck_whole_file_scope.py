"""Record the same reviewer's actual section re-read and frozen-input scope."""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUT = BASE / "metadata-recheck"
OUT.mkdir(exist_ok=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
report_path = ROOT / "docs/technical-reviews/10.2.json"
before_raw = report_path.read_bytes()
report = json.loads(before_raw)
before_sha = sha(before_raw)
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_10_2"
frozen_chapter = BASE / "inputs/course/chapters/10.md"
frozen_file_sha = sha(frozen_chapter.read_bytes())
assert report["source_file_sha256"] == frozen_file_sha

reading = []
for lesson in ("10.1", "10.2", "10.3"):
    current_body, current_file, first_line = facts.original_section(ROOT / "course/chapters/10.md", lesson)
    old_body, _, _ = facts.original_section(frozen_chapter, lesson)
    if lesson == "10.3":
        current_body = b"".join(current_body.splitlines(keepends=True)[:9])
        old_body = b"".join(old_body.splitlines(keepends=True)[:9])
        name = "current-10.3-opening.md"
        scope = "First nine lines only: heading, introductory paragraph and fence setup."
    else:
        name = f"current-{lesson}.md"
        scope = "Complete original UTF-8 section, including heading and fence."
    assert current_body == old_body
    (OUT / name).write_bytes(current_body)
    reading.append({
        "source": f"course/chapters/10.md#{lesson}",
        "first_line": first_line,
        "scope": scope,
        "sha256": sha(current_body),
        "byte_identical_to_original_frozen_input_range": True,
        "current_read_snapshot": (OUT / name).relative_to(ROOT).as_posix(),
        "personally_read_in_current_turn": True,
    })
    if lesson == "10.2":
        assert sha(current_body) == report["source_sha256"]

artifact_checks = []
for artifact in report["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], artifact["id"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

for source in report["sources"]:
    if source["kind"] == "repository_code":
        assert sha((ROOT / source["path"]).read_bytes()) == source["sha256"]

figure = ROOT / "course/figures/rewrite-10-patch-order.svg"
assert sha(figure.read_bytes()) == report["figure_sha256"][figure.relative_to(ROOT).as_posix()]
assert figure.read_bytes() == (BASE / "inputs/course/figures/rewrite-10-patch-order.svg").read_bytes()
implementation = ROOT / "tiny_perceptron/multimodal.py"
assert implementation.read_bytes() == (BASE / "inputs/tiny_perceptron/multimodal.py").read_bytes()
current_whole_hash = sha((ROOT / "course/chapters/10.md").read_bytes())
assert current_whole_hash != frozen_file_sha

receipt = {
    "schema_version": 1,
    "kind": "same_reviewer_scope_metadata_recheck",
    "reviewer_task": report["reviewer_task"],
    "checked_at": "2026-10-05 14:12:03 UTC",
    "time_source": "clock.curr_time actually called in this recheck turn",
    "previous_own_report_sha256": before_sha,
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-10_2-independent/recheck_whole_file_scope.py",
    "cwd": str(ROOT),
    "shell": "bash",
    "login": False,
    "reading": reading,
    "implementation_reading": {
        "path": "tiny_perceptron/multimodal.py",
        "ranges": "lines 1–44 and 177–190, substantively patchify/unpatchify, VisionEncoder and scene",
        "sha256": sha(implementation.read_bytes()),
        "byte_identical_to_original_frozen_implementation": True,
        "personally_read_in_current_turn": True,
    },
    "figure_reading": {
        "path": figure.relative_to(ROOT).as_posix(),
        "sha256": sha(figure.read_bytes()),
        "byte_identical_to_original_frozen_svg": True,
        "personally_read_entire_svg_in_current_turn": True,
        "personally_viewed_original_unchanged_render_in_current_turn": True,
        "render_path": (BASE / "figure/patch-order.png").relative_to(ROOT).as_posix(),
        "render_sha256": sha((BASE / "figure/patch-order.png").read_bytes()),
        "inspection": "Same 640×550 diagram, four rows numbered 0–3/4–7/8–11/12–15, row-order text and 48-value caption visible and consistent.",
    },
    "original_evidence_checks": artifact_checks,
    "original_evidence_reread": "Original fence stdout and selected raw coordinate output: scene p4/p8 shapes/equality, six known RGB coordinates, card sequences and counterexample, and 3648 checked individual coordinates. No CPU/model computation was rerun.",
    "frozen_whole_file": {
        "path": "course/chapters/10.md",
        "snapshot_path": frozen_chapter.relative_to(ROOT).as_posix(),
        "sha256": frozen_file_sha,
        "snapshot_captured_on": "2026-10-05",
        "scope": "Complete original file bytes captured at the first review; actual original reading was 10.1/10.2 and necessary 10.3 opening, not a whole-chapter factual review.",
    },
    "current_whole_file_observation": {
        "path": "course/chapters/10.md",
        "sha256": current_whole_hash,
        "method": "Hash-only observation of complete current bytes; whole chapter not read or reviewed.",
        "is_reviewed_current_whole_chapter_fingerprint": False,
    },
    "decision": "Remove ambiguous top-level source_file_sha256. Retain the actual old whole-file SHA under frozen_whole_source_file with date, real snapshot path and scope. Keep current section/own-figure/implementation fingerprints and six claim verdicts unchanged after actual rereading and unchanged-byte checks.",
    "limits": "Same-section metadata recheck only. No review of 10.5, no expanded claim scope, no new training/inference/CPU rerun, no modification of historical frozen snapshots, manuscript, figure, other report or original opaque pass history.",
    "verdict": "pass",
}
receipt_path = OUT / "reading-unchanged-proof.json"
write_json(receipt_path, receipt)

report.pop("source_file_sha256")
report["frozen_whole_source_file"] = {
    **receipt["frozen_whole_file"],
    "artifact_id": "frozen-whole-source-file",
    "is_current_whole_file_fingerprint": False,
}
new_artifacts = [
    ("frozen-whole-source-file", frozen_chapter, "source_snapshot", "初讀當次完整原章bytes的真实frozen输入快照；旧整章SHA/日期/实际读取范围明确保存，并非目前整章fingerprint。"),
    ("whole-file-scope-recheck-program", Path(__file__).resolve(), "code", "同一reviewer执行的metadata范围复查程序；只核原bytes/历史证据SHA并保存当前真正读过的必要片段。"),
    ("whole-file-scope-recheck", receipt_path, "derivation", "本reviewer本次真实全文/ctx/图/原码重读范围、原证据23项SHA未变、历史whole-file scope澄清及未扩大支持范围的receipt。"),
]
for row in reading:
    suffix = row["source"].split("#")[1]
    new_artifacts.append((f"scope-recheck-current-{suffix}", ROOT / row["current_read_snapshot"], "source_snapshot", f"本次实际重读的{suffix}当前原始UTF-8范围；10.3仅必要开头，范围由reading receipt明确限定。"))
for identifier, path, kind, description in new_artifacts:
    report["artifacts"].append({"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes()), "kind": kind, "description": description})
report["metadata_rechecks"] = [{
    "checked_at": receipt["checked_at"],
    "reviewer_task": report["reviewer_task"],
    "artifact_id": "whole-file-scope-recheck",
    "scope": "Current 10.2 in full; 10.1 in full; necessary 10.3 opening; original SVG and unchanged rendered image; patchify/unpatchify/VisionEncoder/scene source; all original registered evidence hashes unchanged.",
    "result": "Frozen whole-file source fingerprint explicitly separated from current reviewed section. Substantive claims, current section and figure fingerprints remain unchanged; pass retained without unrelated computation.",
}]
write_json(report_path, report)
print(json.dumps({"report_sha256": sha(report_path.read_bytes()), "source_sha256": report["source_sha256"], "formal_receipt_artifact_id": "whole-file-scope-recheck", "receipt_path": receipt_path.relative_to(ROOT).as_posix(), "receipt_sha256": sha(receipt_path.read_bytes()), "original_artifacts_unchanged": len(artifact_checks), "verdict": report["verdict"]}, ensure_ascii=False, indent=2))
