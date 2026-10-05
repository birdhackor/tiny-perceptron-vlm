"""Original-owner correction of the missing top-level raw-intro field.

No lesson edits or new CPU/authority checks; re-read raw lesson and verify the
unchanged existing proof hashes before reusing the initial claim evidence.
"""
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
report_path = ROOT / "docs/technical-reviews/8.1.json"
initial_hash = "a00ad1eef36e2812693220fd19f6cdfb94f3c80fc35b68a4ee4b8e319095673d"
initial_path = BASE / "history/initial-pass-before-intro-field-correction/8.1.initial.json"
digest = lambda raw: hashlib.sha256(raw).hexdigest()
assert digest(initial_path.read_bytes()) == initial_hash
report_raw = report_path.read_bytes()
assert digest(report_raw) == initial_hash
report = json.loads(report_raw)
assert report.get("intro_sha256") is None

chapter = (ROOT / "course/chapters/08.md").read_bytes()
chapter.decode("utf-8")
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
selected = [i for i, h in enumerate(headings) if h[0].startswith(b"## 8.1 ")]
assert selected == [0]
intro = chapter[:headings[0].start()]
section = chapter[headings[0].start():headings[1].start()]
assert digest(intro) == "2f6a1c0019750be969ff609c3ae811e3b72e94ecacd7cfde38126c431f6ff8b3"
assert digest(section) == report["source_sha256"]
assert intro == (BASE / "inputs/chapter-intro.md").read_bytes()
assert section == (BASE / "execution/original/section.md").read_bytes()
assert report["figure_sha256"] == {}
assert not re.findall(rb"!\[[^\]]*\]\(", intro + section)

proofs = []
for artifact in report["artifacts"]:
    observed = digest((ROOT / artifact["path"]).read_bytes())
    assert observed == artifact["sha256"], artifact["id"]
    proofs.append({"id": artifact["id"], "path": artifact["path"], "expected_sha256": artifact["sha256"], "observed_sha256": observed, "unchanged": True})
followup = BASE / "followup-intro-field"
followup.mkdir(exist_ok=False)
(followup / "chapter-intro.raw.md").write_bytes(intro)
(followup / "section.raw.md").write_bytes(section)
summary = "本章從同一題在『用例子解釋』與『只給數字』兩種要求下的合格回答差異出發，先明寫任務判準，再比較提示、示範與少量參數修正對寫法的影響。數字正確、指定範圍與可用格式應各自負責，語氣活潑不能抵消它們；最後用可讀告示連到指定範圍任務。LoRA與自動裁判是依需要選讀的延伸，不是理解明確要求的必經流程。"
reading = {
    "reviewer_task": report["reviewer_task"], "kind": "actual_original_owner_followup",
    "trigger": "Collection gate observed absent top-level intro_sha256 (reported as null). Initial factual PASS and schema exit0 were not yet an accepted current-version PASS.",
    "initial_report_path": initial_path.relative_to(ROOT).as_posix(), "initial_report_sha256": initial_hash,
    "actual_reading_command": "sed -n '1,38p' course/chapters/08.md",
    "read_ranges": ["course/chapters/08.md lines1-6: complete chapter introduction", "course/chapters/08.md lines7-33: complete 8.1 including original fence, exercise and supplementary paragraph", "lines34-38 also displayed at transition into 8.2"],
    "intro_sha256": digest(intro), "intro_byte_count": len(intro), "section_sha256": digest(section), "section_byte_count": len(section),
    "intro_summary": summary, "figure_sha256": {},
    "original_bytes_unchanged": True, "existing_proof_artifact_hashes_rechecked": proofs,
    "proof_reuse": "Reused unchanged initial original-fence CPU execution, bounded variants, authority snapshots/actual inspection and archived-experiment rescore. Only original lesson reading and artifact hash verification were repeated; no CPU example, model, training or external source inspection falsely claimed as rerun.",
    "reader_callback": "not needed: intro, section and figures unchanged; only this reviewer's metadata corrected",
}
(followup / "reading-and-proof-reuse.json").write_text(json.dumps(reading, ensure_ascii=False, indent=2) + "\n")

for identifier, path, kind, description in [
    ("a-initial-report-history", initial_path, "source_snapshot", "自身初版 factual PASS 的 opaque 完整原 bytes；曾因缺 top-level intro_sha256 未獲本輪版本收件接受。"),
    ("a-followup-reading", followup / "reading-and-proof-reuse.json", "source_snapshot", "original-owner 實際重讀導言/完整8.1、raw SHA、獨立摘要與29項既有proof hash核對；明記只重用未變CPU/source證據。"),
    ("a-followup-code", Path(__file__), "code", "原報告owner檢查raw UTF8導言、小節及原證據後修正missing field的實際程式。"),
    ("a-followup-intro", followup / "chapter-intro.raw.md", "source_snapshot", "本次重新讀取並保存的完整導言raw UTF8 bytes。"),
    ("a-followup-section", followup / "section.raw.md", "source_snapshot", "本次重新讀取並保存的8.1完整raw UTF8 bytes；與初版完全相同。"),
]:
    report["artifacts"].append({"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes()), "kind": kind, "description": description})
report["intro_sha256"] = digest(intro)
report["chapter_intro_sha256"] = digest(intro)
report["intro_summary"] = summary
report["intro_source"] = "course/chapters/08.md#chapter-introduction"
report["reviewer_process"]["original_owner_followup"] = reading
report["issues"].append({
    "id": "8.1-intro-field-collection", "status": "resolved", "kind": "review_metadata",
    "original_problem": "Initial report SHA a00ad1eef36e2812693220fd19f6cdfb94f3c80fc35b68a4ee4b8e319095673d omitted top-level intro_sha256, although chapter_intro_sha256, own summary and raw intro proof were present. The schema checker passed but current-version collection did not accept this initial report.",
    "evidence": ["a-initial-report-history", "a-followup-reading", "a-followup-intro"],
    "impact": "Initial factual PASS was historical and not an accepted current-version PASS under this round's first-section intro gate.",
    "resolution": "Actual original-owner followup: opaque initial history retained; personally re-read complete current intro and 8.1; recomputed exact raw UTF8 hashes and independently checked all29 existing proof artifact hashes; wrote own top-level intro_sha256 and own refreshed intro_summary. Source/proof bytes unchanged, no factual verdict or lesson changes, and no CPU/source rerun claim. Checker and updated collection fields validated in a new receipt.",
})
report["checks"]["factual_accuracy"]["details"] += " 本次本人原owner重讀完整未變導言及8.1，修正top-level intro_sha256收件欄位並保留初版與resolved問題。"
for claim in report["claims"]:
    if claim["id"] == "8.1-c4":
        claim["location"] = "course/chapters/08.md:25, demonstration and new-question paragraph"
    elif claim["id"] == "8.1-c5":
        claim["location"] = "course/chapters/08.md:30, JSON supplemental definition"
    elif claim["id"] == "8.1-c6":
        claim["location"] = "course/chapters/08.md:30, complete supplementary empirical paragraph and 8.4 pointer"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"initial_report_history": initial_path.relative_to(ROOT).as_posix(), "initial_report_sha256": initial_hash, "report_sha256": digest(report_path.read_bytes()), "intro_sha256": digest(intro), "section_sha256": digest(section), "proof_hashes_rechecked": len(proofs), "read_ranges": reading["read_ranges"]}, ensure_ascii=False, indent=2))
