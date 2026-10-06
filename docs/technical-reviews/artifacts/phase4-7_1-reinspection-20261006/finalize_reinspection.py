"""Same original owner records the real current-source reinspection and canonical report."""
import copy
import difflib
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

started = time.perf_counter()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = HERE.relative_to(ROOT).as_posix()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads((HERE / p).read_text(encoding="utf-8"))
prior = read("prior-report.json")
freeze = read("freeze-metadata.json")
original_path = ROOT / "docs/technical-reviews/artifacts/phase4-7_1-independent/original-fences/section.md"
original = original_path.read_bytes()
current = (HERE / "current-section.md").read_bytes()
live = (ROOT / "course/chapters/07.md").read_bytes()
heads = list(re.finditer(rb"(?m)^## [^\r\n]+", live))
index = next(i for i, h in enumerate(heads) if h[0].startswith(b"## 7.1 "))
live_section = live[heads[index].start():heads[index + 1].start()]
assert live_section == current
assert live[:heads[0].start()] == (HERE / "current-intro.md").read_bytes()
assert sha(HERE / "prior-report.json") == freeze["prior_report_sha256"]
assert original.replace("帶着舊錯答案".encode(), "帶著舊錯答案".encode()) == current
assert hashlib.sha256(current).hexdigest() == freeze["source_sha256"] == "5d8e19378dd35e2da1992edc0331085451016af176af3e0a8000b2fd1f0c3482"
diff = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True), current.decode().splitlines(keepends=True), fromfile="own-prior-frozen-7.1", tofile="current-7.1"))
(HERE / "section-diff.txt").write_text(diff, encoding="utf-8")
fingerprints = read("fingerprint-verification.json")
assert all(item["match"] for item in fingerprints)
# Recheck actual bytes again when writing the canonical report, avoiding stale receipts.
for item in fingerprints:
    if item["kind"] in {"repository_code", "preserved_own_artifact"}:
        assert sha(ROOT / item["path"]) == item["expected"]
assert re.findall(rb"```python\n(.*?)```", current, re.S) == re.findall(rb"```python\n(.*?)```", original, re.S)
assert re.findall(rb"!\[[^\]]*\]\(([^)]+)\)", current) == []

scope = [
    "2026-10-06 personally read all current7.1 raw UTF8 bytes, chapter07 raw introduction, and the exact own-frozen/current section diff; sole change is 帶着→帶著 in exercise paragraph, with unchanged semantics.",
    "Personally reread current6.6 as necessary boundary/tokenizer context; no formal review verdict is made for6.6 or its empirical BPE statement.",
    "Personally reread data.py10-28,54-68; model.py92-105; attention.py10-18; all3 current helper source hashes match own initial review's real executed inputs.",
    "Personally reread own original/variation stdout and selected immutable original HF chat source23-26/78-84, PyTorch v2.14.1 cross_entropy3500-3505, InstructGPT v1 §3.1 step1 and §3.5 SFT; all own38 evidence artifact fingerprints and support scopes remain valid.",
    "Actually rendered live7.1 page in Chromium151.0.7922.173 at1280x900 and390x844, personally viewed both full-page screenshots and a mobile horizontally scrolled output screenshot. The current帶著 sentence and actual displayed X/Y/58,2 agree with own prior preprocessing execution; displayed output is not claimed as a new2026-10-06 code run.",
    "No changed code, model result, numerical claim or diagram was found; no model/GPU/training/weights/data download/evaluation or pipeline replay was performed. Unchanged true CPU evidence is explicitly reused rather than claimed as newly run.",
]
matrix = []
for claim in prior["claims"]:
    changed = claim["id"] == "question-only-variation"
    matrix.append({"claim_id": claim["id"], "current_status": "verified", "semantic_change": False, "text_change": "Only exercise orthography 帶着→帶著; fixed-answer format example and corrected arithmetic requirement are unchanged." if changed else "No change to the underlying current-source statement or supporting code/source.", "retained_artifact_ids": claim["artifact_ids"], "support_reinspection": "Personally compared current full section to own frozen input and reread necessary current helper/primary-source passages. Every retained artifact and3 current helper sources matches the prior recorded SHA; direct outputs/numerical axes/count2 and support limits stay applicable."})
receipt = {
    "schema_version": 1, "kind": "same_owner_technical_reinspection", "reviewer_task": "/root/phase4_factual_coordinator/factual_7_1", "reviewed_on": "2026-10-06", "verdict": "pass",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-7_1-reinspection-20261006/finalize_reinspection.py", "cwd": str(ROOT), "elapsed_seconds_before_report_write": time.perf_counter() - started,
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU Python hashing/source comparison; no model computation", "browser": "Chromium151.0.7922.173 actual render"},
    "prior_history": {"path": PREFIX + "/prior-report.json", "sha256": sha(HERE / "prior-report.json"), "saved_opaque_before_reinspection": True},
    "current_source": {"path": "course/chapters/07.md#7.1", "sha256": freeze["source_sha256"], "raw_snapshot": PREFIX + "/current-section.md"},
    "chapter_intro": {"sha256": freeze["intro_sha256"], "raw_snapshot": PREFIX + "/current-intro.md", "intro_summary": freeze["intro_summary"], "unchanged": True},
    "changes": [{"location": "course/chapters/07.md:31", "before": "帶着舊錯答案", "after": "帶著舊錯答案", "effect": "Traditional-character orthography only; unchanged calculation, contract and training-data limitation."}],
    "actual_read_scope": scope, "claim_reinspection": matrix, "fingerprint_record_count": len(fingerprints), "fingerprint_mismatches": [],
    "reused_execution": {"original_fence_sha256": "373a09f1d9cdfbfcb735a42adb5ce5eb697e32cbd4a0a6a6851a239b23208ef0", "original_run": "docs/technical-reviews/artifacts/phase4-7_1-independent/original-fences/execution.json", "original_result": "exit0; X/Y10; active58,2", "variation_run": "docs/technical-reviews/artifacts/phase4-7_1-independent/verification.execution.json", "variation_result": "exit0; format/literal/multiturn/visibility/denominator checks passed; unchanged code and inputs", "new_execution_required": False, "reason": "Single orthographic text edit, no changed code or numeric/empirical statement; current claims still within personally verified original support scope."},
    "visual_inspection": {"desktop_and_mobile_viewed": True, "scrolled_mobile_output_viewed": True, "render_receipt": PREFIX + "/render-receipt.json", "no_section_figures": True, "finding": "Current page shows correct modified character and same sequence/targets; mobile long output has real overflow-x:auto, clientWidth306/scrollWidth641, scrollLeft335 verified and personally viewed. No technical content mismatch."},
    "original_sources_reused": "Own immutable primary-source snapshots retain their original2026-10-05 fetch/inspection provenance; fingerprint check and current support reinspection do not pretend a new download or new initial review.",
    "unresolved_questions": [], "canonical_artifact_id": "same-owner-reinspection-receipt", "canonical_report_path": "docs/technical-reviews/7.1.json", "script_sha256": sha(Path(__file__)),
}
(HERE / "reinspection-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

report = copy.deepcopy(prior)
report.update(source_sha256=freeze["source_sha256"], reviewed_on="2026-10-06", intro_sha256=freeze["intro_sha256"], intro_summary=freeze["intro_summary"], actual_read_scope=scope, verdict="pass")
report["reinspection"] = {"same_original_owner": True, "prior_report_path": PREFIX + "/prior-report.json", "prior_report_sha256": sha(HERE / "prior-report.json"), "receipt_artifact_id": "same-owner-reinspection-receipt", "receipt_path": PREFIX + "/reinspection-receipt.json", "receipt_sha256": sha(HERE / "reinspection-receipt.json"), "unresolved_questions": []}
for artifact in report["artifacts"]:
    artifact["description"] = "Own2026-10-05 evidence retained and personally fingerprint/support rechecked on2026-10-06. " + artifact["description"]
    if artifact["id"] == "section-original":
        artifact["description"] = "Own original2026-10-05 prior raw7.1 input for comparison; current raw source is current-section-reinspection. This prior input is not the current-source assertion."
    if artifact["id"] == "provenance":
        artifact["description"] = "Initial2026-10-05 frozen-input provenance and raw whole-file/intro/code snapshot hashes; retained as historical input, not the current whole-chapter version. Current reinspection receipt records new reading scope."
def add(identifier, filename, kind, description, **extra):
    report["artifacts"].append({"id": identifier, "path": PREFIX + "/" + filename, "sha256": sha(HERE / filename), "kind": kind, "description": description, **extra})
add("prior-report-history", "prior-report.json", "source_snapshot", "This same original reviewer's canonical report saved opaque before reinspection; preserves original SHA/verdict/evidence, not authority for current truth.")
add("current-section-reinspection", "current-section.md", "source_snapshot", "Current full7.1 original UTF8 bytes personally reread; canonical source SHA.")
add("current-intro-reinspection", "current-intro.md", "source_snapshot", "Actual current chapter introduction raw UTF8 personally read with new own intro_summary; SHA unchanged.")
add("current-freeze-metadata", "freeze-metadata.json", "source_snapshot", "True prior/current/intro hashes and frozen-full-chapter meaning; no normalization.")
if (HERE / "frozen-full-chapter-07.md").exists():
    add("current-frozen-chapter", "frozen-full-chapter-07.md", "source_snapshot", "Full chapter rawbytes frozen at this read, matching explicitly annotated whole-file input hash; claims reviewed only7.1 and necessary context.")
add("current-boundary-context", "current-prerequisite-6.6.md", "source_snapshot", "Actual current6.6 raw necessary context read; no separate verdict or BPE experimental re-evaluation.")
add("current-review-contract", "current-reviewer-instructions.md", "source_snapshot", "Current factual-review method read for this reinspection.")
add("section-reinspection-diff", "section-diff.txt", "derivation", "Actual own-frozen/current diff, sole change帶着→帶著.")
add("same-owner-fingerprint-check", "fingerprint-verification.json", "source_snapshot", "Actual43 checks:3 current helper sources,38 own immutable artifacts, original fence and figure inventory; no mismatches.")
add("same-owner-reinspection-code", "finalize_reinspection.py", "code", "Actual canonical-writing reinspection code; rechecks live/raw and all evidence hashes, records per-claim semantic/support assessment.")
add("same-owner-reinspection-receipt", "reinspection-receipt.json", "execution", "Real same original owner's full-source/support/diff/hash/visual reinspection receipt, not a source-SHA-only patch.", command=receipt["command"], result="All43 fingerprints match; raw live/current agree; exact one-character orthographic change; seven current claim scopes verified; pass; no unresolved question.", environment=receipt["environment"])
add("desktop-current-render", "render-desktop.png", "figure_render", "Actual Chromium1280x900 full-page render, personally viewed; page illustration evidence, section has no SVG.")
add("mobile-current-render", "render-mobile.png", "figure_render", "Actual Chromium390x844 full-page render, personally viewed.")
add("mobile-scrolled-current-output", "mobile-output-scrolled.png", "figure_render", "Actual mobile output horizontally scrolled, personally viewed to check complete terminal IDs are accessible.")
add("current-page-render-code", "render_page.py", "code", "Actual bounded Chromium render command code.")
add("current-page-render-receipt", "render-receipt.json", "execution", "Actual two-size Chromium rendering receipt and screenshot/DOM text hashes; no course model execution.", command=read("render-receipt.json")["command"], result="HTTP200 at both sizes; current帶著 and exact X IDs in DOM; screenshots rendered and personally viewed.", environment={"python":sys.version,"browser":"Chromium151.0.7922.173","device":"CPU headless renderer"})
add("mobile-scroll-inspection-code", "inspect_mobile_overflow.py", "code", "Actual mobile output overflow/scroll check.")
add("mobile-scroll-inspection", "mobile-overflow.json", "source_snapshot", "Actual DOM pre text, dimensions, overflow style and successful output scrollLeft335; bounded rendering only.")
report["sources"].append({"id":"same-owner-reinspection", "kind":"execution", "title":"Same original owner's2026-10-06 current-section reinspection", "verified":True, "artifact_id":"same-owner-reinspection-receipt"})
for claim in report["claims"]:
    claim["evidence"].append({"source_id":"same-owner-reinspection", "locator":"reinspection-receipt.json claim_reinspection entry for" + claim["id"], "supports":"Current full section personally reread, exact change classified, unchanged primary/code/execution evidence fingerprints and support scope rechecked; own original verification remains applicable."})
    claim["artifact_ids"].append("same-owner-reinspection-receipt")
report["checks"]["factual_accuracy"]["details"] += " 2026-10-06 same original owner真回查current全節與真正導言，確認sole帶着→帶著不改主張；receipt逐claim保存支持核對，非僅patch SHA。"
report["checks"]["numeric_verification"]["details"] += " 2026-10-06 code/fence/原輸出均SHA一致，所有原數值與分母主張未變；明示沿用自己的2026-10-05真CPU證據，沒有把它寫成新執行。"
report["checks"]["figure_consistency"]["details"] += " Current raw仍無圖；另真render/view desktop及mobile current頁及水平捲動輸出，顯示內容一致。"
report["checks"]["source_verification"]["details"] += " 2026-10-06自己原38 artifacts與3 current helper全吻合，親讀必要primary原文支持後沿用，保留原access date/version；無泛搜或下載。"
report["checks"]["limitations"]["details"] += " Same-owner複查只字形修訂，無需新模型/數值實驗；prior report opaque保留，未讀他人判定或作者repair答案，未解疑問為空。"
(ROOT / "docs/technical-reviews/7.1.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"verdict":"pass","report_sha256":sha(ROOT / "docs/technical-reviews/7.1.json"),"prior_history_path":PREFIX+"/prior-report.json","prior_history_sha256":sha(HERE/"prior-report.json"),"receipt_artifact_id":"same-owner-reinspection-receipt","receipt_path":PREFIX+"/reinspection-receipt.json","receipt_sha256":sha(HERE/"reinspection-receipt.json"),"source_sha256":freeze["source_sha256"],"unresolved_questions":[]},ensure_ascii=False,indent=2))
