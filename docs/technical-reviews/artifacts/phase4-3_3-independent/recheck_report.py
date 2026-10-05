"""Record this original reviewer's actual full-section recheck, without rerunning CPU math."""
from pathlib import Path
import difflib
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
REPORT = ROOT / "docs/technical-reviews/3.3.json"
PREFIX = OUT.relative_to(ROOT).as_posix()
def sha(raw):
    return hashlib.sha256(raw).hexdigest()

spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
raw, whole, first_line = helper.original_section(ROOT / "course/chapters/03.md", "3.3")
old = (OUT / "original/section.md").read_bytes()
fences = helper.fences(raw, first_line)
assert len(fences) == 1 and fences[0]["language"] == "python" and fences[0]["closed"]
fence = fences[0]["raw"]
original_fence = (OUT / "original/fence-1.py").read_bytes()
assert fence == original_fence
assert sha(fence) == "4c598ea2d4370f4faafbb47b5e6fb54db1f5f239b075dfbe499f993131ccbba4"
assert (ROOT / "tiny_perceptron/attention.py").read_bytes() == (OUT / "original/attention.py").read_bytes()
diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), raw.decode().splitlines(keepends=True),
                                    fromfile="initial-phase4-3.3", tofile="current-rechecked-3.3"))
print(diff)
old_paragraph = "真實模型讓位置表示 X 各經過一套 Linear，得到 Q 與 K；這種特徵轉換也叫**投影**，這裡先理解為不同的加權配方。若 Q、K 都由同一組文字位置產生，叫**自注意力**。位置來源相同，角色仍不同。"
new_paragraph = "真實模型讓位置表示 X 各經過一套 Linear，得到 Q 與 K；這種特徵轉換也叫**投影**，這裡先理解為不同的加權配方。**自注意力**的「自」是指：提出查詢、提供匹配線索，以及實際取回的內容，都由同一組位置產生。這裡先看 Q、K 的匹配部分；下一節會加入代表取回內容的第三份表示。位置來源相同，角色仍不同。"
assert old_paragraph.encode() in old
assert new_paragraph.encode() in raw
# This checks exact section bytes, including preserved terminal blank lines.
assert old.replace(old_paragraph.encode(), new_paragraph.encode()) == raw
assert b"![" not in raw and b"<img" not in raw

initial = (OUT / "initial-revise-report.json").read_bytes()
assert REPORT.read_bytes() == initial
HISTORY = ROOT / "docs/technical-reviews/history/phase4-3_3-independent-initial-revise.json"
HISTORY.parent.mkdir(parents=True, exist_ok=True)
if HISTORY.exists():
    assert HISTORY.read_bytes() == initial
else:
    HISTORY.write_bytes(initial)
(OUT / "recheck-section.md").write_bytes(raw)
(OUT / "recheck-fence-1.py").write_bytes(fence)
(OUT / "recheck-diff.patch").write_text(diff)

report = json.loads(initial)
before_claim = next(c for c in report["claims"] if c["id"] == "self-attention-definition")
claim_before_revision = json.loads(json.dumps(before_claim))
before_claim.update({
    "statement": "自注意力的查詢、匹配線索及實際取回內容都由同一組位置產生；本節先看Q/K匹配部分，下一節加入第三份取回內容表示。",
    "scope": "標準Transformer的自注意力來源定義；先按角色描述Q/K/V全體同源，未把只驗證QK的玩具code稱作完整自注意力讀取。",
    "status": "verified",
    "evidence": [{"source_id": "vaswani2017", "locator": "§3.2.3 printed p.5; original own extraction lines 230–233; re-read during this recheck",
        "supports": "親自再讀原文all of the keys, values and queries come from the same place。新版按三種角色明說同源，與Q/K/V原始定義吻合；它同時限定本節只展示Q/K匹配部分，因此沒有把原未投影、未softmax、未取V的code當成完整自注意力。"}],
    "artifact_ids": ["paper-pdf", "paper-text", "recheck-section", "recheck-record"],
})
for claim in report["claims"]:
    if claim["kind"] in ("numeric", "software"):
        claim["verification"]["details"] += " Recheck: original fence bytes/SHA-256 unchanged; existing initial-review CPU execution evidence reused, not rerun."
report["source_sha256"] = sha(raw)
report["verdict"] = "pass"
issue = report["issues"][0]
issue.update(status="resolved", resolution="協調者修改充分定義為查詢、匹配線索與取回內容全體同源，並限定這節是Q/K匹配部分。原技術審閱者親讀完整新3.3、再讀原論文§3.2.3 pp.5及檢查raw單段diff後確認解決；原claim/問題與revise判定完整留在history，不重跑未變CPU原fence。")
issue["claim_before_revision"] = claim_before_revision
issue["rechecked_text"] = new_paragraph
report["checks"]["factual_accuracy"].update(status="pass", details="親讀新版完整3.3；自注意力定義已明列查詢、匹配線索、取回內容三者同源，並限縮原碼為Q/K匹配部分，符合親自再讀的原論文§3.2.3。原Q/K角色與投影說明未變。原revise issue保留並記錄親自複查解決。")
report["checks"]["numeric_verification"]["details"] += " 本次僅文字修訂，已親核原fence全bytes及hash完全相同；沿用初次本輪CPU結果，沒有宣稱重跑。"
report["checks"]["limitations"]["details"] += " 新定義直接說這節先看Q/K匹配部分，未宣稱完整Q/K/V讀取已在本fence執行；複查不追加無關測試。"
report["reading_scope"]["recheck"] = "原審閱者在收到正式followup後親讀目前3.3完整raw小節及自身原issue，再讀原Attention Is All You Need v7 §3.2.2–3.2.3原文；逐字比對唯一投影/自注意力段diff。原code/fence bytes相同，其他數值與software claims未受文字修改影響，未重跑CPU。"
report["sources"][0]["inspection_note"] += " In the same original reviewer's formal recheck, personally re-read §§3.2.2–3.2.3 (own text lines 192–240), including complete all-queries/keys/values-same-source definition, and compared the new full section against it."
record = {
    "performed_by": report["reviewer_task"], "context": "same original independent technical reviewer performing actual followup recheck",
    "performed_at": datetime.now(UTC).isoformat(), "source": report["source"],
    "initial_source_sha256": sha(old), "new_source_sha256": sha(raw),
    "initial_report_sha256": sha(initial), "history_path": HISTORY.relative_to(ROOT).as_posix(),
    "full_new_section_read": True, "original_paper_re_read": "§§3.2.2–3.2.3, printed pp.4–5, own extraction lines 192–240",
    "actual_change": "Exactly one paragraph; full original bytes with that old paragraph replaced equal current section bytes.",
    "original_fence_sha256": sha(original_fence), "current_fence_sha256": sha(fence), "fence_bytes_identical": True,
    "repository_attention_unchanged": True, "figure_references": [],
    "numeric_execution_repeated": False,
    "reused_execution_evidence": ["original/execution.json", "original/stdout.txt", "original/environment.json", "probe-execution.json", "probe-results.json"],
    "claim_scope": "Definition alone corrected; all other claim statements and original numeric matrices, exercise, softmax-axis explanation are unchanged. No GPU/full training/data/model retrieval or diagram rendering needed for this text-only recheck.",
    "resolution": "Full self-attention source condition now accurate and original matching-only code scope explicit.",
    "environment": {"python": sys.version, "device": "No tensor execution in recheck; byte/provenance comparison only"},
}
(OUT / "recheck-record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
def add_artifact(identifier, filename, kind, description):
    report["artifacts"].append({"id": identifier, "path": f"{PREFIX}/{filename}", "sha256": sha((OUT / filename).read_bytes()), "kind": kind, "description": description})
add_artifact("recheck-section", "recheck-section.md", "source_snapshot", "Raw UTF-8 current complete 3.3, personally read in original reviewer's formal followup recheck.")
add_artifact("recheck-fence", "recheck-fence-1.py", "code", "Current original fence; complete bytes/hash matched first review, not rerun.")
add_artifact("recheck-diff", "recheck-diff.patch", "source_snapshot", "Own complete before/after raw section diff, exactly one changed definition paragraph.")
add_artifact("recheck-record", "recheck-record.json", "source_snapshot", "Actual full-section/original-paper recheck record, original issue/history hashes, unchanged code and explicit CPU evidence reuse.")
add_artifact("recheck-code", "recheck_report.py", "code", "Executable byte comparisons, own initial-revise history preservation and transparent issue/claim update.")
report["rechecks"] = [record]
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("pass; report", sha(REPORT.read_bytes()))
print("source", sha(raw))
print("history", HISTORY.relative_to(ROOT).as_posix())
print("No numeric execution repeated; original fence bytes and SHA identical.")
