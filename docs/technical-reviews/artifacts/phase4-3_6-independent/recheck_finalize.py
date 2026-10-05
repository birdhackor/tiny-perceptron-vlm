"""Recheck the single authored explanation change, preserving initial evidence."""
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PREFIX = OUT.relative_to(ROOT).as_posix()
RECHECK = OUT / "recheck"
RECHECK.mkdir(exist_ok=True)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text())
def artifact(identifier, relative, kind, description, **extra):
    path = OUT / relative
    return {"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "kind": kind, "description": description, **extra}

initial_manifest = read(OUT / "initial-manifest.json")
for entry in initial_manifest:
    assert digest(ROOT / entry["path"]) == entry["sha256"], entry["path"]
temporary = Path("/tmp/phase4-3_6-recheck")
for path in temporary.rglob("*"):
    if path.is_file():
        target = RECHECK / path.relative_to(temporary)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
initial_extraction = read(OUT / "original/extraction.json")
new_extraction = read(RECHECK / "extraction.json")
old_section = (OUT / "original/section.md").read_bytes()
new_section = (RECHECK / "section.md").read_bytes()
old_sentence = "`tril()` 留下對角線與下方的 True。`[None,None]` 補入筆數與讀法兩軸，資料形狀為 `[1,1,4,2]`：一筆、一種讀法、四位置、每張兩格。".encode()
new_sentence = "`tril()` 留下對角線與下方的 True。`[None,None]` 為許可表補入筆數與讀法兩軸，得到遮罩 `[1,1,4,4]`；後兩軸是四個查詢位置對四個候選位置。Q、K、V 資料形狀則為 `[1,1,4,2]`：一筆、一種讀法、四位置、每張兩格。".encode()
assert old_section.count(old_sentence) == 1
assert old_section.replace(old_sentence, new_sentence) == new_section
assert digest(OUT / "original/fence-1.py") == digest(RECHECK / "fence-1.py")
assert (OUT / "original/fence-1.py").read_bytes() == (RECHECK / "fence-1.py").read_bytes()
assert new_extraction["figure_sha256"] == initial_extraction["figure_sha256"]
for path, expected in initial_extraction["figure_sha256"].items():
    assert digest(ROOT / path) == expected
    assert (ROOT / path).read_bytes() == (OUT / "original/figures" / path).read_bytes()
assert (ROOT / "tiny_perceptron/attention.py").read_bytes() == (OUT / "code/tiny_perceptron/attention.py").read_bytes()
assert (ROOT / "docs/review-tools/section_facts.py").read_bytes() == (OUT / "code/docs/review-tools/section_facts.py").read_bytes()
(RECHECK / "section.diff.txt").write_text("".join(difflib.unified_diff(old_section.decode().splitlines(keepends=True), new_section.decode().splitlines(keepends=True), fromfile="initial-3.6", tofile="rechecked-current-3.6")))

history = ROOT / "docs/technical-reviews/history/phase4-3_6-independent-initial-revise.json"
assert history.read_bytes() == (OUT / "initial-revise.json").read_bytes()
environment = {"python": sys.version, "python_executable": sys.executable, "device": "CPU filesystem/structural validation only; attention computations not rerun", "working_directory": str(ROOT)}
record = {"command_argv": [sys.executable, str(Path(__file__).resolve())], "cwd": str(ROOT), "original_section_sha256": digest(OUT / "original/section.md"), "new_section_sha256": digest(RECHECK / "section.md"), "initial_history_path": history.relative_to(ROOT).as_posix(), "initial_history_sha256": digest(history), "exactly_one_explanation_sentence_changed": True, "fence_bytes_unchanged": True, "fence_sha256": digest(RECHECK / "fence-1.py"), "figure_bytes_unchanged": True, "figure_sha256": new_extraction["figure_sha256"], "attention_bytes_unchanged": True, "attention_sha256": digest(ROOT / "tiny_perceptron/attention.py"), "helper_bytes_unchanged": True, "initial_artifact_hashes_checked": len(initial_manifest), "fence_reexecuted": False, "figure_rendered_again": False, "figure_viewed_again": False, "current_entire_section_read": True, "official_axis_contract_reread": True, "matrix_shape_derivation": "[4,2]@[2,4]=>[4,4] scores/mask/weights; [4,4]@[4,2]=>[4,2] output; B=H=1", "environment": environment, "result": "Structural/immutable-evidence checks passed; revised explanation distinguishes mask L,S from QKV L,E/S,E/S,Ev."}
(RECHECK / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

# This is solely this reviewer's own initial report; no older review is read or used.
report = read(OUT / "initial-revise.json")
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_3_6"
report["verdict"] = "pass"
report["source_sha256"] = new_extraction["source_sha256"]
report["author_tasks"] = ["/root/phase4_factual_coordinator"]
report["artifacts"].extend([
    artifact("initial_report_history", "initial-revise.json", "source_snapshot", "本審閱者親自交出的initial revise；原issue、原claims、原來源指紋與判定永久保留；與history副本bytes/SHA完全相同"),
    artifact("rechecked_section", "recheck/section.md", "source_snapshot", "本審閱者重新讀的目前完整小節原始UTF-8 bytes"),
    artifact("recheck_extraction", "recheck/extraction.json", "source_snapshot", "只extract新節，沒有execute；新原文/不變fence與SVG指紋"),
    artifact("recheck_diff", "recheck/section.diff.txt", "source_snapshot", "親核整節仅許可表與QKV形狀解說句改變，原碼與其他文字不變"),
    artifact("recheck_code", "recheck_finalize.py", "code", "本次raw bytes/fence/figure/helper/既有永久證據全SHA核對與report/checker更新原碼"),
    artifact("recheck_notes", "recheck-notes.md", "derivation", "實際全文複讀與官方尾軸重新讀取記錄、矩陣公式與原issue解決理由；明記未新CPU/render"),
    artifact("recheck_execution", "recheck/execution.json", "execution", "本次只執行結構/證據完整性核對；沒有重跑attention計算或render", command=" ".join(record["command_argv"]), result="PASS: only explanation sentence changed; fence/SVG/attention/helper bytes unchanged; initial 46 evidence hashes unchanged; mask/data axes now explicit", environment=environment),
])
report["sources"].append({"id": "recheck", "kind": "execution", "title": "Same independent reviewer rechecks the actual corrected section", "verified": True, "artifact_id": "recheck_execution"})
claim = next(c for c in report["claims"] if c["id"] == "tensor_and_mask_contract")
claim["status"] = "verified"
claim["statement"] = "tril()[None,None]得到許可表[1,1,4,4]，尾軸四query位置對四候選位置；Q/K/V另為[1,1,4,2]，尾軸四位置與兩特徵，兩種shape在修訂後原文明確區分。factory/tril/clone/manual_attention tuple與對齊契約保持正確。"
claim["verification"]["observed"] = "原次CPU觀測allowed=[1,1,4,4]、QKV/out=[1,1,4,2]、weights=[1,1,4,4]仍有效；本次重新讀新節204、官方L/S與E/Ev契約及矩陣尺寸，文字已完全相符。"
claim["verification"]["details"] += " 修訂複查只改解說句；親核原fence/figure/helper bytes與46證據SHA未變，沿用原次CPU結果，沒有重新執行fence或render。"
claim["evidence"].append({"source_id": "recheck", "locator": "recheck/section.md修訂段；recheck-notes.md全段；recheck/execution.json", "supports": "原同位審閱者完整重讀修訂後小節，官方mask尾軸L,S與資料L,E/S,Ev重新對照；句子已明寫4×4遮罩和4×2資料。"})
claim["artifact_ids"].extend(["rechecked_section", "recheck_execution", "recheck_notes"])
issue = report["issues"][0]
issue["status"] = "resolved"
issue["resolution"] = "協調者只改原204解說句，明寫許可表[1,1,4,4]與兩位置尾軸、Q/K/V[1,1,4,2]與位置/特徵尾軸。原同位審閱者重新親讀目前完整節、官方tril/SDPA原source與manual_attention矩陣公式，確認問題解決；fence/SVG/helper及既有CPU/圖證據bytes/hash未改，沒有新跑attention或render。原問題與initial revise history均保留。"
issue["rechecked_by"] = report["reviewer_task"]
issue["rechecked_source_sha256"] = report["source_sha256"]
issue["initial_report_path"] = history.relative_to(ROOT).as_posix()
issue["initial_report_sha256"] = digest(history)
report["checks"]["factual_accuracy"] = {"status": "pass", "details": "原同位審閱者親自重讀完整修訂節與原issue，許可表4×4與QKV4×2已明確區分；官方尾軸/矩陣公式吻合，其餘因果、shift、−inf、均值/介入內容不變且正確。", "claim_ids": [c["id"] for c in report["claims"]]}
report["checks"]["figure_consistency"]["details"] += " 本次未新render/view：親核SVG bytes/hash完全未變，沿用首次Inkscape親看證據。"
report["checks"]["numeric_verification"]["details"] += " 本次未重新執行fence：原碼bytes/SHA、helper及原證據SHA均未變，沿用首次有界CPU結果。"
report["checks"]["source_verification"]["details"] += " 修訂後再次親讀官方tril/SDPA L,S對E/Ev尺寸、原論文shift/causal段及真helper，來源未變。"
report["reviewed_scope"]["recheck"] = "同一獨立審閱者重讀目前3.6完整原文並核原issue；新raw/逐行diff/官方軸重新親讀/所有既有證據完整性已保存；未新執行fence或render。"
report["review_cycles"] = [{"stage": "initial", "verdict": "revise", "source_sha256": initial_extraction["source_sha256"], "history_path": history.relative_to(ROOT).as_posix(), "history_sha256": digest(history)}, {"stage": "same_reviewer_recheck", "verdict": "pass", "source_sha256": report["source_sha256"], "evidence": PREFIX + "/recheck/execution.json", "reused_cpu_and_figure_evidence": True, "fence_reexecuted": False, "figure_rerendered": False}]
report_path = ROOT / "docs/technical-reviews/3.6.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
command = [sys.executable, "scripts/check_technical_reviews.py", "--lesson", "3.6"]
result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=40)
(RECHECK / "checker.stdout.txt").write_bytes(result.stdout)
(RECHECK / "checker.stderr.txt").write_bytes(result.stderr)
(RECHECK / "checker-receipt.json").write_text(json.dumps({"command_argv": command, "cwd": str(ROOT), "exit_code": result.returncode, "timeout_seconds": 40, "report_sha256": digest(report_path), "new_section_sha256": report["source_sha256"]}, ensure_ascii=False, indent=2) + "\n")
print(result.stdout.decode())
print(json.dumps({"verdict": report["verdict"], "checker_exit_code": result.returncode, "new_source_sha256": report["source_sha256"], "initial_history_path": history.relative_to(ROOT).as_posix(), "initial_history_sha256": digest(history)}, ensure_ascii=False))
assert result.returncode == 0
manifest = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "recheck-manifest.json"]
(OUT / "recheck-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
