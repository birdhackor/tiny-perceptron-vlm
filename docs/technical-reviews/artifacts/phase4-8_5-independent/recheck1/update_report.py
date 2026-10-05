"""Own recheck update; preserves the original issue, verdict, statement and evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
INITIAL_SHA = "7c2aacfff2e628efb36c5c5af0687fc865b00f641bc1e562548b6034f787e3f7"
HISTORY = ROOT / "docs/technical-reviews/history" / ("phase4-8_5-own-initial-revise-" + INITIAL_SHA + ".json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(identifier, filename, kind, description, **extra):
    path = HERE / filename
    return {"id": identifier, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
            "kind": kind, "description": description, **extra}


assert sha(HISTORY) == INITIAL_SHA
report = json.loads(HISTORY.read_bytes())
facts = json.loads((HERE / "reuse-facts.json").read_bytes())
command = json.loads((HERE / "recheck-command.json").read_bytes())
assert command["exit_code"] == 0
assert facts["fence_raw_bytes_unchanged"] is True
assert facts["current_source_sha256"] == sha(HERE / "section.md")
assert report["reviewer_task"] == facts["reviewer_task"]

report["source_sha256"] = facts["current_source_sha256"]
report["verdict"] = "pass"
report["review_history"] = [{
    "phase": "own initial independent review", "verdict": "revise",
    "report_path": HISTORY.relative_to(ROOT).as_posix(), "report_sha256": INITIAL_SHA,
    "source_sha256": facts["initial_source_sha256"],
    "substantive_issue": "parser_standard_boundary; original c3 contradicted; NaN/Infinity format=True was preserved",
    "recheck_phase": "own full section reread plus direct original-source reinspection and SHA-verified reuse of unchanged CPU evidence",
    "current_source_sha256": facts["current_source_sha256"]}]
report["reading_scope"]["section"] = "本人於recheck1完整重讀修訂後8.5 lines146–178，包含新限定句、完整原fence、練習與details；current raw UTF8 bytes由recheck1/section.md保存，不以normalize內容算hash。"
report["reading_scope"]["sources"] += " 本次重新親讀同一原RFC8259§6/9與CPython3.13.5 loads/Standard Compliance/NaN/Repeated Names及decoder段；原bytes逐SHA核驗，LF行號定位見recheck1/recheck.stdout.txt。"
report["reading_scope"]["execution_reuse"] = "未重新執行原fence/14變體/歷史rubric。本人比對新舊原fence raw bytes一致，逐一核驗初稿所有artifacts SHA且沿用本人已執行結果；本次新命令只做原文、来源重读与hash/input證據驗證。"
report["reading_scope"]["figures"] = "當前全文再次確認無任何圖引用；新extraction svg_references=[]、figure_sha256={}，沒有圖可render/view。文字完整呈現原回答字串，不涉及空間或箭頭機制。"

for old in report["artifacts"]:
    if old["id"] == "section":
        old["description"] = "初次revise時的完整8.5原UTF8 bytes，刻意保留，不是當前小節指紋；當前稿另見current_section。"
    if old["id"] == "extract":
        old["description"] = "初次revise時的原稿/fence/helper版本提取結果，保留歷史證據；當前稿另見current_extract。"

report["artifacts"] += [
    artifact("current_section", "section.md", "source_snapshot", "本人完整重讀的修訂後8.5原UTF8 bytes，raw SHA即當前source_sha256。"),
    artifact("current_fence", "fence-1.py", "code", "當前原fence snapshot；raw bytes及SHA與初次執行版本完全一致，未重跑。"),
    artifact("current_extract", "extraction.json", "source_snapshot", "新小節raw SHA、原fence SHA及無圖引用；本次只extract，execution_exit_code=null。"),
    artifact("reuse_code", "verify_reuse.py", "code", "本次實際原文/來源重讀與raw SHA比對的code，不執行模型或原fence。"),
    artifact("reuse_facts", "reuse-facts.json", "source_snapshot", "初稿archive SHA、current source SHA、原code未變、全部舊artifacts SHA驗證與本次重讀范围。"),
    artifact("recheck_run", "recheck.stdout.txt", "execution", "本次完整section及直接來源段落重讀stdout，SHA驗證後列原NaN/Infinity flags與7題rubric；明示原CPU結果重用而非重跑。",
             command=command["command"], result="exit 0；RECHECK ASSERTIONS PASSED；新舊fence bytes一致、全部原artifact SHA吻合，官方RFC/CPython段落支持新限定。",             environment={"python": "3.13.5", "device": "CPU", "libraries": "CPython standard library only", "execution_scope": "Source/section inspection and SHA verification; original program runs reused, not rerun"}),
    artifact("recheck_command", "recheck-command.json", "source_snapshot", "新命令argv/cwd/exit、code/stdout/stderr SHA、環境及RFC LF line定位修正紀錄。"),
    artifact("update_report", "update_report.py", "code", "本人的修訂後報告更新code，保留原revision/issue/NaN反例並只依實際新重讀更新判定。"),
]
report["sources"].append({"id": "reinspection", "kind": "execution", "title": "Own full prose and original-source reinspection with hash-verified reuse", "verified": True, "artifact_id": "recheck_run"})
for source in report["sources"]:
    if source["id"] in {"rfc", "json_docs", "json_decoder"}:
        source["inspection_note"] += " 2026-10-05本次本人重讀此同一原byte snapshot（SHA與source-fetch-receipt吻合）；具體LF行號及原文打印在recheck1/recheck.stdout.txt，未沿用他人摘要。"

c3 = next(claim for claim in report["claims"] if claim["id"] == "c3")
c3["original_review"] = {"statement": c3["statement"], "status": c3["status"],
                         "scope": c3["scope"], "source_sha256": facts["initial_source_sha256"],
                         "initial_report_sha256": INITIAL_SHA,
                         "counterexample": '{"answer":NaN} and {"answer":Infinity}: parse_success=True/format=True/content=False'}
c3["statement"] = "當前稿只聲明本例『答案是：』散文前綴會讓json.loads報錯；並明示『格式』僅指Python預設解析成功且字典只有answer欄位，預設接受NaN等標準外寫法，嚴格JSON規則需要另檢查。"
c3["status"] = "verified"
c3["scope"] = "新prose精確反映本例API/flag範圍，不再聲明default loads拒絕一切不合JSON標準輸入。保留原NaN/Infinity原loop反例作為新限制的直接支持；4.0及重名仍是原fence邊界，不擴大成完整schema/合規驗證器。未刪原實質主張來消除矛盾：original_review與issue原quote/findings以及own initial archive完整保留。"
c3["evidence"].append({"source_id": "reinspection", "locator": "recheck1/recheck.stdout.txt full current section plus RFC8259§6/9 and CPython json Standard Compliance/NaN; reuse-facts.json", "supports": "本人完整重讀新句，重新親讀原權威段落；逐SHA驗證原反例stdout與fence未變，因此據新界定解決先前絕對說法的矛盾，並誠實重用已執行結果。"})
c3["artifact_ids"] += ["current_section", "current_fence", "current_extract", "reuse_facts", "recheck_run", "recheck_command"]

for claim in report["claims"]:
    if claim["id"] in {"c1", "c2", "c4"}:
        claim["recheck_note"] = "本人已重讀目前完整節，該claim與code/既有數字沒有改變；原本人CPU執行證據已逐SHA核對後重用，沒有冒稱新執行。"
issue = report["issues"][0]
issue["status"] = "resolved"
issue["initial_verdict"] = "revise"
issue["initial_source_sha256"] = facts["initial_source_sha256"]
issue["initial_report_path"] = HISTORY.relative_to(ROOT).as_posix()
issue["initial_report_sha256"] = INITIAL_SHA
issue["resolution"] = "協調者修改後，本人完整重讀current source f3948d42d8961345088f0acf6068b0c506080620b1aaee613819ea5a975d8455；150行將廣義格式破損保證改成原例散文前綴，166行緊接原輸出增補default解析/only answer與NaN標準外限制。本人重讀RFC8259§6/9、CPython3.13.5 loads/Standard Compliance/NaN原段並核SHA，確認原NaN format=True反例支持新限定；fence unchanged、沒有新模型執行。問題的original_quote/finding/impact/suggestion與初稿revise及反例均保留。"
issue["actual_recheck"] = {"reviewer_task": report["reviewer_task"], "current_source_sha256": facts["current_source_sha256"], "artifact_ids": ["current_section", "reuse_facts", "recheck_run"], "scope": "Full current section reread, original authority reinspection, exact hash-verified reuse of own original CPU proof."}

report["checks"]["factual_accuracy"] = {"status": "pass", "claim_ids": ["c1", "c2", "c3", "c4"], "details": "本人完整新稿與權威原文複核：c3已從過度概括改為本例prefix與明確預設解析範圍；原API、手寫例及保存7題數字未改，原CPU證據逐SHA驗後重用。無未解實質主張。"}
report["checks"]["limitations"] = {"status": "pass", "claim_ids": ["c1", "c2", "c3", "c4"], "details": "現在正文在原输出後就说明格式是Python預設解析+only answer、NaN標準外仍接受、嚴格JSON須另檢查；保留型別另加、內容閘控、手寫3例與7筆既有實測及不清洗原回答的范围。原初稿問題、反例与revise歷史完整保存。"}
report["checks"]["source_verification"]["details"] += " 本次新增prose本人重讀原RFC/CPython段，正式stdout列直接原文，所有來源及初稿CPU proofs SHA未變；没有把SHA核对冒称重新跑代码。"
report["checks"]["figure_consistency"]["details"] = "當前完整8.5及新extraction仍零圖引用，沒有圖render/view可做，無視覺驗收宣稱；原回答字符串充分给出本節必要素材。"

target = ROOT / "docs/technical-reviews/8.5.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": target.relative_to(ROOT).as_posix(), "report_sha256": sha(target), "source_sha256": report["source_sha256"], "initial_report_sha256": INITIAL_SHA, "verdict": report["verdict"]}, ensure_ascii=False))
