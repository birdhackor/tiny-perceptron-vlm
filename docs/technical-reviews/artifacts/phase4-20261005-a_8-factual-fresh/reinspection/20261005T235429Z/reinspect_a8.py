"""Original reviewer narrowly reinspects changed A.7 context for unchanged A.8."""
import datetime
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = OUT.parents[1]
ROOT = ART.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_a_8"
BASE = OUT.relative_to(ROOT).as_posix()
SHA = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
EXPECTED_PRIOR = "1271846b41106aeb43180e82448b142b3fffb2661df9432ba5bcf570a7d81284"
assert SHA(OUT / "prior-A.8.json") == EXPECTED_PRIOR
prior = json.loads((OUT / "prior-A.8.json").read_text())
assert prior["reviewer_task"] == TASK

def section(path, lesson):
    raw = path.read_bytes()
    heads = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    i = next(i for i, h in enumerate(heads) if h[0].startswith(("## " + lesson + " ").encode()))
    end = heads[i+1].start() if i+1 < len(heads) else len(raw)
    return raw[heads[i].start():end]

current_path = ROOT / "course/chapters/0A.md"
assert section(current_path, "A.8") == (OUT / "A.8-current.md").read_bytes()
assert section(current_path, "A.7") == (OUT / "A.7-current.md").read_bytes()
assert SHA(OUT / "A.8-current.md") == prior["source_sha256"]
assert SHA(OUT / "A.7-current.md") == "32397500c886a31262e690ab72dd5ba34e5fc2936e5dbac9bebb9547d23bb58d"
for item in prior["artifacts"]:
    assert SHA(ROOT / item["path"]) == item["sha256"], item["id"]
assert SHA(ROOT / "course/figures/rewrite-A-clue-position.svg") == prior["figure_sha256"]["course/figures/rewrite-A-clue-position.svg"]
assert SHA(ROOT / "tiny_perceptron/data.py") == SHA(ART / "inputs/tiny_perceptron/data.py")

# Only the changed prerequisite arithmetic is recomputed; no A.8 audit/model run.
C, history, question_format, retrieval, G, history_add = 100, 20, 15, 40, 25, 30
P = history + question_format + retrieval
P_added = history + history_add + question_format + retrieval
overflow_without_answer = P_added - C
removal_preserving_G = P_added - (C - G)
assert (P, P_added, overflow_without_answer, removal_preserving_G) == (75, 105, 5, 30)
assert P_added + 0 > C and P_added - removal_preserving_G + G == C
assert history + history_add + question_format + 10 + G == C
environment = {"python": sys.version, "platform": platform.platform(), "device": "cpu", "scope": "Integer arithmetic and hash inspection only"}
arithmetic = {
    "inputs": {"C": C, "history": history, "question_and_format": question_format, "retrieval": retrieval, "G": G, "history_increase": history_add},
    "computed": {"original_input_P": P, "increased_input_P": P_added, "overflow_at_G_zero": overflow_without_answer, "required_input_removal_if_G_25": removal_preserving_G},
    "equations": ["P=20+15+40=75", "P'=50+15+40=105", "P'+0-C=105-100=5", "P'-(C-G)=105-(100-25)=30", "50+15+10+25=100"],
    "assertions": "all passed", "tolerance": "Exact integer equality", "environment": environment,
    "scope": "Changed A.7 example as necessary A.8 context; not tokenizer measurements or model output."
}
(OUT / "context-arithmetic.json").write_text(json.dumps(arithmetic, ensure_ascii=False, indent=2) + "\n")
now = datetime.datetime.now(datetime.UTC).isoformat()
receipt = {
    "reviewer_task": TASK, "reinspected_at": now, "verdict": "pass",
    "prior_history_path": f"{BASE}/prior-A.8.json", "prior_history_sha256": EXPECTED_PRIOR,
    "current_source": "course/chapters/0A.md#A.8", "current_source_sha256": SHA(OUT / "A.8-current.md"),
    "current_context": "course/chapters/0A.md#A.7", "current_context_sha256": SHA(OUT / "A.7-current.md"),
    "prior_context_sha256": hashlib.sha256(section(ART / "inputs/course/chapters/0A.md", "A.7")).hexdigest(),
    "true_current_read_scope": "本人親讀完整 current A.7（223–252行）與完整 current A.8（253–288行）；親看自己 frozen A.7/current A.7 的逐行diff，唯一內容差異是A.7開頭預算算例。另親讀已變動的 factual-reviewer-instructions.md 全文。沒有閱讀A.9或作者/其他review工作筆記。",
    "changed_claim": "A.7不再說歷史增加後只減回答預留即可：新說輸入105已超C100五個位置，即使G=0仍不合法；若保留G25，需輸入減30。",
    "independent_verification": arithmetic,
    "impact_on_A8": "A.8依賴的是先組成完整角色/資料/問題序列再數ID，固定三版完整輸入長度與相同G，並與位置可用性分開。新A.7數值邊界與原P+G≤C一致，且沒有改這些依賴規則。A.8不宣稱減G可以挽救P>C，也沒有借用A.7的84份短RAG分數來驗證位置。故A.8四個原claims的支持範圍維持；新加context數值claim記錄本人核算。",
    "reuse": "實核原報告27個declared artifact的SHA全部相符，另將current SVG、ByteTokenizer、checker與原親讀frozen bytes比較均相符。原arXiv v3、HF v4.57.1官方固定版本快照及原locator/存取日沿用；原296/296/296等CPU示範、三種寬度的真render/view證據沿用，不冒稱再次執行或再次fetch。reuse-fingerprint-check.json保存逐件SHA核對。",
    "unchanged_measurement_scope": "A.7未變的84份短RAG補充本人讀到以核依賴範圍；沒有把它納入A.8新的能力/實測結論。本輪不新增模型測量、不替A.7整節審閱背書。",
    "original_frozen_wholefile": "初次0A.md frozen snapshot及其真bytes/SHA保留原歷史語義，不改hash冒充現在整章。current依賴slice另存原bytes，無換行正規化。",
    "independence": "同一原owner閱讀自己的報告/原證據是本次明確指派的窄回查；沒有閱讀他人/旧review判定或作者修正摘要。",
    "unresolved_substantive_issues": []
}
(OUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

report = prior
new_artifacts = []
def add(identifier, relative, kind, description, **extra):
    item = {"id": identifier, "path": f"{BASE}/{relative}", "sha256": SHA(OUT / relative), "kind": kind, "description": description}
    item.update(extra); new_artifacts.append(item)
command = f".venv/bin/python {BASE}/reinspect_a8.py > {BASE}/write-stdout.txt 2> {BASE}/write-stderr.txt"
add("a8_prior_own_history", "prior-A.8.json", "source_snapshot", "自己初次canonical報告的opaque/history保存；不是他人答案或本次主張權威")
add("a7_current_context", "A.7-current.md", "source_snapshot", "本人這次親讀的必要A.7 current原始UTF8 bytes")
add("a8_current_reinspection", "A.8-current.md", "source_snapshot", "本人這次完整親讀的A.8 current原始UTF8 bytes，與原節SHA相同")
add("a7_own_context_diff", "A.7-diff.txt", "derivation", "本人對自己的frozen/current必要context作逐行對照，非作者修正摘要")
add("current_factual_method", "factual-reviewer-instructions-current.md", "source_snapshot", "本人重新親讀的current方法全文")
add("reuse_fingerprint_check", "reuse-fingerprint-check.json", "derivation", "本人實核27個原artifact及現有SVG/原code/checkerSHA，明示沿用範圍")
add("context_arithmetic", "context-arithmetic.json", "execution", "只針對改動的A.7必要context核算75/105/5/30；不是模型score", command=command, result="exit 0；105+0>100、需减30保留G25，全部精確整數assert通過", environment=environment)
add("context_reinspection_receipt", "receipt.json", "derivation", "同一original owner的永久窄回查receipt：實讀範圍、變動claim、獨立核算、原證據沿用、未解與history")
add("context_reinspection_code", "reinspect_a8.py", "code", "這次本人窄回查核算/SHA與canonical寫入assert原程式")
report["artifacts"].extend(new_artifacts)
report["sources"].append({"id": "a7_context_derivation", "kind": "derivation", "title": "Original reviewer's changed prerequisite budget verification", "verified": True, "details": "P=20+15+40=75；歷史加30後P'=50+15+40=105，G=0也超100五格；維持G25需把P'减105-(100-25)=30。只核A.8必要context，非A.7其他實測背書。"})
for claim in report["claims"]:
    if claim["id"] == "position_controls":
        claim["artifact_ids"].extend(["a7_current_context", "a8_current_reinspection", "context_reinspection_receipt", "reuse_fingerprint_check"])
        claim["evidence"].append({"source_id": "a7_context_derivation", "locator": "current A.7 lines225–227；receipt.json /impact_on_A8；context-arithmetic.json /equations", "supports": "改動後依然要求先核完整P、再核P+G≤C；P已超C不能靠减G補救，與A.8固定完整輸入與G的方法相容。"})
report["claims"].append({
    "id": "changed_context_budget_boundary", "kind": "numeric", "statement": "必要前文A.7的新算例：歷史增加30後輸入本身達105，G=0仍超C100五格；若保留G25，需要先把輸入縮短30。",
    "location": "course/chapters/0A.md#A.7 current lines225–227；A.8的274行只依賴該節序列化/預算方法",
    "scope": "A.8必要前文的人工token預算邊界核算，不是實際模型tokenization/生成或A.7整節評測；不改A.8既有實測支持範圍。", "status": "verified",
    "evidence": [{"source_id": "a7_context_derivation", "locator": "context-arithmetic.json /inputs /computed /equations；receipt.json /changed_claim /impact_on_A8", "supports": "独立代入正確区分輸入105與含回答總130，以及G=0/G25兩種邊界。"}, {"source_id": "generation_v4571", "locator": "原親讀GenerationConfig max_length/max_new_tokens，saved text lines203–207；原固定版本SHA已實核相同", "supports": "長度限制須區分完整輸入加新增輸出與新輸出上限；沿用原已核API契約，不宣稱重新fetch。"}],
    "artifact_ids": ["a7_current_context", "a7_own_context_diff", "context_arithmetic", "context_reinspection_code", "context_reinspection_receipt"],
    "verification": {"method": "executed", "expected": "P75；P'105；G0時超5；保留G25需輸入减30。", "observed": "本次有界整數核算(75,105,5,30)精確吻合，assert全部通過。", "details": "未重做A.8原CPU、paperfetch或模型生成；只核變更的必要context公式及删retrieval30後P+G=100。", "tolerance": "精確整數相等，沒有四捨五入。"}
})
report["reinspected_at"] = now
report["read_scope"]["manuscript_initial_frozen_scope"] = report["read_scope"]["manuscript"]
report["read_scope"]["manuscript"] = receipt["true_current_read_scope"] + " 初次讀取範圍/原Whole-file frozen input另保留歷史語義。"
report["reinspection_history"] = [{"receipt_artifact_id": "context_reinspection_receipt", "receipt_path": f"{BASE}/receipt.json", "receipt_sha256": SHA(OUT / "receipt.json"), "prior_own_report_path": f"{BASE}/prior-A.8.json", "prior_own_report_sha256": EXPECTED_PRIOR, "current_context_sha256": SHA(OUT / "A.7-current.md"), "current_source_sha256": SHA(OUT / "A.8-current.md"), "verdict": "pass"}]
report["independence_record"] += " 本次same-owner窄回查僅讀自己的prior報告與原證據，以及current A.7/A.8；原初次獨立性記錄保持歷史語義。"
for name in ["factual_accuracy", "numeric_verification", "source_verification", "limitations"]:
    report["checks"][name]["claim_ids"].append("changed_context_budget_boundary")
    report["checks"][name]["details"] += " 本次same original owner親讀current A.7/A.8並獨立核105/5/30的新預算邊界，沒有改變A.8支持範圍；原外部固定版本/原CPU/SVG證據經實核SHA沿用，沒有宣稱重跑。詳context_reinspection_receipt。"
report["checks"]["figure_consistency"]["details"] += " 本次SVG與A.8本文SHA實核不變，沿用本人先前的真render/view證據。"
report["verdict"] = "pass"
report["issues"] = []
report["unresolved_substantive_claims"] = []
target = ROOT / "docs/technical-reviews/A.8.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
written = json.loads(target.read_text())
assert written["reviewer_task"] == TASK and written["source_sha256"] == SHA(OUT / "A.8-current.md")
assert written["verdict"] == "pass" and written["issues"] == []
for item in written["artifacts"]:
    assert SHA(ROOT / item["path"]) == item["sha256"], item["id"]
print(json.dumps({"written": target.relative_to(ROOT).as_posix(), "reviewer_task": TASK, "verdict": written["verdict"], "report_sha256": SHA(target), "prior_history_sha256": EXPECTED_PRIOR, "receipt_artifact_id": "context_reinspection_receipt", "receipt_path": f"{BASE}/receipt.json", "receipt_sha256": SHA(OUT / "receipt.json"), "arithmetic": arithmetic["computed"], "assertions": "all passed"}, ensure_ascii=False))
