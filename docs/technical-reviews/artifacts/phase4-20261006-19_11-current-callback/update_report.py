"""Same original reviewer's current callback; never read a peer's report."""
import hashlib
import json
from pathlib import Path

R = Path.cwd()
A = Path(__file__).resolve().parent
TASK = "/root/phase4_factual_coordinator/factual_19_11"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior_path = A / "prior/19.11.json"
assert sha(prior_path) == "a2d8e1e816bc9547354cbf68d1a6fe3c1574865cc39ff7d67d6796abaaaf3f4c"
# This is my own exact archived report/registry, not another reviewer's answer.
report = json.loads(prior_path.read_bytes())
assert report["reviewer_task"] == TASK and report["source_sha256"] == "68eb2bb942a6481e1e775ee994feba852d16aa7aca94069b1b44da34078a0b3e"
current = json.loads((A / "current-inputs.json").read_bytes())
reuse = json.loads((A / "fingerprint-reuse.json").read_bytes())
assert all(x["unchanged"] for x in reuse["current_original_files"])
assert all(x["exact_match"] for x in reuse["original_report_artifacts"])
assert reuse["all_fence_raw_bytes_identical"]
inspection = json.loads((A / "current-inspection.json").read_bytes())
visual = json.loads((A / "visual-inspection.json").read_bytes())
assert visual["actual_image_view_tool_used"] and len(visual["viewed_images"]) == 4
assert inspection["necessary_context"] == [] and inspection["intro"] is None and inspection["figure_sha256"] == {}
assert len(inspection["actual_changes"]) == 2
env = json.loads((A / "current-inspection-environment.json").read_bytes())
execution = json.loads((A / "current-inspection-execution.json").read_bytes())

report["source_sha256"] = current["own_source_sha256"]
report["initial_reviewed_on"] = report["reviewed_on"]
report["reviewed_on"] = "2026-10-06"
report["review_round"] = "same_original_reviewer_current_callback"
report["reviewer_task"] = TASK
report["figure_sha256"] = {}
report["prior_review"] = {
    "opaque_file": prior_path.relative_to(R).as_posix(), "sha256": sha(prior_path),
    "source_sha256": "68eb2bb942a6481e1e775ee994feba852d16aa7aca94069b1b44da34078a0b3e",
    "proof_opaque_directory": (A / "prior/proof").relative_to(R).as_posix(),
    "opaque_preservation_manifest": (A / "opaque-preservation.json").relative_to(R).as_posix(),
    "rule": "Own complete report and all prior proof/history were copied opaquely before the current callback. Former verdict is not the changed-claim evidence.",
}
report["initial_read_scope_artifact_id"] = report["read_scope_artifact_id"]
report["read_scope_artifact_id"] = "current-inspection-19_11"
report["current_inspection"] = {
    "artifact_id": "current-inspection-19_11", "path": (A / "current-inspection.json").relative_to(R).as_posix(),
    "sha256": sha(A / "current-inspection.json"),
    "own_section": {"source": report["source"], "sha256": current["own_source_sha256"]},
    "intro": None, "figures": {}, "necessary_context": [],
    "html_file": "outputs/site/19.11.html", "html_sha256": current["html_sha256"],
    "http_url": current["http_url"], "http_status": 200,
    "visual_artifact_id": "callback-visual-inspection", "actual_images_viewed": True,
}
report["current_frozen_input"] = {
    "artifact_id": "callback-current-whole-frozen", "sha256": current["frozen_whole_input_sha256"],
    "meaning": current["whole_input_role"],
}
report["evidence_reuse"] = {
    "artifact_id": "callback-fingerprint-reuse", "actual_scope": inspection["reused_scope"],
    "original_execution_date": "2026-10-05", "new_model_cpu_run": False,
    "new_model_generation_or_score": False, "new_complete_recipe_training": False,
    "new_model_download": False, "new_source_fetch": False,
}
report["reviewer_summary"] = "本人完整重讀current19.11與自己初次frozen原節真diff：只有兩處四站改成四份階段／分支稱呼，原程式/數值/五fence/所有命令不變。親讀原STAGES與train_stage必要AST/原方法確認DPO從joint接續、正文保持optional比較分支；HTTP200真current own page與輸出HTML一致，桌面/手機四張改動區渲染本人實查看。原15組概念/測量/短CPU證據在精確指紋與支持範圍核後沿用，不重新跑完整pipeline或模型能力分數。"

new_files = [
    ("current-inspection-19_11", "current-inspection.json", "source_snapshot", "本人的完整current callback讀取/改動/真方法/visual/重用scope与events記錄；actual execution原結果另存callback-render-results。"),
    ("callback-render-code", "render_site.py", "code", "實際執行的原AST/current HTTP頁與desktop/mobile渲染檢查程式；不跑模型。"),
    ("callback-report-code", "update_report.py", "code", "同原reviewer本人更新canonical的程式，只讀自己已opaque保存的registry，不讀peer。"),
    ("callback-execution", "current-inspection-execution.json", "execution", "真新command與exit0/cwd/shell及結果來源。"),
    ("callback-stdout", "current-inspection.stdout.txt", "execution", "新AST/HTML/browser執行的原stdout。"),
    ("callback-stderr", "current-inspection.stderr.txt", "execution", "新AST/HTML/browser執行的原stderr，空。"),
    ("callback-environment", "current-inspection-environment.json", "execution", "實際Python/installed torch/Playwright/Chromium/CPU/browser版本。"),
    ("callback-render-results", "current-inspection-results.json", "execution", "真HTML/code-fence/AST/predecessor和viewport/layout/screens SHA原結果。"),
    ("callback-visual-inspection", "visual-inspection.json", "source_snapshot", "本人view_image實查四圖的觀察，與真渲染原結果分開保存。"),
    ("callback-fingerprint-reuse", "fingerprint-reuse.json", "source_snapshot", "23 current原檔/52 prior正式證據與各opaque副本精確SHA一致的真核對紀錄，原fences不變；非browser命令產物。"),
    ("callback-diff", "section-diff.txt", "source_snapshot", "本人current own section與自己先前raw frozen section的真unified diff。"),
    ("callback-section", "section-current.md", "source_snapshot", "current19.11完整raw UTF8 bytes，SHA62eb0ba9...。"),
    ("callback-current-inputs", "current-inputs.json", "source_snapshot", "current own/HTML/HTTP及實際frozen snapshot的版本含義。"),
    ("callback-current-whole-frozen", "current/course/chapters/19.md", "source_snapshot", "真正saved frozen input；whole chapter只bytes/hash，正式own判定仍按sectionSHA。"),
    ("callback-generated-html", "current/outputs/site/19.11.html", "source_snapshot", "current generated精確own page永久副本，非只引用ignored outputs。"),
    ("callback-http-html", "http-19.11.html", "source_snapshot", "HTTP200真response原bytes，与current generated HTML相同。"),
    ("callback-opaque-report", "prior/19.11.json", "source_snapshot", "本人前canonical完整opaque保存，原SHA a2d8e1e8... 不改舊指紋。"),
    ("callback-opaque-manifest", "opaque-preservation.json", "source_snapshot", "先opaque報告/完整priorproof/history的精確file/SHA/bytes複本清單。"),
    ("callback-current-instructions", "current/docs/review-tools/factual-reviewer-instructions.md", "source_snapshot", "本次先讀latest factual reviewer instructions原UTF8。"),
    ("callback-current-protocol", "current/.agents/skills/clear-tutorial/references/review-protocol.md", "source_snapshot", "本次親讀review protocol原UTF8。"),
    ("callback-current-schema", "current/scripts/check_technical_reviews.py", "code", "current checker schema；與本人前檢查版本SHA同，未修改。"),
]
for identifier, filename, kind, description in new_files:
    p = A / filename
    item = {"id": identifier, "path": p.relative_to(R).as_posix(), "sha256": sha(p), "kind": kind, "description": description}
    if kind == "execution":
        item.update(command=execution["command"], result=execution["result"] + " Fingerprint/visual/write-scope observations are also saved, not represented as model reruns.", environment={k: str(v) for k, v in env.items()})
    report["artifacts"].append(item)
for name in visual["viewed_images"]:
    p = A / name
    report["artifacts"].append({"id": "callback-" + name.replace(".", "_"), "path": p.relative_to(R).as_posix(),
                                "sha256": sha(p), "kind": "figure_render", "description": "本人實際view_image查看的current own page改動區desktop/mobile渲染，沒有新增教材圖。"})
report["sources"].append({"id": "current-page-branch-inspection", "kind": "execution", "verified": True,
                          "title": "本人current own label/AST/HTTP/render inspection", "artifact_id": "callback-render-results"})
for source in report["sources"]:
    if source["id"] != "current-page-branch-inspection":
        source["current_callback_reuse"] = {"date": "2026-10-06", "same_original_reviewer": True,
                                           "exact_fingerprint_and_support_scope_checked": True, "new_fetch_or_model_execution": False}
for claim in report["claims"]:
    claim["current_callback_support_scope_checked"] = True
    if claim["id"] in ("c03", "c11"):
        claim["statement"] = claim["statement"].replace("四站FP32", "四份階段／分支FP32").replace("四站/續訓CLI", "四份階段／分支及續訓CLI")
        claim["evidence"].append({"source_id": "current-page-branch-inspection", "locator": "current-inspection.json /actual_changes /actual_method_read_scope /page; current-inspection-results.json /stage_predecessors_from_original_ast",
                                 "supports": "本人新核兩處稱呼與DPO<-joint原方法，五fences及全部recipe bytes/原支持仍同；非重跑長recipe。"})
        claim["artifact_ids"].append("current-inspection-19_11")
        claim["scope"] += " 本次callback只改四份階段／分支稱呼，原CPU/CLI證據SHA與支持範圍本人核後沿用，未冒充新執行。"
report["claims"].append({
    "id": "c16", "kind": "software", "statement": "兩處新『四份階段／分支模型』稱呼，與原pretrain/sft/joint/dpo格式和DPO<-joint比較分支契約相符；原五個fence/完整recipe沒有改。",
    "location": "current19.11 public names用途表第一列；四行train recipe前開場",
    "scope": "只核actual changed labels/原CLI階段關係，完整重讀own正文保持DPOoptional comparison。沒有新增模型或phase5工程/訓練/能力驗收。",
    "status": "verified", "evidence": [
        {"source_id": "capstone", "locator": "STAGES/DEFAULT_STEPS 24-25 (current SHA identical)", "supports": "四个名pretrain/sft/joint/dpo格式及排程本来存在。"},
        {"source_id": "trainer", "locator": "train_stage100-123 &179-213 (current SHA identical)", "supports": "完成前站檢查dpo<-joint；DPO目標/reference/optimizer是原分支，不把optional變必需。"},
        {"source_id": "current-page-branch-inspection", "locator": "current-inspection.json actual_changes/page; results stage_predecessors/screens; visual-inspection.json", "supports": "本人真diff/AST/HTTP200/desktop&mobile渲染與實際view，兩處label可見且code5fences保留原內容。"},
    ], "artifact_ids": ["current-inspection-19_11", "callback-render-results", "callback-render-code", "callback-stdout", "callback-visual-inspection", "callback-diff"],
    "verification": {"method": "executed", "expected": "四個原stage名、dpo前站joint；兩處newlabel渲染，全部原5fences與命令不變。",
                     "observed": "AST predecessors pretrain:null/sft:pretrain/joint:sft/dpo:joint；真current pageHTTP200与generatedSHA相等；1280x800/390x844四圖本人查看，全部五fence DOM/raw一致。",
                     "details": "新執行只有AST與browser/page inspection。未點執行按钮，0模型CPU/GPU/新generation/長recipe/下載；previous CPU records只明示沿用。"},
})
report["checks"]["factual_accuracy"]["claim_ids"].append("c16")
report["checks"]["factual_accuracy"]["details"] += " 本次完整current own讀取/真diff只兩處label更精確；原16組支持範圍親核，改動另記c16。"
report["checks"]["source_verification"]["claim_ids"].append("c16")
report["checks"]["source_verification"]["details"] += " Callback原23source/52正式proof全部exact SHA；不refetchpaper或重pipeline/11modelCPU。新增證據是own page/AST/render/code/stdout/env/真觀察，正式current_inspection可追踪。"
report["checks"]["figure_consistency"]["details"] += " 本次仍fig={}；指定current own HTML/HTTP200，desktop1280x800/mobile390x844兩處改動區本人真render+view，snapshot永久保存。沒有把頁面截圖冒充新教材圖。"
report["checks"]["limitations"]["claim_ids"].append("c16")
report["checks"]["limitations"]["details"] += " 本callback沒有新模型CPU、GPU、paper fetch、長recipe或新agent/source修改。完整opaque prior/history保留，原作者canonical未知仍[]；仅own current/必要原method，无peer/extra修正筆記讀取。Metadata writer曾relative/absolute path ValueError，保存真事件并修正，非實質來源問題。"
report["issues"] = []
report["verdict"] = "pass"
target = R / "docs/technical-reviews/19.11.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "report_path": target.relative_to(R).as_posix(), "report_sha256": sha(target),
                  "own_source_sha256": report["source_sha256"], "current_inspection": report["current_inspection"],
                  "claims": len(report["claims"]), "issues": []}, ensure_ascii=False))
