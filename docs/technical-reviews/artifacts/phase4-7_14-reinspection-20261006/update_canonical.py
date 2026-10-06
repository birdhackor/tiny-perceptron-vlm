"""Publish original owner's actual reinspection findings to the canonical report."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
RECEIPT_ID = "phase4_7_14_reinspection_20261006_receipt"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


receipt = json.loads((BASE / "reinspection-receipt.json").read_bytes())
assert receipt["verdict"] == "pass" and not receipt["unresolved_substantive_questions"]
prior_path = BASE / "history/prior-report.json"
report = json.loads(prior_path.read_bytes())
report["source_sha256"] = receipt["current_section"]["sha256"]
report["reviewed_on"] = "2026-10-06"
report["reinspection_context"] = "same original independent technical owner; no new reviewer identity claimed"
report["read_scope"] = {
    "current_section": receipt["current_section"],
    "current_necessary_context": receipt["current_context"],
    "frozen_complete_chapter": receipt["frozen_complete_chapter"],
    "current_source_code_and_figures": {"fence_unchanged": True, "fence_sha256": receipt["code_sha256"], "figure_sha256": {}},
    "actual_reinspection_scope": receipt["actual_scope"],
    "prior_own_real_evidence": "All43 predecessor artifact SHA-256 values verified before reuse. Prior raw section, official sources, execution outputs and provenance retained without changing them. This review did not rerun unchanged fences or reconstruct new model measurements.",
    "new_original_provenance_checks": receipt["actual_source_reading"],
    "actual_visual_inspection": receipt["visual_inspection"],
    "reinspection_receipt_artifact_id": RECEIPT_ID,
}
report["reinspection_history"] = [{
    "reviewed_on": "2026-10-06",
    "reviewer_task": report["reviewer_task"],
    "prior_report_path": prior_path.relative_to(ROOT).as_posix(),
    "prior_report_sha256": sha(prior_path),
    "prior_source_sha256": json.loads(prior_path.read_bytes())["source_sha256"],
    "current_source_sha256": receipt["current_section"]["sha256"],
    "receipt_artifact_id": RECEIPT_ID,
    "changed_claims": ["C5"],
    "finding": "Direct SFT baseline link to7.13 and900-from-scratch updates are independently supported by original SFT code/version/recorded training. Traditional-character edits do not alter claims. All other claims retain explicitly checked own prior evidence.",
    "verdict": "pass",
}]
environment = {k: str(v) for k, v in receipt["environment"].items()}
for p in sorted(BASE.rglob("*")):
    if not p.is_file() or p.name.startswith("checker.") or p.name == "checker-receipt.json":
        continue
    relative = p.relative_to(BASE).as_posix()
    identifier = "reinspect_" + relative.replace("/", "_").replace(".", "_").replace("-", "_")
    if relative == "reinspection-receipt.json":
        identifier = RECEIPT_ID
    kind = "code" if p.suffix == ".py" else "figure_render" if p.suffix == ".png" else "source_snapshot"
    description = "2026-10-06 original-owner reinspection artifact: " + relative
    if "attempts/first-ast-check" in relative:
        description += "; preserves failed verifier ordinal assertion, corrected without changing original source"
    if "failed-cli" in relative or relative == "render/current-expanded.html":
        description += "; initial chromium CLI attempt timed out20s; no successful render claimed from it"
    artifact = {"id": identifier, "kind": kind, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "description": description}
    if relative in ["reinspect.stdout.txt", "reinspection-receipt.json"]:
        artifact.update(kind="execution", command="timeout 30 .venv/bin/python " + (BASE / "reinspect.py").relative_to(ROOT).as_posix(), result="Exit0;43 prior evidence hashes, unchanged fence/primary results, direct SFT original code hashes,900-update raw measurement/AST and baseline equality verified. Actual render/view recorded. No model instantiated or training rerun.", environment=environment)
    elif relative == "render/playwright.stdout.txt":
        artifact.update(kind="execution", command=receipt["visual_inspection"]["successful_render_command"], result="Exit0; HTTP200 current page rendered, details opened, baseline7.13/900 text and unchanged table present; screenshot personally viewed.", environment={k: str(v) for k,v in receipt["visual_inspection"]["render_environment"].items()})
    report["artifacts"].append(artifact)

report["sources"].append({"id": "same_owner_reinspection", "kind": "execution", "title": "Original-owner actual current-version reinspection", "verified": True, "artifact_id": RECEIPT_ID})
for identifier, path, locator in [
    ("direct_sft_text", "scripts/course_experiments/text.py", "run_sft556-566: constructs new_lm, saves start step0, fits mode=sft/name=model at _steps(900); _steps34-36"),
    ("direct_sft_common", "scripts/course_experiments/common.py", "seed41-42/new_lm45-47: seeded new TinyLM(ModelConfig), no parent checkpoint load"),
    ("direct_sft_model", "tiny_perceptron/model.py", "TinyLM.__init__53-67: creates embedding, attention blocks, normalization and output modules, no pretrained state load"),
]:
    p = BASE / "inputs/direct-sft-original" / path
    report["sources"].append({"id": identifier, "kind": "repository_code", "title": "Direct SFT original run source: " + path, "verified": True, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "version": "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3", "inspection_note": "Personally read necessary original branch after AST positioning: " + locator + ". Exact git-show bytes and original sft.json code_sha256 matched. No author outcome/correction strings used."})
for source in report["sources"]:
    if source["id"] == "raw_sft":
        source["rechecked_on"] = "2026-10-06"
        source["inspection_note"] += " Original-owner reinspection: full raw bytes unchanged; checked /revision=a253d126..., /step_scale=1.0, /results/checkpoint=model.pt, /results/training/steps=900, /results/training/records=45, and /results/after equality to ablation before.A_attributes. No author notes/review/correction values read."
    elif source["id"] in ["torch_ce_docs", "torch_ce_native", "raw_ablation", "run_text", "run_common", "run_data", "run_model"]:
        source["rechecked_on"] = "2026-10-06"
        source["reinspection_note"] = "Own previously inspected original snapshot and evidence hashes unchanged; current claim support personally reapplied. No fresh external fetch claimed."

claim = next(c for c in report["claims"] if c["id"] == "C5")
claim["statement"] = "局部錯標比較從7.13所指的直接SFT屬性基模各複製一份；該起點由新建模型從零練900次。兩支用同45訓練題、300次更新、batch16、抽樣起點42，四筆circle/square對調約8.9%，集中兩個已教家族，各33733計分目標；留出標籤維持正解。"
claim["scope"] = "直接SFT原run revision a253d126...的new_lm→900次/name=model與model.pt；消融run52964f...的clean/noisy相同起點與300次契約。新增900主張由原碼與原測量指標核實；43件未變自有真證據明示沿用。未下載/讀權重、重訓或逐權重bit-level復驗。"
for ev in claim["evidence"]:
    if ev["source_id"] == "run_text":
        ev["locator"] = "Original ablation source run_sft_ablation614-662, especially base dependency616-617 and clean/noisy641-649"
    if ev["source_id"] == "raw_sft":
        ev["locator"] = "/revision; /step_scale; /results/checkpoint; /results/training/steps; /results/after"
        ev["supports"] = "原SFT版本a253d126...，model.pt由900次直接SFT產生、step_scale1.0；ablation before.A_attributes等於其after"
claim["evidence"].extend([
    evidence("direct_sft_text", "run_sft556-566; _steps34-36", "直接支線新建模型、保存step0起點、mode=sft/name=model、_steps900；沒有先載入pretrain權重"),
    evidence("direct_sft_common", "new_lm45-47", "起點來自seed後新建TinyLM，而非checkpoint續訓"),
    evidence("direct_sft_model", "TinyLM.__init__53-67", "新建參數模組，構造器無pretrained state載入"),
    evidence("same_owner_reinspection", "reinspection-receipt.json changes_inspected/direct_sft_original_sources/original_measurement_pointers_actually_checked", "新連結與900主張由目前7.13脈絡、original SFT版本/AST/JSON核實；未改代價算例及已有實測證據指紋相同"),
])
claim["artifact_ids"].append(RECEIPT_ID)
claim["verification"]["expected"] += "；直接SFT起點從零900次/name=model→model.pt"
claim["verification"]["observed"] += "；原SFT /training/steps=900，step_scale1.0，new_lm構造器與直接SFT分支無parent checkpoint載入"
claim["verification"]["details"] += " 2026-10-06實際複查新增900主張：原始SFT版本三個source code hashes与raw JSON吻合；AST _steps(900)親執行返回900；未改fence/測量沿用前次真CPU證據，不聲稱重跑。"
claim["verification"]["denominators"]["direct_sft_baseline_steps"] = 900

report["checks"]["factual_accuracy"]["details"] += " 本次原owner已完整重讀現7.14及必要7.13；新增直接SFT900起點與連結獨立核實。"
report["checks"]["numeric_verification"]["details"] += " 2026-10-06未變算例/數字沿用SHA核對後的自有真證據；新增900步原記錄與_steps執行返回900。"
report["checks"]["figure_consistency"]["details"] = "Current7.14仍無教材圖片/SVG。現頁HTTP200已由Playwright實際渲染、打開補充並view_image親看；900段落/7.13連結/表格吻合。教材圖核對NA，實頁截圖並非新教材圖或reader/usability通過。"
report["checks"]["source_verification"]["details"] += " 原owner複查保留opaque prior報告；43舊artifacts與兩主測量完整SHA不變。新增直接SFT run a253d126...三原碼hash親核，原JSON只讀具名raw pointers；新paper/外部源未泛抓。"
report["checks"]["limitations"]["details"] += " 本次只做原版本AST/JSON/指紋與實頁核查；未重訓/讀weights/重算CUDA logits。原結果單seed與少量family限制完整保留。"
report["summary"] = "2026-10-06同原technical owner實際複查 current7.14後pass。完整讀現稿與必要7.13；新直接SFT起點從零900次/連結已核實，未变代價/污染/步數/樣本/判準/測量依43件SHA一致自有真證據明示沿用。現頁已render/view，無未解實質問題。"
path = ROOT / "docs/technical-reviews/7.14.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"verdict": report["verdict"], "report_sha256": sha(path), "prior_report_sha256": sha(prior_path), "receipt_artifact_id": RECEIPT_ID, "receipt_path": (BASE / "reinspection-receipt.json").relative_to(ROOT).as_posix(), "receipt_sha256": sha(BASE / "reinspection-receipt.json")}, ensure_ascii=False, indent=2))
