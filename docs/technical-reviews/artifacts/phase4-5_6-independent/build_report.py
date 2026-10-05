"""Construct the fresh reviewer-authored report and immutable evidence manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def read(rel):
    return json.loads((OUT / rel).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(identifier, rel, kind, description, **extra):
    path = OUT / rel
    return dict(id=identifier, path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                kind=kind, description=description, **extra)


def ev(source_id, locator, supports):
    return dict(source_id=source_id, locator=locator, supports=supports)


extract = read("original-run/extraction.json")
original = read("original-run/execution.json")
observed = read("probe-results.json")
probe_env = read("probe-environment.json")
original_env = read("original-run/environment.json")
provenance = read("input-provenance.json")
assert original["exit_code"] == 0 and original["executed"]
assert original_env["attempted_fences"] == [1] and original_env["guard_events"] == []
assert sha(OUT / "original-run/section.md") == extract["source_sha256"]
assert sha(OUT / "original-run/fence-1.py") == extract["python_fences"][0]["sha256"]
assert len(extract["python_fences"]) == 1 and len(extract["svg_references"]) == 1
environment = {"python": probe_env["python"], "torch": probe_env["torch"],
               "device": "CPU; CUDA build None, CUDA unavailable", "threads": "1"}
probe_command = "timeout 30s .venv/bin/python docs/technical-reviews/artifacts/phase4-5_6-independent/probe.py > docs/technical-reviews/artifacts/phase4-5_6-independent/probe.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_6-independent/probe.stderr.txt"
render_command = "inkscape docs/technical-reviews/artifacts/phase4-5_6-independent/inputs/course/figures/rewrite-05-06-learning-rate.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/phase4-5_6-independent/figure-render.png > docs/technical-reviews/artifacts/phase4-5_6-independent/render.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_6-independent/render.stderr.txt"
(OUT / "run-receipts.json").write_text(json.dumps({
    "original": {"command": provenance["original_fence_command"], "exit_code": original["exit_code"],
                 "timeout_seconds": 30, "elapsed_seconds": original["elapsed_seconds"]},
    "probe": {"command": probe_command, "exit_code": 0, "timeout_seconds": 30,
              "scope": "Schedule arithmetic, original fence sentinel, original CLI metadata comparison, SVG coordinates; no training"},
    "render": {"command": render_command, "exit_code": 0, "tool": "Inkscape 1.4 (e7c3feb100, 2024-10-09)",
               "viewed": True, "view_tool": "functions.tools.view_image", "image_pixels": "640x680",
               "warnings": "PangoFT2FontMap/GtkRecentManager initialization warnings; PNG successfully produced and viewed"},
}, ensure_ascii=False, indent=2) + "\n")
artifacts = [
    artifact("section", "original-run/section.md", "source_snapshot", "Current 5.6 original UTF-8 section bytes"),
    artifact("original-fence", "original-run/fence-1.py", "code", "Unmodified complete original Python fence"),
    artifact("original-extraction", "original-run/extraction.json", "source_snapshot", "Original bytes/fence/figure/bootstrap hashes and one-fence/one-SVG counts"),
    artifact("original-execution", "original-run/execution.json", "execution", "Actual current-section original-fence CPU execution receipt", command=provenance["original_fence_command"], result="exit 0; one fence executed; no guard events; 40 rate values computed", environment=environment),
    artifact("original-stdout", "original-run/stdout.txt", "source_snapshot", "Actual first-six and last-three output, copied verbatim from current run"),
    artifact("original-stderr", "original-run/stderr.txt", "source_snapshot", "Empty stderr from current original-fence run"),
    artifact("original-env", "original-run/environment.json", "source_snapshot", "Actual CPU software versions, guard policy/events, module hashes"),
    artifact("probe-code", "probe.py", "code", "Reviewer-written bounded numerical, metadata-contract and SVG checks"),
    artifact("probe-execution", "probe-results.json", "execution", "Actually executed independent schedule and boundary results", command=probe_command, result="exit 0; all assertions passed; 40 base rates, 80 changed-warmup rates, 7 cap boundaries, resume metadata guard and 40 SVG points verified", environment=environment),
    artifact("probe-stdout", "probe.stdout.txt", "source_snapshot", "Actual probe stdout, including exact original fence rerun and results"),
    artifact("probe-stderr", "probe.stderr.txt", "source_snapshot", "Empty stderr from independent probe execution"),
    artifact("probe-env", "probe-environment.json", "source_snapshot", "Actual bounded-probe CPU environment and inspected helper/CLI SHA-256"),
    artifact("figure-render", "figure-render.png", "figure_render", "Inkscape SVG render actually viewed by reviewer", command=render_command, result="640x680 PNG produced, labels and curve viewed"),
    artifact("render-stderr", "render.stderr.txt", "source_snapshot", "Actual renderer warnings, retained as tool limitation evidence"),
    artifact("run-receipts", "run-receipts.json", "source_snapshot", "Actual commands/outcomes and render/view limits"),
    artifact("sgdr-pdf", "sources/sgdr-1608.03983v1.pdf", "source_snapshot", "Original author-paper PDF directly fetched from arXiv, exact v1 printed on first page"),
    artifact("sgdr-text", "sources/sgdr-1608.03983v1.txt", "source_snapshot", "pdftotext -layout original PDF text personally read at title and section 3 Eq5"),
    artifact("hf-source", "sources/hf-v4.57.1-optimization.py", "source_snapshot", "Official pinned Transformers release raw source personally read"),
    artifact("source-acquisition", "source-acquisition.json", "source_snapshot", "Direct HTTPS URLs, access date, final URLs, response versions and hashes"),
    artifact("training-snapshot", "inputs/tiny_perceptron/training.py", "code", "Actual helper source preserving index, cap and endpoint semantics"),
    artifact("train-snapshot", "inputs/scripts/train.py", "code", "Actual CLI metadata/resume/scheduler original code"),
    artifact("figure-snapshot", "inputs/course/figures/rewrite-05-06-learning-rate.svg", "source_snapshot", "Unmodified SVG used in render and point comparison"),
    artifact("input-provenance", "input-provenance.json", "source_snapshot", "Repository HEAD, exact original inputs, read scope and no-weight/no-dataset input declaration"),
    artifact("inspection", "inspection.md", "derivation", "Reviewer reasoning, authority/version/locator inspection, numeric derivation, support scopes and limits"),
    artifact("prepare-code", "prepare_evidence.py", "code", "Executed input preservation and original-authority acquisition program"),
    artifact("report-code", "build_report.py", "code", "Reviewer report-construction program; no prior report loaded"),
]
sources = [
    {"id": "sgdr", "kind": "paper", "title": "SGDR: Stochastic Gradient Descent with Restarts — Loshchilov and Hutter",
     "url": "https://arxiv.org/pdf/1608.03983v1", "version": "arXiv:1608.03983v1, 13 Aug 2016",
     "accessed_on": "2026-10-05", "authority_reason": "Original method paper by its authors, fetched directly from arXiv's versioned PDF endpoint",
     "verified": True, "checked_original": True,
     "inspection_note": "Personally read first-page title/authors/printed v1 identity, §1 Eq1, §3 printed pp3–4 Eq5 and adjacent endpoint definition. Eq5 supports within-run half-cosine decay between explicit endpoints; it does not support this repo's warmup cap or its update-index convention. Saved original PDF/text and HTTPS receipt."},
    {"id": "hf", "kind": "official_source", "title": "Hugging Face Transformers optimization.py warmup/cosine scheduler APIs",
     "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/optimization.py", "version": "Official Transformers release v4.57.1",
     "accessed_on": "2026-10-05", "authority_reason": "Pinned release source in the official huggingface/transformers organization repository",
     "verified": True, "checked_original": True,
     "inspection_note": "Personally read lines132–172,324–384,387–409. Linear warmup + half-cosine is an officially offered schedule and can have a min-rate floor. The later lambda at line324 replaces the earlier same-name definition. Normal and start-rate variants have different indexing; repo semantics are checked separately. Transformers is not installed; this source version was read, not executed."},
    {"id": "helper", "kind": "repository_code", "title": "Course learning_rate helper",
     "path": "tiny_perceptron/training.py", "sha256": sha(ROOT / "tiny_perceptron/training.py"),
     "version": provenance["repo_head"] + " plus exact current-file SHA-256", "verified": True,
     "inspection_note": "Read complete file, especially lines106–115. Function validates positive total/peak, clamps warmup, returns (step+1)/warmup in warmup, and clamps cosine progress to1 with 10%-peak floor. Snapshot preserved."},
    {"id": "cli", "kind": "repository_code", "title": "Course CLI total-schedule metadata and resume contract",
     "path": "scripts/train.py", "sha256": sha(ROOT / "scripts/train.py"),
     "version": provenance["repo_head"] + " plus exact current-file SHA-256", "verified": True,
     "inspection_note": "Personally read parser, training_metadata lines170–185, resume comparison lines229–236, update loop lines303–312 and save calls. Original metadata function and exact resume-comparison AST executed without main/training; changed total40→100 rejected."},
    {"id": "closed-form", "kind": "derivation", "title": "Independent 40-step warmup and cosine calculation", "verified": True,
     "details": "Set W=5,T=40,peak=.001,floor=.0001. For i=0..4 use exact rational (i+1)/5000. For i=5..39 use SGDR Eq5 with phase Fraction(i-5,35), floor +(peak-floor)/2*(1+cos(pi*phase)); last phase34/35. For W10 use warmup(i+1)/10000 and phase(i-10)/30. Compare all outputs at absolute1e-18. SVG mapping x184+416*i/39,y424-300*rate/.001 with two-decimal coordinate tolerance. Code/results saved."},
    {"id": "original-run", "kind": "execution", "title": "Actual unmodified 5.6 fence CPU execution", "verified": True, "artifact_id": "original-execution"},
    {"id": "bounded-probe", "kind": "execution", "title": "Actual bounded schedule/variation/metadata/SVG probe", "verified": True, "artifact_id": "probe-execution"},
]
claims = [
    {"id": "schedule-method", "kind": "concept", "status": "verified",
     "statement": "線性暖身後以餘弦降低學習率，是可選的常見排程；排程按步序給定步幅，沒有自動保證最佳值。",
     "location": "course/chapters/05.md:185,187,200", "scope": "支持方法形態及按預定進度工作，不把SGDR重啟或任何訓練成績移植到本課，亦不宣称0.001/40步是最佳配方。",
     "evidence": [ev("sgdr", "§1 Eq1 and §3 printed pp3–4 Eq5 with endpoint paragraph", "學習率是更新式中的步幅；半餘弦由指定最大/最小率與已進行的週期比例決定。"),
                  ev("hf", "get_cosine_schedule_with_warmup lines141–172; active lambda lines324–332; min-rate API lines335–384", "官方提供線性暖身接半餘弦的選項，輸入為current_step/warmup/total/endpoint，沒有讀取loss或最佳值oracle。")],
     "artifact_ids": ["sgdr-pdf", "sgdr-text", "hf-source", "inspection"]},
    {"id": "base-rates", "kind": "numeric", "status": "verified",
     "statement": "40步、peak0.001、warmup5：前五值為0.0002/0.0004/0.0006/0.0008/0.001；索引5仍為峰值，末三值向0.0001靠近，1e-9只是斷言容差。",
     "location": "course/chapters/05.md:185,192–198", "scope": "核對此40值純時間表與峰值上界；最後實際更新是i39，並未宣稱i39恰等於尾端下限。",
     "evidence": [ev("helper", "learning_rate lines106–115", "暖身使用(i+1)/W，i5下降區相位0，i39相位34/35，下限0.1*peak。"),
                  ev("closed-form", "inspection.md numeric derivation; probe.py independent_cosine and original block", "從原論文封閉式及有理相位獨立代入全部35個下降值，核對所有40項。"),
                  ev("original-run", "original-run/stdout.txt and execution.json", "原fence親跑印出首六、末三並通過max上界assert。"),
                  ev("bounded-probe", "probe-results.json.original", "40項、兩個峰值、單調下降、末相位及i40下限實際數值。")],
     "artifact_ids": ["original-execution", "original-stdout", "probe-execution", "probe-code", "inspection"],
     "verification": {"method": "executed", "expected": "前五為指定數值；i4=i5=.001=max；最後三值高於且接近.0001，i40=.0001；1e-9=0.000000001。",
                      "observed": "首六[.0002,.0004,.0006000000000000001,.0008,.001,.001]；末三[.00011621671268686605,.00010723168513061665,.00010181156770214242]；max=.001。",
                      "tolerance": "所有獨立公式比較absolute1e-18；SVG另計。原assert峰值容差1e-9，非更改配置高峰。",
                      "details": "完整原fence及獨立封閉式皆實際執行，無訓練；浮點0.0006000000000000001與正文十進位0.0006一致於容差內。"}},
    {"id": "fence-software-scope", "kind": "software", "status": "verified",
     "statement": "原fence只用本課learning_rate產生時間表：range40/list comprehension、列印[:6]及[-3:]、max與assert，沒有計算梯度、更新模型或評測。",
     "location": "course/chapters/05.md:187,190–195,198,211; SVG final caption", "scope": "逐项覆盖import/function参数、0-based索引、range40、兩個切片、print、max及assert；僅驗證原fence，不宣稱任何長訓練recipe完成。",
     "evidence": [ev("helper", "learning_rate lines106–115", "函數只返回由step,total,peak,warmup計算的float。"),
                  ev("original-run", "original-run/fence-1.py lines1–6; original-run/environment.json attempted_fences=[1]", "完整原碼在CPU親跑成功，沒有optimizer/backward/train呼叫。"),
                  ev("bounded-probe", "probe.py original fence exec/sentinel block; probe-results.json.original", "原fence再次執行；預先建立的CPU參數/optimizer未變且無梯度。")],
     "artifact_ids": ["original-fence", "original-execution", "original-env", "probe-execution", "probe-code"],
     "verification": {"method": "executed", "expected": "40個float；首六與末三切片；峰值斷言通過；模型/optimizer無更新。", "observed": "原helper exit0,1fence,無guard event；獨立原fence重跑40值且sentinel參數/optimizer不變、grad None。", "details": ".venv Python3.13.5/PyTorch2.14.1+cpu；原fence30秒界限，probe timeout30秒；没有下载data/model。"}},
    {"id": "warmup-variation-and-cap", "kind": "numeric", "status": "verified",
     "statement": "改warmup10：首步0.0001、i4=.0005、i9=.001；total40時warmup30被限成10；total≥4上限floor(total/4)，total1–3上限1。",
     "location": "course/chapters/05.md:204,209", "scope": "本課helper自己的整數步數約定；沒有把25%上限當成原論文或所有cosine排程通則。",
     "evidence": [ev("helper", "learning_rate lines111–115", "warmup=min(requested,max(1,total//4))；暖身/下降公式與所述邊界一致。"),
                  ev("bounded-probe", "probe-results.json.warmup_variation and warmup_cap_boundaries", "全部40個warmup10值及40個warmup30值親跑，exact相等；1,2,3,4,7,8,40的預期cap1,1,1,1,1,2,10核對。"),
                  ev("closed-form", "probe.py Fraction warmup and independent_cosine calculations", "獨立計算暖身10首十值與30項下降；末三接近.0001且路線與暖身5不同。")],
     "artifact_ids": ["probe-execution", "probe-code", "training-snapshot", "inspection"],
     "verification": {"method": "executed", "expected": "warmup10首十為(i+1)*.0001，warmup30與10逐項exact相等，指定總步數cap吻合。", "observed": "i0=.0001,i4=.0005,i9=.001；warmup10末三[.00012202456766718093,.00010983357966978745,.00010246514708427702]；warmup30=10 True；7項總步數邊界均通過。", "tolerance": "數值closed-form absolute1e-18，warmup30/10整列表精確相等；cap整數精確相等。", "details": "80個改暖身排程值與7種短/整數floor邊界有界CPU實際執行；total1–2不會在實際列表到floor，本節只主張其cap。"}},
    {"id": "total-plan-changes-rate", "kind": "numeric", "status": "verified",
     "statement": "同樣第20次更新，總計畫40或100步會給不同學習率。",
     "location": "course/chapters/05.md:200", "scope": "保持peak.001/warmup5不變比較更新20=i19；僅示範總計畫參與排程相位計算。",
     "evidence": [ev("helper", "learning_rate lines114–115", "progress分母為total-warmup，故總計畫改動會改變同一步的率。"),
                  ev("bounded-probe", "probe-results.json.same_20th_update_with_different_plan", "同i19用total40與100實際計算不同值。")],
     "artifact_ids": ["probe-execution", "probe-code"],
     "verification": {"method": "executed", "expected": "相同i19,peak.001,W5時total40與100不相等。", "observed": "total40=.0006890576474687264；total100=.0009526281984193435。", "tolerance": "不同值的精確比較成立；差約.00026357，遠大於1e-18浮點容差。", "details": "保持索引、暖身和高峰固定，未使用不同訓練成績作比較。"}},
    {"id": "resume-keeps-plan", "kind": "concept", "status": "verified",
     "statement": "按此預定排程續訓需保留原總計畫；延長總步數等於改排程計畫。",
     "location": "course/chapters/05.md:200; supplemental linked continuation contract", "scope": "方法概念由原始時間相位/API支持，repo metadata保存與拒絕改計畫由原碼/有界合約執行支持；未宣稱完整resume訓練重現已測。",
     "evidence": [ev("sgdr", "§3 Eq5 and adjacent T_cur/T_i definition", "時間相位以run長度為分母；改長度會改同一進度的排程。"),
                  ev("hf", "get_cosine_schedule_with_warmup lines152–160; active lambda lines324–332", "總training步數與resume進度是排程合約中的明確輸入。"),
                  ev("cli", "training_metadata lines170–185; resume guard lines229–236; step assignment lines303–312; checkpoint save lines362–386", "保存schedule_steps和peak_lr；使用原進度/原總步數；resume metadata不一致會拒絕。"),
                  ev("bounded-probe", "probe-results.json.resume_plan_contract; probe.py exact original AST extraction", "原metadata函數和原comparison block實際接受同40步計畫，拒絕40→100。")],
     "artifact_ids": ["probe-execution", "probe-code", "train-snapshot", "hf-source", "inspection"]},
    {"id": "figure-rates-and-labels", "kind": "numeric", "status": "verified",
     "statement": "圖的x軸是更新次序1–40，y軸為當次學習率；40點呈現前5步暖身、第6步仍峰值、之後向峰值一成靠近，且明示未更新模型。",
     "location": "course/chapters/05.md:200–202; course/figures/rewrite-05-06-learning-rate.svg", "scope": "核對原SVG實際render/view和全部40座標，沒有主張瀏覽器手機頁版式已驗證。",
     "evidence": [ev("bounded-probe", "probe-results.json.figure_numeric_consistency; probe.py SVG parsing/mapping block", "40點逐項對應原rate值，兩位小數座標容差內；tick/caption文字核對。"),
                  ev("closed-form", "inspection.md Figure inspection; x184+416*i/39,y424-300*lr/.001", "明確定義更新/率的座標映射、peak/floor/mid tick位置。")],
     "artifact_ids": ["figure-render", "figure-snapshot", "probe-execution", "probe-code", "run-receipts", "render-stderr", "inspection"],
     "verification": {"method": "executed", "expected": "40點映射原40率；tick.001/.0005/.0001在y124/274/394；最後點略高floor；圖中文字可見且吻合。", "observed": "40點；最大x誤差.003333333333387145px、y誤差.004986193940226258px；Inkscape1.4生成PNG後實際view，所有主要文字和曲線可見。", "tolerance": "SVG座標四捨五入兩位小數容許absolute0.0051px；tick精確坐標。", "details": "static SVG PNG實際檢視，保存renderer的Pango/Gtk warnings；沒有用字串搜尋代替圖視覺判讀。"}},
]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "5.6", "source": "course/chapters/05.md#5.6",
    "source_sha256": extract["source_sha256"], "figure_sha256": extract["figure_sha256"],
    "reviewer_task": "/root/phase4_factual_coordinator/factual_5_6", "reviewer_context": "fresh", "reviewed_on": "2026-10-05",
    "verdict": "pass", "review_scope": provenance["read_scope"], "prior_review_content_read": False,
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "逐項區分排程方法、原helper邊界、純時間表scope、總計畫/resume及图；原始權威來源與原碼親讀後未發現實質錯誤。"},
        "numeric_verification": {"status": "pass", "claim_ids": ["base-rates", "warmup-variation-and-cap", "total-plan-changes-rate", "figure-rates-and-labels"], "details": "原40率與全部35下降closed-form，暖身10/30共80率，7cap邊界、i19總步40/100及40SVG點親跑核對；原JSON中fence/count/hash/CPU/exit等已核對。"},
        "figure_consistency": {"status": "pass", "claim_ids": ["base-rates", "figure-rates-and-labels"], "details": "原SVG已Inkscape1.4 render並親用view_image查看PNG；可見標籤及曲線一致，40點誤差符合兩位小數座標容差；renderer警告留存，未宣稱browser/mobile頁檢查。"},
        "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "直接HTTPS取得SGDR原v1 PDF並核對first-page identity/§3 Eq5；HF官方v4.57.1 source親讀具體API/active definitions；本課helper/CLI契約及實際CPU證據各有hash、定位、支持范围。未使用來源庫摘要或舊報告結論。"},
        "limitations": {"status": "pass", "claim_ids": ["schedule-method", "fence-software-scope", "resume-keeps-plan", "figure-rates-and-labels"], "details": "本節無模型實測成績；玩具檢查只支持排程與合約，未跑長訓練/資料模型下載/完整resume。外部HF source版本為4.57.1、未安裝；當前CPU版本是torch2.14.1+cpu。SVG靜態可見性已核，browser/mobile未驗。"},
    },
}
(ROOT / "docs/technical-reviews/5.6.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
manifest = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"]
(OUT / "manifest.json").write_text(json.dumps({"reviewer_task": report["reviewer_task"], "files": manifest}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": "docs/technical-reviews/5.6.json", "verdict": report["verdict"], "claims": len(claims),
                  "artifacts": len(artifacts), "source_sha256": extract["source_sha256"]}, ensure_ascii=False))
