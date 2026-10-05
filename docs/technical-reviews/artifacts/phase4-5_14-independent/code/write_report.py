"""Assemble this reviewer's inspected claims and hash actual permanent evidence."""
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from importlib.util import module_from_spec, spec_from_file_location

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def relative(path):
    return path.relative_to(ROOT).as_posix()

meta = json.loads((BASE / "inputs/extraction.json").read_text())
runtime = json.loads((BASE / "environment.json").read_text())
results = json.loads((BASE / "bounded-results.json").read_text())
assert results["assertions"] == "all passed"
assert (BASE / "bounded-stderr.txt").read_bytes() == b""
assert json.loads((BASE / "inputs/execution.json").read_text())["exit_code"] == 0
environment = {key: runtime[key] for key in ["python", "torch", "device", "threads"]}
original_command = ".venv/bin/python docs/review-tools/section_facts.py 'course/chapters/05.md#5.14' --output /tmp/phase4-5_14-fresh-original --execute --timeout 30"
bounded_command = "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 30 .venv/bin/python docs/technical-reviews/artifacts/phase4-5_14-independent/code/bounded_checks.py > docs/technical-reviews/artifacts/phase4-5_14-independent/bounded-stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_14-independent/bounded-stderr.txt"
write(BASE / "execution-receipt.json", {
    "shell": "bash; login=false", "cwd": str(ROOT), "environment": environment,
    "original": {"command": original_command, "exit_code": 0, "fences_executed": 1, "timeout_seconds": 30},
    "bounded": {"command": bounded_command, "exit_code": 0, "timeout_seconds": 30, "result": "all assertions passed; no stderr"},
    "source_fetch": {"command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-5_14-independent/code/fetch_sources.py > docs/technical-reviews/artifacts/phase4-5_14-independent/fetch-stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_14-independent/fetch-stderr.txt", "exit_code": 0},
    "additional_paper": {"url": "https://arxiv.org/pdf/1206.5533v2", "accessed_on": "2026-10-05", "path": relative(BASE / "sources/bengio-recommendations-v2.pdf"), "sha256": digest(BASE / "sources/bengio-recommendations-v2.pdf"), "command": "curl --fail --location --max-time 30 https://arxiv.org/pdf/1206.5533v2 --output docs/technical-reviews/artifacts/phase4-5_14-independent/sources/bengio-recommendations-v2.pdf; then pdftotext -layout", "exit_code": 0},
    "large_inputs": "none; no weights/models/data read or downloaded",
})

spec = spec_from_file_location("section_facts_for_provenance", ROOT / "docs/review-tools/section_facts.py")
helper = module_from_spec(spec)
spec.loader.exec_module(helper)
context = {}
for lesson in ["5.1", "5.2", "5.15"]:
    raw, whole, first = helper.original_section(ROOT / "course/chapters/05.md", lesson)
    path = BASE / "inputs" / f"context-{lesson}.md"
    path.write_bytes(raw)
    context[lesson] = {"snapshot": relative(path), "first_line": first, "sha256": digest(path)}

provenance = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_5_14", "accessed_on": "2026-10-05",
    "source": meta["source"], "section_original_utf8_sha256": meta["source_sha256"],
    "whole_chapter_at_original_extraction_sha256": meta["source_file_sha256"],
    "first_line": meta["section_first_line"], "newline_policy": meta["newline_policy"],
    "context": context, "runtime": runtime, "read_old_reports": False,
    "official_sources": json.loads((BASE / "source-receipt.json").read_text())["sources"],
    "additional_paper_receipt": json.loads((BASE / "execution-receipt.json").read_text())["additional_paper"],
    "files": [{"path": relative(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(BASE.rglob("*")) if path.is_file() and path.name not in {"input-provenance.json", "checker-stdout.txt", "checker-stderr.txt", "checker-receipt.json"}],
}
write(BASE / "input-provenance.json", provenance)

artifacts = []
artifact_by_path = {}
for number, path in enumerate(sorted(BASE.rglob("*")), 1):
    if not path.is_file() or path.name.startswith("checker-"):
        continue
    key = "a-" + str(number)
    artifact_by_path[path.relative_to(BASE).as_posix()] = key
    item = {"id": key, "path": relative(path), "sha256": digest(path), "kind": "code" if path.suffix == ".py" else "source_snapshot", "description": "本輪永久證據：" + path.relative_to(BASE).as_posix()}
    if path.relative_to(BASE).as_posix() in {"inputs/execution.json", "inputs/stdout.txt"}:
        item.update(kind="execution", command=original_command, result="exit 0；原始 fence 印出起點4，新位置1.0399999618530273/1.399999976158142/9與代價3.841600179672241/2.56000018119812/36。", environment=environment)
    if path.relative_to(BASE).as_posix() in {"bounded-results.json", "bounded-stdout.txt", "execution-receipt.json", "exercise-stdout.txt"}:
        item.update(kind="execution", command=bounded_command, result="exit 0；所有純量/tensor assertions通過；只執行短CPU機制查核，未訓練LM。", environment=environment)
    if path.name == "inspection.md":
        item.update(kind="derivation", description="親讀原始來源定位、逐項支持限制與平方函數獨立推導")
    artifacts.append(item)

def aid(name):
    return artifact_by_path[name]

def official(identifier, title, filename, url, version, kind, inspection):
    return {"id": identifier, "title": title, "kind": kind, "url": url, "version": version, "accessed_on": "2026-10-05", "authority_reason": "作者原始arXiv論文" if kind == "paper" else "PyTorch官方pytorch/pytorch儲存庫固定tag的原始程式/文件", "verified": True, "checked_original": True, "inspection_note": inspection, "snapshot_path": relative(BASE / "sources" / filename), "snapshot_sha256": digest(BASE / "sources" / filename)}

sources = [
    official("bengio", "Bengio: Practical Recommendations for Gradient-Based Training of Deep Architectures", "bengio-recommendations-v2.pdf", "https://arxiv.org/pdf/1206.5533v2", "arXiv:1206.5533v2; 2012-09-16", "paper", "親讀原PDF首頁版本；§2.1 p.5、§3.3.1/3.3.2 p.17、§3.3.3 p.18；自行pdftotext定位。支持局部下降方向、log尺度探索與粗到細搜尋，不保證幾步找到最終最佳LR。"),
    official("smith", "Smith: Cyclical Learning Rates for Training Neural Networks", "smith-clr-v6.pdf", "https://arxiv.org/pdf/1506.01186v6", "arXiv:1506.01186v6; 2017-04-04", "paper", "親讀首頁與§1 p.1、§3.3 pp.3–4。原named LR range test單次run線性增LR數epochs且畫accuracy；本文是候選重置pilot，不能稱Smith方法重現；只引用探索有效邊界的機制與限制。"),
    official("sgd", "PyTorch official SGD algorithm and step implementation", "pytorch-v2.8.0-sgd.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/sgd.py", "PyTorch tag v2.8.0; tested installed 2.14.1+cpu separately inspected", "official_source", "親讀step105–150、公式153–184、single-tensor320–379；另親讀installed-sgd.py347–379；負梯度更新與momentum歷史語義一致。"),
    official("adamw", "PyTorch official AdamW algorithm", "pytorch-v2.8.0-adamw.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/adamw.py", "PyTorch tag v2.8.0; runtime 2.14.1+cpu", "official_source", "親讀21–58委派Adam與61–108矩公式、歷史m/v；不是只重設參數就重設優化器歷史。"),
    official("adam", "PyTorch official Adam per-parameter optimizer state", "pytorch-v2.8.0-adam.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/adam.py", "PyTorch tag v2.8.0; tested installed 2.14.1+cpu separately inspected", "official_source", "親讀139–190 state[step,exp_avg,exp_avg_sq]與405–447更新；installed150–196亦保存同樣歷史。"),
    official("optimizer", "PyTorch official Optimizer.state_dict contract", "pytorch-v2.8.0-optimizer.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/optimizer.py", "PyTorch tag v2.8.0; installed2.14.1+cpu state_dict700–721", "official_source", "親讀666–688：state不含參數值，param_groups保存LR等設定；狀態復原後須另設試驗LR。"),
    official("randomness", "PyTorch official reproducibility notes", "pytorch-v2.8.0-randomness.rst", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/randomness.rst", "PyTorch tag v2.8.0; CPU runtime2.14.1+cpu", "official_docs", "親讀6–15與27–69：固定seed在同環境控制亂數序列；跨平台/版本不保證，其他RNG亦需控制；沒有把同seed當成同資料順序的充分條件。"),
    official("crossentropy", "PyTorch official CrossEntropyLoss ignore/reduction contract", "pytorch-v2.8.0-loss.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/loss.py", "PyTorch tag v2.8.0; small CPU runtime2.14.1+cpu", "official_source", "親讀CrossEntropyLoss1158–1197公式及1236–1239 ignore_index對梯度/平均的契約。repository helper先拒絕zero-count另親讀與實跑。"),
    {"id": "section-code", "kind": "repository_code", "title": "5.14 exact original Python fence", "path": relative(BASE / "inputs/fence-1.py"), "sha256": digest(BASE / "inputs/fence-1.py"), "version": "section SHA256 " + meta["source_sha256"], "verified": True, "inspection_note": "親讀全部251原始bytes；每輪new=w-lr*gradient，w無覆蓋；gradient手算、requires_grad=False、無optimizer/訓練。"},
    {"id": "model-code", "kind": "repository_code", "title": "Repository loss_sum/masked_loss actual contract", "path": relative(BASE / "inputs/model.py"), "sha256": digest(BASE / "inputs/model.py"), "version": "current snapshot SHA256 " + digest(BASE / "inputs/model.py"), "verified": True, "inspection_note": "親讀92–105：有效label計數，count0直接ValueError，cross_entropy ignore_index=-100 sum，再除count；短tensor實跑。"},
    {"id": "train-code", "kind": "repository_code", "title": "Repository original training loop contract", "path": relative(BASE / "inputs/train.py"), "sha256": digest(BASE / "inputs/train.py"), "version": "current snapshot SHA256 " + digest(BASE / "inputs/train.py"), "verified": True, "inspection_note": "親讀297–316、346–363：復原optimizer/RNG、設LR排程、backward與有train才step；loss記錄是本batch更新前，無把它轉作本節曲線；未跑長recipe。"},
    {"id": "quadratic", "kind": "derivation", "title": "Independent quadratic error/loss-ratio derivation", "verified": True, "details": "L=(w-3)^2，g=2(w-3)，w_new-3=(1-2η)(w-3)，非零初值L_new/L_old=(1-2η)^2；0<η<1下降，η=.5一步到3；與bounded-results.json兩起點及8步算例核對。"},
    {"id": "original-run", "kind": "execution", "title": "Unchanged original fence CPU run", "verified": True, "artifact_id": aid("inputs/execution.json")},
    {"id": "bounded-run", "kind": "execution", "title": "Independent bounded CPU scalar/tensor probes", "verified": True, "artifact_id": aid("bounded-results.json")},
]

def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}

def claim(identifier, kind, statement, location, scope, references, names, verification=None):
    value = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope, "status": "verified", "evidence": references, "artifact_ids": [aid(name) for name in names]}
    if verification is not None:
        value["verification"] = verification
    return value

def verification(expected, observed, details, tolerance=None):
    value = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance:
        value["tolerance"] = tolerance
    return value

claims = [
    claim("local-direction", "concept", "w減學習率乘梯度是梯度下降；沿初始下降方向仍不保證任意大步能降代價。", "course/chapters/05.md:522–537", "平方函數手工梯度下降機制；局部下降方向不承諾大步效果，更不等於所有AdamW步都沿當前負梯度。", [evidence("bengio", "§2.1 printed p.5, extracted lines236–262", "參數減LR乘梯度；梯度只給局部最陡下降方向，大步方向可能不合適。"), evidence("sgd", "algorithm lines153–184; single-tensor320–379", "zero momentum/decay情況的負梯度更新；不是本文呼叫optimizer。"), evidence("quadratic", "inspection.md independent calculation", "η=2從1到9、代價4到36具體反例。")], ["inspection.md", "bounded-results.json"]),
    claim("start1-numbers", "numeric", "起點1、g=-4、L=4；LR .01/.1/2得新位置1.04/1.4/9、新代價3.8416/2.56/36。", "course/chapters/05.md:522–537", "一個float32 scalar的一步候選數字，未訓練模型。", [evidence("quadratic", "g=2(w-3), new=w-ηg", "自己代入得原文小數。"), evidence("original-run", "inputs/stdout.txt four printed lines", "原碼實際float32輸出，起點不覆蓋。")], ["inputs/execution.json", "inputs/stdout.txt", "bounded-results.json"], verification("g=-4、new1.04/1.4/9、loss3.8416/2.56/36", "float32 new1.0399999618530273/1.399999976158142/9；loss3.841600179672241/2.56000018119812/36", "原fence不修改地跑完；獨立CPU代入再核對。", "new絕對容差1e-6、loss絕對容差2e-6；整數4/9/36精確相等。")),
    claim("fence-contract", "software", "代碼每輪只計算new而不覆蓋w，三候選都從相同起點比較。", "course/chapters/05.md:527–537", "手工計算，不是autograd backward或真正optimizer parameter update。tensor/square/item的實際使用逐项由原fence完整執行cover。", [evidence("section-code", "entire fence, lines1–9", "w只賦值一次，gradient手工2*(w-3)，循環只賦new與print。"), evidence("original-run", "inputs/execution.json exit_code=0; stdout.txt", "全部候選成功執行，數字符合相同w。")], ["inputs/fence-1.py", "inputs/execution.json", "inputs/stdout.txt"], verification("原fence成功，三個候選不累積更新。", "exit0；new1.04/1.4/9對應同w1。", "完整原始251bytes由helper保存SHA後執行；未引入optimizer。")),
    claim("exercise-ratio", "numeric", "改w為4後g=2，三新位置3.98/3.8/0，L .9604/.64/9；兩起點新舊L比都 .9604/.64/9，η=.5都一步到3。", "course/chapters/05.md:541–543", "只限L=(w-3)^2且初始代價非零；練習不支持最佳LR隨起點改變或LM最佳LR。", [evidence("quadratic", "w_new-3=(1-2η)(w-3), ratio=(1-2η)^2", "新舊代價比不含起點；η=.5將誤差置零。"), evidence("bounded-run", "numeric rows start4, exercise_stdout, eight_step_scalar_trajectories", "原fence僅將1.0換4.0實跑，另獨立兩起點及η=.5核對。")], ["code/exercise-original-single-change.py", "exercise-stdout.txt", "bounded-results.json", "inspection.md"], verification("start4 new3.98/3.8/0，L .9604/.64/9；兩起點η=.5 new3,L0", "new3.9800000190734863/3.799999952316284/0，L .9604000449180603/.6399999260902405/9；两起點η=.5 new3、L0；比例吻合。", "原fence单值替換；獨立float32檢查與8步純数学模型防止誤把crossing/oscillation本身當loss必增。", "new絕對1e-6、loss與ratio絕對2e-6；η=.5 endpoint3/L0精確。")),
    claim("pilot-search", "concept", "真模型可先用小份固定資料探索LR量級，再在有效範圍細分；看多步loss/異常而非一次幸運下降定案。", "course/chapters/05.md:539", "可行的受控pilot/coarse-to-fine搜尋建議，不是已實跑模型結果、不保證幾步找到最佳，也不聲稱復現Smith單run線性LR range test。1e-4/1e-3/1e-2是候選例子而非通用有效區間。", [evidence("bengio", "§3.3.1 p.17 lines926–952; §3.3.2 p.17 lines947–968; §3.3.3 p.18 lines971–1025", "log尺度、先少量廣範圍候選再細分、分開超參數實驗。"), evidence("smith", "§3.3 pp.3–4 lines129–170,191–205", "探索合理邊界、觀察曲線粗糙/變差的診斷動機；其原算法不同，不能支持本文等同named LR range test。"), evidence("bengio", "§1.3 p.4 lines205–219; §2.1 p.5", "訓練objective改善沒有泛化保證，heuristic建議要限定而非保證最佳。")], ["inspection.md"]),
    claim("comparison-controls", "concept", "每候選控制初參數、optimizer歷史、資料順序、更新步數及亂數設定；最終參數/loss可以不同。", "course/chapters/05.md:539", "受控短程比較原則；同seed不自動保證跨架構同batch或跨裝置確定性，本文另獨立要求同資料順序並連5.15。", [evidence("optimizer", "state_dict lines666–688; installed700–721", "optimizer歷史和參數不同；LR亦在param_groups，復原後要另設候選。"), evidence("adam", "lines139–190,405–447", "state step/m/v會影響更新，需控制相同歷史。"), evidence("adamw", "lines21–58,72–91", "AdamW保存moment歷史。"), evidence("randomness", "lines6–15,27–69", "seed只控制對應RNG序列且需同環境/其他亂數控制。"), evidence("bounded-run", "history_control; four_update_fixed_scalar_pilots; rng_reset", "相同初參數不同momentum給1.4對1.31；同初值/AdamW歷史/四同序標量的候選有不同final，重跑相同候選逐值一致。")], ["bounded-results.json", "inspection.md", "environment.json"]),
    claim("update-diagnostic", "software", "看似不動時直接比同格前後並保留小數位；確認optimizer.step真的執行。", "course/chapters/05.md:541", "合理診斷步驟，不宣稱step只要呼叫就一定移動或所有不動都因未step；未執行大型模型診斷。", [evidence("sgd", "step105–150; parameter add320–379", "step執行parameter update。"), evidence("train-code", "lines346–363", "repository backward後只有train模式才step；本batchloss是preupdate。"), evidence("bounded-run", "backward_vs_step; display_precision", "backward留w1但梯度-4；step到1.4；1→1.0000004四位顯示仍1.0000。")], ["bounded-results.json", "bounded-stdout.txt"], verification("無step參數不動，step按梯度移動；小變化可能顯示不出。", "before_step1、grad-4，after_step1.4；完整值1.0000004、same_four_decimal_display=true。", "float64一個參數與普通格式化，未以打印外觀判斷數值相等。")),
    claim("answer-diagnostic", "software", "確認本批至少一答案位置參與代價；有效答案計數與平均依5.2規則。", "course/chapters/05.md:541", "本repo -100忽略契約；空答案會直接拒絕。不是所有框架空target都自動更新，也不稱label存在就保證學習。", [evidence("model-code", "loss_sum/masked_loss lines92–105", "count非ignoredlabel；0報錯；sum/count。"), evidence("crossentropy", "1158–1197,1236–1239", "ignore_index不貢獻input gradient與平均分母。"), evidence("bounded-run", "target_denominator", "一有效一忽略count1，loss log3；ignored row梯度零；全ignored ValueError。")], ["inputs/model.py", "bounded-results.json"], verification("有效count1、loss log3、ignored row grad0；全ignored直接error。", "count1、loss1.0986122886681098、ignored gradient[0,0,0]；ValueError『所有 labels 都被忽略：没有可學習的答案』。", "只用形狀[1,2,3]分數与两個label實際呼叫repo helper。")),
    claim("nonfinite-diagnostic", "software", "無窮/NaN是非有效有限值，應追第一次異常輸入/運算並檢查分母是否零。", "course/chapters/05.md:541", "除零為應查原因之一，沒有說所有NaN由除零或LR造成；小tensor檢查不能查證未實跑模型的異常源。", [evidence("bounded-run", "nonfinite_denominator_probe", "finite1,0在第一次division變成inf/nan；用isfinite/isinf/isnan實驗核對。"), evidence("model-code", "92–105", "本課loss平均分母count0已在除法前拒絕；應先讀helper契約。")], ["bounded-results.json", "bounded-stdout.txt"], verification("finite inputs1,0；1/0 inf、0/0 nan；找得到第一異常op。", "inputs_finite=true，first_nonfinite_operation='division by zero'；isinf/isnan斷言通過。", "pure scalar tensor arithmetic，結果以字串記入JSON以免把nonfinite值寫成非標準JSON。")),
]

report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "5.14", "source": meta["source"],
    "source_sha256": meta["source_sha256"], "figure_sha256": {}, "reviewer_task": "/root/phase4_factual_coordinator/factual_5_14",
    "reviewer_context": "fresh", "author_tasks": [], "verdict": "pass", "reviewed_on": "2026-10-05",
    "actual_read_scope": "Full5.14；完整本輪方法/schema/helper/review protocol；前置5.1/5.2與5.15、相關training/model/train實作與bootstrap；Smith v6/Bengio v2原PDF指定段落；固定tag官方PyTorch源码及installed relevant implementations。無舊報告讀取。",
    "applicability": {"substantive_claims": True, "empirical_model_results": False, "figures": False},
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "負梯度、局部下降/大步反例、粗到細pilot与控制變項由親讀原始來源及small CPU機制分別核對。本文未把候選重置pilot稱成Smith原算法重現。", "claim_ids": [x["id"] for x in claims]},
        "numeric_verification": {"status": "pass", "details": "原fence与w4單一替換短CPU真跑；两起點三候選、.5一步到3、loss ratio由獨立數值核對；absolute2e-6可涵蓋float32；8步math示例只驗證本函數判準。", "claim_ids": ["start1-numbers", "exercise-ratio"]},
        "figure_consistency": {"status": "not_applicable", "details": "原始section extraction與親讀全文皆確認0圖/curve/SVG引用；沒有需render/view的圖，沒有用其他節圖補充本節。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "Bengio v2/Smith v6原arXiv PDF直接取得、首頁版本與具體段落親讀；官方PyTorch pinnedv2.8.0源码親讀，runtime2.14.1+cpu差異與實際installed源码亦保存核對；原repository函式契約實讀。", "claim_ids": ["local-direction", "pilot-search", "comparison-controls", "update-diagnostic", "answer-diagnostic"]},
        "limitations": {"status": "pass", "details": "純量手工new無模型訓練/autograd/optimizer；短tensor只支持機制；.1不是LM最佳、起點練習不證最佳LR改變、pilot幾步無最佳/泛化保證；沒有引用原實驗JSON/score/checkpoint可作分母核對，NA而非改寫模型成績。No GPU/data/model download/full recipe。", "claim_ids": ["fence-contract", "exercise-ratio", "pilot-search", "comparison-controls", "nonfinite-diagnostic"]},
    },
    "summary": "本輪9項實質主張已核對，無未解決實質矛盾；原始CPU fence與全部有界變化通過；無圖/模型实测JSON；支援範圍有明確限定。",
}
write(ROOT / "docs/technical-reviews/5.14.json", report)
print(json.dumps({"report": "docs/technical-reviews/5.14.json", "verdict": report["verdict"], "source_sha256": report["source_sha256"], "claims": len(claims), "artifacts": len(artifacts)}, ensure_ascii=False))
