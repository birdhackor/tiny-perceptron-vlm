"""Persist this reviewer's authored findings, evidence links, and exact file hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
PREFIX = BASE.relative_to(ROOT).as_posix()
REV = "52964f650787cef393d18647a350471637208f67"
TORCHREV = "5c4886908584029761b579af026dcfb627c84070"


def hash_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


env = json.loads((BASE / "results/environment.json").read_bytes())
execution_environment = {k: str(v) for k, v in env.items()}
fence_env = json.loads((BASE / "results/original-fence/environment.json").read_bytes())
extract = json.loads((BASE / "inputs/current/extraction.json").read_bytes())
verification_command = "CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 45 .venv/bin/python " + PREFIX + "/code/verify.py"
commands = {
    "working_directory": str(ROOT),
    "shell": "bash, login:false",
    "commands": [
        {"command": ".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.14 --output /tmp/phase4-7_14-extract", "exit_code": 0, "purpose": "Extract raw UTF-8 section and unaltered Python fence"},
        {"command": ".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.14 --output /tmp/phase4-7_14-fence-run --execute --timeout 45", "exit_code": 0, "purpose": "Execute complete original fence in CPU helper; permanent inputs/results under results/original-fence"},
        {"command": "git show " + REV + ":<original path>", "exit_code": 0, "purpose": "Immutable source snapshots; all recorded code hashes equal the original result code_sha256; prepare_data.py is pinned by git tree because run hashing did not include scripts/prepare_data.py"},
        {"command": "curl --fail --location --silent --show-error --max-time 25 https://raw.githubusercontent.com/pytorch/pytorch/" + TORCHREV + "/torch/nn/modules/loss.py --output " + PREFIX + "/sources/pytorch-loss.py", "exit_code": 0, "purpose": "Original authoritative PyTorch document/source; fetched bytes exactly equal installed wheel loss.py"},
        {"command": "curl --fail --location --silent --show-error --max-time 25 https://raw.githubusercontent.com/pytorch/pytorch/" + TORCHREV + "/aten/src/ATen/native/LossNLL.cpp --output " + PREFIX + "/sources/pytorch-LossNLL.cpp", "exit_code": 0, "purpose": "Original installed-revision native implementation; independently read branch and axis"},
        {"command": verification_command, "exit_code": 0, "purpose": "Float64 analytic loss/gradient, score8 perturbation, bounds, immutable data hash/sampler/ID/NLL audit; no training or weights"},
        {"command": ".venv/bin/python -m scripts.course_experiments.run --help", "exit_code": 0, "purpose": "Linked long shell entrypoint CLI contract only; training recipes deliberately not executed"},
    ],
    "limits": "Only read-only external official sources were fetched. No new training data, model downloads, GPU, paid runs, uploads, full training, or neural weights. Reconstructed original JSONL bytes were admitted only after matching original SHA-256."
}
dump(BASE / "results/commands.json", commands)

inspection = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_7_14",
    "accessed_on": "2026-10-05",
    "section_read": {"source": "course/chapters/07.md#7.14", "first_line": 493, "last_content_line": 533, "entire_section_read": True, "source_sha256": extract["source_sha256"], "raw_policy": "original UTF-8 bytes, no newline normalization", "first_section": False},
    "other_current_text_read": ["course/chapters/07.md#7.12 whole section, lines 419-456", "course/training.md#T.4 whole section, lines 123-205, to inspect linked recipe and baseline distinction"],
    "independence": "Fresh single-section technical-review identity. Did not read previous technical/reader report text or conclusions. Locator index was used only to locate immutable original PyTorch bytes; original commit URLs were independently fetched and read. Checker necessarily scans owner IDs but no other report conclusions were used.",
    "official_source_inspection": [
        {"url": "https://raw.githubusercontent.com/pytorch/pytorch/" + TORCHREV + "/torch/nn/modules/loss.py", "file": "sources/pytorch-loss.py", "sha256": hash_file(BASE / "sources/pytorch-loss.py"), "version_check": "Installed torch.__version__=2.14.1+cpu and torch.version.git_version=" + TORCHREV + "; fetched complete loss.py SHA is exactly equal to installed torch.nn.modules.loss source", "actually_read": "CrossEntropyLoss lines 1199-1244, 1270-1330, 1383-1407; class-index -log(exp target/sum exp) formula, ignore_index mean denominator, input/target shape and long dtype, default label smoothing, forward to F.cross_entropy"},
        {"url": "https://raw.githubusercontent.com/pytorch/pytorch/" + TORCHREV + "/aten/src/ATen/native/LossNLL.cpp", "file": "sources/pytorch-LossNLL.cpp", "sha256": hash_file(BASE / "sources/pytorch-LossNLL.cpp"), "version_check": "Fetched immutable URL at the installed runtime git revision; independently read native implementation (index only supplied locator)", "actually_read": "cross_entropy_loss_symint lines 632-666: same-shape soft-target path, separate label-smoothing path, ordinary class-index path selects class_dim=1 for batched logits and applies log_softmax then NLL"},
    ],
    "original_run_inspection": "Personally read original source tree at 52964f650787cef393d18647a350471637208f67: text.py _steps, _save_splits, _evaluations, run_sft and entire run_sft_ablation; common.py Context.dependency, split_records, records_sha256, text_examples, _nll, fit_lm, evaluate_lm and load_lm; data.py ByteTokenizer/render_chat/pad_batch; model.py loss_sum/masked_loss/generate; prepare_data.py conversation/generate_records; run.py complete CLI and step-scale contract. Relevant recorded code hashes exactly match git-show bytes. Personally inspected clean/noisy results, all 30 original raw-ID samples, data manifest and corruptions; recomputed all corresponding counts/hashes/averages. Original CUDA logits and model weights were not rerun or opened.",
    "figure_inspection": {"references": [], "svg_references": 0, "figure_sha256": {}, "render_and_view": "not_applicable: section has no referenced raster/SVG figure; no visual rendering claimed", "spatial_need": "The four score candidates and before/after score changes are explicit text and executable tensors. No spatial mapping, image asset, or figure is necessary to understand this section."},
    "scope": "Tests demonstrate supplied-label objective, gradients with respect to logits, and accounting of stored experiment. They do not re-estimate model behavior, seed variance, general label-noise tolerance, or a universal causal dose-response."
}
dump(BASE / "results/inspection.json", inspection)

artifacts = []
ids = {}
for p in sorted(BASE.rglob("*")):
    if not p.is_file() or p.suffix == ".pt":
        continue
    rel = p.relative_to(BASE).as_posix()
    if rel.startswith("results/checker.") or rel == "results/receipt.json":
        continue  # Checker output and receipt are post-report observations.
    identifier = "a_" + rel.replace("/", "_").replace(".", "_").replace("-", "_")
    ids[rel] = identifier
    kind = "code" if p.suffix == ".py" else "source_snapshot"
    description = "Permanent review input/output: " + rel
    entry = {"id": identifier, "kind": kind, "path": p.relative_to(ROOT).as_posix(), "sha256": hash_file(p), "description": description}
    if rel in ["results/verify.stdout.txt", "results/numeric.json", "results/original-result-audit.json"]:
        entry.update(kind="execution", command=verification_command, result="Exit 0; every assertion passed. Original result accounting and independent scalar/gradient checks completed; no retraining.", environment=execution_environment)
    elif rel == "results/original-fence/stdout.txt":
        entry.update(kind="execution", command=commands["commands"][1]["command"], result="Exit 0; original fence printed label2=0.21099762618541718, label3=2.2109975814819336, score5 label3=0.13872769474983215; no guard events.", environment={k: str(fence_env[k]) for k in ["python", "torch", "torch_git_version", "device_requested", "cuda_build", "cuda_available"]})
    elif rel == "results/recipe-help.stdout.txt":
        entry.update(kind="execution", command=".venv/bin/python -m scripts.course_experiments.run --help", result="Exit 0; parser documents experiment/device/dependencies/step-scale; no experiment executed.", environment={"python": env["python"], "device": "CPU CLI only; no torch model loaded"})
    artifacts.append(entry)

def aid(rel):
    return ids[rel]

sources = [
    {"id": "torch_ce_docs", "kind": "official_source", "title": "PyTorch CrossEntropyLoss original documented source at installed commit", "url": "https://raw.githubusercontent.com/pytorch/pytorch/" + TORCHREV + "/torch/nn/modules/loss.py", "version": "torch 2.14.1+cpu, git " + TORCHREV, "authority_reason": "PyTorch upstream defines its own cross-entropy contract and formula; original bytes match installed wheel source exactly", "verified": True, "checked_original": True, "accessed_on": "2026-10-05", "inspection_note": inspection["official_source_inspection"][0]["actually_read"] + "; independently fetched immutable HTTPS URL, SHA matched installed module", "artifact_ids": [aid("sources/pytorch-loss.py")]},
    {"id": "torch_ce_native", "kind": "official_source", "title": "PyTorch ordinary class-index cross entropy native implementation", "url": "https://raw.githubusercontent.com/pytorch/pytorch/" + TORCHREV + "/aten/src/ATen/native/LossNLL.cpp", "version": TORCHREV, "authority_reason": "Original upstream implementation of the installed runtime's loss operation", "verified": True, "checked_original": True, "accessed_on": "2026-10-05", "inspection_note": inspection["official_source_inspection"][1]["actually_read"] + "; independently fetched immutable URL", "artifact_ids": [aid("sources/pytorch-LossNLL.cpp")]},
    {"id": "fence_execution", "kind": "execution", "title": "Complete unmodified section Python fence, CPU/offline helper", "verified": True, "artifact_id": aid("results/original-fence/stdout.txt")},
    {"id": "bounded_audit", "kind": "execution", "title": "Independent original-result audit and bounded loss/gradient/ID checks", "verified": True, "artifact_id": aid("results/verify.stdout.txt")},
    {"id": "numeric_derivation", "kind": "derivation", "title": "Loss and gradient derivation", "verified": True, "details": "For one unweighted class-index target with no smoothing, L=log(sum_c exp z_c)-z_y in nats; dL/dz_c=softmax(z)_c-1[c=y]. B=1,C=4, class axis1 and denominator1. Increasing false target z3 increases p3 and lowers label3 loss while lowering p2 and increasing label2 loss. Supervised-target sums count assistant bytes+EOS, so repetition/length can raise counted target share without determining final shared-model behavior."},
]
for key, path, note in [
    ("run_text", "scripts/course_experiments/text.py", "Read _steps lines 35-37, _save_splits 48-64, _evaluations 67-73, run_sft 550-574, complete run_sft_ablation 614-662: separate deepcopies from same sft model.pt, deterministic first 4 eligible circle/square labels, same seed, clean/noisy 300 updates, clean held-out evaluations."),
    ("run_common", "scripts/course_experiments/common.py", "Read Context.dependency 25-29, split_records 51-70, records_sha256 73-74, text_examples 77-102, _nll 106-119, fit_lm 122-204, evaluate_lm 243-287, load_lm 307-309: train sampling Random(seed), batch16, 300 steps supplied by caller, token sum normalization and EOS-stripped raw token equality; greedy generation uses original own outputs."),
    ("run_data", "tiny_perceptron/data.py", "Read ByteTokenizer 14-28, render_chat 54-68, pad_batch 71-87: specials offset8, assistant bytes+EOS only, ignored user/role targets, shift once, right padding."),
    ("run_model", "tiny_perceptron/model.py", "Read loss_sum/masked_loss 92-105 and generate 109-136: summed CE/number of nonignored targets; generation argmax default, appended prior outputs, EOS/context stopping."),
    ("run_prepare", "scripts/prepare_data.py", "Read conversation 11-19 and generate_records attributes branch 22-45: 3 colors x2 shapes x2 pitches=12 families, 5 queries each; explicit external ground truth rules. This script is pinned by original git tree; original runner did not hash this path."),
    ("run_cli", "scripts/course_experiments/run.py", "Read entire original CLI, including execute 40-55 and main 135-162: seed42, step_scale default1.0, GPU fail rather than fallback, original dependency root and model.pt requirement. Current --help was run but no experiment was launched."),
]:
    full = BASE / "inputs/original" / path
    sources.append({"id": key, "kind": "repository_code", "title": "Original run code: " + path, "verified": True, "path": full.relative_to(ROOT).as_posix(), "sha256": hash_file(full), "version": REV, "inspection_note": note + " Extracted from git show immutable revision; relevant code_sha256 equals original JSON where recorded."})
for key,path,title in [("raw_ablation", "sft_ablation.json", "Original full clean/noisy result JSON"), ("raw_sft", "sft.json", "Direct SFT baseline result JSON")]:
    full = BASE / "inputs/current/docs/course-experiments/results" / path
    sources.append({"id": key, "kind": "repository_code", "title": title, "verified": True, "path": full.relative_to(ROOT).as_posix(), "sha256": hash_file(full), "version": "Tracked result blob at repo HEAD 607dc4287d12a6043de59457785aa91bf13af331; ablation code-run revision " + REV, "inspection_note": "Personally parsed raw JSON, not a review conclusion. Inspected relevant data/clean/noisy/corruptions and all 30 clean/noisy validation/test raw generated-ID samples. Auditor proves ablation before A_attributes exactly equals direct SFT after; NLL sums retained as original CUDA measurements."})

def ev(source,locator,supports):
    return {"source_id":source,"locator":locator,"supports":supports}

def verify(expected,observed,details,denominators=None,tolerance=None):
    out={"method":"executed","expected":expected,"observed":observed,"details":details}
    if denominators is not None: out["denominators"]=denominators
    if tolerance is not None: out["tolerance"]=tolerance
    return out

claims = [
    {"id":"C1","kind":"concept","status":"verified","statement":"交叉熵按提供的標籤求 -log p(label)，不檢查 1+1 的外部正解；錯標同樣產生指向該目標的 logit 梯度。","location":"7.14 lines495,512-516","scope":"未加權、單一類別、無label smoothing的交叉熵；梯度方向指logits的直接局部方向，不聲稱所有共享参数模型必然記住每個錯例或固定程度。","evidence":[ev("torch_ce_docs","CrossEntropyLoss lines1199-1244 and shape lines1302-1320","提供的class index決定 -log softmax 目標，公式沒有外部算術驗證"),ev("torch_ce_native","cross_entropy_loss_symint lines632-666","普通class-index路徑只對scores做log_softmax後NLL"),ev("numeric_derivation","L=logsumexp(z)-z_y; derivative p-onehot","錯標y=3時dL/dz3<0且dL/dz2>0，局部梯度下降有利錯標"),ev("bounded_audit","results/numeric.json rows wrong_score=1 and one_logit_step","親算gradient matches解析值，單logit更新使錯標loss由2.210998降至2.067387")],"artifact_ids":[aid("results/numeric.json"),aid("code/verify.py")]},
    {"id":"C2","kind":"software","status":"verified","statement":"原fence是四候選分類代價示範，輸入[1,4]浮點logits，目標[1] long ID；候選ID0..3與byte數字字符ID不同，未訓練模型。","location":"7.14 lines497-510","scope":"完整原fence實跑；僅求代價與列印，沒有backward/optimizer或語言模型。CPU為torch2.14.1+cpu；不把它當成CUDA原實驗重訓。","evidence":[ev("torch_ce_docs","CrossEntropyLoss shapes/defaults lines1210-1244,1302-1320","[N,C]分數与[N]long目標合法，class indices在[0,C)"),ev("torch_ce_native","class_dim selection lines658-660","此[1,4]輸入使用類別軸1"),ev("run_data","ByteTokenizer lines14-28","數字字符2→58、3→59，與獨立候選ID2/3不同"),ev("fence_execution","results/original-fence/stdout.txt and environment.json","完整原fence exit0三個loss，沒有guard事件，沒有更新"),ev("bounded_audit","numeric.json boundary_class4 and original-result-audit.json token_id_boundary","四候選label4被拒絕，byte IDs親算58/59")],"artifact_ids":[aid("results/original-fence/stdout.txt"),aid("results/numeric.json"),aid("results/original-result-audit.json")],"verification":verify("合法[1,4]/[1]分類CE能執行；ID4越界；digit byte IDs58/59","完整fence exit0；boundary Target4 is out of bounds；digit2=[58],digit3=[59]","原fence原bytes SHA保留；tensor整數建立long，類別軸與候選ID檢查見verify.py")},
    {"id":"C3","kind":"numeric","status":"verified","statement":"原scores[0,0,3,1]的p2≈.810,p3≈.110，label2 loss≈.211,label3≈2.211；錯候選分數5時label3≈.139；改8後錯標loss下降而真標loss上升。","location":"7.14 lines497-514,518","scope":"一個樣本、四類別、自然對數nats，分母1。這是改scores的算例與有意義小變化，不是任何訓練成績。","evidence":[ev("numeric_derivation","logsumexp-z_label","逐一代入scores1、5、8求loss，softmax分母包括四分數"),ev("fence_execution","原fence全部三行stdout","float32原語義得到.210997626、2.210997581、.138727695"),ev("bounded_audit","numeric.json rows and tolerance","float64獨立exp/log與torch絕對誤差≤1e-12；分數8的label3=.007381561,label2=5.007381561")],"artifact_ids":[aid("results/original-fence/stdout.txt"),aid("results/numeric.json")],"verification":verify("所列3位小數符合四捨五入；score5→8 false CE下降、true CE上升","p2=.809775992,p3=.109591263；初始CE=.210997623/2.210997623；score5 CE3=.138727649,CE2=2.138727649；score8 CE3=.007381561,CE2=5.007381561","原fencefloat32與另行float64對公式核對；梯度解析值同時一致",tolerance="3位小數絕對誤差≤0.0005；float64公式/梯度絕對1e-12")},
    {"id":"C4","kind":"concept","status":"verified","statement":"重複出現或較長答案可讓錯例的有效目標份額超出筆數份額，實際影響仍須量測。","location":"7.14 line514","scope":"有效token機會/計數，不是gradient norm、因果效應量或模型成功率。本節四筆circle/square同樣6 bytes，實際兩支target總數一致。","evidence":[ev("torch_ce_docs","CrossEntropyLoss reduction/ignore_index formula lines1230-1242,1273-1281","未忽略目標在代價中的分母/求和份額，並非每筆平均一個答案"),ev("run_data","render_chat lines54-68","助手每byte及EOS成為有效目標，非助手目標ignore"),ev("run_common","fit_lm lines125-144","每次抽樣的有效目標可重複累計；實際模型影響不由筆數直接決定"),ev("bounded_audit","original-result-audit.json bounded_target_length_variation and sampling","ASCII短/長答案親算2/7個target含EOS；本例各33733target且更改rows實抽416次")],"artifact_ids":[aid("results/original-result-audit.json"),aid("code/verify.py")]},
    {"id":"C5","kind":"empirical","status":"verified","statement":"局部錯標比較從直接SFT基模各複製一份；同45訓練題、300次更新、batch16、抽樣起點42，四筆circle/square對調約8.9%，集中於兩個已教屬性家族，兩支各33733個計分目標；留出label保持正解。","location":"7.14 lines516,523","scope":"原run revision52964f...的clean/noisy兩支契約与原資料hash/抽樣重算；未下載/打開model.pt，初始化相同由原始copy.deepcopy(base)契約與SFT before結果吻合支持，不聲稱逐權重獨立bit-level復驗。","evidence":[ev("run_text","run_sft_ablation lines614-661 and run_sft lines550-568","唯一load_lm依賴sft/model.pt；每支seed/deepcopy並update300次；污染最先4個eligible shape答案；評測原attributes"),ev("run_common","Context.dependency lines25-29; fit_lm lines122-144; split_records lines51-70","batch16、local sampler seed42、family分割与targets累計契約"),ev("run_prepare","attributes-sft branch lines30-45","規則的60題來自12家族×5要求，shape/color/pitch/joint外部正解明確"),ev("raw_ablation","results.data.attributes, results.corruptions, results.runs.clean/noisy.training","45/5/10分割與四筆row0,1,5,6和training record hashes/steps33733"),ev("raw_sft","results.after and results.checkpoint","ablation before.A_attributes完全等於直接SFT after且checkpoint=model.pt"),ev("bounded_audit","original-result-audit.json splits/corruptions/fraction/sampling/base_evaluation_equal","所有原JSONL SHA重現；9/1/2家族無交叉；四筆僅更改答案；兩支4800抽樣順序hash相同且33733target")],"artifact_ids":[aid("results/original-result-audit.json"),aid("results/verify.stdout.txt"),aid("inputs/provenance.json")],"verification":verify("四筆/45≈8.9%；300×16draws；labels/queries與leaveouts遵從原設計","4/45=8.888888889%；rows0,1,5,6，families green:square:low/green:circle:low；4800draws each，33733targets each；原split/records hashes全吻合","只重建原版本既有資料與抽樣token計數，沒有重訓。四筆採按順序污染，並非各任務均勻随机noise。",denominators={"train_records":45,"validation_records":5,"test_records":10,"train_families":9,"validation_families":1,"test_families":2,"changed_records":4,"changed_families":2,"seed":42,"steps_per_branch":300,"batch_size":16,"draws_per_branch":4800,"effective_targets_per_branch":33733})},
    {"id":"C6","kind":"empirical","status":"verified","statement":"clean/noisy驗證完整答對2/5與1/5，最後題7/10與6/10，最後平均代價.37101與.34698；所有留出生成有EOS，錯標支線少答對1題但平均代價略低。","location":"7.14 lines516,525-529","scope":"同固定題集原生成IDs與原測量NLL sums重新彙總；精確匹配只去第一EOS之後内容，按raw byte ID完全相等，不strip文字；本次全部完成EOS，因此和completed exact相同。未由原weights重新計算token logits/NLL。","evidence":[ev("raw_ablation","results.runs.clean/noisy.attributes.validation/test including all samples.generated_ids","30份原ID串與NLL sums支持計數与彙總"),ev("run_common","evaluate_lm lines243-287 and _nll lines106-119","匹配按第一EOS之前raw IDs，EOS另外記錄；NLL總和除非ignored targets，不用例数做分母"),ev("run_data","ByteTokenizer/render_chat lines14-28,54-68","每個expected答案bytes+EOS給36或69有效目標"),ev("bounded_audit","original-result-audit.json evaluations","每題親核prompt/expected/raw ID/decoded/exact/EOS；36/69分母重算，五位小數吻合表格")],"artifact_ids":[aid("results/original-result-audit.json"),aid("results/verify.stdout.txt")],"verification":verify("兩支validation 2/5,1/5；test7/10,6/10；NLL5dp .37101/.34698；每支validation5+test10皆EOS","由30份IDs計得2/5与1/5、7/10与6/10；clean25.59955406188965/69=.3710080298824587；noisy23.941728591918945/69=.34698157379592676；各split全EOS","全數據子集核對，未挑樣本；NLL原sum受原CUDA浮點語義限制，但只做除法和rounding不存在重新模型測量。",denominators={"validation_records_per_branch":5,"test_records_per_branch":10,"validation_effective_targets_including_eos":36,"test_effective_targets_including_eos":69,"audited_generations":30,"seeds":1})},
    {"id":"C7","kind":"concept","status":"verified","statement":"正確前文下的各位置平均代價與沿自己生成的完整答案匹配是不同判準；本節的單種子、兩污染家族与少量新題不支持固定污染比例必然造成某量損失或低loss证明資料無害。","location":"7.14 lines514,516,529","scope":"方法差異由真正nll/generation契約与本次 counterexample支持；普遍化限制是對證據設計的限制，不增添重訓結果、統計顯著性或通用label-noise率。","evidence":[ev("torch_ce_docs","CrossEntropyLoss per-target negative log likelihood formula lines1216-1244","CE評估提供目標的概率，不保證argmax自由生成整體正確"),ev("run_data","render_chat lines54-68","NLL輸入含給定助手前文，目標是指定answer bytes/EOS"),ev("run_model","generate lines109-136","後續模型輸入使用自己追加的argmax token，与給定答案前文不同"),ev("run_common","_nll lines106-119 vs evaluate_lm lines243-287","平均target NLL与raw生成精確匹配是分開執行与彙總"),ev("bounded_audit","original-result-audit.json original_run_environment/fraction/splits/evaluations/limitations","single seed42，两污染family，validation1/test2family；test lower noisy NLL仍少1 exact，EOS不能替內容正確")],"artifact_ids":[aid("results/original-result-audit.json"),aid("results/numeric.json")]},
]
report = {
    "schema_version":1,"review_stage":"technical","lesson_id":"7.14","source":"course/chapters/07.md#7.14","source_sha256":extract["source_sha256"],"figure_sha256":{},
    "reviewer_task":"/root/phase4_factual_coordinator/factual_7_14","reviewer_context":"fresh","author_tasks":[],"verdict":"pass","reviewed_on":"2026-10-05",
    "read_scope":inspection,"artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"提供目標的交叉熵/梯度、分類IDs、有效target權重与低loss≠完整正解均有原始文件/實作与局部反例；没有找到未解實質錯誤。","claim_ids":["C1","C2","C4","C7"]},
        "numeric_verification":{"status":"pass","details":"原fence亲跑；score8/true-target對照、解析梯度/單logit步、class4边界；4/45比例、300×16、33733targets、45/5/10与家族/ID/EOS/NLL全重算。","claim_ids":["C3","C5","C6"]},
        "figure_consistency":{"status":"not_applicable","details":"本節沒有引用任何SVG/圖片。四候選与scores变更已在文字/fence明確；无必要空间映射，不声称已render/view不存在的图。","claim_ids":[]},
        "source_verification":{"status":"pass","details":"親fetch/read官方PyTorch immutable commit文件且loss.py bytes吻合CPU輪子；原實驗git tree與recorded code hashes吻合；原JSON inputs永久保存，自己检查30生成IDs與重建split SHA，不沿用旧review结论。","claim_ids":["C1","C2","C3","C4","C5","C6","C7"]},
        "limitations":{"status":"pass","details":"实际支持范围是single-seed固定合成属性任务与原测量audit；baseline同初始由原code/dependency契约和before结果吻合，未讀weights。原NLL sums未重算logits。沒有新訓練/下載模型/GPU/上傳；linked longrecipe仅inspect+CLI help，有界替代验证已完成。","claim_ids":["C4","C5","C6","C7"]}
    },
    "summary":"現稿7.14通過。原CE算例、score8練習、四筆污染的8.9%比例、相同步數/有效目標、驗證與最後題成績及平均代價均核實；現稿已明確限制污染因果與普遍化，沒有新訓練結論。"
}
dump(ROOT / "docs/technical-reviews/7.14.json", report)
print(json.dumps({"report":"docs/technical-reviews/7.14.json","sha256":hash_file(ROOT / "docs/technical-reviews/7.14.json"),"claims":len(claims),"artifacts":len(artifacts),"verdict":"pass"},ensure_ascii=False))
