"""Assemble this task's independently checked factual report; no source or checker edits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

receipt = json.loads((OUT / "source-receipt.initial.json").read_text())
run = json.loads((OUT / "probe-results.json").read_text())
env = run["environment"]
artifact_base = str(OUT.relative_to(ROOT))
probe_command = ".venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/2.5/probe.py > docs/technical-reviews/artifacts/natural-v4-factual/2.5/probe.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/2.5/probe.stderr.txt"
figure_command = ".venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/2.5/figure_check.py > docs/technical-reviews/artifacts/natural-v4-factual/2.5/figure-check.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-factual/2.5/figure-check.stderr.txt"
artifacts = []
def artifact(id, kind, filename, description, command=None, result=None, environment=None):
    value = {"id": id, "kind": kind, "path": artifact_base + "/" + filename,
        "sha256": sha(OUT / filename), "description": description}
    if command is not None:
        value.update(command=command, result=result, environment=environment)
    artifacts.append(value)

artifact("a_source", "source_snapshot", "source.initial.md", "Own exact raw UTF-8 assigned section including all terminal blank lines, preserved before review; 2.5 is not a first section.")
artifact("a_receipt", "source_snapshot", "source-receipt.initial.json", "Initial source SHA, byte count, complete directly referenced SVG map, own canonical task and introduction applicability.")
artifact("a_prior_unread", "source_snapshot", "prior-report.unread.json", "Byte-copy of prior assigned report before replacement; never read or used as evidence of truth or verdict.")
artifact("a_probe_code", "code", "probe.py", "Own bounded CPU code: exact lesson block, extra slices, split fingerprints, parameter counts, 3*200 updates and independent scalar NLL.")
artifact("a_probe_stdout", "execution", "probe.stdout.txt", "Actual execution stdout with environment, all cases and successful bounded reruns.", probe_command,
    "Exit 0; all assertions passed. Every rounded section table entry matched; maximum post-update difference from original record was 8.940696716308594e-08. Empty probe.stderr.txt.", env)
artifact("a_probe_results", "execution", "probe-results.json", "Actual execution result with full document split, vocabulary, per-target NLLs, configuration and historical/current environment distinction.", probe_command,
    "Own CPU rerun completed 200 updates per tiny MLP. Independent float64 scalar means matched float32 PyTorch within 2e-6; target denominators 103/11/22; historical data SHA matched for all three splits.", env)
artifact("a_environment", "source_snapshot", "environment.json", "Runtime versions captured by the actual probe: Python3.13.5, torch2.14.1+cpu, CPU, float32, two threads.")
artifact("a_original_record", "source_snapshot", "existing-simple_models.initial.json", "Existing official project experiment record actually read and copied, distinct from the reviewer rerun; historical CPU torch2.14.1+cu126/Python3.13.3.")
artifact("a_derivation", "derivation", "derivations.initial.md", "Own tracked mathematical calculations of visible information, affine weights, total parameters, mean loss and limitations.")
artifact("a_authorities", "source_snapshot", "authorities.initial.md", "Own summaries of actual original-source reading, precise locators, repository-code inspection and personal figure viewing; no complete third-party paper copied.")
artifact("a_retrieval", "source_snapshot", "retrieval-receipts.json", "Original HTTPS URL, HTTP status, retrieval date and exact SHA receipts for actual authoritative downloads kept only in ignored research.")
artifact("a_installed_match", "source_snapshot", "installed-authority-match.stdout.txt", "Actual runtime inspection confirms both used PyTorch Python source files are byte-identical to official v2.14.1 originals.")
artifact("a_figure_code", "code", "figure_check.py", "Own code recovering the SVG point values from tick positions, accounting for decimal quantization.")
artifact("a_figure_stdout", "execution", "figure-check.stdout.txt", "Actual six-point SVG coordinate check stdout.", figure_command,
    "Exit0; all six recovered values differ from original stored losses by less than 1e-7, finer than plotted 0.001 labels.", env)
artifact("a_figure_values", "derivation", "figure-coordinate-check.initial.json", "Checked numeric values, original SVG SHA, calibration and personal visual-inspection record.")
artifact("a_figure_render", "figure_render", "window_training.initial.png", "Original SVG rendered by Inkscape at1296px and personally viewed with view_image; checked six point labels and directions, context positions, parameters, legend and non-time horizontal axis.")
artifact("a_render_stderr", "source_snapshot", "render.stderr.txt", "Actual Inkscape stderr. GTK/Pango wrapper warnings did not prevent rendering; personally viewed PNG text and chart geometry were correct.")
for section in ["1.3", "2.2", "2.4", "T.3"]:
    artifact("a_prereq_" + section.replace(".", "_"), "source_snapshot", f"prerequisite-{section}.initial.md",
        f"Full direct prerequisite{section}, independently read as necessary background; no claim of reviewing all unrelated T.3 software instructions.")

sources = [
    {"id": "s_bengio", "kind": "paper", "title": "A Neural Probabilistic Language Model",
        "url": "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf", "version": "JMLR3 (2003),1137–1155, publisher PDF",
        "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
        "authority_reason": "Original authors' peer-reviewed publication at the JMLR publisher; introduces the finite-context neural probability network.",
        "inspection_note": "Actually downloaded PDF and read Section1 finite-window approximation and Section2 pp1141–1143, feature table, conditional mapping, Figure1, average log likelihood, softmax and equation(1). With optional W=0, concatenated features -> tanh -> output match the conceptual architecture. Does not prove project numbers, training success, or character-level quality. Retrieval SHA in a_retrieval; own reading notes a_authorities."},
    {"id": "s_python", "kind": "official_docs", "title": "Python3.13 Common Sequence Operations and Tuples",
        "url": "https://docs.python.org/3.13/library/stdtypes.html#common-sequence-operations", "version": "Official Python3.13 branch retrieved2026-10-04; actual runtime3.13.5",
        "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
        "authority_reason": "Python Software Foundation's official language/library documentation, applicable to the executed Python3.13 runtime.",
        "inspection_note": "Actually read original HTML sequence notes3–5 and Tuples subsection. Negative start is relative to length, omitted end uses length, omitted step is+1; tuple items keep order. Server redirects to /3.13/builtins/stdtypes.html. Exact local examples additionally executed; receipt in a_retrieval."},
    {"id": "s_linear", "kind": "official_source", "title": "PyTorch Linear source",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/linear.py", "version": "v2.14.1; installed2.14.1+cpu source byte-identical",
        "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
        "authority_reason": "Versioned implementation and API contract in official pytorch/pytorch repository, matched to installed CPU package.",
        "inspection_note": "Actually read lines53–80,96–115,130–134: y=xA^T+b, weight(out,in), optional bias(out), and allocated dimensions; used for H*C*D weight count separately from bias. Actual installed byte match in a_installed_match."},
    {"id": "s_ce", "kind": "official_source", "title": "PyTorch functional cross_entropy source",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py", "version": "v2.14.1; installed2.14.1+cpu source byte-identical",
        "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
        "authority_reason": "Official versioned PyTorch implementation corresponding to the actual installed package.",
        "inspection_note": "Actually read cross_entropy lines3478–3572: logits/class-index labels, mean reduction, ignore_index=-100, no default class weights, label_smoothing=0 and backend call. Concrete unmasked means were separately reconstructed from every emitted logit row with Python scalar log-sum-exp."},
    {"id": "s_math", "kind": "derivation", "title": "Independent transparent calculations for2.5", "verified": True,
        "details": "See a_derivation. Equal visible input v has one fixed g(v) or q(.|v); balanced two-answer accuracy is (q(圓|v)+q(方|v))/2<=1/2. Strings count5 or8, requiring full window to retain first distinguishing character. C vectors of D values ->C*D inputs; H*C*D matrix weights plus optional H bias. V17,D=H16:17*16+(256C+16)+(16*17+17)=577+256C ->833,1345,1857. Mean NLL=-sum log p_y/N; exp(-mean NLL) is geometric mean correct-target probability. Four 三角 train documents contribute4*12 targets and five other train docs5*11, giving103; validation11/test22. Only fixed-configuration representability and this small experiment are supported."},
    {"id": "s_execution", "kind": "execution", "title": "This reviewer's bounded CPU rerun and scalar verification", "verified": True, "artifact_id": "a_probe_results"},
]
repo_specs = [
    ("s_model", "tiny_perceptron/simple.py", "Read complete28-line file: ContextMLP embedding(17,16), hidden Linear(C*16,16), output Linear(16,17), fixed-context check, flatten and tanh. No dropout or recurrent hidden state."),
    ("s_data", "tiny_perceptron/data.py", "Read complete122-line file, especially split_documents97–108 and toy_documents113–115. Exact duplicate removal, local seed shuffle and 4color*3shape corpus; independently reconstructed fingerprints."),
    ("s_exp", "scripts/course_experiments/text.py", "Actually read lines1–209, especially _save_splits, _simple_examples102–111 and run_simple_models128–203. Seed reset, same split, width16, 200full-batch updates, AdamWlr.01, clip1.0, post-update means. Unrelated later experiments not needed."),
    ("s_common", "scripts/course_experiments/common.py", "Actually read lines1–108 for Context, write_json and seed delegation; relevant seed(value) calls seed_everything. Unrelated later workflows not reviewed."),
    ("s_runner", "scripts/course_experiments/run.py", "Actually read lines1–175: execute establishes seed42 and torch.set_num_threads(2), dispatches one selected experiment and records environment/code hashes. Runner not executed; no Git fallback used."),
    ("s_seed", "tiny_perceptron/training.py", "Read complete115-line file; seed_everything23–27 seeds Python/PyTorch and conditionally CUDA. Our CPU replay uses the same relevant seeds and no CUDA."),
]
for id, path, note in repo_specs:
    sources.append({"id": id, "kind": "repository_code", "title": path, "path": path,
        "sha256": sha(ROOT / path), "version": "Current repository bytes independently inspected; full-file SHA binds actual source",
        "verified": True, "inspection_note": note})

claims = []
def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}
def claim(id, kind, statement, location, evidence, artifact_ids, scope, verification=None, denominators=None):
    c = {"id": id, "kind": kind, "statement": statement, "location": location,
        "status": "verified", "evidence": evidence, "artifact_ids": artifact_ids, "scope": scope}
    if verification is not None: c["verification"] = verification
    if denominators is not None: c["denominators"] = denominators
    claims.append(c)
def executed(expected, observed, details, tolerance="Exact strings/integers; numeric tolerances stated where relevant"):
    return {"method": "executed", "expected": expected, "observed": observed, "tolerance": tolerance, "details": details}
def hand(expected, observed, details):
    return {"method": "hand_calculation", "expected": expected, "observed": observed,
        "tolerance": "Exact integer algebra; decimals at displayed precision", "details": details}

claim("c01", "concept", "兩個題目若只給相同的可見前文，固定模型得到相同候選分配，不能據被刪除的顏色穩定區分兩答案。", "段1與切片程式後段2",
    [ev("s_math", "Visible information proof", "g(v) is fixed for identical v; the answer labels do not add unavailable color to input."), ev("s_bengio", "Section2 p1141 conditional distribution mapping", "Network maps the visible context's feature sequence to one distribution.")], ["a_derivation"], "Fixed parameters and no extra state/metadata exposing color. The two examples' obstruction does not apply to different visible inputs.")
claim("c02", "concept", "隨機選字不能補回沒有輸入的顏色線索。", "切片程式後段2末句",
    [ev("s_math", "Balanced two-example randomized prediction proof", "With same q(.|v), averaged correctness is(q圓+q方)/2<=1/2, so independent randomness cannot reliably select each required answer.")], ["a_derivation"], "Randomness independent of hidden color; no claim that randomized outputs themselves are identical.")
claim("c03", "concept", "本節上下文是可用前文，固定窗口只取最近有限C位置；較長窗口可能保留更遠線索。", "段2及2.2直接連結",
    [ev("s_bengio", "Section1 finite-window approximation; Section2 p1141", "Next-word conditional distribution is approximated with the last n-1 tokens and their feature sequence.")], ["a_prereq_2_2", "a_authorities"], "Local character-level fixed-window model; this definition does not assert a universal beneficial window length.")
claim("c04", "software", "程式的兩項tuple保存前文/答案次序，prefix[-length:]取後綴且仍按正向順序讀取。", "唯一Python區塊與緊接解釋",
    [ev("s_python", "Common Sequence Operations notes3–5; Tuples", "Negative index/omitted stop/default+1 step and ordered tuple contents."), ev("s_execution", "probe exact lesson block", "Exact original block executed in Python3.13.5.")], ["a_probe_code", "a_probe_stdout"], "These ordinary Chinese strings have one relevant code point per character; no claim that slicing arbitrary Unicode grapheme clusters is language-aware.")
claim("c05", "numeric", "兩個五字前文在窗口1/3/4均相同後綴，是/物體是/色物體是；窗口5保留紅/藍差別。", "程式輸出解釋及可辨識性段",
    [ev("s_execution", "slices output1,3,4,5", "Executed exact block and extra4-character case."), ev("s_math", "five-character counting", "Positions1–5 identify earliest differing character.")], ["a_probe_results", "a_derivation"], "Only the specified two prefix/answer pairs.", executed("1:是;3:物體是;4:色物體是;5:紅色物體是/藍色物體是", "All exact expected strings and lengths matched", "Source strings sliced with positive C; exact equality assertions, including omitted4 case."))
claim("c06", "concept", "五字窗口消除了輸入不可區分的障礙，但不保證已學會或能在新題泛化。", "可辨識性段後半",
    [ev("s_math", "Distinct inputs versus learned function", "Distinct v values permit different outputs without proving optimized parameters implement the desired rule."), ev("s_bengio", "Section2 opening p1141", "Out-of-sample likelihood is a separate modeling objective, not guaranteed by representation.")], ["a_derivation", "a_probe_results"], "Representability and data availability are necessary considerations, separate from actual learning and held-out quality.")
claim("c07", "numeric", "C個D維字向量拼接有C*D输入，接H個Linear輸出有H*C*D權重；D2時C3增至C6使該層權重翻倍。", "擴大固定窗口段",
    [ev("s_linear", "lines53–80 and108–112", "Matrix weight(out,in) and separate bias(out)."), ev("s_math", "Dense mixing derivation", "C*D inputs yield H*C*D entries;6H becomes12H.")], ["a_derivation"], "Fixed H,D and fully dense concatenation architecture. Bias adds H; total model size or runtime need not exactly double.", hand("C*D inputs;H*C*D weights;D2,C3->6 gives6H->12H", "Algebra matches exactly", "Each of H outputs has one weight per C*D input coordinate; optional H biases separate."))
claim("c08", "concept", "此固定拼接架構擴大C會增加首個稠密層的參數與計算。", "擴大固定窗口段首尾",
    [ev("s_bengio", "Section2 pp1142–1143 dimensional/parameter discussion", "Dense hidden matrix has h*(n-1)*m weights and window-dependent work."), ev("s_math", "Dense mixing operation count", "H*C*D multiplications and corresponding additions scale with C.")], ["a_derivation"], "Dense arithmetic count and stored parameters; no measured latency, memory or universal architecture scaling claim.")
claim("c09", "concept", "MLP串聯線性層和非線性轉換，能表示純線性組合無法表示的彎曲規則。", "實測介紹及2.4直接連結",
    [ev("s_bengio", "Section2 p1142; Eq(1) p1143 withW=0", "Feed-forward hidden tanh plus output linear mapping."), ev("s_math", "MLP/nonlinearity affine-composition and absolute-value derivation", "Affine maps compose to affine; specified ReLU pair produces|x| impossible for single affine rule.")], ["a_prereq_2_4", "a_derivation"], "Representational possibility; no guarantee optimization finds a good rule, and no universal approximation claim.")
claim("c10", "software", "本實驗MLP先以寬度16嵌入查表、按窗口拼接，再經Linear/tanh/Linear輸出下一字logits。", "實測介紹的模型流程",
    [ev("s_model", "ContextMLP classlines16–28", "Inspected exact module ownership and forward; probe runs actual class."), ev("s_exp", "run_simple_models constructor andbatches", "Each of contexts 1/3/5 uses width 16 with the same vocabulary.")], ["a_probe_code", "a_probe_results"], "Current project code; character-level toy architecture, not a description of all language models.")
design_den = {"seed": 42, "width": 16, "windows": [1,3,5], "updates_per_model": 200,
    "documents": {"total":12,"train":9,"validation":1,"test":2},
    "next_character_targets_including_eos": {"train":103,"validation":11,"test":22},
    "batch": "all103train targets each update", "split": "deduplicated whole documents before windows; exact split SHA matched original", "timing": "No reviewer timing/throughput claim"}
claim("c11", "empirical", "三組tiny MLP使用相同12篇規則語料與種子42的9/1/2整篇切分，寬度16，各200次更新。", "實測介紹與表前設定；T.3指向",
    [ev("s_data", "toy_documents andsplit_documents", "Corpus and deterministic whole-document split."), ev("s_exp", "run_simple_models128–203", "Same data, seed reset and200updates withwidth16."), ev("s_execution", "documents/split_fingerprints/denominators/runs", "Actual own execution reconstructs fingerprints and completes all boundedupdates.")], ["a_original_record", "a_probe_results", "a_probe_stdout"], "Historical officialCPU run separately validated and ownCPU rerun completed. Originaltorch2.14.1+cu126/Python3.13.3; owntorch2.14.1+cpu/Python3.13.5. One small fixed corpus/seed; no GPU or longtraining claim.", denominators=design_den)
for c, count in [(1,833),(3,1345),(5,1857)]:
    claim(f"c_param_{c}", "numeric", f"{c}字MLP的總參數為{count}。", f"表格{c}字列的參數格數",
        [ev("s_math", "parameter formula577+256C", "Independent sum includes embedding, both dense weights and biases."), ev("s_execution", f"runs/{c}/parameters", "Actual tensor numel count matches hand calculation.")], ["a_derivation", "a_probe_results"], "V17,D=H16 and this exactContextMLP, not just first-layer weights.", executed(str(count), str(run["runs"][str(c)]["parameters"]), "Compare actual sum(p.numel()) with577+256C exact integer count."))
for c in [1,3,5]:
    for split, display in [("train",{1:"0.33352",3:"0.21302",5:"0.19883"}[c]),("validation",{1:"0.43103",3:"0.30577",5:"0.51252"}[c])]:
        observed=run["runs"][str(c)]["after"][split]
        claim(f"c_loss_{c}_{split}", "empirical", f"{c}字窗口更新200次後的{split}平均下一字代價為{display}（五位小數）。", f"實測表{c}字列/{split}欄",
            [ev("s_ce", "cross_entropy3478–3572", "Mean loss on unmasked class-index targets."), ev("s_exp", "post-update afterevaluation", "Evaluation occurs after finaloptimizer.step, not last preupdate batchloss."), ev("s_execution", f"runs/{c}/after/{split}", "Actual ownCPU rerun and independent scalar calculation match originalrecord.")], ["a_original_record", "a_probe_results", "a_probe_stdout"], "Existingrecord measurement verified, with distinct freshCPU replay. Five-decimal example result applies only to this split/seed/200updates; CPU floating point tail can differ.",
            executed(display, f"PyTorch{observed['nll']}; independentScalar{observed['manual_nll_python_float64']}", "Saved every targetloss from scalar stablelogsumexp; arithmetic mean over statednonignored targets; compared with original before/aftervalues.", "Rounding to5decimal exact; raw historic replay difference<2e-6, scalar vsPyTorch<2e-6."),
            {"window":c,"seed":42,"updates":200,"width":16,"documents":9 if split=="train" else 1,"targets_including_eos":103 if split=="train" else 11,"weighting":"equal per target; not equal per document","device":"CPU","full_training_batch_targets":103,"optimizer":"AdamW lr=0.01, weight_decay=0.01, gradient norm clip=1.0","timing_scope":"No time/throughput claim"})
claim("c12", "concept", "較低的平均下一字代價表示整批正確目標機率的幾何平均較高。", "實測表前最後一句",
    [ev("s_bengio", "Section2 openingp1141", "AverageNLL and its exponential/perplexity relation."), ev("s_math", "MeanNLL derivation", "exp(-L)=geometricmeancorrecttargetprobability."), ev("s_ce", "defaultunweightedmean", "Thisexperiment uses equal targetweights.")], ["a_derivation", "a_probe_results"], "Average claim; it need not improve every character, arithmeticmean probability or argmaxaccuracy.")
claim("c13", "numeric", "本次1->3字訓練/驗證代價都降；3->5字訓練下降但驗證由.30577升至.51252。", "表後解釋段",
    [ev("s_math", "Displayed loss differences", "Train:-.12050 then-.01419; validation:-.12526 then+.20675."), ev("s_execution", "unroundedpost-update NLLs", "All directions hold for full precision values.")], ["a_derivation", "a_probe_results"], "Specific observed configuration, not a monotonic window rule.", hand("1->3bothdecrease;3->5train-.01419/validation+.20675", "Signs and subtraction match table and rawrecords", "Independent subtraction of all six displayed values and comparison to unrounded ownCPU results."))
claim("c14", "concept", "窗口與參數一起改變、僅一篇驗證文及單種子，不能推出所有任務3字最佳或只因窗口造成品質差異。", "圖後限制段",
    [ev("s_math", "Parameterconfound andlimits", "Totalparameters increase833->1857; onevalidationdocument cannot support universaloptimum."), ev("s_execution", "documents/validation andrunparametercounts", "Validationexactly顏色=紅；形狀=圓。; explicit jointvariation confirmed.")], ["a_derivation", "a_probe_results"], "The section appropriately limits the inference to a toyexample and recommends held-out checks; no generalqualityclaim is present.")
claim("c15", "numeric", "練習前文各8字；窗口7同為色的小小物體是，8才保留紅/藍首字。", "末段練習",
    [ev("s_math", "Eight-characterexercisecount", "Enumerated eachposition independently."), ev("s_execution", "exercise7/8exactassertions", "Actual slices andcounts checked.")], ["a_derivation", "a_probe_results"], "The exact exercise strings andunchangedanswerlabels; no modeltraining occurs.", executed("len8;7same;8different", "len8forboth,7=[色的小小物體是,色的小小物體是],8=fulloriginalstrings", "ExactPython len/slicing equality assertions and independentcharacter enumeration."))
claim("c16", "concept", "圖的橫軸為1/3/5字窗口，藍/橘曲線表示表格train/validation代價，標示對應833/1345/1857參數；不表示時間演變。", "window_training.svg與圖後首句",
    [ev("s_math", "Six-pointSVGcoordinatecalibration", "Recover actual plottedvalues from tickpositions within1e-7; rounded0.001labels agree."), ev("s_execution", "postupdatewindowconfigurationcomparison", "Values correspond to separatelytrainedwindowconfigs.")], ["a_figure_render", "a_figure_values", "a_figure_stdout", "a_original_record"], "Personally viewed actual render: ascendingcontexts, linearcroppedyaxis, bluefalling/orangefallingthenrising, correctlegend andparametercaptions; noarrows or temporalinterpretation.")

all_ids=[c["id"] for c in claims]
numeric_ids=[c["id"] for c in claims if c["kind"] in ["numeric","empirical"]]
report={"schema_version":1,"review_stage":"technical","lesson_id":"2.5","source":"course/chapters/02.md#2.5",
    "reviewer_task":"/root/v4_review_coordinator/factual_v4_2_5","reviewer_context":"fresh",
    "source_sha256":receipt["source_sha256"],"figure_sha256":receipt["figure_sha256"],"verdict":"pass",
    "claims":claims,"sources":sources,"artifacts":artifacts,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"Every substantive statement independently checked; same-input information obstruction proved, finite-window/MLP methods located in original publication and versioned source, actual project model/configuration executed. No inherited reader or technical verdict used.","claim_ids":all_ids},
        "numeric_verification":{"status":"pass","details":"All slices/counts, C*D andH*C*D algebra, exactparameter totals and tabledifferences checked. ThreefulltinyCPU reruns each completed200updates; originalsplit SHA reconstructed; all relevant lossvalues agree atfive decimals with independent scalarNLL denominators103/11/22.","claim_ids":numeric_ids},
        "figure_consistency":{"status":"pass","details":"Only direct SVG is window_training.svg and its current SHA is fullymapped. Original SVG personally rendered and viewed; all six numeric point coordinates, rounded labels, axes, legend, parameter ticks and directions checked. Lines compare windows, not updates/time; no arrows present.","claim_ids":["c16"]},
        "source_verification":{"status":"pass","details":"Actually read JMLR publisher PDF §2 pp1141–1143/Eq1 and Python 3.13 official slicing/tuple docs; official PyTorch v2.14.1 Linear/CE source byte-matches installed package. Registered repository-code full-file SHA and actual inspection locators saved; complete existing official run record inspected independently and own CPU execution separately identified.","claim_ids":all_ids},
        "limitations":{"status":"pass","details":"Distinguishesavailability from learning, fixed MLP conditional distribution from independent random sampling, first matrix weights fromtotal model/work/latency, averaged NLL fromper-target accuracy, historical record fromown CPU run, andsingle-seed/one-validation-document comparisons fromuniversal best window. NoGPU/speed or paper benchmark claim.","claim_ids":all_ids}},
    "inspection_scope":{"introduction_applicable":False,"reason":"2.5 is not the first numbered section; chapter introduction is not used as 2.5 evidence.","raw_section_read":True,"direct_prerequisites_read":["1.3","2.2","2.4","T.3"],"historical_report":"Copied bytes unread; not used","source_current_check":"Bound to own raw initial source and full current SVG map"}}
destination=ROOT/"docs/technical-reviews/2.5.json"
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
initial=OUT/"report.initial.json"
if initial.exists(): raise RuntimeError("Never overwrite the genuine first report")
initial.write_bytes(destination.read_bytes())
print(json.dumps({"verdict":report["verdict"],"source_sha256":report["source_sha256"],"report_sha256":sha(destination),"claims":len(claims),"sources":len(sources),"artifacts":len(artifacts),"issues":len(report["issues"])},indent=2))
