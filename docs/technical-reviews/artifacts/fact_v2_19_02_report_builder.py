"""Write only the independently reviewed initial version; do not adopt a new hash."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_02_"
ENV = {"python": "3.13.5", "torch": "2.14.1+cpu", "device": "cpu", "threads": "1"}
COMMAND = (
    "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_02_audit.py "
    "> docs/technical-reviews/artifacts/fact_v2_19_02_audit_stdout.json "
    "2> docs/technical-reviews/artifacts/fact_v2_19_02_audit_stderr.txt"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


artifacts = []


def artifact(identifier, filename, kind="source_snapshot", description="", command=None, result=None, env=None):
    path = OUT / (PREFIX + filename)
    entry = {"id": identifier, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "kind": kind,
             "description": description or filename}
    if kind == "execution":
        entry.update(command=command or COMMAND, result=result or "Exit 0; assertions completed; raw results retained.", environment=env or ENV)
    artifacts.append(entry)


artifact("reviewed_body", "section_19_2.md", description="First genuinely read body, extracted by checker.sections before locator-label revision.")
artifact("read_manifest", "read_manifest.json", description="Initial source/full prerequisite/code/figure SHA receipts; no old review or author history used.")
artifact("inspection", "inspection.md", description="Original-source reading, hand derivation, full empirical scope and actual SVG visual inspection notes.")
artifact("fetch", "fetch_result.json", "execution", "Original fixed PDFs, RFC and installed-commit official PyTorch code were really fetched.",
         "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_02_prepare.py", "Exit 0; five original HTTPS sources downloaded and checksummed; PDFs extracted with pdftotext.")
for identifier, filename in [("mixtral_pdf", "mixtral_v1.pdf"), ("mixtral_text", "mixtral_v1.txt"),
                              ("switch_pdf", "switch_jmlr.pdf"), ("switch_text", "switch_jmlr.txt"),
                              ("rfc_text", "rfc3629.txt"), ("torch_module", "torch_module.py.txt"),
                              ("torch_adam", "torch_adam.py.txt")]:
    artifact(identifier, filename)
artifact("figure_render", "capstone_resources.png", "figure_render", "Inkscape render at 1800x1140, actually visually viewed; four boxes/two arrows/shared rules and limits agree with prose.")
artifact("replay_code", "audit.py", "code", "Final import-formatted CPU replay source; its actual execution SHA is in audit_completion.")
artifact("prepare_code", "prepare.py", "code", "Initial checker.sections freeze and verified original-source retrieval source.")
artifact("original_code", "original.py", "code", "Verbatim fenced Python program of first reviewed section.")
artifact("exercise_code", "exercise.py", "code", "Only total*4 was replaced with total*16 as instructed.")
for identifier, filename, description in [
    ("snippet", "snippet_execution.json", "Original and exercise truly executed with stdout/stderr, source/program SHA and environment."),
    ("numeric", "numeric_audit.json", "All parameter tables counted including bias; proxy, dtype bytes, untied storage and Adam state allocation verified."),
    ("weights", "weights_audit.json", "Strictly loaded real joint and anonymous fixed-pin public joint; whole-file SHA/non-tensor metadata/53 finite tensor checks/shape/dtype/hash and exact tensor equality retained, no tracked checkpoint binary."),
    ("dataset", "dataset_audit.json", "Seed42 family splits and exact24 batch IDs,94 grid length,1836 valid inputs/474 labels independently rebuilt."),
    ("pad", "pad_balance_audit.json", "Only8 valid positions enter each FFN; 16 top2 assignments; formula reconstruction and gradient checks; appended PAD invariance."),
    ("gpu", "gpu_report_audit.json", "All10 saved GPU samples recalculated; median/mean and baseline-subtracted allocator peak checked, with original environment/time/memory scope."),
    ("cpu_benchmark", "cpu_mechanism_benchmark.json", "Real CPU warm3/measured10 forward/backward with same frozen batch/config, zero optimizer updates and unchanged weights; no GPU latency inference."),
    ("per_row", "per_row_audit.json", "Every saved84/90 joint row reaggregated and actual full CPU generated-ID/score replay compared, with zero differences."),
    ("cpu_validation", "cpu_validation.json", "Fresh CPU84-row validation raw traces; 75 correct, exact historical generated IDs."),
    ("cpu_test", "cpu_test.json", "Fresh CPU90-row test raw traces; 78 final/80 action correct, exact historical generated IDs."),
    ("old_training", "moe_training_audit.json", "Historical MoE/Dense complete configurations, update/effective-token denominators, aggregate medians and2.27682495 ratio; not retimed on GPU."),
    ("inventory", "experiment_inventory.json", "All capstone results parsed, relevant configs/steps/objectives and exact code SHA comparisons retained; selection checkpoint linkage checked."),
    ("completion", "audit_completion.json", "Actual final replay-code SHA, initial reviewed section SHA, completion date and environment."),
]:
    artifact(identifier, filename, "execution", description)
artifact("stdout", "audit_stdout.json", "execution", "Raw final CPU audit summary and per-row checks.")
artifact("first_run_manifest", "first_execution_manifest.json", description="Preserves full first real audit execution prior to changing only import-list formatting; never substitutes fresh hashes for actual executed source versions.")
artifact("old_code", "moe_run_architecture.py.txt", "code", "Exact execution code from report revision48a4f3e...; SHA2561ad0b078... matches its report, unlike the current newer architecture.py.")
for identifier, rel in [
    ("gpu_raw", "docs/course-experiments/capstone-evidence/deployment/mechanism-benchmark.json"),
    ("deployment_raw", "docs/course-experiments/results/capstone_deployment.json"),
    ("old_moe_raw", "docs/course-experiments/results/moe.json"),
    ("selection_raw", "docs/course-experiments/capstone-selection.json"),
    ("joint_raw", "docs/course-experiments/results/capstone_joint.json"),
    ("historical_validation", "docs/course-experiments/capstone-evidence/joint/validation.json"),
    ("historical_test", "docs/course-experiments/capstone-evidence/deployment/test-joint.json"),
    ("public_manifest", "docs/course-experiments/capstone-public.json"),
]:
    artifacts.append({"id": identifier, "path": rel, "sha256": sha(ROOT / rel), "kind": "source_snapshot",
                      "description": "Original persistent experiment/raw evidence actually read and independently checked, not a prior review."})

sources = []


def external(identifier, kind, title, url, version, authority, note):
    sources.append({"id": identifier, "kind": kind, "title": title, "url": url, "version": version,
                    "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
                    "authority_reason": authority, "inspection_note": note})


external("mixtral", "paper", "Mixtral of Experts", "https://arxiv.org/pdf/2401.04088v1", "arXiv2401.04088v1, 8 January2024",
         "Original authors' paper from arXiv, separately downloaded as original PDF.",
         "Read §2.1 pp2–3 sparse topK gated expert sum and specialized kernels/Expert Parallelism; §3 Size and Efficiency p4 states serving memory follows total47B, routing/increased memory-load overhead and batched utilization caveats. Not a tiny-model performance or FLOPs-match guarantee.")
external("switch", "paper", "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
         "https://jmlr.org/papers/volume23/21-0998/21-0998.pdf", "JMLR23(120),2022 final PDF; SHA8831edca...",
         "Original publication PDF from JMLR; directly fetched rather than relying on a candidate-source summary.",
         "Read §2.1 Simplifying Sparse Routing pp3–4 and §2.2 Efficient Sparse Routing pp5–6. The Differentiable Load Balancing Loss passage and Eq4–6 are §2.2 p6, contradicting initial lesson's2.1 locator. Eq4 alpha*N*sum f_iP_i: f top1 assignment is nondifferentiable, P mean full router softmax is differentiable. Capstone extends assignment normalization to top2 and excludes PAD; it is dropless rather than Switch capacity/drop behavior.")
external("rfc", "official_docs", "RFC3629 UTF-8, a transformation format of ISO10646", "https://www.rfc-editor.org/rfc/rfc3629.txt", "RFC3629, November2003",
         "RFC Editor's original standards text.", "Read §3 encoding-range table: code points use1–4 octets; common BMP Chinese uses3 and supplementary Chinese can use4. ByteTokenizer's CPU examples map octets, not characters, to token IDs.")
commit = "5c4886908584029761b579af026dcfb627c84070"
external("pytorch_module", "official_source", "PyTorch Module.parameters/named_parameters", f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/module.py",
         "Installed torch2.14.1+cpu original git commit" + commit, "PyTorch project's original source, fetched at the installed distribution's immutable git_version.",
         "Read lines2650–2729: recursive parameter iterator delegates to named_parameters and removes duplicates by default. Compared count with all named tensors and distinct embedding/output objects on CPU.")
external("pytorch_adam", "official_source", "PyTorch Adam state initialization", f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/optim/adam.py",
         "Installed torch2.14.1+cpu original git commit" + commit, "PyTorch original source for the exact installed version.",
         "Read Adam defaults and lines152–189 lazy exp_avg/exp_avg_sq zeros_like, scalar step and optional AMSGrad third moment. Verified default FP32 state allocation with zero-lr, zero-gradient initialization, no learning; total*16 omits scalar steps and real runtime allocations.")
for identifier, rel, note in [
    ("capstone_code", "tiny_perceptron/capstone.py", "Read default_config lines34–49, PadSafeBlock52–71, CapstoneModel80–121, batch321–341, evaluation and weights-only strict loading644–654. Frozen snapshot retained; exact report SHA match."),
    ("core_code", "tiny_perceptron/model.py", "Read ModelConfig, Block and TinyLM; RMS/GELU/GQA branches, two layers, untied input/output and summed auxiliary. Frozen execution-source snapshot retained."),
    ("moe_code", "tiny_perceptron/modern.py", "Read biased GELU DenseFFN and MoEFFN; bias-free router, four independently stored experts, renormalized top2, index_add dispatch, detached load times mean full probabilities. Exact report SHA match."),
    ("attention_code", "tiny_perceptron/attention.py", "Read GQA query/output64x64 and key/value32x64 weights, masks and SDPA path; all projection biases false."),
    ("data_code", "tiny_perceptron/data.py", "Read ByteTokenizer UTF8 encode/decode and eight special IDs, vocabulary264; CPU token sequence verified."),
    ("benchmark_code", "scripts/course_experiments/capstone_deployment.py", "Read benchmark_training_step150–202 and construction348–357. Fixed first24 rows,3/10 iterations, pre-timer zero_grad and synchronized forward+CE+.01aux+backward, no optimizer, preserved weights; peak includes warmups and live baseline. This exact file SHA matches deployment report."),
    ("training_code", "scripts/course_experiments/capstone.py", "Read stage initialization, AdamW, auxiliary logging/loss inclusion,600-step joint schedule and report/strict immediate-predecessor rules. No long training performed."),
    ("old_training_code", "docs/technical-reviews/artifacts/fact_v2_19_02_moe_run_architecture.py.txt", "Read exact historical report execution version, _train152–284 and run_moe427–486; latency scope includes gradient checks/clipping/AdamW, median latencies[3:]. Saved SHA exactly matches original report; no author history read."),
]:
    sources.append({"id": identifier, "kind": "repository_code", "title": rel, "path": rel, "sha256": sha(ROOT / rel),
                    "version": ("report revision48a4f3e912b483d70aee57c42c2aac226534a9a6" if identifier == "old_training_code" else "actual CPU inspection/execution SHA256:" + sha(ROOT / rel)),
                    "verified": True, "inspection_note": note})
sources.append({"id": "derivation", "kind": "derivation", "title": "Explicit parameter and bytes calculation", "verified": True,
                "details": "One FFN64x256 has33088=64*256+256+256*64+64; eight experts264704, routers512, other62912=33792 tables+24576 attention+320 RMS+4224 modality. Total328128; active=328128-2layers*2inactive*33088=195776; Dense64=62912+2*33088=129088; Dense80=42240+38400+400+5280+103200=189520. FP32*4=1312512; full FP32 weight+grad+two Adam moments*16=5250048, Dense2065408. See inspection.md for scope and exact breakdown."})
for identifier, target in [("exec_numeric", "numeric"), ("exec_snippet", "snippet"), ("exec_weights", "weights"), ("exec_dataset", "dataset"),
                            ("exec_pad", "pad"), ("exec_gpu", "gpu"), ("exec_cpu", "cpu_benchmark"), ("exec_old", "old_training"), ("exec_rows", "per_row")]:
    sources.append({"id": identifier, "kind": "execution", "title": "Independent actual CPU audit:" + target, "verified": True, "artifact_id": target})

claims = []


def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, scope, refs, aids, verification=None, status="verified"):
    entry = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope,
             "status": status, "evidence": refs, "artifact_ids": aids}
    if verification is not None:
        entry["verification"] = verification
    claims.append(entry)


def verification(expected, observed, details, tolerance=None, denominators=None):
    value = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance is not None:
        value["tolerance"] = tolerance
    if denominators is not None:
        value["denominators"] = denominators
    return value


claim("dense_moe", "concept", "Dense各位置用同一FFN；本成品MoE每層四份獨立FFN，每位置選兩份，共享字表/attention/output。",
      "前置段與主架構段；圖", "本模型FFN替換；router按token特徵選擇，不保證領域專長或品質優勢。",
      [evidence("mixtral", "§2.1 pp2–3; Figure1; sparse gated expert-sum formula", "每token路由少數獨立FFN，FFN以外部分仍共享。"), evidence("capstone_code", "default_config34–49; description110–121", "本成品4 experts/top2具體配置。")], ["inspection", "figure_render", "numeric"])
claim("byte_tokens", "software", "byte tokenizer以UTF8每個byte為位置，192位置不等於192中文字。", "前置段",
      "包含protocol特殊編號；中文常見BMP字3bytes，補充平面也可4bytes；不是每字固定3。",
      [evidence("rfc", "§3 encoding-range table", "UTF8每codepoint1–4octets。"), evidence("data_code", "ByteTokenizer.encode19–20", "逐octet加8形成ID。")], ["numeric", "rfc_text"],
      verification("中/文各3，𠀀4個byte IDs", "IDs長度3/3/4，A長度1；default max_length192", "真CPUencode與原始RFC對照。"))
claim("config_description", "software", "default_config是兩層width64、4experts/top2；description數實際參數，輸入與輸出表分開存。", "主配方段、Python與參數段",
      "只描述此配置；active是全共享表加選中expert的容量代理，未量精確FLOPs。",
      [evidence("capstone_code", "default_config34–49; CapstoneModel.description110–121", "兩層64/4/2與total-inactive公式。"), evidence("core_code", "TinyLM.__init__55–66", "tied=False，output與embedding分開。"), evidence("pytorch_module", "parameters2670–2697; named_parameters2699–2729", "遞迴且去重的參數iterator。")], ["snippet", "numeric", "weights"],
      verification("配置2/64/4/2、tied=False；描述數實際張量", "53張主模型FP32張量、embedding/output不同物件與storage、code/config與實報吻合", "教材逐字程式真跑；逐張量numel/dtype/storage核對。"))
claim("total", "numeric", "主MoE共有328128個參數。", "正文參數段與表MoE列", "含bias、兩層expert與所有共享模組；非檔案bytes。",
      [evidence("derivation", "inspection hand calculation", "264704expert+512router+62912其他。"), evidence("exec_numeric", "result.moe64.groups/parameter_tables", "實際枚舉全部參數。")], ["numeric", "weights", "snippet"], verification("328128", "328128", "手算後以原始程式和真權重逐tensor核對。", "整數精確相等"))
claim("active", "numeric", "主MoE的邏輯active代理為195776。", "正文參數段與表MoE列",
      "含全字表/輸出/共享規則，不是每token實際讀取格數、FLOPs、runtime memory或速度。",
      [evidence("derivation", "328128-2*2*33088", "扣除每層未選兩份expert。"), evidence("capstone_code", "description110–121", "結構容量代理實作。")], ["numeric", "snippet"], verification("195776", "195776", "one_expert33088；兩層各扣2份；top2。", "整數精確相等"))
claim("weight_bytes", "numeric", "主MoE FP32權重數字需1312512bytes。", "FP32說明與正文參數段", "只張量數值；容器/梯度/optimizer/activation/框架額外。原inference檔1345023bytes，公開整理版1327950bytes。",
      [evidence("exec_weights", "original.tensor_inventory/public.tensor_inventory", "真權重float32且每element4bytes，逐tensorbytes相加。"), evidence("derivation", "328128*4", "FP32weight-only bytes。")], ["numeric", "weights"], verification("1312512bytes", "1312512bytes", "原與公开模型53張量皆finiteFP32，whole-fileSHA不同但每tensorexact相同。", "整數精確相等"))
claim("dense64_count", "numeric", "同寬度層數Dense為129088參數，並非equal-active/equal-FLOPs比較。", "正文Dense參數段", "Dense每層1份FFN，MoE每位置2份；只是結構基準，不是已訓練公平品質比較。",
      [evidence("derivation", "62912+2*33088", "共享模組加兩份FFN、無router。"), evidence("core_code", "Block FFN conditional", "Dense branch每層一份DenseFFN。")], ["numeric", "snippet"], verification("129088", "129088；logical_active129088；FP32bytes516352", "同default_config僅dense=True；含bias。", "整數精確相等"))
claim("mixtral_cost", "concept", "Mixtral成本討論指出服務記憶體隨total参数增加，routing與單device多expert帶來overhead；可以分散儲存/使用專用kernel。", "Mixtral引用段",
      "原Mixtral47B大型系統的機制與限制；本單機小Python模型沒有重現專用運算或大型分散效能。",
      [evidence("mixtral", "§3 Size and Efficiency p4; §2.1 p3 specialized kernels/Expert Parallelism", "直接原文memory proportional sparse count、routing/memory loads和EP/kernel。")], ["mixtral_pdf", "mixtral_text", "inspection"])
claim("balance_concept", "concept", "Switch使用可微分auxiliary loss鼓勵分派均衡，不是指定expert學科專長。", "Switch引用段",
      "鼓勵而非保證均衡；Switch原式top1，capstone拓展top2且dropless；f不微分，P可微分。",
      [evidence("switch", "§2.2 p6 Eq4–6", "alpha*N*sum f_iP_i，以mean router probabilities可微分。"), evidence("moe_code", "MoEFFN.forward70–86", "top2 assignment負載除以T*k，load.detach()*importance。")], ["switch_pdf", "switch_text", "pad", "inspection"])
claim("switch_locator", "concept", "本節將Switch可微分負載平衡誤差定位於JMLR論文第2.1節。", "初讀正文：『Switch Transformer第2.1節』",
      "這是被明確連結的JMLR final版定位，不是arXiv其他版；不影響均衡誤差概念，但需修訂引用編號。",
      [evidence("switch", "§2.1 pp3–4 vs §2.2 p6 heading A Differentiable Load Balancing Loss and Eq4–6", "2.1是Simplifying Sparse Routing；負載平衡誤差實在2.2，原文contradicts locator。")], ["reviewed_body", "switch_pdf", "switch_text", "inspection"], status="contradicted")
claim("pad_balance", "software", "成品記錄load-balance誤差，PAD位置不進FFN或其auxiliary統計。", "Switch引用段",
      "valid mask正確提供時成立；auxiliary按valid輸入token，不等於僅assistant labels。測試是帶PAD的非空valid batch。",
      [evidence("capstone_code", "PadSafeBlock52–71", "normalized[valid]後才呼叫ffn並scatter。"), evidence("training_code", "training objective/history194–239", "CE+.01aux與auxiliary_before_update記錄。"), evidence("exec_pad", "records[0]/[1]", "補4PADcolumns仍8validtokens/16assignments且aux完全相同。")], ["pad", "weights"],
      verification("每層僅8valid，16top2 assignments；PAD不改aux且router可微分", "shape[1,8,64]；aux差0，valid logits最大差2.3841858e-6；router梯度非零有限", "8valid的2x5與2x9真checkpointforward/backward；手重建N*sum fP完全相等；logit容忍1e-5、aux1e-6。"))
claim("older_gpu_training", "empirical", "先前L4單種子短訓top2更新27.24ms，近active匹配Dense11.96ms，約慢2.28倍。", "架構段中15.13連結句",
      "舊MoE实验完整更新含梯度檢查/裁剪/AdamW；與本節no-optimizer機制不同。存檔聚合median未保存逐steplatencies，未在本CPU重測L4。單seed/180updates不能外推品質/速度普遍性。",
      [evidence("exec_old", "branches.*.training.warm_step_median_seconds; ratio", "保存實報聚合時間與比值核對。"), evidence("old_training_code", "_train152–284; run_moe427–486", "精確原執行版scope與前三步暖機排除。")], ["old_training", "old_moe_raw", "old_code", "inspection"],
      verification("27.24ms/11.96ms≈2.28", "27.24195099999349/11.96488600000123=2.2768249526147337", "原報告L4/FP32/seed42；top2 width64 vs Dense80 total207680近proxy208256。各180次AdamW更新337761targets，median177次。", "時間round2decimalms，ratio2decimal", {"seed": 42, "branches": 2, "updates_each": 180, "warmup_steps": 3, "timed_steps_each": 177, "batch_size": 16, "context_limit": 128, "effective_training_targets_each": 337761, "validation_targets": 39256, "test_targets": 41914}))
claim("mechanism_configuration", "software", "同batch24x94、FP32、warm3/measured10，前後向測量且不做optimizer更新，兩邊權重不變。", "本輪分派成本段與表後說明",
      "計時含forward/CE+.01aux/backward與CUDA同步；zero_grad/batch準備/model建立在計時外。Dense80隨機未等訓練，僅機制基準；原GPU環境與新CPU不同。",
      [evidence("benchmark_code", "benchmark_training_step150–202; construction348–357", "固定第一24rows、synchronized forward/backward、noneoptimizer、state相等檢查。"), evidence("exec_cpu", "moe/dense80", "真CPU13迴圈跑通且no updates/unchanged。"), evidence("exec_dataset", "denominators", "全部24IDs與shape真重建。")], ["cpu_benchmark", "gpu", "dataset", "gpu_raw"],
      verification("24x94、warm3/measured10、updates0、weightsunchanged", "兩邊精確24IDs/shape94、3/10、0 updates、unchanged true；1836valid inputs/474labels", "read原code+重建batch+CPU真13cycles；保存所有新CPUtimes並不当GPU數值。"))
claim("gpu_latency_moe", "empirical", "主MoE前向加反向GPU中位時間21.464ms。", "機制對照表MoE時間欄", "原L4同一fixedbatch的10samples，未重新量GPU，不含optimizer更新；只能說此機制此次比Dense80慢。",
      [evidence("exec_gpu", "checks.moe.raw_report.seconds/median_ms", "10savedtimes真statistics.median。")], ["gpu", "gpu_raw", "deployment_raw", "cpu_benchmark"],
      verification("21.464ms", "21.464479999998787ms", "原10samples中第5/6順序中位數；与报告median_seconds exact相等；按3小數ms呈現。", "顯示四捨五入±0.0005ms；rawmedian精確相等", {"batch_rows": 24, "grid_length": 94, "valid_input_positions": 1836, "effective_answer_targets": 474, "warmup": 3, "measured": 10, "optimizer_updates": 0, "seed": 42}))
claim("gpu_latency_dense", "empirical", "random Dense80前向加反向GPU中位時間11.338ms。", "機制對照表Dense時間欄", "無同等訓練或能力對照；w80總/active189520約比MoE195776代理少3.20%，非equalFLOPs。",
      [evidence("exec_gpu", "checks.dense80.raw_report.seconds/median_ms", "10savedtimes中位數重算。"), evidence("exec_numeric", "result.dense80.description", "真实模型189520count。")], ["gpu", "gpu_raw", "numeric", "cpu_benchmark"],
      verification("11.338ms；189520parameters", "11.338393500000876ms；189520parameters", "10timesmedian exact；参数枚举includebias。", "時間顯示±0.0005ms，count精確相等", {"batch_rows": 24, "grid_length": 94, "valid_input_positions": 1836, "effective_answer_targets": 474, "warmup": 3, "measured": 10, "optimizer_updates": 0, "seed": 42}))
claim("gpu_extra_moe", "empirical", "MoE相對起點GPUallocated peak增量為80174592bytes。", "機制表MoE額外記憶體欄",
      "PyTorchallocated peak-baseline，峰值含warmup及measurement13cycles；baseline有其他常駐物與stateclones；非reserved/driver/全卡或最低部署需求。",
      [evidence("exec_gpu", "checks.moe.extra_bytes/raw_report.cuda_*", "117236736-37062144=80174592。"), evidence("benchmark_code", "baseline/reset158–161; peak178; return197–201", "allocator起點扣除scope。")], ["gpu", "gpu_raw"],
      verification("80174592bytes", "117236736-37062144=80174592bytes", "原allocatedpeak减原baseline精確整數重算；CPU無CUDA未聲稱重測此peak。", "整數精確相等", {"iterations_within_peak_scope": 13, "warmup": 3, "measured": 10, "batch_rows": 24, "valid_input_positions": 1836, "optimizer_updates": 0}))
claim("gpu_extra_dense", "empirical", "Dense80相對起點GPUallocated peak增量為38589952bytes。", "機制表Dense額外記憶體欄",
      "與MoE相同allocator口徑但baseline不同；只比較新增配置，不是獨立resident模型全記憶體需求。",
      [evidence("exec_gpu", "checks.dense80.extra_bytes/raw_report.cuda_*", "108650496-70060544=38589952。"), evidence("benchmark_code", "baseline/reset158–161; peak178; return197–201", "allocatedpeak-baselineScope。")], ["gpu", "gpu_raw"],
      verification("38589952bytes", "108650496-70060544=38589952bytes", "原记录完整10times/context与bytes读核；CPU不替代L4peak。", "整數精確相等", {"iterations_within_peak_scope": 13, "warmup": 3, "measured": 10, "batch_rows": 24, "valid_input_positions": 1836, "optimizer_updates": 0}))
claim("trained_checkpoint", "software", "機制表的主MoE是實際joint已訓練權重，random Dense80只是機制基準。", "表列名稱與表後限制",
      "joint驗證選擇紀錄與benchmark來源SHA8f7e...相連；本次只載入驗證而不重做600updates，未聲稱Dense能力相同。",
      [evidence("exec_weights", "original.non_tensor_payload/public.all_tensors_equal_to_original", "真checkpoint strict載入、stagejoint、step600、有效targets249100與53tensorSHA。"), evidence("exec_rows", "result.validation/test", "84/90真CPU逐題生成與GPU保存ID/score全相同。"), evidence("benchmark_code", "construction348–357", "Dense由torchmanualseed42随机新建，未訓練。")], ["weights", "per_row", "cpu_validation", "cpu_test", "selection_raw", "joint_raw", "inventory"],
      verification("實際joint來源8f7e...而非random權重；Dense隨機", "原8f7e...与公开d85c...所有tensorexact同；jointstep600；真CPU75/84、78/90每題生成IDs无差异", "检查完整config/metadata/finite/statekey/SHAs，重建family切分，逐題重評；無新增訓練。"))
claim("adam_estimate", "numeric", "total*16粗估FP32權重加梯度及Adam兩份形狀相同的moment狀態。", "最後練習段",
      "常見全量FP32、AMSGrad false的parameter-shaped數值估算；scalarstep212bytes已被明確省略，亦不含activation/temp/allocator/framework。",
      [evidence("pytorch_adam", "Adam defaults37–43; state initialization152–189", "兩份zeros_likeexp_avg/exp_avg_sq与scalarstep；AMSGrad另有第三state。"), evidence("derivation", "328128*16 and129088*16", "主5250048/Dense2065408。")], ["numeric", "snippet", "exercise_code"],
      verification("主5250048/Dense2065408bytes；三份额外同形FP32数字", "練習stdout精確两值；zero-lrAdam每份moment与grad1312512bytes、coarse5250048；额外step212bytes", "真CPU原練習只替换乘数；独立zero-lrstateallocation不学习、不改权重。", "整數精確相等"))
claim("figure", "concept", "圖四套expert都儲存，本token只用兩套，共享規則仍需存和算，不保证更小記憶體/更快/專業分工。", "capstone_resources.svg XML與實際渲染",
      "示意本層四選二，不畫時間/参数比例；router共享於expert選項，但不表示層與層router權重綁定，global字表/output是概念分組。",
      [evidence("mixtral", "§2.1 Figure1/top2; §3 Size and Efficiency p4", "每token稀疏路由与total-memory/routeroverhead。"), evidence("switch", "§2.2 Eq4–6", "均衡統計不指定語義專長。")], ["figure_render", "inspection", "numeric", "pad"])

report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "19.2", "source": "course/chapters/19.md#19.2",
          "reviewer_task": "/root/integration_technical_coordinator/fact_v2_19_02", "reviewer_context": "fresh",
          "source_sha256": sha(OUT / (PREFIX + "section_19_2.md")),
          "figure_sha256": {"course/figures/capstone_resources.svg": sha(ROOT / "course/figures/capstone_resources.svg")},
          "verdict": "revise", "claims": claims, "sources": sources, "artifacts": artifacts,
          "issues": [{"claim_id": "switch_locator", "status": "contradicted", "details": "初讀正文把JMLR負載平衡誤差定位第2.1節；原PDF在第2.2節p6 Eq4–6。應改為2.2（最好列p6 Eq4–6）。協調者已通知新版定位標籤，本報告仍綁初讀9537d5de...真版本；不自動採新SHA，待另次完整重讀再判。"}],
          "checks": {
              "factual_accuracy": {"status": "revise", "details": "方法/軟體/數據成立；Switch章節號被原始PDF直接反證。主模型four/top2与普通Dense、PAD有效位置及全部tensor已真核。", "claim_ids": ["dense_moe", "config_description", "balance_concept", "pad_balance", "switch_locator", "trained_checkpoint"]},
              "numeric_verification": {"status": "pass", "details": "total328128、proxy195776、FP32bytes1312512、Dense129088/189520與Adam*16手算及真CPU一致。原L4全部10times的兩median及兩peak-baseline整数重算吻合；旧更新ratio2.27682495→2.28。", "claim_ids": ["total", "active", "weight_bytes", "dense64_count", "older_gpu_training", "gpu_latency_moe", "gpu_latency_dense", "gpu_extra_moe", "gpu_extra_dense", "adam_estimate"]},
              "figure_consistency": {"status": "pass", "details": "完整讀SVG XML、Inkscape1800x1140實際渲染後親看；四等尺寸expert框都寫需儲存，兩箭頭只到1/3且顏色選中兩位，shared/成本/無專長footer與正文同scope，無裁切或假比例。", "claim_ids": ["figure", "dense_moe", "active"]},
              "source_verification": {"status": "revise", "details": "匿名親讀Mixtralv1/SwitchJMLR原PDF、RFC3629與installedcommit官方PyTorch；本地core及deploymentcodeSHA吻合實報，舊MoE核精確execution原碼SHA。Switch第2.1引用定位錯誤，應2.2；其餘supports定位與範圍均已核。", "claim_ids": ["mixtral_cost", "balance_concept", "switch_locator", "byte_tokens", "config_description", "adam_estimate"]},
              "limitations": {"status": "pass", "details": "分清結構proxy與FLOPs/真正memory、FP32values與容器、旧完整更新與新nooptimizer13cycle機制、allocator增量與整卡/最低部署、随机Dense機制與同訓練品質。CPU重播與L4原測明分、單seed不外推；沒有同分或84/90重播推論普遍能力。", "claim_ids": ["active", "weight_bytes", "mixtral_cost", "balance_concept", "older_gpu_training", "mechanism_configuration", "gpu_latency_moe", "gpu_latency_dense", "gpu_extra_moe", "gpu_extra_dense", "trained_checkpoint", "adam_estimate"]}},
          "reviewed_version_note": "This initial review is deliberately bound to frozen first-read body via checker.sections. A newer locator-label body was announced during audit; it has not been reviewed in this turn and is not approved by hash replacement."}
(ROOT / "docs/technical-reviews/19.2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"verdict": report["verdict"], "source_sha256": report["source_sha256"], "claims": len(claims),
                  "sources": len(sources), "artifacts": len(artifacts), "issues": report["issues"]}, ensure_ascii=False, indent=2))
