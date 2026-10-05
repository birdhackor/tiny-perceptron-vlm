"""Assemble this reviewer's own report from the preserved independent evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PREFIX = OUT.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_16_6"
REV = "5c4886908584029761b579af026dcfb627c84070"
RUNREV = "48a4f3e912b483d70aee57c42c2aac226534a9a6"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def artifact_id(path):
    return "a_" + path.replace("/", "_").replace(".", "_").replace("-", "_")

data = json.loads((OUT / "cpu-results.json").read_text())
original_execution = json.loads((OUT / "original-execution.json").read_text())
original_env = json.loads((OUT / "original-environment.json").read_text())
cpu_execution = json.loads((OUT / "cpu-execution.json").read_text())
extraction = json.loads((OUT / "extraction.json").read_text())

artifacts = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file() or path.is_symlink():
        continue
    rel = path.relative_to(OUT).as_posix()
    kind = "code" if path.suffix == ".py" or rel.endswith(".py.txt") else "source_snapshot"
    description = "保留原始來源／本人核對的定位與版本：" + rel
    if rel == "frozen-chapter-16.md":
        description = "首次取得整章原始 bytes 的 frozen input actual snapshot；本審閱只判定16.6，非整章／導言核准。"
    elif rel == "section.md":
        description = "16.6 正式 source_sha256 所對應的原始 UTF-8 bytes。"
    elif rel.startswith("prerequisite-"):
        description = "實際讀取的必要前文原始 bytes；不代替該節的獨立技術審閱。"
    elif rel == "primary/efficiency-original.json":
        description = "完整未刪改的原始 GPU 測量 JSON；只讀 journal/cpu-results 所列測量、樣本與 provenance pointers。"
    elif rel == "verify_cpu.py":
        description = "實際執行的原 fence、有界 CPU 變體與具名原始量測 pointer 核對碼；不重訓／評測完整模型。"
    elif rel == "derivation.md":
        kind = "derivation"
        description = "獨立推導平均分母、28/3、7.5、11.5、token 份量與 ms/MiB 單位。"
    elif rel == "inspection-journal.md":
        description = "本人的實讀範圍、原始來源定位、存取／執行失敗與真實限制。"
    item = {"id": artifact_id(rel), "path": PREFIX + "/" + rel, "sha256": sha(path),
            "kind": kind, "description": description}
    if rel == "original-stdout.txt":
        item.update(kind="execution", command=original_execution["command"],
                    result="exit 0；未改原 fence：9.3333、9.3333、7.5。",
                    environment={"python": original_env["python"], "torch": original_env["torch"],
                                 "device": "cpu", "cuda_build": original_env["cuda_build"],
                                 "torch_git": original_env["torch_git_version"],
                                 "cwd": original_execution["cwd"]})
    elif rel == "cpu-stdout.txt":
        item.update(kind="execution", command=" ".join(cpu_execution["command_argv"]),
                    result="exit 0；原 fence、必要短 CPU 變體及原始量測核對 assertions 全部通過；cpu-results.json保存結果。",
                    environment=data["environment"] | {"cwd": cpu_execution["cwd"]})
    artifacts.append(item)

def aid(path):
    return artifact_id(path)

def official(identifier, title, path, inspection):
    return {"id": identifier, "kind": "official_source", "title": title,
            "url": "https://raw.githubusercontent.com/pytorch/pytorch/" + REV + "/" + path,
            "version": "PyTorch installed git revision " + REV + " (CPU wheel 2.14.1+cpu)",
            "verified": True, "checked_original": True, "accessed_on": "2026-10-05",
            "authority_reason": "PyTorch 官方 pytorch/pytorch 原始碼／原始 API 文件；固定為已安裝 wheel 回報的 git revision。",
            "inspection_note": inspection + "；親讀保留的 raw excerpt，完整 response 與 excerpt 指紋列於 primary retrieval metadata。"}

sources = [
    official("s_backward", "Tensor.backward original implementation and documentation", "torch/_tensor.py",
             "親讀lines566-625：chain rule、leaf .grad 累加與預設圖釋放；支持梯度計算而非參數更新"),
    official("s_torchdocs", "torch.tensor, mean, square, sum original API documentation", "torch/_torch_docs.py",
             "親讀tensor9582-9635、mean7195-7261、square11031-11052、sum11266-11321；支持leaf建立、elementwise平方與all-element reductions"),
    official("s_tensordocs", "Tensor mean, square, sum, item original API documentation", "torch/_tensor_docs.py",
             "親讀item2788-2805、mean3245-3252、square4890-4897、sum5027-5034；supports tensor method delegates and scalar extraction"),
    official("s_ce", "CrossEntropyLoss original class-index reduction formulas", "torch/nn/modules/loss.py",
             "親讀CrossEntropyLoss1200-1379，尤其class-index ignore_index指示函數、nonignored mean與sum公式；本節不使用class weights或probability-target平均"),
    official("s_amp", "Official gradient accumulation example", "docs/source/notes/amp_examples.md",
             "親讀Gradient accumulation126-165：有效大batch累積完才unscale/clip/step/update；例內按迴圈除只支持相同權重的micro-batches，異長token分母另由CE公式和獨立推導核對"),
    official("s_optimizer", "Optimizer.zero_grad and step original contracts", "torch/optim/optimizer.py",
             "親讀zero_grad1048-1109和真正step1117-1124；前者清梯度、後者一次更新；未以overload stub作證據"),
    official("s_clip", "clip_grad_norm_ original implementation and contract", "torch/nn/utils/clip_grad.py",
             "親讀185-232：所有參數梯度視為串成一向量的total norm，原地裁剪；支持最後裁剪整份梯度"),
    official("s_bn", "BatchNorm1d original training statistics", "torch/nn/modules/batchnorm.py",
             "親讀306-377：訓練均值／變異數在mini-batch上計算；拆批不保證保持相同per-example計算"),
    official("s_dropout", "Dropout original mask semantics", "torch/nn/modules/dropout.py",
             "親讀35-67：訓練時random zero，each forward call獨立選取；支持可能抽到不同遮罩，沒有聲稱每次必然不同"),
    official("s_memory", "CUDA allocated tensor memory and reset peak original APIs", "torch/cuda/memory.py",
             "親讀memory_allocated525-539、max_memory_allocated542-560、reset_peak_memory_stats381-397；bytes限tensor allocator，非nvidia-smi總量或單模型部署大小"),
    {"id": "s_python", "kind": "official_docs", "title": "Python 3.13 Common Sequence Operations",
     "url": "https://docs.python.org/3.13/library/stdtypes.html#common-sequence-operations",
     "version": "Python 3.13 documentation, original page accessed 2026-10-05",
     "verified": True, "checked_original": True, "accessed_on": "2026-10-05",
     "authority_reason": "Python 官方3.13語言文件，對應CPU環境的Python3.13。",
     "inspection_note": "親讀Common Sequence Operations表與Notes3-4：索引從0、slice取i<=k<j、省略邊界用0或len(s)；Tensor上相同切片另實際執行。"},
]
for identifier, title, rel, note in [
    ("s_arch", "Original efficiency experiment method", "primary/architecture-original.py",
     "AST定位後親讀original-code-inspection.json所列方法；尤其_train152-284、_accumulation_probe687-706、run_efficiency916-976。原檔SHA匹配GPU JSON /code_sha256；相關當前方法AST一致。"),
    ("s_common", "Original shared text/evaluation contracts", "primary/scripts_course_experiments_common.py",
     "親讀text_examples77-97、_nll106-119、evaluate_lm243-304、records_sha25673-74；完整SHA匹配GPU JSON；exact用原token身份、EOS另計，NLL按有效target總數。"),
    ("s_data", "Original ByteTokenizer, render_chat and pad_batch", "primary/tiny_perceptron_data.py",
     "親讀IGNORE/SPECIALS及ByteTokenizer14-28、render_chat54-68、pad_batch71-86；完整SHA匹配原實測，assistant byte targets加EOS，PAD和其他直接target=-100。"),
    ("s_model", "Original loss_sum, masked_loss and generation contracts", "primary/tiny_perceptron_model.py",
     "親讀loss_sum92-100、masked_loss103-105、generate109-131；完整SHA匹配原實測，flatten batch與sequence保留末軸vocab，ignore=-100，sum/count，generation遇EOS停止。"),
]:
    sources.append({"id": identifier, "kind": "repository_code", "title": title,
                    "path": PREFIX + "/" + rel, "sha256": sha(OUT / rel), "version": "Original measurement revision " + RUNREV,
                    "verified": True, "inspection_note": note})
sources.extend([
    {"id": "s_derivation", "kind": "derivation", "title": "Independent weighted-mean and gradient derivation", "verified": True,
     "details": "derivation.md：L=w²Σx²/N，dL/dw=2wΣx²/N；完整分母3、token共同分母、錯誤micro-batch等權、浮點與單位範圍。"},
    {"id": "s_cpu", "kind": "execution", "title": "Bounded CPU variants and preserved original measurement audit", "verified": True,
     "artifact_id": aid("cpu-stdout.txt")},
    {"id": "s_original", "kind": "execution", "title": "Unmodified original 16.6 Python fence", "verified": True,
     "artifact_id": aid("original-stdout.txt")},
])

def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}

def claim(identifier, kind, statement, location, scope, evidence, files, verification=None):
    c = {"id": identifier, "kind": kind, "statement": statement,
         "location": "course/chapters/16.md:" + location, "scope": scope, "status": "verified",
         "evidence": evidence, "artifact_ids": [aid(f) for f in files]}
    if verification:
        c["verification"] = verification
    return c

cpu_proof = ["verify_cpu.py", "cpu-stdout.txt", "cpu-results.json", "cpu-execution.json"]
claims = [
    claim("c_accumulation", "concept",
          "Gradient accumulation在相同權重處將各份梯度加到同一份.grad，最後更新一次；保留原樣本份量需按共同分母求和。",
          "238,260,262", "單裝置、可分解樣本／token loss、固定參數和計算規約；不推到任意batch-dependent或隨機模型的逐bit等價。",
          [ev("s_backward", "torch/_tensor.py:566-625", "backward用chain rule累加leaf gradients，需先清.grad"),
           ev("s_amp", "docs/source/notes/amp_examples.md:126-165", "effective batch累積完才做一次step／update"),
           ev("s_derivation", "derivation.md:weighted token mean", "S_j共同除N保留每target原權重")],
          ["primary/official-backward.py.txt", "primary/official-amp-gradient-accumulation.md", "derivation.md"]),
    claim("c_scalar_numbers", "numeric",
          "w=1、x=[1,2,3]、零目標：L=14/3、梯度28/3；拆1+2正確累積9.3333，兩份平均再平均的梯度7.5。",
          "240,258", "這個一維平方代價手工算例；每筆一個loss、N=3，9.3333是梯度而非loss。",
          [ev("s_derivation", "derivation.md:first three paragraphs", "逐項代入和錯誤權重[1/2,1/4,1/4]"),
           ev("s_cpu", "cpu-results.json:/split_1_2", "float64獨立核對28/3、28/3、7.5")],
          cpu_proof + ["derivation.md"],
          {"method": "executed", "expected": "loss14/3；gradients28/3,28/3,7.5；原FP32四位小數9.3333,9.3333,7.5。",
           "observed": "float64 grads9.333333333333332,9.333333333333332,7.5；原fence输出一致。",
           "details": "sum/mean沿唯一example軸；wrong實際分配1/2,1/4,1/4，不能將9.3333當誤差平均。",
           "tolerance": "float64與28/3 abs<=1e-12；原示範按round(...,4)精確核對。"}),
    claim("c_fence_api", "software",
          "原Python fence實際執行：三個獨立leaf初值1；tensor切片、square、sum/mean、backward累加、grad.item和round產生宣稱输出，沒有step更新。",
          "242-258,262", "相關API合組coverage：torch.tensor(requires_grad=True)、x[:1]/x[1:]、逐元素乘平方、全元素sum/mean、len(x)、Tensor.backward/.grad/.item、Python round；僅CPU梯度示範。",
          [ev("s_original", "original-stdout.txt; original-execution.json", "未改fence成功執行，输出9.3333/9.3333/7.5"),
           ev("s_cpu", "cpu-results.json:/original_fence_no_update", "三leaf仍是1且切片值[1]/[2,3]"),
           ev("s_torchdocs", "tensor9582-9635; mean7195-7261; square11031-11052; sum11266-11321", "建leaf、逐元素平方、全元素平均／和"),
           ev("s_tensordocs", "item2788-2805; mean3245-3252; square4890-4897; sum5027-5034", "method委派與single-element.item取Python數字"),
           ev("s_backward", "torch/_tensor.py:566-625", "leaf.grad累加而未執行optimizer update"),
           ev("s_python", "Common Sequence Operations table and Notes3-4", "0-based索引與不含終點的slice、len")],
          cpu_proof + ["fence-1.py", "original-stdout.txt", "original-execution.json", "original-environment.json", "original-stderr.txt"],
          {"method": "executed", "expected": "原fence退出0；四位小數9.3333、9.3333、7.5；權重保持1。",
           "observed": "獨立helper和CPU verifier皆退出0；三權重1.0；切片assertions通過。",
           "details": "Python3.13.5／torch2.14.1+cpu；以bootstrap執行未改fence，再在獨立CPU verifier核對權重與切片；stdout/env/code/命令皆留永久證據。"}),
    claim("c_mean_variants", "numeric",
          "改拆2+1仍得到正確梯度9.3333而錯誤等份平均11.5；有效數相同時份平均可等權，1與10個loss全1時兩種平均偶然皆1。",
          "260,264", "平均的保證條件與巧合反例，非不等token數時數值必定不同；換拆法只改份量計算。",
          [ev("s_derivation", "derivation.md:exercise and weighted-mean paragraphs", "錯誤權重[1/4,1/4,1/2]給11.5；等count與不等count全1推導"),
           ev("s_cpu", "cpu-results.json:/exercise_split_2_1,/unequal_count_coincidence; verify_cpu.py:equal_parts assertion", "改拆法、11個loss1、等count變體實際核對")],
          cpu_proof + ["derivation.md"],
          {"method": "executed", "expected": "2+1 grads28/3,28/3,11.5；counts1/10全1時兩個mean=1；等count份平均等於全平均。",
           "observed": "9.333333333333332,9.333333333333332,11.5；兩個mean都是1.0；等countassert通過。",
           "details": "11.5=2*(1/4+4/4+9/2)；等數例每份2個值；不等count反例不推成權重正確。",
           "tolerance": "float64 grads abs<=1e-12；全1與等count例精確相等。"}),
    claim("c_target_denominator", "concept",
          "本課助手答案的token-mean loss只計有效label；PAD及其他ignore位置不計直接答案loss。跨micro-batch以有效target總數作共同分母，而非loop次數或PAD表格元素數。",
          "260,269", "本課無class weights的class-index cross entropy／assistant SFT；一般任務若定義其他加權目標需按其規約。",
          [ev("s_ce", "CrossEntropyLoss1200-1379:class-index equations and ignore_index", "sum排除ignore；mean按非ignored class-index targets計算，無class weight时就是target數"),
           ev("s_data", "render_chat54-68; pad_batch71-86; IGNORE=-100", "助手content+EOS作target，PAD／其他直接target忽略"),
           ev("s_model", "loss_sum92-100; masked_loss103-105", "flatten(batch,sequence)保留vocab末軸，loss_sum配有效count"),
           ev("s_cpu", "cpu-results.json:/valid_answer_token_variant", "異長token的shared denominator和ignored-position零直接梯度核對")],
          cpu_proof + ["primary/official-CrossEntropyLoss.py.txt", "derivation.md"]),
    claim("c_training_order_limits", "concept",
          "正確順序先zero_grad、分份backward、整份clip、一次step；逐份step使用不同權重。Batch統計與dropout遮罩使任意模型拆批不保證等價。",
          "262", "只說條件與可能差別；clip限制梯度norm而不保證任意optimizer更新或整體訓練穩定，dropout不宣稱每次遮罩必定不同。",
          [ev("s_optimizer", "zero_grad1048-1109; step1117-1124", "清除梯度與更新參數分開"),
           ev("s_clip", "clip_grad_norm_185-232", "整個參數梯度串成向量的共同norm後原地裁剪"),
           ev("s_amp", "Gradient accumulation126-165", "累積完才unscale／clip／step"),
           ev("s_bn", "BatchNorm1d306-377", "mini-batch共同計均值／變異數"),
           ev("s_dropout", "Dropout35-67", "training mask每forward獨立隨機選取"),
           ev("s_cpu", "cpu-results.json:/one_vs_two_demonstration_updates", "一次SGD累積與逐份SGD更新確實不同，僅有界示範")],
          cpu_proof + ["primary/official-clip_grad_norm_.py.txt", "primary/official-amp-gradient-accumulation.md", "primary/official-BatchNorm1d.py.txt", "primary/official-Dropout.py.txt"]),
    claim("c_original_probe", "empirical",
          "原文字機制核對三段助手內容拆1+2，有效目標7/13，共同分母20；原保存梯度最大差2.08616e-7。正式SFT則batch8拆3+5，全部backward後clip與step。",
          "269", "讀取既有單seed L4原始測量與其原方法；小probe助手內容text例和正式問答SFT不同，不當成SFT評測。",
          [ev("s_arch", "_mechanism_examples709-715; _accumulation_probe687-706; _gradients558-568; _gradient_error571-574; _train152-284; run_efficiency955-966", "probe抽assistant內容，fixed copies/shared denominator；正式sft3+5與更新順序"),
           ev("s_cpu", "primary/efficiency-original.json:/results/accumulation; cpu-results.json:/existing_measurement_audit/accumulation_probe", "原始7/13/20、梯度最大差的pointer、共同分母相加和五位科學記號核對")],
          cpu_proof + ["primary/efficiency-original.json", "primary/architecture-original.py", "primary/original-code-inspection.json"],
          {"method": "executed", "expected": "micro_sizes[1,2]；valid counts7,13，N20；maxgradient2.08616e-7；正式batch8,micro[3,5]。",
           "observed": "原測量counts[7,13],N20,maxgradient2.086162567138672e-7；原方法與正式原量測micro[3,5]一致。",
           "details": "讀取原始aggregate並核20=7+13與格式，不重跑GPU／checkpoint；原程式max取所有可求導權重梯度差的絕對最大值。",
           "denominators": {"probe_examples": 3, "probe_micro_examples": [1, 2], "probe_valid_targets": [7, 13], "probe_shared_denominator": 20, "formal_batch_examples": 8, "formal_micro_examples": [3, 5], "seeds": 1}}),
    claim("c_original_update_and_quality", "empirical",
          "原普通與累積各40次更新、2328有效目標；保存權重差1.38730e-5；heldout test NLL兩者四捨五入0.41778、raw-token完整答案各5/10。",
          "271", "既有L4 seed42的特定小SFT；2328是40步累計target讀取次數，不是獨立樣本數；0.41778是test有效目標平均NLL，不是training final NLL。",
          [ev("s_arch", "_train187-223; run_efficiency943-972", "同base深拷貝、同seed抽樣、各更新40步；權重max按state_dict全部table差計算"),
           ev("s_common", "_nll106-119; evaluate_lm243-284", "有效target平均NLL，greedy生成raw token身份exact與EOS分開"),
           ev("s_cpu", "primary/efficiency-original.json:/results/update_variants/{ordinary,accumulated}/training/{steps,optimizer_updates,effective_tokens}; /heldout/test/{nll_sum,nll,effective_tokens,matches,records,samples}; /accumulated/weight_max_error_from_ordinary_after_updates", "原始40/2328和權重差；由10筆generated_ids重數5 matches、10EOS、69target，nll_sum/69和四捨五入核對")],
          cpu_proof + ["primary/efficiency-original.json", "primary/scripts_course_experiments_common.py", "primary/architecture-original.py"],
          {"method": "executed", "expected": "40成功更新/無skip；2328targets；maxweight格式1.38730e-5；test69targets、5/10matches、NLL0.41778。",
           "observed": "各40update、0skip、2328targets；原保存1.3872981071472168e-5；每支raw token matches5/10、EOS10/10、69targets；28.82705307006836/69=0.4177833778270777，28.826644897460938/69=0.4177774622820426。",
           "details": "原始测量aggregate audit，不加载权重或重评模型；对每个sample按ByteTokenizer编码expected，比到EOS前token身份，EOS另计；训练最终NLL约0.01878与test0.41778明确区分。",
           "denominators": {"updates_each": 40, "training_effective_target_occurrences_each": 2328, "batch_examples": 8, "test_records_each": 10, "test_effective_targets_each": 69, "training_unique_records": 45, "seeds": 1}}),
    claim("c_original_time_and_memory", "empirical",
          "原每步中位数14.070/24.064ms；峰值相对各自起点新增8.824/6.474MiB；完整峰值75.988/74.724MiB、起点67.164/68.250MiB，不能当单独部署大小。",
          "271", "特定L4原更新测量；median排除前三步，CUDA tensor allocated peak含同进程驻留对象，increment=peak-baseline；没有重测时间或GPU内存，也未从缺失latency样本重造median。",
          [ev("s_arch", "_train176-182,193-194,220-236,275-281; run_efficiency923-975", "同步timing、排除前三步median、reset peak后baseline与peak差、模型序列驻留；micro可先释放各份图"),
           ev("s_backward", "torch/_tensor.py:retain_graph argument", "默认backward释放本份图，故不用同时保留全部份中间计算图"),
           ev("s_memory", "memory_allocated525-539; max_memory_allocated542-560; reset_peak_memory_stats381-397", "allocated tensors bytes与reset后的峰值范围"),
           ev("s_cpu", "primary/efficiency-original.json:/results/update_variants/{ordinary,accumulated}/training/{warm_step_median_seconds,memory_allocated_before_bytes,peak_memory_allocated_bytes,peak_additional_allocated_bytes}; cpu-results.json:/existing_measurement_audit/variants", "秒×1000、bytes/2^20、peak-before、各3位小数均由原raw核对")],
          cpu_proof + ["primary/efficiency-original.json", "primary/official-memory_allocated.py.txt", "primary/official-max_memory_allocated.py.txt", "primary/official-reset_peak_memory_stats.py.txt", "derivation.md"],
          {"method": "executed", "expected": "ms14.070/24.064；incrementMiB8.824/6.474；full75.988/74.724；baseline67.164/68.250。",
           "observed": "全部单位与3位小数一致；ordinary79678976-70426624=9252352bytes，accumulated78353920-71565312=6788608bytes。",
           "details": "timing是原保存median不是新CPU测量；原40步各略过前三步后37步median，但individual latencies不在JSON，未声称重新求得median；1MiB=1048576bytes。追加峰值较低与累计图释放的方法一致，比较只限本run。",
           "denominators": {"updates_each": 40, "excluded_warmup_updates": 3, "median_updates_each_by_method": 37, "MiB_bytes": 1048576, "milliseconds_per_second": 1000, "seeds": 1}}),
]

report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "16.6",
    "source": "course/chapters/16.md#16.6", "source_sha256": extraction["source_sha256"],
    "reviewer_task": TASK, "reviewer_context": "fresh", "verdict": "pass", "reviewed_on": "2026-10-05",
    "figure_sha256": {}, "issues": [], "sources": sources, "artifacts": artifacts, "claims": claims,
    "checks": {
        "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "本人逐项核实梯度累加、共同分母、API语义、操作顺序与等价条件，原实测限定其真正证据范围。"},
        "numeric_verification": {"status": "pass", "claim_ids": ["c_scalar_numbers", "c_mean_variants", "c_original_probe", "c_original_update_and_quality", "c_original_time_and_memory"], "details": "手工推导+原fence+短CPU变体；原raw测量pointer、10个token样本、69targets、NLL商、40/2328原aggregate、ms/MiB换算亲核，未重训或新造缺失观测。"},
        "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "16.6无任何image/SVG/animation/screen引用或画面主张。值、slice与example权重在原文和代码明确显示，无需想象空间素材；没有作图render或desktop/mobile页面检查，亦不代核必要前文中其他节的图。"},
        "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "亲读官方同torch git版本的raw源与Python3.13文档；保存实际locator/response与excerpt SHA。PyTorch文档403后有相应官方原码替代；原测量版本代码SHA匹配且当前相关AST相同。未用旧报告或作者结果解释。"},
        "limitations": {"status": "pass", "claim_ids": ["c_accumulation", "c_target_denominator", "c_training_order_limits", "c_original_probe", "c_original_update_and_quality", "c_original_time_and_memory"], "details": "明确梯度示范不更新、toy SGD不是模型能力验收；拆批等价需固定per-example计算，BN/dropout另核；单seed/GPU原量测不逐bit承诺；原aggregate无per-step samples/checkpoint重算，原始持久证据不由outputs代替。"},
    },
    "review_scope": {
        "primary": "16.6 entire original section, sole original Python fence, all substantive prose including optional empirical paragraph",
        "prerequisites_read": ["16.1", "7.3", "7.7", "W.6"], "introduction_reviewed": False,
        "frozen_full_chapter_input": {"path": PREFIX + "/frozen-chapter-16.md", "sha256": sha(OUT / "frozen-chapter-16.md"), "meaning": "实际首次读取的full-file原bytes快照；不冒充其他节后来版本或全章核准。"},
        "read_journal": PREFIX + "/inspection-journal.md", "visual_review": "not applicable to primary16.6; no render attempted",
        "existing_measurement_scope": "原CUDA JSON audit only; test sample/count/unit arithmetic independently recomputed, no raw model checkpoints or complete evaluation/training rerun",
        "unresolved_substantive_questions": [], "contamination": "none; no prohibited result interpretations exposed",
    },
}

# Do not validate an old report: write this task's complete new report first.
path = ROOT / "docs/technical-reviews/16.6.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
own = json.loads(path.read_text())
assert own["reviewer_task"] == TASK and own["source_sha256"] == extraction["source_sha256"]
print(json.dumps({"report": str(path), "reviewer_task": own["reviewer_task"], "verdict": own["verdict"],
                  "source_sha256": own["source_sha256"], "report_sha256": sha(path), "claims": len(claims), "artifacts": len(artifacts)}, ensure_ascii=False))
