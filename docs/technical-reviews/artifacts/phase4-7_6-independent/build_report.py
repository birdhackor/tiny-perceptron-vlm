"""Write this independent review from my personally inspected evidence."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
PREFIX = BASE.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_7_6"
COMMIT = "5c4886908584029761b579af026dcfb627c84070"
VERSION = "PyTorch 2.14.1+cpu; upstream git commit " + COMMIT
repo_version = "working-tree bytes at HEAD 022dc9b2ffde92c14ca133406849c1c378bb8e8f on 2026-10-05"
extraction = json.loads((BASE / "original-run/extraction.json").read_text())
original_execution = json.loads((BASE / "original-run/execution.json").read_text())
probe = json.loads((BASE / "probe-results.json").read_text())
environment = {"python": "3.13.5", "torch": "2.14.1+cpu", "torch_git_version": COMMIT,
               "device": "cpu", "cuda_build": "None", "cuda_available": "False"}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

commands = [
    {"command": ".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.6 --output outputs/reviewer-tools/phase4-7_6-independent-original --execute --timeout 60", "exit_code": 0, "stdout": "original-run/stdout.txt", "stderr": "original-run/stderr.txt", "facts": "original-run/execution.json", "permanent_copy": "prepare_evidence.py copied raw files unchanged"},
    {"command": ".venv/bin/python " + PREFIX + "/cpu_probe.py > " + PREFIX + "/probe.stdout.json 2> " + PREFIX + "/probe.stderr.txt", "exit_code": 0, "stdout": "probe.stdout.json", "stderr": "probe.stderr.txt", "facts": "probe-results.json", "environment": "probe-environment.json"},
    {"command": ".venv/bin/python " + PREFIX + "/acquire_sources.py > " + PREFIX + "/acquire.stdout.json 2> " + PREFIX + "/acquire.stderr.txt", "exit_code": 0, "receipt": "source-acquisition.json"},
    {"command": ".venv/bin/python " + PREFIX + "/acquire_exact_sources.py > " + PREFIX + "/acquire-exact.stdout.json 2> " + PREFIX + "/acquire-exact.stderr.txt", "exit_code": 0, "receipt": "exact-source-acquisition.json"},
    {"command": ".venv/bin/python " + PREFIX + "/prepare_evidence.py > " + PREFIX + "/prepare.stdout.json 2> " + PREFIX + "/prepare.stderr.txt", "exit_code": 0, "receipt": "input-provenance.json"},
]
write(BASE / "commands.json", {"cwd": str(ROOT), "shell": "bash login:false", "commands": commands})
files = [p for p in sorted(BASE.rglob("*")) if p.is_file() and p.name != "manifest.json" and not p.name.startswith("checker") and not p.name.startswith("report-write")]
write(BASE / "manifest.json", {"files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in files], "models_or_dataset_files": [], "all_inputs_retained_under_docs": True})
files.append(BASE / "manifest.json")
artifacts = []
artifact_ids = {}
for i, p in enumerate(files):
    rel = p.relative_to(BASE).as_posix()
    identifier = "a" + str(i + 1)
    artifact_ids[rel] = identifier
    kind = "source_snapshot" if "sources/" in rel or "inputs/" in rel else "code" if p.suffix == ".py" else "source_snapshot"
    item = {"id": identifier, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "kind": kind,
            "description": "永久保存本節實際 " + rel + "；用途見 commands.json / input-provenance.json / 各 claim locator。"}
    if rel == "original-run/execution.json":
        item.update(kind="execution", command=commands[0]["command"], result="exit 0; fence 1 executed; before=after=0.8635237216949463; stdout/stderr/environment retained", environment=environment)
    if rel == "probe-results.json":
        item.update(kind="execution", command=commands[1]["command"], result="exit 0; all float64 math, gradient, PAD-label, all-ignored and two-length-batch assertions passed", environment=environment | {"dtype": "float64", "optimizer_steps": "0", "model_forward": "none"})
    artifacts.append(item)

sources = []
def official(identifier, title, filename, upstream, note):
    sources.append({"id": identifier, "kind": "official_source", "title": title, "url": "https://raw.githubusercontent.com/pytorch/pytorch/" + COMMIT + "/" + upstream,
        "version": VERSION, "accessed_on": "2026-10-05", "authority_reason": "PyTorch 官方維護的 pytorch/pytorch 原始碼與 API docstrings，commit 完全等於本次 CPU wheel 的 torch.version.git_version。",
        "verified": True, "checked_original": True, "inspection_note": note + "；親讀 raw HTTPS 原始內容，完整原始 bytes 和 acquisition receipt 已保存在 sources/" + filename,
        "artifact_id": artifact_ids["sources/" + filename]})
official("pt_loss", "CrossEntropyLoss 原始公式與 ignore_index 定義", "loss-installed.py", "torch/nn/modules/loss.py", "親讀 1200–1275 class CrossEntropyLoss 的 logits、類別索引、ignored indicator、sum/mean 公式，以及 1320–1380 的示例/限制。只引用 class-index、無權重、無 smoothing 分支")
official("pt_nll", "CPU NLL/CE 原始前向與反向實作", "LossNLL-installed.cpp", "aten/src/ATen/native/LossNLL.cpp", "親讀 175–310：ignored target skip、num_ignored、未加權 total_weight=batch_size-num_ignored、mean 除 total_weight；346–435：ignored target 不寫梯度、grad_input.zero_；630–666：CE 無 smoothing = log_softmax + NLL")
official("pt_functional", "F.cross_entropy 原始 API 契約", "functional-installed.py", "torch/nn/functional.py", "親讀 3478–3603：ignore_index 預設 -100、ignored target 不貢獻 input gradient、nonignored average、class-index shape、reduction='sum' 與 C++ dispatch")
official("pt_torch_api", "torch.tensor / zeros / cat / allclose / reshape / sum 原始 API", "torch-docs-installed.py", "torch/_torch_docs.py", "親讀 allclose 835–874 的 atol+rtol 公式；cat 2532–2570 的指定既有軸串接；tensor 9583–9625 的資料與 leaf/requires_grad=False；zeros 12618–12652 的 size/0；reshape 9834–9862；sum 11267–11326 的全元素/指定軸加總。每個原 fence 的 torch API 都已覆蓋")
official("pt_tensor_api", "Tensor.item / reshape / sum 原始 API", "tensor-docs-installed.py", "torch/_tensor_docs.py", "親讀 item 2788–2805：單元素轉 Python number、非可微；reshape 4163–4179：元素不變改 shape；sum 5027–5034 轉指 torch.sum；原 fence .item 僅展示結果，不是 backward")
official("pt_padding", "pad_sequence 的變長 batch padding 契約", "rnn-installed.py", "torch/nn/utils/rnn.py", "親讀 pad_sequence 405–460：將不同 L 序列排成 B×T，T 是最大長度，以 padding_value 右側補齊；此權威契約支持一般 padding 概念，不冒充本課 pad_batch 實作")
def repository(identifier, name, note):
    sources.append({"id": identifier, "kind": "repository_code", "title": name + " 實際契約", "path": PREFIX + "/inputs/" + name, "sha256": sha(BASE / "inputs" / name), "version": repo_version, "verified": True, "inspection_note": note})
repository("repo_model", "tiny_perceptron/model.py", "全文親讀；92–105 loss_sum 用 labels!=IGNORE 數有效項，reshape 同一位置 flatten，CE reduction=sum，再由 masked_loss 除 count；不再 shift；95–96 count=0 提前拒絕。")
repository("repo_data", "tiny_perceptron/data.py", "全文親讀；10 IGNORE=-100；18 ByteTokenizer.pad_id=0；54–68 render_chat 只 shift 一次；71–86 pad_batch x補0、y補-100、valid補False，保留真實 user 前文的 valid=True，並拒絕某筆無有效答案。")
repository("repo_attention", "tiny_perceptron/attention.py", "全文親讀；10–18 attention_mask 用 causal & valid-key 許可，與 loss 的 labels mask 無關。真實 user token 即使 label=-100 仍為可讀 key。")
repository("repo_train", "scripts/train.py", "只親讀 320–352；343–346 將 pad_batch 的 x/valid 傳 model，另把 y 傳 masked_loss。這是靜態契約核對，沒有執行訓練入口。")
sources += [
    {"id": "run_original", "kind": "execution", "title": "本節原始 fence CPU 實跑", "verified": True, "artifact_id": artifact_ids["original-run/execution.json"]},
    {"id": "run_probe", "kind": "execution", "title": "獨立有界 CPU 算例／梯度／batch 核對", "verified": True, "artifact_id": artifact_ids["probe-results.json"]},
    {"id": "math", "kind": "derivation", "title": "四候選 softmax NLL 獨立代入", "verified": True, "details": "p1=e²/(e²+3)=0.7112345942275938；l1=ln(e²+3)-2=0.3407529539131313 nats；l2=ln4=1.3862943611198906 nats；sum=1.727047315033022，count=2，mean=0.863523657516511；多加 label0 的零 logits：第三项=ln4，count=3，mean=1.0377805587176374；舍入三位=0.711/0.341/1.386/0.864/1.038；dL/dz=(softmax-onehot)/count。實際 math.exp/log 計算保存在 cpu_probe.py 与 probe-results.json。"},
]
def ev(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}
def claim(identifier, kind, statement, location, scope, evidence, files, verification=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope, "status": "verified", "evidence": evidence, "artifact_ids": [artifact_ids[f] for f in files]}
    if verification:
        item["verification"] = verification
    return item
original = "original-run/execution.json"
probe_file = "probe-results.json"
claims = [
claim("padding_convention", "concept", "變長對話 batch 以 PAD 補齊；本課輸入 PAD ID0 與答案忽略值-100用途不同，補齊格不是真實答案。", "course/chapters/07.md:184–186", "本課 ByteTokenizer/render_chat/pad_batch 的約定。PAD 所謂輸入專用指正常答案標籤不使用它，沒有宣稱從 logits 候選或 softmax 分母刪除 PAD 類別；也不把所有 tokenizer 的0視為PAD。", [ev("pt_padding", "pad_sequence 405–440", "變長序列以 padding_value 補成同一 T"), ev("pt_functional", "cross_entropy 3478–3503", "ignore_index 是答案值，不是輸入 padding 自動規則"), ev("repo_data", "10/18/54–86", "本課真實輸入ID與答案忽略值分欄"), ev("run_probe", "batch_contract padded_x/padded_y/valid", "短序列右補兩個0、兩個-100與False")], [probe_file]),
claim("original_numbers", "numeric", "第一格ID1機率約0.711、NLL約0.341；第二格機率1/4、NLL約1.386，兩格平均約0.864。", "course/chapters/07.md:192–193,202", "人工四候選 logits 的自然對數交叉熵；不是模型能力、訓練或評測指標。", [ev("pt_loss", "CrossEntropyLoss 1217–1243 class-index formula", "softmax再負自然對數，無權重平均只數非ignored項"), ev("math", "p1/l1/l2/sum/count/mean", "獨立代入與三位舍入"), ev("run_original", "stdout.txt", "float32 原 fence =0.8635237216949463"), ev("run_probe", "independent_math/original_observed", "float64 mean=0.8635236575165111")], [original, probe_file], {"method": "executed", "expected": "0.7112345942 / 0.3407529539 / 1.3862943611 / 0.8635236575；三位舍入與原文一致。", "observed": "float64逐項與math公式差<=1e-12；原float32=0.8635237216949463，差6.42e-8。", "details": "[B,T,C]=[1,2,4]，答案[1,2]；單位nats/有效目標，count=2。", "tolerance": "float64絕對1e-12；原float32絕對1e-6；三位舍入精確。"}),
claim("original_api", "software", "tensor建立a與labels，zeros建立三格候選，cat dim=1串接成[1,5,4]，masked_loss與.item/allclose核對兩平均。", "course/chapters/07.md:189–199,202", "涵蓋原 fence 全部 PyTorch API、repo helper、軸與label shape；原碼僅求loss與assert，沒有backward、optimizer或模型前向。", [ev("pt_torch_api", "tensor 9583–9625; zeros 12618–12652; cat 2532–2570; allclose 835–874", "建立/填零/既有dim串接/容差比較逐API"), ev("pt_tensor_api", "item 2788–2805; reshape 4163–4179; sum 5027–5034", "結果轉number，helper不改元素而flatten/count"), ev("repo_model", "92–105", "helper flatten到[N,C]與[N]、CE求和後除有效數"), ev("run_original", "execution.json exit_code/attempted_fences/stdout.txt", "原fence1完整執行成功")], [original, "original-run/fence-1.py", "original-run/environment.json", probe_file], {"method": "executed", "expected": "a[1,2,4]→b[1,5,4]；labels[1,2]→extended[1,5]；before=after，assert通過。", "observed": "exit0；before=after=0.8635237216949463；獨立shape與分母檢查通過。", "details": "PyTorch2.14.1+cpu，API原始docstrings同git commit；allclose預設rtol1e-5/atol1e-8，本次數值精確相同。"}),
claim("ignore_invariance", "concept", "額外label=-100的格不加入loss分子或有效分母；原兩格固定時，忽略格的有限logits怎樣變都不增加直接答案代價。", "course/chapters/07.md:186,202,204", "無class weights、無label smoothing、整數label與有限logits；ignore是目標位置mask，不是刪掉一個候選類別。梯度结論只關於這些位置logits的直接loss。", [ev("pt_loss", "1217–1243 ignored indicator/sum/mean", "ignored項不加入loss或有效mean"), ev("pt_nll", "175–310/346–435/630–666", "target ignore skip與有效count，梯度初始化0且skip ignored"), ev("repo_model", "92–105", "sum/count兩者同用-100"), ev("run_probe", "extended_observed", "三格全0或不同极端有限分數的loss與原兩格梯度相同；ignored梯度全0")], [probe_file]),
claim("pad_target_direction", "concept", "若PAD格label誤設0，它被當正常類別0答案並產生新的梯度方向；只填輸入0不能讓loss自動識別padding。", "course/chapters/07.md:204", "0是四候選的合法class index，而ignore_index=-100。原示範沒有實際參數更新；新的更新方向指若反向時得到的loss梯度。", [ev("pt_functional", "3478–3520 ignore_index/target", "CE接受logits與target，僅指定ignored target被忽略"), ev("pt_nll", "193–207/257–267/346–435", "0通過合法class index檢查並累加loss/寫梯度"), ev("repo_data", "78–83", "填充輸入和答案分別設定"), ev("run_probe", "pad_label_variation new_gradient", "第三格梯度[-0.25,1/12,1/12,1/12]，原梯度縮2/3")], [probe_file]),
claim("scope_of_demo", "software", "原段只驗證loss统计，不證明模型讀入PAD後前向預測相同，讀取許可與位置另在7.7檢查。", "course/chapters/07.md:206,213", "親讀本節与前置7.1–7.5/相鄰7.7–7.9文字的範圍限制；7.7圖與算法不是本節正式核實對象。沒有跑完整訓練recipe、下載模型/資料、評測既有權重或GPU。", [ev("run_original", "fence-1.py 1–11; execution.json", "原碼全為人工logits/labels/loss/assert，沒有model()或backward/step"), ev("repo_model", "68–105", "TinyLM前向與loss helper是分開操作；masked_loss不處理valid或positions"), ev("repo_attention", "10–18", "讀取mask另外建立")], [original, "original-run/fence-1.py", probe_file], {"method": "executed", "expected": "執行原loss fence成功；不产生模型前向或參數更新的證據。", "observed": "原helper attempted_fences=[1]且exit0；原碼無model/backward/optimizer；probe也只對人工logits求梯度。", "details": "教材限制正確；短CPU替代只核算例、梯度與helper契約，未把它擴大為整體模型等價/能力驗收。"}),
claim("empty_mean", "software", "整批有效目標數0不能算平均；masked_loss在除以有效數前拒絕全ignored labels。", "course/chapters/07.md:206", "本課helper拒絕此batch；原生PyTorch無權重CE sum為0而mean為NaN，不能把NaN當正常結果。", [ev("repo_model", "94–96/103–105", "count=0先raise ValueError"), ev("pt_nll", "255–304", "所有項skip令sum=0/count=0，mean除0"), ev("run_probe", "all_ignored", "repo error明確、原生torch mean_is_nan=true、sum0")], [probe_file], {"method": "executed", "expected": "有效數0，repo ValueError；torch sum0/meanNaN。", "observed": "ValueError 所有 labels 都被忽略：没有可學習的答案；torch_mean_is_nan=true、torch_sum=0.0。", "details": "人工[1,5,4]logits與全-100的[1,5]long labels，不涉及空訓練資料或下載。"}),
claim("exercise", "numeric", "把第一個-100改0增為第三個目標，平均約1.038且原相等assert應失敗；恢復-100後兩值再相同。", "course/chapters/07.md:208", "原兩有效位置及所有logits保持固定，第三格四候選0分數的loss=ln4；失敗是預期反例，不是程序缺陷。", [ev("math", "wrong_pad_mean=(l1+l2+ln4)/3", "count3與1.0377805587176374"), ev("run_probe", "pad_label_variation", "改label後不allclose，恢復後精確相等")], [probe_file], {"method": "executed", "expected": "count3、mean1.0377805587176374，與before不allclose；restore精確相等。", "observed": "sum3.1133416761529125、count3、mean1.0377805587176374，restore=0.8635236575165111；全部assert通過。", "details": "只變extended[0,2]；第三格NLL=ln4 nats，平均分母從2變3。", "tolerance": "mean絕對1e-12；三位舍入1.038精確；restore精確相等。"}),
claim("two_lengths_and_masks", "software", "變長batch只數真實監督答案，不數所有物理格或所有attention有效格；render_chat已shift，pad_batch與loss不再shift。", "course/chapters/07.md:184–186,202,206；必要前置7.3–7.5及實作契約", "對本課 Q→A 與 QQ→BC 的有界integration核對：PAD的valid=False/label=-100一致；真user內容valid=True/label=-100合法，因此兩個mask範圍不相等。未宣稱attention mask等同loss mask。", [ev("repo_data", "54–86", "shift後補x/y/valid，真實前文不抹掉"), ev("repo_model", "92–105", "跨筆flatten sum/有效token count"), ev("repo_attention", "10–18", "valid禁止PAD key，不禁止真实user key"), ev("repo_train", "343–346", "模型讀x/valid，loss讀y，靜態使用分開"), ev("run_probe", "batch_contract", "兩筆長6/8、有效14格、监督2+3=5，首答案位置4/5；user key可讀、PAD key不可讀；token weighted mean不同於平均row mean")], [probe_file], {"method": "executed", "expected": "batch shape[2,8]、PAD0/-100/False；valid_count14，loss_count5；first4/5；batch mean=sumNLL/5。", "observed": "所有契約assert通過；weighted_token_mean5.185514210024927，unweighted_example_mean5.08790548674458；沒有多shift。", "details": "兩筆完全literal合成對話，不下載或準備課程資料；人工[2,8,264]logits單獨核合併分母，非模型評測。"}),
]
all_ids = [c["id"] for c in claims]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "7.6", "source": "course/chapters/07.md#7.6", "source_sha256": extraction["source_sha256"], "figure_sha256": {},
    "verdict": "pass", "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": [],
    "scope_record": {"reviewed_section": "原UTF-8 bytes，182–216；完整原fence親跑一次；全部實質claim逐項覆蓋。", "actual_reads": ["course/chapters/07.md:1–320，重點7.1–7.6和相鄰7.7–7.9；7.6以外僅前置/範圍核對", "tiny_perceptron/model.py、data.py、attention.py全文", "scripts/train.py:320–352", "scripts/build_course.py:1–90及extract出的BOOTSTRAP", "course/training.md:1–90：僅確認外鏈工作性質，没有執行長recipe", "pyproject.toml全文；factual-reviewer-instructions.md/review-protocol.md/check_technical_reviews.py/section_facts.py完整方法與schema", "各sources的inspection_note所列原始官方內容"], "chapter_intro_review": {"status": "not_applicable", "reason": "7.6不是章首小節；intro只隨上下文讀取，不代替7.1的章首审查。"}, "figures": "7.6原小節沒有image/SVG引用，因此無需render/view；沒有把相鄰7.7圖的字串閱讀寫成視覺驗證。", "independence": "全新單節task；未讀舊technical/reader報告正文/結論或他人判定；無spawn，無正文/圖修改，無commit。", "external_access": "docs.pytorch.org 2.14/2.11頁面回403，保存真實receipt；改親讀同installed git commit的官方原始碼/docstrings，權威查證已完成，無搜尋摘要支持。2.11官方原碼取得留作來源探索但未用作同版本證據。", "measurements": "本節沒有既有模型實測主張；只有人工logits數值示範。未讀權重、重訓、GPU、評測模型、準備新課程資料或下載資料/模型。"},
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "逐項核實PAD/ignore_index、sum/有效count、有限ignored logits及直接梯度、PAD誤標方向、no-forward限制與mask範圍；無未確定或矛盾的實質主張。", "claim_ids": all_ids},
        "numeric_verification": {"status": "pass", "details": "math.exp/log獨立計算；原float32與float64短CPU親跑；轴B/T/C、nats、有效分母2/3/5、舍入、梯度和restore均核對。", "claim_ids": ["original_numbers", "exercise", "ignore_invariance", "two_lengths_and_masks"]},
        "figure_consistency": {"status": "not_applicable", "details": "完整原7.6没有圖引用；本節文字和簡短tensor足以回答分子/分母問題，沒有需要猜圖片素材或空間箭頭的實質主張。相鄰7.7圖不屬於這份判定。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "原始repo helper親讀，PyTorch官方raw code/docstrings匹配2.14.1+cpu git commit且亲读公式/API/CPU实现；每claim獨立locator与支持範圍。403不是已查證頁面；使用已取得原始源码完成核實。", "claim_ids": all_ids},
        "limitations": {"status": "pass", "details": "示範保持原logits不變，僅確認loss統計與必要梯度变化；非TinyLM前向等價或回答能力。whole batch count0明確拒絕；权重/smoothing/非有限logits不由此例支持。無既有實測、圖或長recipe本節驗收需求。", "claim_ids": ["padding_convention", "ignore_invariance", "scope_of_demo", "empty_mean", "two_lengths_and_masks"]},
    },
    "required_artifacts": [a["path"] for a in artifacts],
}
write(ROOT / "docs/technical-reviews/7.6.json", report)
print(json.dumps({"report": "docs/technical-reviews/7.6.json", "report_sha256": sha(ROOT / "docs/technical-reviews/7.6.json"), "source_sha256": extraction["source_sha256"], "claims": len(claims), "verdict": "pass"}, indent=2))
