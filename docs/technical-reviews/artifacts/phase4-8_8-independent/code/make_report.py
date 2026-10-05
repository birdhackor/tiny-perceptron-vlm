"""Serialize the independently reviewed section 8.8 evidence and scope."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
REL = OUT.relative_to(ROOT).as_posix()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

cpu = json.loads((OUT / "cpu.stdout.json").read_text())
extraction = json.loads((OUT / "inputs/extraction.json").read_text())
original_receipt = json.loads((OUT / "inputs/execution.json").read_text())
env = cpu["environment"]
cpu_command = "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python " + REL + "/code/verify_cpu.py > " + REL + "/cpu.stdout.json 2> " + REL + "/cpu.stderr.txt"
original_command = ".venv/bin/python docs/review-tools/section_facts.py course/chapters/08.md#8.8 --output /tmp/phase4-8_8-independent-original --execute --timeout 60"

special = {
    "inputs/stdout.txt": ("original-fence-run", "execution", {"command": original_command,
        "result": "exit 0; 可訓練參數 56; 初始相同 True; A梯度為零 True; B梯度非零 True", "environment": env}),
    "cpu.stdout.json": ("bounded-cpu-run", "execution", {"command": cpu_command,
        "result": "exit 0; all_assertions_passed；原fence必要變體、矩陣/縮放、凍結、11處參數量、saved JSON分母/逐題/資料SHA/有效目標數全部一致", "environment": env}),
    "figure/lora-paths.png": ("figure-render", "figure_render", {"command": "inkscape course/figures/rewrite-08-lora-paths.svg --export-type=png --export-filename=" + REL + "/figure/lora-paths.png", "result": "exit 0; personally viewed render", "environment": {"renderer": "Inkscape 1.4 (e7c3feb100, 2024-10-09)"}}),
    "inspection.md": ("inspection-derivation", "derivation", {}),
    "inputs/fence-1.py": ("original-fence-code", "code", {}),
    "code/verify_cpu.py": ("bounded-cpu-code", "code", {}),
    "inputs/current/docs/course-experiments/results/lora.json": ("saved-lora-json", "source_snapshot", {}),
    "inputs/style-base-evidence.json": ("saved-base-excerpt", "source_snapshot", {}),
    "sources/lora-2106.09685v2.pdf": ("lora-paper-bytes", "source_snapshot", {}),
    "sources/lora-2106.09685v2.txt": ("lora-paper-text", "source_snapshot", {}),
    "figure/render.receipt.json": ("figure-render-receipt", "source_snapshot", {}),
}
artifacts = []
ids = {}
for i,p in enumerate(sorted(OUT.rglob("*"))):
    if not p.is_file() or p.name in ["checker.receipt.json", "manifest.sha256.json", "check.stdout.txt", "check.stderr.txt"]:
        continue
    local = p.relative_to(OUT).as_posix()
    identifier,kind,extra = special.get(local,("snapshot-"+str(i), "code" if p.suffix==".py" else "source_snapshot", {}))
    ids[local] = identifier
    artifacts.append({"id":identifier,"kind":kind,"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p),
        "description": "8.8本輪正式原始bytes／執行或查證證據："+local, **extra})

sources = []
def repo(identifier, path, version, note):
    p=ROOT/path
    sources.append({"id":identifier,"kind":"repository_code","title":path,"path":path,
        "sha256":sha(p),"version":version,"verified":True,"inspection_note":note})

sources.append({"id":"lora-paper","kind":"paper","title":"Hu et al. LoRA: Low-Rank Adaptation of Large Language Models",
    "url":"https://arxiv.org/pdf/2106.09685v2","version":"arXiv:2106.09685v2; 16 Oct 2021; Version 2",
    "accessed_on":"2026-10-05","authority_reason":"LoRA作者原論文，原始方法與初始化/縮放契約。",
    "checked_original":True,"verified":True,"inspection_note":"我親讀保存的原PDF first-page版本、Section1、4.1 p.4 Eq.(3)及其初始化/alpha-rank段落、4.2 p.5實務收益/限制；來源locator僅用於找到bytes，並核SHA。","artifact_ids":["lora-paper-bytes","lora-paper-text"]})

torch_commit="5c4886908584029761b579af026dcfb627c84070"
for identifier,name,original,loc in [
    ("torch-linear","torch--nn--modules--linear.py","torch/nn/modules/linear.py","lines53–134: y=xW.T+b, shapes, default bias"),
    ("torch-module","torch--nn--modules--module.py","torch/nn/modules/module.py","lines2670–2697 parameters and 2934–2958 requires_grad_ freezing"),
    ("torch-parameter","torch--nn--parameter.py","torch/nn/parameter.py","lines30–55 Parameter default requires_grad=true and registration"),
    ("torch-backward","torch--_tensor.py","torch/_tensor.py","lines566–620 backward chain-rule gradient accumulation"),
    ("torch-basic-apis","torch--_torch_docs.py","torch/_torch_docs.py","lines4241–4264 equal and 8814–8834 numel")]:
    sources.append({"id":identifier,"kind":"official_source","title":"PyTorch official source "+original,
        "url":"https://raw.githubusercontent.com/pytorch/pytorch/"+torch_commit+"/"+original,
        "version":"immutable git "+torch_commit+"; installed 2.14.1+cpu matches torch.version.git_version",
        "accessed_on":"2026-10-05","authority_reason":"PyTorch官方專案原始程式，commit與本次CPU執行版本一致。",
        "verified":True,"checked_original":True,"inspection_note":"我本人重讀原始bytes "+loc+"；逐API核對本節實際program contract，原始bytes SHA匹配提供的immutable locator，非來源庫摘要。","artifact_ids":[ids["sources/"+name]]})

repo("alignment", "tiny_perceptron/alignment.py", "current bytes, same code SHA as original lora run", "親讀LoRALinear lines11–28，W與bias凍結、A random/B zero、row-batch forward与merged_weight。")
repo("model", "tiny_perceptron/model.py", "current bytes, same SHA as original run", "親讀ModelConfig、Block、TinyLM構造及forward；完整模型僅構造數參數，未執行推論。")
repo("attention", "tiny_perceptron/attention.py", "current bytes, same SHA as original run", "親讀CausalAttention lines33–83；q、v、out皆Linear，q/k配分、weighted v與out變換符合插入位置說明。")
repo("ffn", "tiny_perceptron/modern.py", "current bytes, same SHA as original run", "親讀DenseFFN lines35–52；64→256→64與bias/norm計數。")
repo("data", "tiny_perceptron/data.py", "current bytes, same SHA as original run", "親讀ByteTokenizer lines15–33、render_chat/pad_batch lines62–103；UTF8 byte decode errors=replace、assistant labels含EOS。")
runver="original immutable experiment revision a7cdffec4dc1d2356264f26e37da5705a298877e; git show bytes match JSON code_sha256"
repo("run-behavior", REL+"/inputs/run-version/scripts/course_experiments/behavior.py",runver,"親讀原版_style_metrics21–90和LoRA163–358；same base copy、11位置、only A/B optimizer、450步、首次gradient contract、base digest比較與full-SFT對照。當前rubric已加hidden-special檢查，未替換原版判準。")
repo("run-common",REL+"/inputs/run-version/scripts/course_experiments/common.py",runver,"親讀split_records/text_examples、fit_lm122–202、evaluate_lm243–284；seed42 sampler、batch16、有放回、有效assistant token/EOS累加與逐題generation IDs判準。")
repo("run-text",REL+"/inputs/run-version/scripts/course_experiments/text.py",runver,"親讀_steps/_save_splits34–67與arithmetic_records596–612；64道加法、symmetric family、精確JSONL serialization指紋。")
repo("saved-lora", "docs/course-experiments/results/lora.json",runver+"; seed42; CUDA/L4; PyTorch2.14.1+cu126; complete_run", "親讀原JSON metadata/training/layers/data/source hashes及六組sample；獨立redecode/re-score/count，未重訓或載入權重。formal完整JSON snapshot保存。")
repo("saved-base", "docs/course-experiments/results/style.json", "original style run revision ae7bbbf95537d228a44810041d2a9e978360d369; seed42; complete_run", "親讀原JSON arithmetic_data/content_training/content_evaluation.test，七題sample與generated_ids重算0/7；formal excerpt保留這些欄位，整體來源SHA亦核對。")
sources += [
    {"id":"derivation","kind":"derivation","title":"8.8本人的矩陣、梯度與參數推導","verified":True,"details":"inspection.md：rank(BA)≤r；dL/dA=s sum B.T1x.T；B=0導致A梯度0；r(in+out)和完整模型9w bias/norm計數，完整ratio重算。"},
    {"id":"original-execution","kind":"execution","title":"本節未修改fence CPU執行","verified":True,"artifact_id":"original-fence-run"},
    {"id":"bounded-execution","kind":"execution","title":"本人有界CPU變體及原JSON重算","verified":True,"artifact_id":"bounded-cpu-run"},
]

def ev(s,l,support):return {"source_id":s,"locator":l,"supports":support}
def ver(expected,observed,details,tolerance=None,denominators=None):
    d={"method":"executed","expected":expected,"observed":observed,"details":details}
    if tolerance:d["tolerance"]=tolerance
    if denominators:d["denominators"]=denominators
    return d

claims=[
    {"id":"mechanism","kind":"concept","statement":"LoRA保留凍結原W，訓練低秩A/B修正ΔW=(alpha/rank)BA；兩路用同一輸入並相加，中間rank限制矩陣變化。","location":"course/chapters/08.md:250–254","scope":"標準LoRA線性重參數化；rank是中間寬度和rank上界，未保證修正實際rank恰等於設定。",
     "status":"verified","evidence":[ev("lora-paper","Section4.1 p.4 Eq.(3)與其前後段落","凍結W、A/B shapes、共享x、add、B zero初始化和alpha/r scaling。"),ev("alignment","LoRALinear lines11–28","此repo確按同一公式實作。")],"artifact_ids":["lora-paper-bytes","inspection-derivation"]},
    {"id":"single-layer-numbers","kind":"numeric","statement":"16→12的W為12×16且192值；A2×16/B12×2共56值；偏置12另凍結；r=alpha=4則112值，倍率仍1。","location":"course/chapters/08.md:250–254,275–281","scope":"這一層與練習；不把192視為含bias總參數，不把56/112視為整個模型總量。","status":"verified",
     "evidence":[ev("torch-linear","Linear lines53–134","out×in權重、out bias與batch last-dimension契約。"),ev("alignment","lines16–28","A/B dimensions、bias跟隨base凍結與縮放。"),ev("derivation","inspection.md Derivations and execution scope","BAx次序與r(16+12)計數。")],"artifact_ids":["bounded-cpu-run","bounded-cpu-code","figure-render"],
     "verification":ver("192 weight, 12 bias, trainable56/112；輸出[3,12]；alpha/r=1","全部一致；另以alpha2/rank4=0.5檢查非零修正，row-column最大差2.384185791015625e-7","seed0原例與rank4變體；原linear/LoRA shapes及requires_grad逐參數檢查。","整數與初始equal精確相等；非零row-column/merged浮點rtol=atol=1e-6。")},
    {"id":"program-coverage","kind":"software","statement":"原fence得到56和三個True；B zero令初始輸出相同和首次A梯度0、B非零；backward只是梯度，不更新、不訓練新寫法。B更新後A才可能收到梯度。","location":"course/chapters/08.md:258–277","scope":"涵蓋整段程式與必要PyTorch API合約；nonzero B梯度限於seed0本例，不泛化到任意loss/input。單步synthetic SGD不是風格實驗。","status":"verified",
     "evidence":[ev("original-execution","inputs/fence-1.py与stdout.txt; helper execution.json exit0","未修改原fence直接得到文中輸出。"),ev("alignment","LoRALinear lines11–28","初始化、可學A/B與凍結base。"),ev("torch-module","parameters2670–2697/requires_grad_2934–2958","列舉參數及requires_grad篩選/凍結。"),ev("torch-parameter","Parameter30–55","A/B參數預設可學並被module列舉。"),ev("torch-backward","Tensor.backward566–620","chain-rule計算並累積grad，無更新器step。"),ev("torch-basic-apis","equal4241–4264、numel8814–8834","same sizes/elements與元素數。"),ev("derivation","inspection.md gradient derivation","B=0導致A梯度0、更新B後可傳A訊號。")],"artifact_ids":["original-fence-code","original-fence-run","bounded-cpu-run","bounded-cpu-code"],
     "verification":ver("56、True、True、True；backward不改任何值、W/bias無grad；一步B更新後A可非零","原fenceexit0；rank2 first maxA=0/maxB=.0987018272，second maxA=.4269385934；rank4亦一致","參數clone逐值相等、SGD只對A/B且一個step；W/bias保持精確相同。")},
    {"id":"full-model-parameters","kind":"numeric","statement":"兩層每層q/v/out/up/down五處再加output共11處；原141568全凍結；rank4/alpha4只學9504，約原量6.7%。","location":"course/chapters/08.md:286–288","scope":"本保存實驗的width64/vocab264/context128/two-block TinyLM配置；不宣稱其他架構相同量。","status":"verified",
     "evidence":[ev("run-behavior","_add_lora172–187、run_lora296–358","明示11處與全base.requires_grad_(False)。"),ev("model","ModelConfig、Block、TinyLM construction","width/layers/vocab/position/norm/output計數。"),ev("attention","CausalAttention33–83","q/v/out各為64→64線性投影；其資料流支持Query/Value的教學描述。"),ev("ffn","DenseFFN35–52","up64→256/down256→64。"),ev("derivation","inspection.md full-model accounting","原參數141568和LoRA9504的獨立公式。")],"artifact_ids":["bounded-cpu-run","bounded-cpu-code","inspection-derivation"],
     "verification":ver("base141568；11處；adapter9504；占base约6.7%","構造但不forward模型逐module得512×6+1280×4+1312=9504；base141568 frozen，ratio6.713381555153706%，combined151072","編譯原版_add_lora unedited AST，僅構造數參數，與saved layers完全一致。","整數精確相等，百分比四捨五入至1小數。")},
    {"id":"saved-training-contract","kind":"empirical","statement":"兩adapter從相同加法base讀相同49題，450步；rank4 alpha4；首次A梯度全0/B非0，保存原權重未改。FullSFT以同base/比喻資料/450步/318991有效回答目標對照。","location":"course/chapters/08.md:286–290","scope":"親核保存run的原JSON及指定版本實作，重算accounting；沒有重訓、沒有checkpoint tensor inspection。原權重不變是saved digest檢查的成功紀錄。","status":"verified",
     "evidence":[ev("saved-lora","revision/code_sha256；results.runs.*.training/data/layers/base_parameters_unchanged，results.full_sft.training","保存實際run的條件、原權重digest檢查成功與目標數。"),ev("run-behavior","_fit_adapter234–293、run_lora296–358、_state_digest163–169/_base_state209–215","只更新A/B、first gradients檢查、before/after base digest比較，full取same original base。"),ev("run-common","fit_lm122–202；split_records50–68；text_examples77–96","seed42/batch16/replacement、450步、有效labels含EOS的累加方式。"),ev("run-text","_save_splits47–64；arithmetic_records596–612","精確data SHA與symmetric family切分。"),ev("data","render_chat62–80","回答token和EOS計入，其他labels忽略。")],"artifact_ids":["saved-lora-json","bounded-cpu-run","bounded-cpu-code"],
     "verification":ver("same 49 input questions，450×16抽樣；concise16591/vivid與full318991目標；11位置9504；原base成功unchanged","原版bytes全match JSON code SHA；六個data SHA全match；重算seed sampling targets16591/318991；full records SHA與vivid相同；保存first-gradient/freeze assertions一致","只在記憶體序列化原函式的資料以check SHA，不準備資料檔，不跑訓練函式。當前_style_metrics有後加hidden-special條件，此次使用原版，且全部45 sample無hidden-special。",denominators={"train_questions_per_adapter":49,"train_families":28,"validation_questions":8,"test_questions":7,"steps_per_run":450,"batch_size":16,"example_draws_per_run":7200,"concise_answer_token_targets":16591,"vivid_and_full_answer_token_targets":318991})},
    {"id":"saved-scores-EOS","kind":"empirical","statement":"LoRA短答風格8/8、7/7；固定比喻5/8、6/7；full8/8、7/7；各最後七題算術0/7；base原亦0/7；所有回答含EOS，失敗句1�，像�兩組積木合在一起再數。未過固定句。","location":"course/chapters/08.md:279,286,290–298","scope":"僅此兩個保存實驗中的固定算術和模板判準；未推論/重新評測模型；未把EOS作為正確性或正常UTF8保證。","status":"verified",
     "evidence":[ev("saved-lora","results.runs.{concise,vivid}.evaluation.{validation,test}.samples/rubric；results.full_sft.evaluation","逐題原generated與generated_ids支撐table和失敗例。"),ev("saved-base","results.content_evaluation.test.samples/records/matches；arithmetic_data","同49題base的原七题為0/7。"),ev("run-behavior","_style_metrics21–90","數字isdigit/固定substring與獨立content判準。"),ev("run-common","evaluate_lm243–284","EOS ID/停止token、raw exact判準、decode方式。"),ev("data","ByteTokenizer15–33","UTF8解碼errors=replace產生U+FFFD。")],"artifact_ids":["saved-lora-json","saved-base-excerpt","bounded-cpu-run","bounded-cpu-code"],
     "verification":ver("六組table風格和算術分數、EOS与失敗原文全部match","逐題decode/re-score得到8/8,7/7;5/8,6/7;8/8,7/7，所有content0；45/45 EOS，base0/7；失敗句完全match","依原implementation外再以arithmetic truth自行評分，保留逐題sample支持範圍；family不跨train/validation/test。",denominators={"validation_per_model":8,"test_per_model":7,"evaluation_models":3,"generated_answers":45,"base_final_test":7,"failed_vivid_final_template":1})},
    {"id":"limitations","kind":"concept","statement":"較少可學參數是结构收益；本小例未證明風格學成，小測試僅部分固定寫法、算術未改善；基模/forward/中間資料仍存在，6.7%不是總模型大小或總訓練記憶體比率。","location":"course/chapters/08.md:258,275,279,298","scope":"標準LoRA凍結模型與參數效率的限定；本run結論不推廣通用算術、比喻或LoRA普遍優劣，未量測總memory savings。","status":"verified",
     "evidence":[ev("lora-paper","Section1 benefits p.2；Section4.2 practical benefits/limitations p.5","少計base gradients/optimizer state而保留base weights，trainable比例與memory比例不同。"),ev("alignment","LoRALinear.forward24–25","未merge示例仍執行base和修正路。"),ev("saved-lora","六組evaluation.samples/rubric與base_parameters/trainable parameters","固定句學成程度與內容結果分開，不將小型模板成績當方法通用效能。")],"artifact_ids":["bounded-cpu-run","inspection-derivation"]},
]

report={"schema_version":1,"review_stage":"technical","lesson_id":"8.8","source":"course/chapters/08.md#8.8",
    "source_sha256":extraction["source_sha256"],"figure_sha256":extraction["figure_sha256"],"verdict":"pass",
    "reviewer_task":"/root/phase4_factual_coordinator/factual_8_8","reviewer_context":"fresh",
    "reviewer_fork":"none","reviewed_on":"2026-10-05","read_scope":"現完整8.8 raw UTF8 lines248–301；必要前置8.3和2.4；引用SVG真render+view；指定原版程式及原權威段落，見inspection.md；未讀舊報告/作者notes/readability答案。",
    "sources":sources,"artifacts":artifacts,"claims":claims,"issues":[],
    "checks":{
      "factual_accuracy":{"status":"pass","details":"LoRA公式、凍結、rank上界、共享input與插入路徑符合原paper和原實作；逐主張支持範圍獨立列出。","claim_ids":["mechanism","program-coverage","full-model-parameters"]},
      "numeric_verification":{"status":"pass","details":"56/112、192/12、11/9504/141568、6.7134%→6.7%及全部saved分母/318991自算；整數精確、矩陣浮點容忍1e-6。","claim_ids":["single-layer-numbers","full-model-parameters","saved-training-contract","saved-scores-EOS"]},
      "figure_consistency":{"status":"pass","details":"Inkscape640×845 render+本人view_image；同input兩路、W frozen、A16→2→B2→12、192與32+24=56、相加和最後12皆一致。GTK/Pango警告不阻斷render；未聲稱browser/mobile頁檢查。","claim_ids":["mechanism","single-layer-numbers"]},
      "source_verification":{"status":"pass","details":"亲读arxiv2106.09685v2原PDF Sec4.1/4.2與PyTorch immutable5c488690版本原source；原JSON/code SHA/資料SHA均重核。來源locator只找bytes。","claim_ids":[x["id"] for x in claims]},
      "limitations":{"status":"pass","details":"原fence不是訓練；有界synthetic一步不是風格實驗；saved JSON核證而非重訓/推論，base-unchanged是saved digest記錄。固定模板與EOS/内容分開，未算總memory savings，未保留神經權重。","claim_ids":["program-coverage","saved-training-contract","saved-scores-EOS","limitations"]}
    }}
(ROOT/"docs/technical-reviews/8.8.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
print(json.dumps({"report":"docs/technical-reviews/8.8.json","report_sha256":sha(ROOT/"docs/technical-reviews/8.8.json"),"source_sha256":extraction["source_sha256"],"artifacts":len(artifacts),"claims":len(claims)},ensure_ascii=False))
