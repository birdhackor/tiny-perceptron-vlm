"""彙整本人逐項判斷，先在記憶體檢查格式，再以新檔保存正式報告。"""

import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = Path("docs/technical-reviews/artifacts/review-20261019-curriculum")
DEST = Path("docs/course-review-20261019/reviews/rv1019_technical_curriculum/technical-curriculum-original-01.json")
local = json.loads((ROOT / BASE / "local-evidence.json").read_text())
upstream = json.loads((ROOT / BASE / "upstream-evidence.json").read_text())
mechanics = json.loads((ROOT / BASE / "mechanics-evidence.json").read_text())
records = {r["id"]: r for r in upstream["records"]}


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


artifacts = []
for identifier, filename, kind, description in [
    ("a-local-code", "inspect_local.py", "code", "本地教材清單、Notebook同步、參數量、固定結果及Auto規則核對程式。"),
    ("a-local", "local-evidence.json", "execution", "當次執行結果，含輸入完整SHA、實際重算數字、模型設定及既有實驗紀錄的取值。"),
    ("a-upstream-code", "inspect_upstream.py", "code", "用curl讀取固定上游版本與匿名HF metadata，只保存支持主張的摘錄和全文指紋。"),
    ("a-upstream", "upstream-evidence.json", "execution", "當次55個HTTPS原始來源的URL、版本、讀取日期、curl指令、exit code、指紋及定位摘錄。"),
    ("a-mechanics-code", "verify_mechanics.py", "code", "CPU流程小算例、BPB透明推導及本地訓練入口探查。"),
    ("a-mechanics", "mechanics-evidence.json", "execution", "causal可讀性、指定計分位置、梯度清零／累積和SGD更新的當次結果；另保存trainer help失敗stderr。"),
    ("a-scope", "scope-disclosure.txt", "source_snapshot", "原始判斷存檔前意外看到既有integrity-review路徑／版本片段的範圍揭露；不作主張證據。"),
    ("a-report-code", "write_report.py", "code", "建立本報告與執行記憶體格式預檢的程式，內含逐項判斷。"),
]:
    path = BASE / filename
    item = {"id": identifier, "kind": kind, "path": path.as_posix(), "sha256": digest(path), "description": description}
    if kind == "execution":
        value = {"a-local":local, "a-upstream":upstream, "a-mechanics":mechanics}[identifier]
        item.update(command=value["command"], environment=value["environment"], result={
            "a-local":"20章與A/B/C共312份Notebook、338個編號小節；312份同步；兩版參數重算相符；Auto規則已執行。資料包為133-byte LFS pointer，未抽取原始資料。",
            "a-upstream":"55個curl讀取皆exit 0；HF tree列四目錄／16檔；上游固定版本摘錄支持所列程式與方法主張。",
            "a-mechanics":mechanics["result"]}[identifier])
    artifacts.append(item)

sources = {
    "s-local": {"id":"s-local", "kind":"execution", "title":"本回合本地清單、原始碼與固定實驗結果核對", "verified":True, "artifact_id":"a-local"},
    "s-mechanics": {"id":"s-mechanics", "kind":"execution", "title":"本回合CPU小算例與CLI入口核對", "verified":True, "artifact_id":"a-mechanics"},
}


def register_upstream(identifier):
    sid = "s-"+identifier
    if sid in sources:
        return sid
    r = records[identifier]
    assert r["exit_code"] == 0
    authority = "上游專案維護者發布的固定commit原始碼／文件；只核對此版本的實作或文件內容，未外推其訓練效益。"
    if identifier.startswith("hf-"):
        authority = "本專案發布者在Hugging Face固定revision的原始檔案／官方tree API回應，直接支持公開檔案與發布metadata。"
    elif identifier.startswith("issue-"):
        authority = "提問者本人在原始GitHub issue的直接記錄；只證明有人提出這類問題，不證明提問中的因果猜測或困難盛行率。"
    elif identifier.startswith(("cs336-", "mit-", "harvard-")):
        authority = "授課組織官方repository的固定版本課綱／教材，是相應課程內容的原始來源；未宣稱觀看影片。"
    elif identifier == "ntu-mirror-topics":
        authority = "鏡像筆記維護者本人發布的固定版本索引；只支持鏡像列出的主題線索，不作台大官方課綱或影片日期證據。"
    notes = {
        "nano-bpb":"閱讀evaluate_bpb的sum nats、有效target bytes、分散式加總與除以ln(2)*bytes；特殊token和計分邊界必須一致。",
        "nano-gpt":"閱讀MLP.forward的ReLU平方、獨立wte與lm_head宣告及Block；輸入與輸出權重未共享。",
        "mv-default-llm":"閱讀MiniMindConfig的silu與tie_word_embeddings=True、gated FFN、FeedForward/MOEFeedForward選擇及實際權重共享指派。",
        "mv-trainer-trainer_utils.py":"閱讀init_vlm_model的from_weight載入、vision_proj保留梯度、freeze_llm=1只開layers.0和最後layer，以及vision_encoder維持凍結。",
        "mv-dataset-lm_dataset.py":"閱讀VLMDataset.generate_labels：先全部-100，再填assistant內容／結束區段；padding和截斷的細節不作一般最佳策略宣稱。",
        "mini-trainer-train_distillation.py":"閱讀distillation_loss和train_epoch，確認softmax/log_softmax溫度、teacher detach、T平方與有效loss_mask上的CE/KL。",
        "safe-rlhf":"閱讀Constrained Value Alignment via Safe RLHF中reward與cost及helpful/high reward、harmless/low cost的分開建模。",
        "ao-docs-source-contributing-quantization_overview.rst":"閱讀primitive ops、efficient kernels及Quantized Tensors/packing分層；int4兩值可pack入uint8不等於所有硬體自動加速。",
        "ao-docs-source-workflows-qat.md":"閱讀QAT的prepare插入fake quant與convert依base config轉換，確認模擬與實際量化的階段差別。",
        "torch-attention":"閱讀PyTorch v2.14.1 scaled_dot_product_attention對is_causal與attention mask的定義；不混用MultiheadAttention布林mask的相反慣例。",
        "torch-zero":"閱讀Optimizer.zero_grad的Reset gradients說明；只對optimizer收錄參數清除梯度。",
        "torch-backward":"閱讀Tensor.backward的graph leaves梯度計算；與CPU實際梯度累積觀察交叉核對。",
        "mit-style":"閱讀Lab3風格微調與positive/negative judge controls文字；不是重現其微調成績。",
        "mit-cv":"閱讀Lab2 facial detection/debiasing及按膚色／性別分項評估的材料；不外推其模型公平性保證。",
    }
    sources[sid] = {"id":sid, "kind":"official_source", "title":identifier+"：原始固定版本", "url":r["url"], "version":r["version"], "verified":True, "checked_original":True, "accessed_on":r["accessed_on"], "authority_reason":authority, "inspection_note":notes.get(identifier, "實際以curl讀取原始URL並閱讀upstream-evidence.json中此id的excerpts／metadata／issue_excerpt；核對定位與下列逐項evidence的支持範圍，沒有用本地來源摘要代替原文。")}
    return sid


claims = []


def claim(kind, statement, location, expected, observed, locator, scope, upstream_ids=(), artifact="a-local", denominators=None, concept=False):
    identifier = "c"+str(len(claims)+1)
    evidence = []
    if artifact == "a-local":
        evidence.append({"source_id":"s-local", "locator":locator, "supports":observed})
    elif artifact == "a-mechanics":
        evidence.append({"source_id":"s-mechanics", "locator":locator, "supports":observed})
    for ext in upstream_ids:
        sid = register_upstream(ext)
        evidence.append({"source_id":sid, "locator":"upstream-evidence.json records[id="+ext+"]："+locator, "supports":observed})
    aids = [artifact]
    if "mechanics-evidence.json" in locator and "a-mechanics" not in aids:
        aids.append("a-mechanics")
    if upstream_ids and "a-upstream" not in aids:
        aids.append("a-upstream")
    item = {"id":identifier,"kind":kind,"statement":statement,"location":location,"status":"verified","evidence":evidence,"artifact_ids":aids,"scope":scope}
    if not concept:
        item["verification"] = {"method":"executed", "expected":expected, "observed":observed, "details":locator+"；當次指令、環境及取值保存在所列execution artifact。"}
        if kind == "numeric":
            item["verification"]["tolerance"] = "整數／集合精確相等；不使用浮點容忍差。"
        if kind == "empirical":
            item["verification"]["denominators"] = denominators or {"scope":"固定版本既有實驗報告的分項樣本／步數；未重新訓練"}
    claims.append(item)
    return identifier


def num(text, where, expected, observed, loc, scope="只支持目前凍結頁面對應的教材或固定實驗版本。", upstream_ids=(), artifact="a-local"):
    return claim("numeric",text,where,str(expected),str(observed),loc,scope,upstream_ids,artifact)


def sw(text, where, observed, loc, upstream_ids=(), artifact="a-local", scope="只支持所讀固定版本的原始碼／文件；靜態程式核對與指令實際讀取已保存，沒有執行上游完整訓練。"):
    return claim("software",text,where,text,observed,loc,scope,upstream_ids,artifact)


def emp(text, where, observed, loc, den, scope="核對固定實驗的原始配置與已保存結果，不重跑完整訓練；不能外推到其他seed、資料、硬體或開放領域。", upstream_ids=()):
    return claim("empirical",text,where,text,observed,loc,scope,upstream_ids,"a-local",den)


def con(text, where, observed, loc, upstream_ids, artifact="a-mechanics", scope="限正文所述用途與條件；局部機制不保證一般品質、速度或記憶體效益。"):
    return claim("concept",text,where,text,observed,loc,scope,upstream_ids,artifact,concept=True)


num("有20章與A/B延伸、C推理計算單元。","開頭教材規模","20+3","實際列出20個數字章與0A、0B、0C，共23個章檔。","inventory.numeric_chapters/extension_chapters")
num("逐課Notebook共312份。","開頭教材規模",312,"312個章節小節、312個Notebook檔。","inventory.chapter_sections/notebook_files")
num("含閱讀指南、暖身、訓練操作、名詞頁，共338個編號小節。","開頭教材規模",338,"按同一編號正則重新枚舉得338。","inventory.all_numbered_sections/sections")
sw("網站與逐課Notebook由同一Markdown正文產生。","開頭與教材呈現方式","重產312份Notebook的預期bytes，與現檔全數相符；build_course.notebook讀正文，export_course輸出網站及Notebook/Colab入口。","notebook_sync; code_excerpts.scripts/build_course.py / scripts/export_course.py",scope="核對生成來源與目前同步；沒有將同步當作每個kernel或Colab都執行成功。")
emp("v2的文字、視覺、OCR、語音神經模組及接頭來自隨機初始化訓練。","開頭第19章現況","四份公開inference manifest origin.kind為all-neural-weights-random；固定結果、來源選擇與模型建構沒有載入外部預訓練神經模型。","selftrained.architectures; training_selection; data_manifest; LimitedAssistant.__init__",{"models":2,"modalities":["text","vision","ocr","voice"],"origin_seed":20261006},upstream_ids=("hf-manifest-moe-pretrain","hf-manifest-moe-sft","hf-manifest-moe-joint","hf-manifest-dense-joint"))
num("MoE共5,447,107參數。","開頭參數數量",5447107,local["recounted_models"]["moe"]["parameters"],"recounted_models.moe.parameters；以公開config重新建立LimitedAssistant再sum(numel)")
num("Dense共2,288,067參數。","開頭參數數量",2288067,local["recounted_models"]["dense"]["parameters"],"recounted_models.dense.parameters；以公開config重新建立LimitedAssistant再sum(numel)")
num("兩版同固定資料包的訓練題數為28,876。","開頭資料數量",28876,local["selftrained"]["data_scope"]["train_rows"],"selftrained.data_scope.train_rows; data_manifest.records; training_stages的manifest_sha256",scope="核對固定實驗紀錄與兩版相同的資料manifest SHA；未由LFS pointer取得原始包逐列重算。")
num("validation題數為2,435。","開頭資料數量",2435,local["selftrained"]["data_scope"]["validation_rows"],"selftrained.data_scope.validation_rows; training_selection.private_moe_validation.count",scope="核對固定結果和validation執行索引的一致值；未重新生成所有validation輸出。")
emp("凍結後兩版各有一次3,734題最後測試。","開頭最後測試","moe/dense各evaluation_complete=true、count=3734，per_task_final_reply的count加總也各3734；各有不同單次final_test run與固定權重SHA。","selftrained.architectures.*.count/task_count_sum/run_id; docs/selftrained/results/v2-final-public-results.json final_test",{"moe_test_rows":3734,"dense_test_rows":3734,"attempts_per_architecture":1},scope="核對發布的單次測試紀錄與分母；不聲稱取得雲端所有歷史執行以排除未記錄的私下實驗。")
emp("兩版仍未通過全部原定能力判準。","開頭未達標結論","capability_acceptance_passed均false；語音QA分別42/90與37/90，低於各意圖0.8門檻所能允許的整體下限，足以確認未全達標。","selftrained.architectures.*.voice/thresholds/capability_acceptance_passed",{"moe_voice_rows":90,"moe_correct":42,"dense_voice_rows":90,"dense_correct":37,"per_intent_minimum":0.8})
num("最後MoE段完成4,000步，選第1,000步。","開頭訓練步數","completed=4000, selected=1000","native-balanced-4000 stage index與公開joint manifest均支持completed/selected的區別。","training_stages.v2-moe-native-balanced-4000-training-stage-index.json; training_selection.moe_joint",upstream_ids=("hf-manifest-moe-joint",))
num("最後Dense段完成10,000步，選第1,000步。","開頭訓練步數","completed=10000, selected=1000","weighted-10000 stage index與公開joint manifest均支持completed/selected的區別。","training_stages.v2-dense-weighted-10000-training-stage-index.json; training_selection.dense_joint",upstream_ids=("hf-manifest-dense-joint",))
num("固定HF revision公開四組推論輸出，共16個配對檔案。","開頭公開輸出","4 groups, 16 files","匿名HF tree恰有moe-pretrain/moe-sft/moe-joint/dense-joint四目錄，各有model.safetensors、model-config、tokenizer、inference-manifest。","HF tree entries，僅計type=file；不把四個directory當檔案。",upstream_ids=("hf-v2-tree",),artifact="a-upstream")
sw("這四組推論輸出採MIT授權。","開頭公開輸出授權","實際發布job列16個檔案且每檔license=MIT、redistribution_approved=true；公開權重SHA與這些job及final results相符。","four_export_release_job; selftrained.architectures.*.safe_weights",upstream_ids=("hf-manifest-moe-joint","hf-manifest-dense-joint"),scope="按這16個檔案的明示發布授權核對；未將HF根README的程式碼MIT推及所有其他資料與權重。")
sw("公開safetensors是推論輸出，不含optimizer/RNG/sampler續訓狀態。","開頭公開權重限制","HF每組僅推論權重／設定／tokenizer／manifest；exporter以model state_dict生成safetensors，完整checkpoint另由train_local_stage.local_source要求best/latest與receipt。","code_excerpts.scripts/selftrained/hf_transport.py; four_export_release_job; code_excerpts.scripts/selftrained/train_local_stage.py",upstream_ids=("hf-v2-tree",),scope="核對匯出邊界與檔案種類，沒有下載約22MB權重逐tensor重算，也不把推論檔當exact resume。")
sw("重做主線可接自己的完整本機checkpoint，不需要作者私人Volume。","開頭重做方式","本機wrapper讀data-root、manifest與own best/latest，backend=local-subprocess；wrapper help成功，trainer parser有init-checkpoint/resume參數。","code_excerpts.scripts/selftrained/train_local_stage.py; mechanics-evidence.json local_training_help/trainer_parser_options",scope="確認本機程式路徑與CLI登記；未跑完整訓練。額外trainer --help的argparse格式錯誤另列incidental_observations，與本頁未提供的help指令分開。")
num("各章公開清單保留30組／120份權重。","開頭歷史實驗清單","30 groups, 120 weights",f"{local['public_experiments']['count']}組，對所有files計.pt/.safetensors共{local['public_experiments']['weight_count']}份。","public_experiments.count/weight_count; docs/course-experiments/public-models.json")
emp("舊部署joint與DPO各78/90題。","開頭舊部署成績","讀取capstone_deployment.results.stages，joint與dpo的end_to_end_correct均78，count均90。","old_experiments.capstone_deployment.*stages*.count/end_to_end_correct",{"joint_rows":90,"joint_correct":78,"dpo_rows":90,"dpo_correct":78,"seed":42,"device":"NVIDIA L4"})
emp("舊Dense示範學生為62/90題。","開頭舊Dense成績","capstone_student.results.evaluations.test_ce是62/90；不是validation_ce的59/84。","old_experiments.capstone_student.evaluations.test_ce",{"test_rows":90,"end_to_end_correct":62,"seed":42,"steps":350})
emp("舊Dense蒸餾學生為61/90題。","開頭舊Dense成績","capstone_student.results.evaluations.test_kd是61/90；與v2最後考卷分開。","old_experiments.capstone_student.evaluations.test_kd",{"test_rows":90,"end_to_end_correct":61,"seed":42,"steps":350})
num("舊整合清單有8份階段／量化檔。","開頭舊整合清單",8,"pretrain/sft/joint/dpo/joint-int4/joint-int8/dpo-int4/dpo-int8共8項。","public_capstone.models.id")
num("舊整合清單另有3份Dense學生檔。","開頭舊整合清單",3,"student-ce/student-kd/student-kd-int4共3項。","public_capstone.models.id")
sw("第20章固定清單指定Qwen3-VL-2B-Instruct及Whisper-large-v3-turbo，沒有LoRA修正。","開頭第20章","public-release.json selected_variant=base，base_model/asr_model對應兩個名稱與固定revision，adapter_parameters=0。","natural_release.base_model/asr_model/selected_variant/adapter_parameters",scope="核對本版選擇與固定清單；沒有重跑自然素材模型或宣稱這兩個上游模型從零訓練。")
sw("各主題定位及C/14/19的閱讀次序與教材實際小節相符。","共同基礎、排序原則、閱讀與實作路線、目前教材與新增主題","重新枚舉的標題涵蓋SFT、MTP、思考預算、C.1–C.7、Looped、型別判斷、三成品與Auto；0C章與zensical導航分成第7與第13章後的兩段，14與19的小節順序相符。","inventory.sections; scripts/export_course.py; zensical.toml導航與course/README.md閱讀路線",scope="屬導覽位置核對；不代表初學者可理解或每一節方法已由本報告重審。")
con("MoE讓每個token只選部分專家執行。","MoE設計取捨與縮小部署成本","上游MOEFeedForward用top-k路由並執行選中的experts；本專案PaddingSafeMoE也是topk後依chosen派送。","MOEFeedForward.forward; local code_excerpts PaddingSafeMoE",("mini-model-model_minimind.py","mv-default-llm"),artifact="a-local")
con("MoE仍須儲存全部專家，不能只由稀疏啟動推定筆電上一定更快或更省記憶體。","MoE設計取捨","experts以ModuleList全部註冊且有router；本地重算顯示total與active counts不同，運算與儲存不等價。","MOEFeedForward.__init__/forward; recounted_models.moe.expert_parameters/active_language_parameters_per_token",("mini-model-model_minimind.py",),artifact="a-local")
sw("v2 top-2 MoE使用兩個FFN，Dense使用一個，最後目標與選定歷史不同。","第19章架構比較限制","config.top_k=2，Dense分支使用一個DenseFFN；兩版參數數量不同，native_voice/tool/numeric objective與來源選擇紀錄也不同，不能視為等參數／等計算受控對照。","recounted_models; training_selection; training_stages; MaskedBlock",scope="只確認這兩個固定成品不構成隔離架構因果的比較；不否定其他匹配預算的MoE比較。")
sw("nanochat串接tokenizer、pretrain、evaluation、SFT與inference。","參考專案策略表","speedrun.sh依序tok_train/tok_eval/base_train/base_eval/chat_sft/chat_eval，README與CLI路徑支持完整流程。","runs/speedrun.sh與README流程",("nano-pipeline","nano-readme"),artifact="a-upstream")
sw("nanochat評估包括生成樣本、BPB與任務分數。","參考專案策略表","speedrun.sh明列CORE、train/val BPB、samples，再呼叫base_eval。","runs/speedrun.sh base_eval註解／命令",("nano-pipeline",),artifact="a-upstream")
sw("nanochat另有tokenizer壓縮率與還原檢查。","參考專案策略表與原始碼參考","tok_eval.py對原文UTF-8 bytes與token數計ratio，並有decode檢查；與模型任務評分分開。","scripts/tok_eval.py encode/decode/ratio",("nano-tokenizer-eval",),artifact="a-upstream")
con("BPB以所計分原文UTF-8 byte數為共同分母，避免直接比較不同切詞的平均token loss。","參考專案策略表BPB定義","evaluate_bpb累加有效target nats與bytes，回傳nats/(ln2*bytes)；本回合例子中文A為7 bytes，3.5 nats得到0.7213475204444817 BPB。","evaluate_bpb; mechanics.bpb_derivation",("nano-bpb",),scope="必須匹配計分原文、邊界與上下文；不同tokenizer的不同條件機率分解仍可改變BPB，不保證模型排名公平性的所有條件已滿足。")
sw("nanochat提供CPU/MPS小型示範。","參考專案策略表","README連runcpu.sh並明示小模型效果有限；腳本以depth6和小迭代量執行。","README smaller models; runs/runcpu.sh",("nano-readme","nano-cpu"),artifact="a-upstream",scope="只確認示範配置存在；未重跑訓練或將它當能力／速度保證。")
sw("nanochat以depth作為主要規模旋鈕。","參考專案策略表","固定README將--depth定為主要scale dial，其他寬度／heads等隨其導出。","README depth說明",("nano-readme",),artifact="a-upstream")
sw("nanochat GPT採Dense ReLU² MLP，輸入與輸出權重未共享。","參考專案策略表後架構差異","MLP用兩個Linear與F.relu(x).square()；wte與lm_head分別配置／初始化。","gpt.py MLP.forward; GPT.__init__/init_weights",("nano-gpt",),artifact="a-upstream")
sw("MiniMind-V預設SiLU gating與共享輸入／輸出權重。","架構差異","固定MiniMind-V內model_minimind.py hidden_act='silu'、gate_proj乘up_proj、tie_word_embeddings=True且指派共享weight。","MiniMindConfig; FeedForward.forward; MiniMindForCausalLM.__init__",("mv-default-llm",),artifact="a-upstream")
sw("MiniMind-V由已訓練MiniMind LLM加凍結SigLIP2視覺編碼器延伸。","參考策略與模態來源","README指定llm_768.pth與siglip2-base-p32-256-ve；get_vision_model以from_pretrained載入並requires_grad=False。","README快速開始與結構說明; MiniMindVLM.get_vision_model",("mv-README.md","mv-model-model_vlm.py","mv-trainer-trainer_utils.py"),artifact="a-upstream")
sw("MiniMind-V投影視覺特徵並替換圖片佔位符。","參考專案策略表","VLMConfig含image_pad/image_ids，vision_proj把視覺表示轉為LLM寬度，count_vision_proj組合embedding。","VLMConfig; MiniMindVLM.__init__/count_vision_proj/forward",("mv-model-model_vlm.py",),artifact="a-upstream")
sw("MiniMind-V labels僅assistant區段計分，其餘為忽略標籤。","參考專案策略表","generate_labels起始全-100，只為assistant內容與其結束區段填token ID。","VLMDataset.generate_labels",("mv-dataset-lm_dataset.py",),artifact="a-upstream")
sw("此版MiniMind-V可直接SFT，projector預對齊為選配。","參考專案策略表","README推薦直接SFT；caption資料已合入SFT，pretrain可跳過。","README直接SFT與訓練階段說明; train_pretrain_vlm.py/train_sft_vlm.py",("mv-README.md","mv-trainer-train_pretrain_vlm.py","mv-trainer-train_sft_vlm.py"),artifact="a-upstream")
sw("MiniMind-V SFT預設訓練projector與LLM首末層。","參考專案策略表","train_sft_vlm預設freeze_llm=1；init_vlm_model先凍結非vision_proj，再開layers.0和最後layer。","train_sft_vlm.py freeze_llm; trainer_utils.init_vlm_model",("mv-trainer-train_sft_vlm.py","mv-trainer-trainer_utils.py"),artifact="a-upstream")
sw("MiniMind-V混入純文字對話的目的是保留語言能力。","參考專案策略表","README資料段列約23萬純文字對話、黑圖佔位與維持語言能力的設計目的。","README SFT資料與純文字說明",("mv-README.md",),artifact="a-upstream",scope="核對資料設計與作者陳述的目的；不把混入資料直接當作已證明不會遺忘。")
sw("MiniMind提供LoRA行為／領域調整與手寫DPO。","參考專案策略表","README有LoRA身份／領域訓練介紹；model_lora手寫A/B低秩層；train_dpo有ref_model凍結與logsigmoid偏好loss。","MiniMind README LoRA; model_lora.LoRA; train_dpo.py",("mini-README.md","mini-lora","mini-trainer-train_dpo.py"),artifact="a-upstream")
sw("Smol Course將指令SFT與偏好對齊分成不同教學單元。","參考專案策略表","unit1介紹instruction tuning/SFT，unit2介紹preference alignment/DPO。","units/en/unit1/1.md、unit1/3.md、unit2/1.md、unit2/2.md",("smol-units-en-unit1-1.md","smol-units-en-unit1-3.md","smol-units-en-unit2-1.md","smol-units-en-unit2-2.md"),artifact="a-upstream",scope="只採單元分工；沒有採信上游SFT永遠不能學新事實等過強口號。")
sw("PKU Safe RLHF分開建模helpfulness reward與safety cost。","參考專案策略表","README的Constrained Value Alignment用R和C兩個偏好代理，目標高reward、低cost。","README Constrained Value Alignment via Safe RLHF",("safe-rlhf",),artifact="a-upstream")
sw("MiniMind/MiniMind-V能在同一FFN位置選Dense或routed experts。","參考專案策略表","MiniMindBlock依use_moe選FeedForward或MOEFeedForward；後者保留gate、experts和負載均衡輔助loss。","model_minimind.py MiniMindBlock/MOEFeedForward",("mini-model-model_minimind.py","mv-default-llm"),artifact="a-upstream")
sw("MiniMind區分教師回答的黑盒蒸餾與CE+KL白盒蒸餾。","參考專案策略表","README區分硬標籤教師回答與分布蒸餾；train_distillation.py實作teacher/student、temperature、T²及有效mask的CE/KL。","README蒸餾; distillation_loss/train_epoch",("mini-README.md","mini-trainer-train_distillation.py"),artifact="a-upstream")
sw("TorchAO分開處理數值量化、低位元packing與執行kernel。","參考專案策略表","quantization_overview區分primitive ops、efficient kernels、derived dtype與packing format，例子int4兩元素pack入uint8。","quantization_overview.rst primitive ops/efficient kernels/Quantized Tensors",("ao-docs-source-contributing-quantization_overview.rst",),artifact="a-upstream")
sw("TorchAO QAT有prepare插fake quant及convert轉換兩階段。","參考專案策略表","qat.md明示prepare與convert，範例以QATConfig step='prepare'/'convert'呼叫。","qat.md prepare/convert文字與範例",("ao-docs-source-workflows-qat.md",),artifact="a-upstream")
con("模態對齊、行為對齊與架構比較是可分開定義及驗收的目標。","三類目標","MiniMind-V投影／凍結策略對接視覺與語言；Smol分開指令／偏好；Dense/MoE分支改計算模組，不能由一者成功替另一者驗收。","MiniMind-V init_vlm_model; Smol instruction/preference units; MiniMindBlock",("mv-trainer-trainer_utils.py","smol-units-en-unit1-3.md","smol-units-en-unit2-1.md","mini-model-model_minimind.py"),artifact="a-upstream")
con("量化以較低位元表示權重等數值，通常保留邏輯參數數量。","縮小部署成本","TorchAO表示由高精度Tensor建立低精度Tensor並加入scale metadata；packing改儲存排列，不等於刪除模型權重元素。","quantization_overview.rst Quantized Tensors與packing",("ao-docs-source-contributing-quantization_overview.rst",),artifact="a-upstream",scope="邏輯參數元素與實際打包tensor／metadata數量不同；不由位元數直接保證執行速度。")
con("蒸餾由教師提供學習訊號；配合較小學生架構才減少參數與計算。","縮小部署成本","上游黑盒／白盒定義和teacher/student訓練入口支持訊號來源；本專案學生分支明示較小Dense架構，縮小來自學生選擇而非KL本身。","MiniMind README蒸餾; train_distillation.py; local old capstone_student配置",("mini-README.md","mini-trainer-train_distillation.py"),artifact="a-upstream")
con("LoRA能減少需要微調的參數。","縮小部署成本","LoRA用in×rank與rank×out兩個可訓練低秩矩陣；MIT Lab3亦明示凍結基礎模型、僅更新子集。","model_lora.LoRA; MIT apply_lora說明",("mini-lora","mit-style"),artifact="a-upstream",scope="rank足夠小並凍結原權重時成立；未宣稱推論總參數、記憶體或延遲按同等比例下降。")
sw("notebook dependency group含ipykernel、matplotlib、JupyterLab與nbclient。","教材呈現方式的首跑指南","實際解析pyproject.toml notebook group，恰有這四個套件規格。","notebook_group",scope="核對依賴宣告；未重新安裝或測試Colab雲端。")
con("可讀前文的attention mask與決定哪些答案計分的loss mask用途不同。","文字訓練流程圖與其後正文","PyTorch causal attention限制可讀key；CE ignore_index只排除計分位置。CPU例子改未來V不影響過去輸出；只計labels中間位置的loss和該位置CE相等。","scaled_dot_product_attention is_causal; mechanics.causal/loss_positions",("torch-attention","mv-dataset-lm_dataset.py"),scope="loss忽略不表示上下文不可見，亦不表示作為其他位置上下文時完全沒有間接梯度；圖與正文沒有做這種錯誤外推。")
con("不累積舊梯度的一次更新可先forward/loss，再zero_grad、backward、optimizer.step；刻意累積則另排清零。","文字訓練流程圖與其後正文","zero_grad清梯度、backward算梯度；CPU把舊梯度99清掉後得-4，SGD lr0.1把1改為1.4；不清零的兩次backward得到雙倍梯度。","Optimizer.zero_grad; Tensor.backward; mechanics.update/accumulation",("torch-zero","torch-backward"),scope="一般沒有依賴param.grad的forward；只清optimizer所含參數，gradient accumulation的縮放與清零需另行設計。")
emp("v2視覺僅三類服飾與公開兩格位置。","目前教材與新增主題的主線範圍","manifest scope與final data_scope列Fashion-MNIST3與public two-slot relations，模型image入口固定兩槽與公開coordinates。","data_manifest.scope; selftrained.data_scope.vision_classes; LimitedAssistant._embeddings",{"vision_classes":3,"public_slots":2})
emp("v2 OCR限12個已知繁體字、提供的連續1–4字ROI，Sans/Serif都見於訓練。","目前教材與新增主題的OCR範圍","tokenizer OCR_CHARACTERS與manifest/固定結果的data_scope一致；明記provided ROI且兩種字型已見，沒有宣稱未知字、文字偵測或未知字型。","selftrained.data_scope.ocr_glyph_count/ocr_fonts/ocr_region_source; data_manifest.scope",{"glyphs":12,"min_roi_cells":1,"max_roi_cells":4,"fonts":["Sans","Serif"],"test_ocr_rows":324})
emp("v2語音限地址/App/卡片客服主題，聲音特徵直接產生回答；錄音切分不等於說話者隔離。","目前教材與新增主題的語音範圍","data_scope列三intent、train/validation/test錄音62/15/30且speaker unknown；AudioEncoder接共享LM，沒有runtime ASR或開放逐字聽寫。","selftrained.data_scope; data_manifest.scope; LimitedAssistant.__init__/_embeddings",{"voice_intents":3,"train_recordings":62,"validation_recordings":15,"test_recordings":30,"test_qa_rows":90},scope="紀錄單位是錄音而非獨立說話者；任務成績不支持開放聽寫或speaker-disjoint泛化。")
emp("新增B/C已完成GPU訓練並有設定凍結後最後測試。","新增成品現況","initial-b/c-execution記錄completed、NVIDIA L4、cuda、selected_step=1000/800；final-report綁settings_freeze_sha256，pretest freeze與設定紀錄存在。","product_training; products_final.settings_freeze_sha256/recorded_at; products_final.gates",{"products":2,"training_device":"NVIDIA L4","b_selected_step":1000,"c_selected_step":800})
emp("B的三數加法泛化失敗有保留。","新增成品現況","正式B各輪test sum共42題，1–6輪correct為0/0/1/2/2/3，gates.b_secondary_target=false。","products_b.*.sum; products_final.gates.b_secondary_target",{"sum_test_rows":42,"loops":[1,2,3,4,5,6],"correct":[0,0,1,2,2,3]})
emp("C在換說法時仍有錯誤。","新增成品現況","保存wording_shift錯例：天氣被判boxes、地址被判none、卡片進度被判boxes，包含原始scores與gold。","products_failure_examples.examples; products_final.shifts.c_wording",{"stored_distinct_examples":3,"split":"wording_shift"})
sw("B/C題目依原始情境家族繼承切分，仍可能有跨切分同文字／同標籤。","新增成品資料切分限制","生成器使用family split；data audit joint_family_overlap=0，但C cross_split_shared_observations非零；卡片進度;申請編號=?在四個split具有相同choice/info/rubric標籤。","products_data_audit; products_duplicate_example; code_excerpts.tiny_perceptron/lesson_products/data.py",scope="核對生成器及既有稽核紀錄，未從原始c.jsonl重跑完整去重；family隔離不等於觀測字串未見。")
sw("第一版Auto由C判斷任務與資訊充分性，再以固定規則選B預算。","新增成品Auto","auto_solve先呼叫choice和info，非B任務拒絕／缺資訊追問，auto模式用rule_depth；本回合執行公開policy得1–4事件用3輪、5–6事件用5輪、sum用1輪。","code_excerpts.tiny_perceptron/lesson_products/inference.py; executed_auto_rules",scope="固定規則由policy開發集選定；不是預測各預算收益的學習式控制器，且多輪不保證更準。")
sw("公開課來源確有evaluation/data/multimodality、style/post-training/CV分項實驗及RAG/Agents內容。","公開課補入內容與原始碼參考","直接讀CS336 lecture12/14/17、MIT兩年課站與Lab2/3、Harvard固定課綱，主題與頁面來源歸屬一致；台大僅核對鏡像索引的遺忘／推理／工具線索。","上游各課綱／講義／Notebook的已保存原始摘錄",("cs336-eval","cs336-data","cs336-multimodal","mit-2025","mit-2026","mit-style","mit-cv","harvard-2025","harvard-2026","ntu-mirror-topics"),artifact="a-upstream",scope="不核定台大完整官方課綱、影片內容／上傳日期，亦不把作者過往代理限制敘事當成本回合網路現況。")
sw("公開提問確實包含mask/shift、UTF-8、有效監督／截斷、資料配比及OCR需求。","學習者回饋補強與原始碼參考","讀原始issue：MiniMind476問loss_mask與attention_mask，633問第一token，CS33621問UTF-8，nanochat590報SFT NaN，MiniMind732問原始資料／配比，MiniMind-V36問OCR。","GitHub issue title/body與created_at/updated_at",("issue-jingyaogong/minimind-476","issue-jingyaogong/minimind-633","issue-stanford-cs336/assignment1-basics-21","issue-karpathy/nanochat-590","issue-jingyaogong/minimind-732","issue-jingyaogong/minimind-v-36"),artifact="a-upstream",scope="只支持定性的公開問題存在；未推定每位發文者身分、困難盛行率或NaN的單一成因。LR與梯度細節是設計動機，未當作因果效果證據。")
sw("歷史實驗紀錄保留CUDA Flash補驗、有限候選PPO及舊MoE整合的配置、結果與限制。","開頭歷史輔助證據","歷史索引直接連到CUDA Flash、reward/PPO、capstone及capstone部署／學生的配置與結果；公開清單的30組與額外整合清單分開。","historical_index; public_experiments; public_capstone",scope="只核對紀錄與索引保留，不將舊有限候選PPO當完整逐token語言PPO，也不拿其成績代替v2最後測試。")
sw("三類模態使用共享語言核心與同一種對話紀錄介面，各有不同特徵入口。","目前教材與新增主題的主線範圍","LimitedAssistant只建一個lm及vision/ocr/audio三個encoder；RecordEncoder將messages與模態槽轉成同一input_ids/labels介面。","code_excerpts.tiny_perceptron/selftrained/model.py / tiny_perceptron/selftrained/dataset.py",scope="僅限有限助理的實作介面，不表示所有模態可沿用相同原始素材或訓練訊號。")

assert local["inventory"]["all_numbered_sections"] == 338
assert not local["notebook_sync"]["stale"]
assert local["recounted_models"]["moe"]["parameters"] == 5447107
assert local["recounted_models"]["dense"]["parameters"] == 2288067
assert local["public_experiments"]["weight_count"] == 120
assert local["four_export_release_job"]
assert all(r["exit_code"] == 0 for r in records.values())

report = {
    "schema_version":1, "review_stage":"technical", "lesson_id":"curriculum", "source":"docs/curriculum.md#curriculum",
    "reviewer_task":"/root/rv1019_technical_curriculum", "reviewer_context":"fresh", "author_tasks":["/root"],
    "source_sha256":"e7e30fb141095edb5b9d95a4659e37285f1b4fef7a7935066c407f3bf7ac75e9",
    "source_freeze":{"path":"docs/course-review-20261019/freeze-04.json", "sha256":"a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"},
    "figure_sha256":{"course/figures/curriculum_learning_flow.svg":"55c76a1ce7d7482025064283219b7449ab13ebd7c4c518979f5a55fa16d41da4"},
    "reviewed_at":datetime.datetime.now(datetime.UTC).isoformat(), "verdict":"pass", "claims":claims, "sources":list(sources.values()), "artifacts":artifacts, "issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"逐項核對整份凍結原文的實質主張；分開教材數量、固定產品事實、上游版本與方法定義。教學應然規則與作者歷史行程不冒充已證明的教學效益。","claim_ids":[c["id"] for c in claims]},
        "numeric_verification":{"status":"pass","details":"重新枚舉20章+A/B/C、312份Notebook、338小節；重建兩版模型數參數；加總各final task得到3734；直接核對固定訓練／選定步數與歷史分母。原始LFS資料未逐列重算，資料筆數驗證限固定紀錄交叉一致性。","claim_ids":[c["id"] for c in claims if c["kind"] in ("numeric","empirical")]},
        "figure_consistency":{"status":"pass","details":"已用view_image實際看outputs/course-review-20261019/figure-renders-04/55c76a1ce7d7482025064283219b7449ab13ebd7c4c518979f5a55fa16d41da4.png（PNG SHA d67c4934240c17bb43659e993c7bfee4edc4db97b18b69e3d98c0a76fb72e4fc），並讀SVG。順序為raw→roles/IDs→X/Y→forward→指定位置loss→zero_grad→backward→step；兩種mask分開，梯度累積例外明示，沒有數值或方向矛盾。PNG不列artifact；檢視記錄與SHA在local-evidence.figure_viewed。","claim_ids":[c["id"] for c in claims if "mask" in c["statement"] or "zero_grad" in c["statement"]]},
        "source_verification":{"status":"pass","details":"上游以curl直接讀六個指定專案commit與官方課綱／教材；另讀PyTorch v2.14.1原始碼、HF固定tree與inference manifests、原始提問。URL、實讀版本、日期、定位與完整原檔SHA都留收據；未採摘要作方法證據。","claim_ids":[c["id"] for c in claims if any(e["source_id"].startswith("s-") and e["source_id"] not in ("s-local","s-mechanics") for e in c["evidence"])]},
        "limitations":{"status":"pass","details":"正文已區分歷史小世界與v2、錄音與說話者、已知字型／ROI與開放OCR、語音QA與逐字聽寫、模型能力與執行通路；top-2/Dense不同目標歷史不作受控比較。B加法失敗、C換句錯誤與跨家族同文字限制保留；Auto為規則而非學習式收益控制。沒有以本回合CPU機制例子推論GPU效益或一般品質。","claim_ids":[c["id"] for c in claims if c["kind"] in ("empirical","concept")]},
    },
    "claim_counts_by_kind":dict(collections.Counter(c["kind"] for c in claims)),
    "independence":{"prior_authoring":False,"other_reader_reports_opened":False,"disclosure_artifact_id":"a-scope","details":"原始判斷存檔以前，意外rg命中docs/selftrained/infrastructure內兩份integrity/preparation review的檔案路徑、HF版本片段，沒有顯示其verdict或問題處置；沒有把命中當證據。明確揭露訊號C有。未開啟本輪他人報告或派工禁止的歷輪目錄。"},
    "scope_limits":[
        "只做事實與技術審閱，未做初學讀者理解審閱，也沒有真人學生測試。",
        "未重跑完整GPU/CPU訓練、所有逐題推論、Colab雲端或上游六專案訓練；固定成績以已保存實驗配置／結果與公開manifest核對。",
        "assets/training/selftrained-v2.tar.gz目前只有133-byte LFS pointer，未下載大型資料／權重；訓練及validation資料筆數採固定紀錄核對，未逐列重算。",
        "未重新驗證作者過往網路阻擋、瀏覽器實測、是否親自讀過來源及四角度審閱等行政歷程；不將它們當現時技術契約或已證明的學習成效。",
        "台大資料只以鏡像原始索引確認線索，未核定官方完整2026課綱；沒有觀看影片或驗證上傳日期。",
    ],
    "incidental_observations":[{"path":"scripts/selftrained/train.py","identifier":"parser","details":"額外執行uv run --no-sync python scripts/selftrained/train.py --help，Python 3.13.16 argparse回報ValueError: incomplete format。wrapper help成功，parser登記resume/init-checkpoint。該help指令不在本頁，未當作本頁實質主張的反證；stderr完整留在a-mechanics。本次未修改程式。"}],
    "escalation_signals":{
        "A":"無 — 查核方式：核對freeze-04原文／圖SHA、check_technical_report.py、scripts/check_technical_reviews.py及本回合三個驗證程式輸出；未發現派工宣稱被推翻。",
        "B":"無 — 查核方式：本單未附已駁回／不要重提清單，無可核對的編號區間。",
        "C":"有 — 授權搜尋時glob未完整排除兩份既有selftrained infrastructure review的版本／路徑片段，見a-scope；沒有開啟全文或見到其裁決，且未作證據。只寫指定兩目錄。",
        "D":"無 — 查核方式：只新增本人審閱與artifact，讀取凍結原文、SVG及必要實作時未見與本次判斷對應的裁決編號或被改寫註解；未修改任何修復位置。",
    },
}
assert digest("docs/course-review-20261019/freeze-04/sources/curriculum.md") == report["source_sha256"]
assert digest(report["source_freeze"]["path"]) == report["source_freeze"]["sha256"]
spec = importlib.util.spec_from_file_location("technical_report_validator", ROOT / "scripts/check_technical_reviews.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
errors = module._validate(ROOT, ROOT/"docs/curriculum.md", "curriculum", (ROOT/"docs/course-review-20261019/freeze-04/sources/curriculum.md").read_text(), report, {report["reviewer_task"]:["curriculum"]}, {"/root/rv1019_reader_ref"})
assert not errors, errors
(ROOT/DEST).parent.mkdir(parents=True, exist_ok=True)
with (ROOT/DEST).open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"report":DEST.as_posix(),"sha256":digest(DEST),"verdict":report["verdict"],"claims":len(claims),"counts":report["claim_counts_by_kind"]},ensure_ascii=False))
