import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parents[2]
E = B / "evidence/technical-modalities"
M = json.loads((B / "manifest.json").read_text())
RAW = json.loads((B / "checks/technical-inputs-manifest.json").read_text())
RUN = json.loads((E / "execution.json").read_text())
TASK = "/root/repair_tech_modalities"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = []
def source(id, path, version, locator, note, kind="repository_code", url=None):
    s = {"id": id, "kind": kind, "path": str(path.relative_to(B)), "sha256": sha(path),
         "version": version, "locator": locator, "inspection_note": note,
         "checked_original": True, "verified": True}
    if url:
        s["url"] = url
    sources.append(s)

for id, path, locator, note in [
    ("impl-multimodal", "tiny_perceptron/multimodal.py", "49–83 mel_filter_bank/log_mel/AudioEncoder；103–135 MultiModalLM；scene", "核對固定三角頻帶、Hz/mel 換算、矩陣方向、模態零件與圖形位置。"),
    ("impl-model", "tiny_perceptron/model.py", "15–30 ModelConfig；33–69 Block/TinyLM", "核對 width/layers 預設與計數用的模組。"),
    ("impl-alignment", "tiny_perceptron/alignment.py", "29–36 sequence_log_probability；39–43 dpo_loss", "核對 -100 mask、log_softmax、gather、sum 與固定參考差值；helper 不自行 shift。"),
    ("impl-data", "tiny_perceptron/data.py", "ByteTokenizer；65–101 render_chat/pad_batch", "核對 UTF-8 byte、assistant 內容與 EOS 的有效目標；前文及 PAD 為 -100。"),
    ("impl-modalities", "scripts/course_experiments/modalities.py", "_fit；_vision_records；_loss_fn/_evaluate；_freeze；584–769 run_contrastive/run_vqa；985–994 _resample_8_to_16", "核對原試驗 code SHA 與凍結實作相同；實讀抽樣、更新、逐題 exact-match、位置留出、接頭承接與預算停止。"),
    ("impl-common", "scripts/course_experiments/common.py", "50–69 split_records；205–245 fit", "核對家族切分；DPO 每個更新只調用一次 loss_fn，沒有額外 probe 算入曝光。"),
    ("impl-train", "scripts/train.py", "263–291 多模態、freeze partial 與優化器", "核對 CLI partial 為兩接頭與首末文字區塊，與固定實驗的最後區塊約定不同。"),
    ("impl-audio-utils", "scripts/audio_utils.py", "10–79 resample_waveform/load_mono_audio", "核對新時間點、cutoff=min(1,target/source)、單聲道與振幅檢查；本次只執行既有 8k→16k 算例。"),
    ("impl-text", "scripts/course_experiments/text.py", "44–64 _save_splits；596–612 arithmetic_records", "只用確定性資料生成與固定 JSONL 序列化；重建後三份 DPO split SHA 與原紀錄相同。"),
]:
    p = B / "freeze/implementation" / path
    assert sha(p) == M["implementation_sha256"][path]
    source(id, p, "freeze-01 / " + M["baseline_commit"], locator, note)

for name, locator, note in [
    ("contrastive", "results.data；results.variants.{one_way,two_way}.{training,validation,test}", "實讀所有 split records、兩向逐題樣本、訓練 history 及 config；最後兩向 6/6，驗證圖找文 5/6。"),
    ("vqa", "results.data；variants.{projector_only,partial,all,direct_vqa}；two_stage_alignment_training；direct_vs_two_stage_budget", "實讀 records、history、config 及相關逐題答案，重加總 3830、14408、18256 與答對分子；直接版確無 text_after。"),
    ("real_modal", "results.fsdd.resampling[0]；code_sha256", "核對 0_jackson_5.wav 的 8000→16000、4591→9182、來源 SHA 與方法；本頁不作語音答題能力驗收。"),
    ("dpo", "results.data；results.runs.{model,beta1}.training；code_sha256", "核對兩支各 250 更新、兩側 9448 有效回答目標；重建 train/validation/test JSONL 的 SHA 全吻合並按原抽樣重算。"),
]:
    relative = "docs/course-experiments/results/" + name + ".json"
    p = B / "freeze/technical-data" / relative
    assert sha(p) == RAW["files_sha256"][relative]
    d = json.loads(p.read_text())
    source("raw-" + name, p, d["revision"], locator, note, "raw_experiment_record")
    sources[-1]["environment"] = {k: d[k] for k in ["seed", "device", "torch_version", "python_version", "gpu"]}

for id, name, version, locator, note, kind, url in [
    ("original-clip", "clip-radford21a.pdf", "ICML 2021 / PMLR 139, radford21a", "§2.2 / Figure 3（PDF p.3；擷取文字 173–225）", "實讀成對相似度、兩軸交叉熵與平均；只支持雙向機制，未把 CLIP 規模或品質轉移至本課。", "paper", "https://proceedings.mlr.press/v139/radford21a/radford21a.pdf"),
    ("original-dpo", "dpo-2305.18290v2.pdf", "arXiv:2305.18290v2", "§4 Eq.(5)–(7)，Appendix B dpo_loss（PDF pp.4、20）", "實讀同篇 log(pi/pi_ref) 與兩篇的差；支持後續參考分數方向，不是本頁所有偏好效果的證明。", "paper", "https://arxiv.org/pdf/2305.18290v2"),
    ("official-torchaudio", "torchaudio-functional-v2.8.0.py", "pytorch/audio v2.8.0", "429–590 Hz/mel、triangular filterbank；1383–1420 sinc resampling", "實讀 HTK 2595×log10 正反換算、等 mel 點、重疊三角與兩種尺度。此官方未面積歸一 HTK 矩陣方向與本課相反，不能直接混用 shape。重取樣用較低源/目標帶寬與 sinc 插值。未安裝或呼叫 torchaudio。", "official_source", "https://raw.githubusercontent.com/pytorch/audio/v2.8.0/src/torchaudio/functional/functional.py"),
    ("official-scipy", "scipy-signaltools-v1.16.1.py", "SciPy v1.16.1", "3808–3868 resample_poly；3965–3972 FIR cutoff", "實讀升取樣、低通 FIR、降取樣與相對 Nyquist 的 cutoff；核對一般順序，未執行 SciPy。", "official_source", "https://raw.githubusercontent.com/scipy/scipy/v1.16.1/scipy/signal/_signaltools.py"),
    ("original-smith", "smith-resample-theory.html", "Julius O. Smith III / CCRMA；copyright 2020-09-17；2026-10-08 固定快照", "Theory of Ideal Bandlimited Interpolation；HTML 60–164；Eq.(1)–(2)", "實讀 |ω|≥πFs 頻譜為0的理想條件、唯一 sinc 重建、按新時間點求值及降取樣截止。數學圖片 ALT 亦已讀。", "author_published_text", "https://ccrma.stanford.edu/~jos/resample/Theory_Ideal_Bandlimited_Interpolation.html"),
]:
    source(id, E / "original-sources" / name, version, locator, note, kind, url)
source("installed-torch", E / "torch-inspection.json", "PyTorch " + RUN["environment"]["torch"],
       "Module.requires_grad_；Adam._init_group",
       "實讀本機官方模組：遞迴設參數開關；grad 非 None 的參數才進更新組並懶建立一階/二階狀態。原檔 SHA、起始行及 source 存於快照；AdamW 使用此 Adam 狀態初始化。",
       "installed_official_source")

def ref(id, locator, support):
    return {"source_id": id, "locator": locator, "supports": support}

def run(locator, support):
    return {"artifact_id": "execution", "locator": "checks." + locator, "supports": support}

def claim(id, kind, quote, lines, checked, evidence, scope, executed=False):
    return {"id": id, "kind": kind, "original_quote": quote, "source_location": lines,
            "status": "verified", "check": checked, "evidence": evidence, "support_scope": scope,
            "execution_status": "executed" if executed else "source_or_derivation_checked"}

pages = []
def page(id, commitments, claims, four, taught, remaining_note, limits):
    entry = next(p for p in M["inventory"]["pages"] if p["page_id"] == id)
    p = B / "freeze/sources" / (id + ".md")
    assert sha(p) == entry["source_sha256"]
    for file, expected in entry["figures_sha256"].items():
        assert sha(B / "freeze/original" / file) == expected
    pages.append({"page_id": id, "source": entry["source"], "source_sha256": sha(p),
        "figures_sha256": entry["figures_sha256"], "original_question": entry["title"],
        "read_scope": "指定凍結正文與全部選讀；不是逐段首讀實驗",
        "original_commitments": commitments, "verdict": "supported_with_stated_limits",
        "technical_checks": claims, "four_question_relationship_check": four,
        "taught_relationships": taught, "remaining_relationships": [],
        "remaining_relationships_note": remaining_note, "necessary_findings": [], "optional_findings": [],
        "no_findings_note": "在所列原句承諾及實際核對範圍，未發現必要缺口；沒有為填問題另要求無效果承諾的全面實驗。",
        "unknowns_and_limits": limits})

page("10.11", [
    {"quote": "同一張配對分數表，找文字時在每橫列選候選；找圖片時則在每直欄選候選。", "location": "L3", "depth": "兩種查詢的候選競爭及雙向用途"},
    {"quote": "雙向平均表達兩種查詢都重要，但不保證每次效果勝過單向。", "location": "L25", "depth": "有限實驗與能力邊界"},
], [
    claim("10.11-c1", "concept", "原表算「圖找文」，轉置 scores.T 讓文字變橫列、圖變直欄，算「文找圖」。", "L3、L21–23",
        "行內正確/錯誤差為1.9、1.4；欄內為1.6、1.7。兩軸 softmax 分母不同，非同式重複。CLIP Figure 3 確用兩軸 CE 平均。",
        [ref("original-clip", "§2.2 / Figure 3", "兩向候選競爭及平均"), ref("impl-modalities", "run_contrastive loss_fn/evaluate", "本課候選軸")],
        "同批一圖一描述且正確索引同列的入門例；多正確配對另處理。"),
    claim("10.11-c2", "numeric", "兩個交叉熵約 0.1799、0.1758，平均約 0.1779；梯度符號是 [[-1,1],[1,-1]]。", "L9–21",
        "按 -log softmax 兩行/兩列均值核對；原程式實跑0.1799020469、0.1758433878，均值0.1778727174，梯度符號相同。只對手設 scores backward，無權重更新。",
        [run("snippet-10.11", "原程式 stdout"), run("weighted-10.11", "精確值及梯度")],
        "CPU float32 算例；四位小數舍入容差5e-5。", True),
    claim("10.11-c3", "empirical", "兩種版本的最後題、兩個方向都為 6/6，沒有觀察到答對數優勢；驗證也非全部正確。", "L25",
        "重數兩版 test.image_samples/text_samples，均6/6；validation 圖找文均5/6、文找圖6/6。各250更新、seed42、每批六標籤，曝光1500。train offset[-2,-1,0]、validation[1]、test[2]，位置家族不交叉，屬性組合仍相同。",
        [ref("raw-contrastive", "data、variants.*", "原條件及逐題資料"), run("historical-recounts.contrastive", "逐題及 history 重數"), ref("impl-modalities", "584–646 run_contrastive", "相同seed、初始化規則及單/雙向差異")],
        "六種已知顏色形狀的新位置檢索；非大型檢索或新屬性泛化，未重作模型推論。", True),
    claim("10.11-c4", "concept", "檢索是在既有卡片中選一張，生成描述要逐步寫出文字。", "L27",
        "本實驗 evaluate 是六候選 argmax，無自由描述生成；10.10 已教同描述的假負例，因此沿用多正確規則有前文。",
        [ref("impl-modalities", "607–635 evaluate", "只選有限候選"), ref("prior-10.10", "正文假負例段", "多正確配對")],
        "不能由檢索分數推出描述能力。"),
    claim("10.11-c5", "numeric", "練習改成 (2*image_to_text + text_to_image)/3 ... 代價更靠近圖找文。", "L29",
        "實跑加權均值0.1785491705，距圖找文0.0013528764、距文找圖0.0027057827；梯度形狀仍[2,2]。只改權重，沒有新增觀測。",
        [run("weighted-10.11", "加權結果及 shape")], "線性加權算例，非品質優勢。", True),
], {
    "identity_role": "兩方向 CE 分別為圖查詢與文查詢評錯；正確對角標籤由10.10已教。",
    "mechanism_property": {"answer": "行與列分母不同，同分數收到兩向導數；非對稱表令差異可見。", "basis": "L3、L21–23", "status": "已足夠"},
    "need_use": {"answer": "來源已指出兩種查詢都重要，平均使兩種查找參與目標；沒有改成效果必勝。", "basis": "L3、L25", "status": "已足夠"},
    "operation": "scores 與 scores.T 對同 labels 求 CE，平均並只對手設表 backward。",
    "example_support": "非對稱表教兩向競爭；有限歷史結果教不可保證勝過單向的邊界。"
}, ["正確配對→行/欄候選競爭→各向代價→平均梯度。", "非對稱表解釋兩向不必相等。", "檢索及描述生成分需證據。"],
    "原問題在L23–27回收，不需審閱者另造查詢用途才連起理由。",
    ["未重訓或載入歷史 checkpoint；同起點依原seed/初始化碼核對。", "未驗證大型自然圖文或多正確配對的實際訓練。", "本頁無圖；2×2矩陣與兩組競爭由文字完整定位，不要求裝飾圖。"])

page("11.5", [
    {"quote": "先數每種方案允許改幾個數字，再用相同資料與預算比較答案；參數多不代表新題一定更好。", "location": "L3", "depth": "成本線索、更新範圍及有限效果"},
], [
    claim("11.5-c1", "software", "寬 8 的預設模型印 projector 136、partial 976、all 8296。", "L9–19",
        "實跑相同；136=8×16+8，partial=136+840（最後block）。all包含language、vision、audio與兩接頭，每輪 requires_grad_ 遞迴重設。",
        [run("snippet-11.5", "三方案原stdout"), ref("impl-model", "ModelConfig、Block/TinyLM", "寬8一層預設"), ref("impl-multimodal", "MultiModalLM.__init__", "16寬入口/兩接頭"), ref("installed-torch", "Module.requires_grad_", "遞迴開關")],
        "可訓練名單計數；無 forward、optimizer step 或顯存測量。", True),
    claim("11.5-c2", "concept", "all 包含聲音零件，純圖片輸入未必讓它們收到梯度。", "L19–21",
        "forward 僅 waveform 非None才算 audio/audio_projector；純圖分支不使用它們。Adam 僅為 grad 非None參數懶建歷史狀態，所以參數數是成本線索，非實測記憶體。",
        [ref("impl-multimodal", "111–135 forward", "聲音分支條件"), ref("installed-torch", "Adam._init_group", "狀態建立條件")],
        "此實作與所查PyTorch版本；沒有量測訓練峰值。"),
    claim("11.5-c3", "empirical", "文字寬 64、兩層，圖片入口寬 16 ... 每版都使用3,830個有效回答目標。", "L23–25",
        "三variant配置為width64/layers2/vision_width16；各160更新、seed42。history加總3830；依每步seed及四次抽樣離線重算也3830，含答案byte和EOS。實驗傳入非空replay池，即使比例0仍先消耗rng.random()，本次已照該規則重算。",
        [ref("raw-vqa", "三variant.training", "配置、參數與history"), run("historical-recounts.vqa.variants", "history及sampler重算"), ref("impl-modalities", "_loss_fn / run_vqa", "抽樣與零比例分支")],
        "3830是重複曝光，非不同樣本數；不把width8例當成歷史訓練。", True),
    claim("11.5-c4", "empirical", "圖片接頭 ... 3/12；接頭＋最後文字區塊 ... 12/12；全部 ... 9/12。", "L27–33",
        "逐題 exact_match 重數：驗證3/12、10/12、12/12；test3/12、12/12、9/12。可訓練數1088、50816、145664。每評測72有效答案目標、EOS各12/12。train為18張位置圖×2題=36；validation/test各6張新位置×2題=12，屬性組合不變。",
        [ref("raw-vqa", "data及三variant.validation/test.samples", "原逐題資料"), run("historical-recounts.vqa", "分子分母/家族重數"), ref("impl-modalities", "_evaluate", "raw答案byte exact-match及EOS分列")],
        "完整短答exact-match，非token accuracy；12題一seed不推出普遍最佳範圍。", True),
    claim("11.5-c5", "software", "固定工具的 partial 是最後區塊；T.6的訓練CLI則把partial定義為接頭加首末文字區塊。", "L35",
        "實驗 _freeze 只開blocks[-1]；CLI開blocks[0]/[-1]及image/audio兩接頭。正文已交代入口差異。兩層練習實跑projector136、partial976、all9136；最後一層未變而all加840。",
        [ref("impl-modalities", "374–384 _freeze", "固定實驗範圍"), ref("impl-train", "278–288", "CLI範圍"), run("layers-exercise-11.5", "兩層計數")],
        "首末同一block不重數；兩層CLI partial不等同最後單層partial。", True),
], {
    "identity_role": "requires_grad 範圍決定允許學的參數；11.2已教名單、優化器與真正更新的差別。",
    "mechanism_property": {"answer": "先重設再開接頭/最後區塊/全部，改變參數清單；開放和收到梯度分開。", "basis": "L10–21", "status": "已足夠"},
    "need_use": {"answer": "原需求是比較問答效果與成本；數量給成本線索，留出題給效果，原文字題另判保留。", "basis": "L3、L21、L23–33", "status": "已足夠"},
    "operation": "按入口設開關、數numel，固定配置/資料/有效目標預算再看答案。",
    "example_support": "width8程序教計數；width64歷史表另教效果。all的9/12顯示參數排名不能直接作答案排名。"
}, ["允許更新數量→成本線索，不等於實際梯度/記憶體。", "更新範圍、模型寬深、預算、評估題分別說明。", "同圖兩題同組→新位置留出→有限短答表。"],
    "成本與效果各有依據，未只用參數計數放行效果。",
    ["未量測AdamW狀態、FLOPs、GPU/MPS記憶體；正文未給這些實測數。", "未重新推論或用blank/shuffle證明視覺因果作用；此表是有限VQA答案紀錄。", "同初始模型由copy.deepcopy(initial)核對，未驗歷史checkpoint bytes。", "本頁無圖；更新範圍用程式/表格可定位，此問題不依賴觀看像素。"])

page("11.6", [
    {"quote": "如果第一方案多花一整段訓練，最後較好可能只是多用了資料。先把兩者的總有效回答位置排成相同預算。", "location": "L3", "depth": "比較條件與資源混淆"},
    {"quote": "第一階段接頭需要帶到第二階段，不能花完預算又從隨機接頭重新開始。", "location": "L18", "depth": "階段承接與用途"},
], [
    claim("11.6-c1", "numeric", "兩階段分 4,000＋6,000，直接問答用 10,000，程式都印總數 10,000、True。", "L6–16、L24",
        "原碼實跑兩計畫均10000 True；4000改2000並保留6000為8000 False，第二項改8000才10000；練習變體為整數手算，未另執行。這是預算表，無訓練。有效目標是labels非-100，包含EOS。",
        [run("snippet-11.6", "預算原stdout"), ref("impl-data", "render_chat", "回答/EOS位置"), ref("impl-modalities", "_fit/_loss_fn", "曝光計量")],
        "只對曝光目標量，不保證樣本數、FLOPs、時間或統計難度相同。", True),
    claim("11.6-c2", "concept", "實際比較還要同一初始模型、明確的更新範圍、同一批留出圖文題，以及基模原本會做的文字題。", "L18",
        "原需求維持比較先描述再問答與直接問答；同曝光減少額外訓練量混淆，再固定起點/題與記明範圍；文字保持另需文字題。正文沒有說曝光相同便已控制所有因素。",
        [ref("impl-modalities", "run_projector/run_vqa", "同SFT/encoder起點、留出split及範圍"), ref("raw-vqa", "scope、direct_vs_two_stage_budget", "有限比較")],
        "一seed課程配方；階段訓練任務及開放範圍本來不同。"),
    claim("11.6-c3", "software", "第一階段接頭需要帶到第二階段。", "L18–20",
        "run_vqa從完整projector MultiModalLM checkpoint載入initial，copy.deepcopy至問答；direct用_modal從同SFT與encoder建模。描述只開image_projector，後續all/direct均all，沒有對齊後再隨機接頭。",
        [ref("impl-modalities", "696–718、736–753", "載入與承接"), ref("raw-vqa", "two_stage_alignment_training", "1088可訓練、300更新")],
        "核對程式資料流及紀錄；未逐bit重構初始checkpoint。"),
    claim("11.6-c4", "empirical", "兩階段合計 18,238 個有效目標，直接方案到最後一批為 18,256，差 18 個。最後圖片問答分別 9/12、10/12。", "L20",
        "alignment history=14408，all問答=3830，總18238。direct762更新history=18256，超額18，最後一批達預算才停。兩版test逐題重數9/12、10/12，材料同一新位置留出。",
        [ref("raw-vqa", "two_stage_alignment_training、variants.all/direct_vqa、budget", "原預算與樣本"), run("historical-recounts.vqa", "曝光加總、逐題重數"), ref("impl-modalities", "_fit 97–109 / run_vqa 742–766", "最後批預算停止")],
        "約0.099%曝光差已明說；一題差不支持普遍優劣或完全無影響。", True),
    claim("11.6-c5", "empirical", "直接版本未另列原文字留出分數，所以這個比較不支持文字保留結論。", "L20",
        "direct結果只有training/before/validation/test，無text_before/text_after；其他all分支的文字分數不能補成direct的測量。",
        [ref("raw-vqa", "variants.direct_vqa", "實際缺文字保持紀錄"), run("historical-recounts.vqa.two_stage.direct_has_text_after", "False")],
        "未知已在正文界定；不把缺字段判成退化或保持。", True),
], {
    "identity_role": "描述對齊教圖→描述，問答SFT教圖＋問題→指定答案；都可用文字預測損失。",
    "mechanism_property": {"answer": "兩階段分配曝光並承接已學接頭，直接版全用問答；範圍需明記。", "basis": "L3、L16–22", "status": "已足夠"},
    "need_use": {"answer": "正文先指出額外訓練量混淆；同曝光/留出題使結果能解釋至這份安排，原文字題另驗保留。", "basis": "L3、L18、L20", "status": "已足夠"},
    "operation": "累積實際回答目標、帶接頭到第二段、記最後批差，不能手改測量值。",
    "example_support": "預算表只驗計畫；原資料支持18238/18256與9/12/10/12，正文解讀一題差及缺文字測量。"
}, ["額外階段→總曝光混淆→同有效目標預算。", "已學接頭延續才是先對齊再問答。", "圖片問答與原文字保持各需證據。"],
    "來源已建立比較目標、資源及承接關係，未讓預算算術代答方法理由。",
    ["未以同曝光證明同FLOPs、同時間或嚴格統計等價。", "未補direct文字留出或重訓；未知已在正文明說。", "本頁無圖；預算表和文字足以表示分配/先後，真承接另查程式。"])

page("12.2", [
    {"quote": "拿到 1600 個聲音樣本，還不能知道錄了多久。", "location": "L3", "depth": "時間單位與取樣率"},
    {"quote": "降低取樣率時，先把新奈奎斯特界線以上的成分減弱或濾除 ... 接著在新的時間點計算數值", "location": "L15", "depth": "低通目的與重取樣動作"},
], [
    claim("12.2-c1", "numeric", "時長是 樣本數／取樣率；440 Hz 每週期的樣本數是 取樣率／440，約 36.36 與 18.18。", "L3、L8–13",
        "N=rate×t，故t=N/rate；每週期1/440秒乘rate得rate/440。原程式印0.1/0.2、界線8000/4000、每週期36.36/18.18。",
        [run("snippet-12.2", "時間與頻率換算stdout"), ref("original-smith", "Fs=1/Ts，Eq.(1)–(2)", "取樣時間定義")],
        "音訊緩衝時長N/Fs約定；不把末點時刻(N-1)/Fs當整個播放時長。", True),
    claim("12.2-c2", "concept", "高於界線的變化可能混成較低頻率，不能由這串樣本唯一重建。", "L5",
        "CCRMA明示|ω|≥πFs譜為0的理想唯一重建條件。透明反例cos(2π(Fs-f)n/Fs)=cos(2πfn/Fs)：Fs8000時7000Hz與1000Hz樣本相同，CPU float64最大誤差1.83e-12；僅樣本不能唯一分辨。",
        [ref("original-smith", "HTML60–111", "帶限與唯一重建"), run("alias-12.2", "高/低頻同樣本算例")],
        "理想均勻取樣及只憑這串樣本；不是所有ADC或有先驗推斷的驗證。", True),
    claim("12.2-c3", "concept", "直接以 8000 Hz 播放 ... 440 Hz 聽起來變 220 Hz。正確改取樣率需要重取樣。", "L13–15",
        "原每週期16000/440點不變，播放每秒只走8000點，頻率=8000/(16000/440)=220；N/Fs也加倍。與兩份各自正確記440Hz的錄音不同，正文已區分。",
        [ref("original-smith", "Fs及新時間點", "時間軸"), {"kind": "derivation", "details": "f_play=f_source×Fs_play/Fs_source=440×8000/16000=220；t_play/t_source=2。"}],
        "算術推導，未播放或作聽覺實驗。"),
    claim("12.2-c4", "concept", "保留低頻、壓低高頻的動作叫低通處理；否則它們可能在新樣本中混成假的低頻。", "L15–19",
        "CCRMA明示降取樣cutoff在新率一半以下，再按新時間點算sinc；SciPy polyphase官方先低通再降。升取樣是在更密時間點求已有帶限重建，不創造已失資訊。上取樣亦可用濾波插值/去鏡像，L17未承諾完全不用濾波。",
        [ref("original-smith", "HTML141–164", "新時間點及降取樣cutoff"), ref("official-scipy", "resample_poly文檔", "低通與降取樣"), ref("official-torchaudio", "1383–1420", "較低源/目標帶寬及sinc")],
        "理想關係與有限窗近似；非有限FIR/邊界精確無誤保證。"),
    claim("12.2-c5", "empirical", "4591 點的 8000 Hz 聲音重取樣成 9182 點的 16000 Hz 聲音，兩者時長仍為 0.573875 秒。", "L21",
        "原resampling[0]為0_jackson_5.wav；本地來源SHA=5b47a9df...915吻合。實讀4591/8000，按原sinc函式求9182；N/Fs同0.573875。current audio_utils對同輸入陣列max error=0。重建WAV容器SHA與歷史derived不同，未當位元組重現。",
        [ref("raw-real_modal", "results.fsdd.resampling[0]", "來源及歷史紀錄"), run("fsdd-example-12.2", "讀檔/重取樣/時長"), ref("impl-modalities", "985–994", "原8→16k函式")],
        "僅該讀檔/重取樣，非語音模型重訓或能力驗收。", True),
], {
    "identity_role": "取樣率把序號轉時間，Nyquist限制可唯一表示的頻率；12.1已教振幅/週期。",
    "mechanism_property": {"answer": "改播放rate拉伸時間，真重取樣換時間點與數量，降取樣先減弱新界線以上成分。", "basis": "L13–17", "status": "已足夠"},
    "need_use": {"answer": "要知道時長並維持同聲音的時長/保留頻率，且避免高頻混成假低頻；低通定義作用當場建立。", "basis": "L3、L5、L15、L21", "status": "已足夠"},
    "operation": "N/Fs讀時間、Fs/2判界線，按目標時間點求值並核對rate/振幅。",
    "example_support": "1600點教rate；440Hz重標與真重取樣分開，真人例驗時間而非語音能力。"
}, ["樣本數＋每秒點數→時長。", "同陣列改rate→時間/頻率縮放；重取樣換時間點。", "新界線→低通防混疊；升取樣不找回已失資訊。"],
    "時間換算、低通動作及用途都有正文因果；選讀不負責補必要關係。",
    ["未測ADC、實際播放、CPU以外後端、所有比率/頻率響應。", "歷史derived WAV SHA=b982a86a5234e727511efa25b81b6accd404b5486b6d9ca2d994fb6caabd93eb；本機WAV SHA=e7b53e826e5768f8778142c154ca2d6f69c51804d75a5b96af99bca02ce3b69e，原因未確定，不判教材缺陷。", "本頁無新圖；時間與頻率由12.1波形背景及本頁明示數字推得，不冒稱看過原位前文圖。"])

page("12.6", [
    {"quote": "頻譜每框有 201 個頻率格，想用 16 個較寬區段表示能量。", "location": "L3", "depth": "頻帶摘要操作/用途"},
    {"quote": "這是本例採用的表示約定，16 不是最佳帶數的結論，答題效果仍要由後續訓練與檢查判定。", "location": "L9", "depth": "配置關係與效果邊界"},
], [
    claim("12.6-c1", "concept", "中間權重最大、兩側逐漸降至0 ... 相鄰三角形可以重疊，共同接收同一頻率。", "L3、L27",
        "mel_filter_bank以bands+2等mel點換Hz，取左右斜率最小值clamp≥0，為重疊三角。Torchaudio HTK同構；官方(201,16)，本課(16,201)，不直接混方向。",
        [ref("impl-multimodal", "49–61", "固定非負三角"), ref("official-torchaudio", "496–590", "等mel點及重疊")],
        "HTK式、無面積歸一、16000Hz/n_fft400；其他mel尺度/歸一可不同。"),
    claim("12.6-c2", "numeric", "mel＝2595×log10(1＋f／700)；0、700、2100 Hz 約為 0、781、1562 mel。", "L7",
        "官方HTK公式一致；實算0、781.1728387、1562.3456775。兩mel差相同，Hz差700、1400；反函式Hz=700(10^(m/2595)-1)，等delta隨m增大對應較大Hz跨度，故密低疏高可追蹤。",
        [ref("official-torchaudio", "429–478", "正反換算"), run("mel-12.6", "Hz/mel數值")],
        "整數近似容差0.5；非聽覺/答題收益證明。", True),
    claim("12.6-c3", "software", "矩陣相乘得到 (16,3)，時間框沒有被平均掉。權重非負印 True", "L17–25",
        "原碼實跑bank(16,201)、output(16,3)、True。每帶一列，只合併201頻率軸；三時間列各自保留。全1功率明確示範，非真人特徵。",
        [ref("impl-multimodal", "49–72", "矩陣軸/使用方向"), run("snippet-12.6", "原shape輸出")],
        "變換與軸意義，未證明聲音辨識品質。", True),
    claim("12.6-c4", "numeric_and_figure", "第一帶權重 [1,0.5,0]，得到 5；第二帶權重 [0,0.5,1]，得到 2。", "L5、L27",
        "toy矩陣×power實得[5,2]。實看640/360靜態圖，格0/1/2的4/2/1對應上方；中間2分兩路各2×0.5=1，A/B合流分別4+1=5、1+1=2。零路未畫、底注列完整權重；兩個1的來源可沿箭頭追。",
        [run("mel-12.6.toy_band_output", "手算重算"), {"artifact_id": "figure-640", "locator": "格1兩路與A/B合流", "supports": "已實看圖文路徑"}, {"artifact_id": "figure-360", "locator": "全圖及底注", "supports": "360寬數值/標籤可辨"}],
        "圖只用小表，不是實際16帶濾波曲線；未看原位頁面。", True),
    claim("12.6-c5", "concept", "頻帶變少會合併差異，因此不能由 16 帶唯一還原所有 201 格。", "L29",
        "201→16線性映射rank≤16必有非零零空間v。選正基底z及小ε，使z、z+εv皆非負而輸出相同，故非負功率也非唯一。實算rank16、各帶非空，168頻率格貢獻超過一帶。固定bank無參數，後續AudioEncoder Linear/feature才學。",
        [ref("impl-multimodal", "49–83", "固定前處理/可訓練入口"), run("mel-12.6", "rank/重疊/非空"), {"kind": "derivation", "details": "Av=0、v≠0；A(z+εv)=Az，足夠小ε可保非負。"}],
        "壓縮的一般非唯一性；非任何16帶都無法辨任務。", True),
    claim("12.6-c6", "software", "bands 改成 32，預期權重表 (32,201)、輸出 (32,3)，並保持三個時間框。", "L29",
        "實跑shape均吻合。此練習只改前處理表；不把32帶輸出說成可直接塞入已訓練的16帶Linear。",
        [run("mel-12.6", "32帶shape")], "不主張32帶最佳或不用配入口。", True),
], {
    "identity_role": "固定矩陣逐框把頻率功率壓成少量摘要；12.5已教201頻率格功率。",
    "mechanism_property": {"answer": "非負三角合併頻率；同格可分兩路，時間列保留；等mel令低頻細高頻粗。", "basis": "L3、L7、L25–27", "status": "已足夠"},
    "need_use": {"answer": "來源先提出少數字逐框摘要，16減少201輸入；mel沿常見約定以便對照及分配細分，無最佳帶數/效果保證。", "basis": "L3、L9、L29", "status": "已足夠"},
    "operation": "(16,201)bank乘(201,3)power，逐列出16帶，再由可訓練入口使用。",
    "example_support": "程序示軸保留，非對稱三格圖使重疊分流可見；文字另教資訊損失及效果邊界。"
}, ["201格→少量摘要需要與矩陣改動。", "mel等間隔對應Hz密低疏高。", "同格雙路→頻帶合流；時間列不變。", "固定前處理/可訓練入口分工；壓縮非可逆。"],
    "主文有獨立摘要需要及mel約定，不靠公式正確代答用途，也未從示範推出任務收益。",
    ["只看凍結SVG與640/360靜態圖，未驗原位網頁、DOM、平台或整頁可讀性。", "未重訓16/32帶或比較mel尺度/歸一的品質，正文亦未承諾優劣。", "外部套件只讀官方源，沒有安裝、呼叫或新增依賴。"])

page("13.3", [
    {"quote": "第二步的條件已含第一步，因此連乘不要求兩個文字彼此獨立。", "location": "L3", "depth": "條件鏈式法則"},
    {"quote": "下面只把答案位置的log機率相加，前文和補齊用的PAD位置不計分。", "location": "L5", "depth": "有效位置及整段分數"},
    {"quote": "本章的整段約定也包括EOS結束目標。", "location": "L22、L29", "depth": "正式實驗分母"},
], [
    claim("13.3-c1", "concept", "0.8×0.8=0.64 ... 連乘不要求兩個文字彼此獨立。", "L3–5",
        "P(a1,a2|x)=P(a1|x)P(a2|x,a1)由條件機率定義；第二項已條件於第一目標，不需獨立。ln(product)=sum ln。0.8明確是假設，不追實測。",
        [{"kind": "derivation", "details": "條件機率P(A∩B|X)=P(A|X)P(B|A,X)；ln乘積等於ln相加；0.8²=0.64。"}],
        "給定各自目標前綴的序列機率，沒有隨機生成。"),
    claim("13.3-c2", "software", "標籤[-100,0,2]忽略第一位置，第二取候選0，第三取候選2。工具先轉成log機率，再按標籤取值、加總。", "L11–18",
        "helper labels!=-100得mask，無效label先置0以免gather越界；log_softmax末詞表軸、gather、乘mask、sum位置軸，沒有再shift。13.2已教render_chat對齊；PAD也標-100。",
        [ref("impl-alignment", "29–36", "mask/log_softmax/gather/sum"), ref("impl-data", "render_chat/pad_batch", "前文、PAD、EOS"), ref("prior-13.2", "目標對齊段", "只shift一次")],
        "手設三候選logits；不推完整模型vocab為3。"),
    claim("13.3-c3", "numeric", "整段分數約−0.4791，exp()還原機率約0.6193。若多加一個同樣機率的位置，總log約−0.7186", "L20",
        "p=e²/(e²+2)=0.786986，lnp=-0.2395448；兩項sum=-0.4790896、exp≈0.619347。原stdout-0.4791/0.6193；首label改1實得sum=-0.7186344、mean=-0.2395448。",
        [run("snippet-13.3", "原sum/exp"), run("extra-label-13.3", "三項與平均")],
        "四位舍入容差5e-5，非真模型生成或偏好测量。", True),
    claim("13.3-c4", "concept", "長回答常有更負的總log，不能單靠它說人類更不喜歡。後面會先對每篇答案減去固定參考的同篇分數", "L20–22",
        "每條件機率≤1，新增有效項添非正log；等單步機率的兩/三項已示sum與mean不同。DPO Eq7確比logπθ(yw|x)-logπref(yw|x)及同yl項。此為後續方向，未說參考相減已消除全部長度/偏好偏差。",
        [ref("original-dpo", "Eq.(5)–(7)、Appendix B", "同篇政策/參考差，再比兩篇"), ref("impl-alignment", "39–43 dpo_loss", "本課相同差值")],
        "‘常有’非不同回答必按長度排序；參考機制細節待後教，不影響本頁sum。"),
    claim("13.3-c5", "empirical", "4 ... 兩個目標，10則有三個。兩支各250次更新 ... 9,448；這是重複抽樣的訓練曝光量", "L22、L29",
        "原model/beta1各250更新與9448。render_chat實數'4'為byte＋EOS=2，'10'=3。重建家族split49/8/7 JSONL，三SHA相同。依原Random(42).choices(k=8)重複250次、兩側有效標籤相加實算9448；fit只在更新迴圈調loss，無額外probe。",
        [ref("raw-dpo", "data、runs.*.training", "原分母/更新"), ref("impl-data", "render_chat/pad_batch", "EOS/mask"), ref("impl-common", "205–245", "每更新一次loss"), run("dpo-denominator-13.3", "資料SHA及曝光重算"), {"artifact_id": "historic-dpo-behavior", "locator": "552–604 _pair_examples/_dpo_train", "supports": "當時原碼SHA94ab...dc0吻合原紀錄，雙側sum約定"}],
        "含重複及EOS的兩側曝光，非不同答案數；沒有模型更新。", True),
    claim("13.3-c6", "concept", "實際第一位置若屬提問，就仍須忽略，不能為了改分數把它算入回答。", "L24",
        "練習改mask僅教求和；真計分角色規則由render_chat先定，不能為改分數把prompt當answer。此限定直接回應目前練習可產生的誤用。",
        [ref("impl-data", "render_chat targets", "角色限定有效位置"), run("extra-label-13.3", "數學增加有效項")],
        "手設標籤變體與真回答計分分開。"),
], {
    "identity_role": "序列log分數在各自已對齊目標前綴下累加回答與EOS；13.2已教各續寫配各自目標。",
    "mechanism_property": {"answer": "條件鏈把乘轉加，mask篩有效項；同機率加一項令sum更負、mean不變。", "basis": "L3–5、L18–24", "status": "已足夠"},
    "need_use": {"answer": "要合成整篇答案機率並避免小數連乘過小，僅評回答及停止，給後續偏好比較一致序列約定。", "basis": "題目、L5、L22、L29", "status": "已足夠"},
    "operation": "已對齊logits/labels→log_softmax→按候選取值→mask→sum；exp僅還原。",
    "example_support": ".8假設教條件鏈；手設表教mask/候選/shape；加有效項對照sum/mean；EOS曝光另核正式約定。"
}, ["條件前綴→鏈式機率→log加總，無獨立假設。", "目標/前文/PAD及一次對齊約定。", "sum、mean與人類偏好不能混一量。", "EOS目標；曝光與不同答案數分開。"],
    "sum/mask/長度關係正文已完成，正式配方留選讀不替正文補洞。參考分數明確作後續預告，未用未教機制證明當前數字。",
    ["未執行生成、DPO優化、自然/人類偏好測量。", "選讀未直接重列DPO JSON鏈，本次依本章方法與原實作定位dpo紀錄；未把定位負擔自動判技術錯。", "本頁無圖；shape、候選、有效位置逐項明說，可直接追蹤手算。"])

priors = []
for id in ["10.10", "11.2", "11.3", "11.4", "12.1", "12.5", "13.2"]:
    p = B / "freeze/sources" / (id + ".md")
    priors.append({"id": "prior-" + id, "path": str(p.relative_to(B)), "sha256": sha(p),
                   "scope": "完整實讀凍結頁，只用作指定頁必要前文；未開該頁判斷紀錄。"})

artifacts = []
for id, p, kind, note in [
    ("execution", E / "execution.json", "execution", "六段原碼、主要變體、alias、真人例、分母/資料重構；CPU，無模型權重更新。"),
    ("execution-stdout", E / "execution.stdout.txt", "execution_stdout", "同次完整stdout。"),
    ("execution-code", E / "run_checks.py", "code", "相稱離線查核腳本；首次VQA sampler少消耗零比例replay隨機檢查，照原碼修正後3830，不把3799誤報教材矛盾。"),
    ("input-identities", E / "input-identities.json", "identity_check", "實際freeze、原資料及歷史git SHA對manifest/原紀錄。"),
    ("historic-dpo-behavior", E / "historical/dpo/scripts/course_experiments/behavior.py", "original_repository_code", "git show原DPO revision，SHA吻合原紀錄，實讀資料/訓練相關函式。"),
    ("historic-dpo-common", E / "historical/dpo/scripts/course_experiments/common.py", "original_repository_code", "原fit源SHA與freeze/原紀錄相同。"),
    ("figure-640", B / "renders/multimodal_overlapping_bands-640.png", "figure_render", "實際view_image看640寬靜態圖。"),
    ("figure-360", B / "renders/multimodal_overlapping_bands-360.png", "figure_render", "實際view_image看360寬靜態圖。"),
]:
    a = {"id": id, "kind": kind, "path": str(p.relative_to(B)), "sha256": sha(p), "description": note}
    if id == "execution":
        a.update(command=RUN["command"], environment=RUN["environment"],
                 result="六頁stdout及主要變體符合本文；VQA抽樣3830與DPO抽樣9448吻合；歷史WAV容器SHA未重現，原因未定，另列限制。")
    artifacts.append(a)

criteria = {}
for file in ["SKILL.md", "references/review-protocol.md", "references/calibration.md", "references/project-context.md"]:
    actual = sha(B / "freeze/criteria" / file)
    assert actual == M["criteria_sha256"][file]
    criteria[file] = {"sha256": actual, "read": True}
assert all(x["matches"] for x in RUN["imported_frozen_source_identities"].values())
assert all(x["matches"] for x in json.loads((E / "input-identities.json").read_text()).values())

report = {
    "schema_version": 1, "review_stage": "technical-source-initial", "group": "modalities",
    "reviewer": TASK, "reviewer_task": TASK, "at": datetime.now(timezone.utc).isoformat(),
    "scope": {"assigned_pages": M["groups"]["modalities"]["pages"], "repository": "/workspace/tiny-perceptron-vlm",
              "freeze": "docs/course-repair-20261008/reviews/freeze-01",
              "method": "獨立技術來源初判；指定正文/選讀、必要前文、原資料/官方源及相稱CPU計算。"},
    "version": {"baseline_commit": M["baseline_commit"], "manifest_sha256": sha(B / "manifest.json"),
                "freeze_created_at": M["created_at"], "source_hash_scheme": M["inventory"]["source_hash_scheme"]},
    "criteria": {"frozen_files": criteria,
                 "general_conventions_read": ["CLAUDE.md", "docs/editorial-guide.md", "docs/technical-review-guide.md"],
                 "precedence": "本輪grouped派工與封存流程優先，不照舊單節task格式冒充六個身分。"},
    "independence": {
        "reader_source_gate": {"path": "checks/reader-source-all-sealed.json", "sha256": sha(B / "checks/reader-source-all-sealed.json"), "existed_before_source_judgment": True},
        "raw_input_manifest": {"path": "checks/technical-inputs-manifest.json", "sha256": sha(B / "checks/technical-inputs-manifest.json")},
        "other_role_judgments_read": False, "author_or_old_findings_read": False,
        "limitations": "技術審閱非首讀：完整頁及外部說明不回算成讀者當時理解；初判前沒讀reports/traces/notes/authors/diagnosis或root判斷。"},
    "prerequisites_actually_read": priors, "sources": sources, "artifacts": artifacts,
    "executed_imports_identity_check": "execution.json.imported_frozen_source_identities全吻合manifest；直接用freeze implementation，未用live差異背書。",
    "pages": pages, "findings": {"necessary": [], "optional": [], "note": "所列原承諾在實際核對範圍有具體支持，無必要待修項；沒有為消除差異重寫他人觀察。"},
    "unknowns_and_limits": [
        "未重訓模型、安裝依賴或呼叫成熟模型；只作短碼/數值/抽樣及一段音訊讀檔重取樣。",
        "歷史CUDA/L4品質來自有revision/seed/config/SHA的原紀錄及逐題重數，不是本次重現訓練或推論。",
        "只實看12.6兩寬靜態渲染及SVG源；未驗原位整頁、手機瀏覽器/DOM或平台。",
        "真人例來源SHA與sample/rate吻合，重取樣陣列與當前函式一致；歷史derived WAV容器SHA未重現，不能聲稱字節相同。",
        "最初CCRMA Sampling_Theorem路徑404，DSPRelated對應路徑回通用首頁，均未作證據；之後實讀CCRMA Theory_Ideal_Bandlimited_Interpolation原作者頁。",
        "未開交叉覆核；將來同行同意不能替代本次獨立重現。"],
    "no_mutations": "未修教材、Notebook、圖、skill、凍結來源或歷史結果；只存本審閱者證據/報告，未Git提交。"
}

path = B / "reports/technical-modalities-initial.json"
assert not path.exists(), "不覆寫已有來源初判"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": str(path), "sha256": sha(path), "pages": len(pages),
      "claims": sum(len(p["technical_checks"]) for p in pages), "necessary": 0, "optional": 0}, ensure_ascii=False, indent=2))
