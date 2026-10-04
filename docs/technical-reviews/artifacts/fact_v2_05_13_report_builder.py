"""Build this review from evidence personally inspected and executed."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_05_13_"
audit = json.loads((OUT / (PREFIX + "audit-result.json")).read_text())
raw = json.loads((OUT / (PREFIX + "text-foundation-raw.json")).read_text())
body = dict(sections(ROOT / "course/chapters/05.md"))["5.13"]
assert hashlib.sha256(body.encode()).hexdigest() == audit["source_sha256"]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


artifacts = []
for path in sorted(OUT.glob(PREFIX + "*")):
    if path.name in (PREFIX + "report-build-result.json", PREFIX + "checker-result.txt", PREFIX + "completion.json"):
        # These receipts are written after the report; avoid hashing an open stdout file.
        continue
    if path.suffix == ".py":
        kind = "code"
    else:
        kind = "source_snapshot"
    identifier = path.name.removeprefix(PREFIX).replace(".", "_").replace("-", "_")
    if identifier in ("audit_result_json", "fetch_result_json", "ruff_result_txt"):
        kind = "execution"
    item = {"id": identifier, "kind": kind, "path": str(path.relative_to(ROOT)),
            "sha256": digest(path.relative_to(ROOT)),
            "description": {
                "audit_result_json": "真執行原碼、參數拆帳、正式資料指紋、抽樣分母及四個公開權重的 CPU 全集 NLL 複算。",
                "fetch_result_json": "原始來源取得與來源快照 hash；NIST 403 的失敗也如實保留，不列為已核實來源。",
                "text_foundation_raw_json": "親讀的完整正式 GPU 原始結果；含四組完整配置、history、全評估與來源 hash。",
                "export_manifest_json": "正式完整 checkpoint 的 SHA 到公開推論權重 SHA 的可追查映射。",
                "download_manifest_json": "公開模型 repository、精確 revision 與下載 SHA。",
                "section_txt": "以 sections() 擷取的本節全部原文，包含所有 LF 到下一個 ## 前。",
                "report_builder_py": "本審閱報告的組裝程式。",
                "ruff_result_txt": "本審閱者新增 Python 的 Ruff 檢查結果，包含 import 排序。",
            }.get(identifier, f"審閱者親讀並保存的 {path.name}；供報告定位與版本重核。")}
    if kind == "execution":
        item.update(environment=audit["environment"])
        if identifier == "audit_result_json":
            item.update(command=audit["command"] + " > docs/technical-reviews/artifacts/fact_v2_05_13_audit-result.json",
                        result="exit 0; all assertions passed; 12 個 CPU NLL 與正式 GPU 結果最大差 5.53e-8；無訓練。")
        elif identifier == "fetch_result_json":
            item.update(command=".venv/bin/python docs/technical-reviews/artifacts/fact_v2_05_13_fetch.py > docs/technical-reviews/artifacts/fact_v2_05_13_fetch-result.json",
                        result="exit 0; pinned PyTorch 原碼取得成功，候選原始快照保存；NIST HTTP 403，未用於任何 verified 主張。")
        else:
            item.update(command=".venv/bin/python -m ruff check --no-force-exclude docs/technical-reviews/artifacts/fact_v2_05_13_fetch.py docs/technical-reviews/artifacts/fact_v2_05_13_audit.py docs/technical-reviews/artifacts/fact_v2_05_13_report_builder.py",
                        result="exit 0; All checks passed!")
    artifacts.append(item)

sources = []


def original(identifier, kind, title, url, version, authority, note):
    sources.append({"id": identifier, "kind": kind, "title": title, "url": url, "version": version,
                    "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
                    "authority_reason": authority, "inspection_note": note})


original("chinchilla", "paper", "Training Compute-Optimal Large Language Models",
         "https://arxiv.org/pdf/2203.15556v1", "arXiv:2203.15556v1 (2022-03-29)",
         "Hoffmann 等作者的原始研究；親讀已保存 PDF 的原文文字與 arXiv 版本 metadata。",
         "親讀 §1 Eq(1), §3.1–3.3 Eq(2), Appendix D.2 pp24–25, Appendix F pp27–28, §5 pp15–16。"
         "研究用多模型與 token 長度做經驗擬合；F 明說 multiply-accumulate 計2次。D.2 定義以前文預測下一 token。"
         "原研究少於一 epoch、大規模設定，不支持用本節重複小資料四點直接套係數或宣稱普遍法則。"
         "完整原文快照 chinchilla_v1_txt/pdf 與 metadata 保留。")
torch_commit = "5c4886908584029761b579af026dcfb627c84070"
original("torch_linear", "official_source", "PyTorch Linear 官方原碼",
         f"https://github.com/pytorch/pytorch/blob/{torch_commit}/torch/nn/modules/linear.py",
         f"PyTorch 2.14.1+cpu, git {torch_commit}", "PyTorch 專案本身，與已安裝 torch.version.git_version 相同。",
         "親讀 class Linear L53–139：y=xA^T+b，weight=(out_features,in_features)，bias=(out_features)，forward=F.linear。"
         "保存 pinned raw GitHub 原碼，不以一次執行代替 API 契約。")
original("torch_ce", "official_source", "PyTorch CrossEntropyLoss 官方原碼與公式",
         f"https://github.com/pytorch/pytorch/blob/{torch_commit}/torch/nn/modules/loss.py",
         f"PyTorch 2.14.1+cpu, git {torch_commit}", "PyTorch 專案本身，與本機版本的官方原碼相符。",
         "親讀 class CrossEntropyLoss L1200–1250：class-index 目標的 -log softmax、ignore_index 指示式、sum/mean 分母。"
         "本節採無 class weight 的整數 labels，ignore=-100，不用機率 targets 或 label smoothing。")
original("sklearn_cv", "official_docs", "scikit-learn Cross-validation: evaluating estimator performance",
         "https://scikit-learn.org/stable/modules/cross_validation.html", "1.9.1 官方頁面，保存快照；讀取2026-10-04",
         "scikit-learn 官方方法說明，支持訓練/驗證/測試使用時機與選擇偏差。",
         "親讀 3.1 開頭與 hyperparameter selection/validation/test 三分頁段（文字快照 L167–194）。"
         "反覆以 test 選 hyperparameter 會泄漏；保留 test 作 final evaluation。此處不聲稱必須用 sklearn API。")
original("sklearn_leakage", "official_docs", "scikit-learn Common pitfalls: Data leakage",
         "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage", "1.9.1 官方頁面保存快照；讀取2026-10-04",
         "scikit-learn 官方文件，直接討論訓練與留出資料混用造成樂觀估计。",
         "親讀 §12.2–12.2.1（文字快照 L235–274）：leakage、never fit test、先分資料；支援本節保持共同留出集合及不拿留出題訓練。")
original("cawley", "paper", "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation",
         "https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf", "JMLR 11 (2010), pp2079–2107",
         "Cawley 與 Talbot 的原始研究、出版方 PDF。",
         "親讀 abstract、§1、§4–4.1 pp2084–2086、§5–5.1 pp2094–2095。有限樣本評分的 variance 讓最佳化選擇過擬合；"
         "不對稱地挑最好 seed 屬額外模型選擇，不能與無同等選擇的另一方法視為僅改容量/資料量。未借用論文實驗數字作本節數字。")


def repo(identifier, path, title, note, version="目前檔案，全檔SHA固定"):
    sources.append({"id": identifier, "kind": "repository_code", "title": title, "path": path,
                    "sha256": digest(path), "version": version, "verified": True, "inspection_note": note})


repo("model", "tiny_perceptron/model.py", "TinyLM/ModelConfig/description/loss_sum",
     "完整親讀131行，尤其L14–30預設、L34–72零件、L95–112總參數及有效答案；CPU實建且逐named parameter拆帳。與正式實验 source SHA 完全相同。")
repo("attention", "tiny_perceptron/attention.py", "CausalAttention 的四個無bias Linear",
     "親讀L32–75，width/heads預設對QKV/out權重；正式SHA相同。")
repo("modern", "tiny_perceptron/modern.py", "DenseFFN 的4倍中間寬度",
     "親讀L36–57，兩個带bias Linear、GELU無學習參數；正式SHA相同。")
repo("data", "tiny_perceptron/data.py", "ByteTokenizer/shifted/pad_batch",
     "完整親讀128行：UTF-8 byte+8，BOS/EOS、shifted只移一次、pad labels=-100，valid bool。正式SHA相同；執行確認ASCII每字元1 byte、另加EOS目標。")
for name, note in {
    "common": "親讀L45–95 new_lm seed及 split_records/text_examples；L106–202 _nll/fit_lm（seed42私有sampler、with replacement、150×4、masked loss、clip1、AdamW lr.003、最後重算全集）；L243–305評估。",
    "text": "親讀L34–72步數/正式JSONL序列化與manifest、L262–315全run_text_foundation；pool80按object family先split64/8/8，前16為64訓練子集，width16/32 layers1各new_lm seed42。",
    "run": "親讀execute L58–163：Context seed42、正式step_scale1、threads2、source SHA、CUDA同步、計時與completed結果範圍。",
}.items():
    repo(f"formal_{name}", f"docs/technical-reviews/artifacts/{PREFIX}formal-{name}.py", f"正式GPU實驗時的 {name}.py",
         note + " 由git show正式revision取出，全檔SHA與原始結果code_sha256一致。",
         raw["revision"] + "; " + raw["code_sha256"][f"scripts/course_experiments/{name}.py"])
repo("training", "tiny_perceptron/training.py", "load_checkpoint 的CPU載入與strict state_dict",
     "親讀全部保存/載入实现，特別L78–102 weights_only=True、map_location=cpu、依config建model、strict=True載入；CPU實載四組公開權重。")
sources.append({"id": "calculation", "kind": "derivation", "title": "預算、參數與密集Linear成本推導",
                "verified": True,
                "details": "2寬度×2資料量=4格。4筆/步×8有效目標/筆×100步=3200；改4目標/筆時3200/(4×4)=200步，100×4×4=1600。"
                "Linear每輸出D乘+D-1加（不含bias），共H(2D-1)；標準2 FLOPs/MAC記2DH，忽略低階項：2×8×8=128、2×16×16=512，4倍。"
                "TinyLM固定部分=(2×264+128+2)w=658w；block=4w²+8w²+5w+4w=12w²+9w；一層總12w²+667w。"
                "w8=768+5336=6104，w16=3072+10672=13744，w32=12288+21344=33632；CPU named parameters精確相等。"
                "實際抽樣150×4=600次記錄曝光；16與64平均曝光37.5與9.375，但各記錄次數不同。有效目標20765-20623=142。"
                "抽象例貓/看/狗，指定的前文與下一答案對為(貓,看)、(貓看,狗)，有2對故2個有效位置；未指定EOS或BOS，不額外計它們。"})
sources.append({"id": "execution", "kind": "execution", "title": "本審閱者CPU數值與軟體實執行",
                "verified": True, "artifact_id": "audit_result_json"})

claims = []


def claim(identifier, kind, statement, location, evidence, scope, artifact_ids=None, verification=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "status": "verified",
            "evidence": [{"source_id": source, "locator": locator, "supports": support}
                         for source, locator, support in evidence],
            "artifact_ids": artifact_ids or [], "scope": scope}
    if verification is not None:
        item["verification"] = verification
    claims.append(item)


def verify(expected, observed, details, tolerance="exact", denominators=None):
    value = {"method": "executed", "expected": expected, "observed": observed,
             "tolerance": tolerance, "details": details}
    if denominators:
        value["denominators"] = denominators
    return value


claim("c1", "concept", "同時改模型容量、資料量與訓練量的改善不能單獨歸因於其中一個因素；應分開變化。", "開頭第1段",
      [("chinchilla", "§1 Eq(1), §3.1–3.3", "把最終損失建為N與D的函數，計算限制另列；多配置分開研究。")],
      "控制其他訓練設定且比較共同評估集下的局部因素效果；四格不分離所有潛在交互作用與隨機誤差。", ["chinchilla_v1_txt"])
claim("c2", "numeric", "兩種寬度乘兩種不同記錄數形成4格。", "第2段及迴圈",
      [("calculation", "2×2=4", "組合數"), ("execution", "snippet_stdout", "原文迴圈實輸出4份")],
      "僅本示意2×2計畫。", ["audit_result_json"], verify("4份計畫", "4份計畫", "親執行原文完整Python區塊並數輸出。"))
claim("c3", "concept", "有效答案是參與loss的下一token預測位置；忽略padding不納入分母，並非整道問題。", "第2段『貓』/『貓看』及有效token定義",
      [("chinchilla", "Appendix D.2 Loss decomposition, p24", "定義下一token y 與前文x"),
       ("torch_ce", "CrossEntropyLoss class-index formula L1200–1250", "ignore_index指示式與平均分母")],
      "貓/看/狗為抽象單位的兩次預測示例，不是宣稱UTF-8 tokenizer有兩個byte。本節正式ASCII資料使用ByteTokenizer且計EOS目標。",
      ["chinchilla_v1_txt", "torch_loss_py"])
claim("c4", "software", "原文程式只建立模型數參數及印計畫，品質保持『尚待訓練與量測』。", "唯一Python區塊及其後第1段",
      [("model", "ModelConfig/TinyLM.description L14–30,L95–97", "預設一層模型與parameters=sum(numel)"),
       ("execution", "snippet_stdout", "四格print未呼叫optimizer或訓練")],
      "建構有隨機初值；此輸出不含任何品質測量。", ["snippet_py", "audit_result_json"],
      verify("w8:6104,w16:13744，各16/64；品質未測", "四份輸出相符", "完整原碼CPU執行，另逐named parameter加總。"))
claim("c5", "numeric", "假設每步4筆、每筆8個有效答案、100步，每格有效答案預算為3200。", "程式乘式與後第1段",
      [("calculation", "4×8×100=3200", "逐單位相乘"), ("execution", "snippet_stdout", "四格3200")],
      "每筆有效長度8是計畫假設，不是正式資料測得長度。", ["audit_result_json"],
      verify("3200", "3200，四格相同", "直接執行batch_size*answers_per_record*steps。"))
claim("c6", "concept", "固定更新步數、有效token數、時間或FLOPs是不同预算；同token預算不保證同計算成本。", "『公平預算』段及Linear例後結論",
      [("chinchilla", "§1 Eq(1); Appendix F", "FLOPs依模型與token量及算子不同；compute受模型大小影響")],
      "固定有效監督量不包含所有padding工作；時間還依硬體、實作和負載。此處不把2DH等同整模型訓練成本。", ["chinchilla_v1_txt"])
claim("c7", "concept", "密集Linear以H×D權重把D個輸入轉成H個輸出並可加bias。", "公平預算段D、H、權重表及bias說明",
      [("torch_linear", "Linear L53–139", "y=xA^T+b，weight=(out,in)，bias=(out)")],
      "描述稠密層數學；學成零權重仍是同一稠密運算。", ["torch_linear_py"])
claim("c8", "numeric", "每位置Linear約2DH FLOPs，8進8出128、16進16出512，寬度倍增使此項約4倍。", "公平預算段數字",
      [("chinchilla", "Appendix F pp27–28", "multiply-accumulate計2 FLOPs且linear成本2T·D·H"),
       ("calculation", "H(2D-1)與2DH", "精確無bias相加次數與標準近似分開")],
      "128/512採標準2FLOPs/MAC記帳近似；不等同CPU實際指令數或整模型wall time，忽略bias和低階項。", ["audit_result_json", "chinchilla_v1_txt"],
      verify("128,512;比4", "128,512;比4", "代入2×D×H CPU計算；無bias最小乘加則120/496，正文明示約。"))
claim("c9", "concept", "共同留出評估與避免訓練混入驗證/測試，是比較訓練條件的必要控制。", "『還要用共同的驗證與測試資料』段",
      [("sklearn_cv", "§3.1開頭及validation/test段", "留出評估避免把記憶資料当泛化"),
       ("sklearn_leakage", "§12.2–12.2.1", "训练混入留出資料導致樂觀估計")],
      "共同評估是必要而非單獨充分的因果控制；本文也要求seed/预算/版本與其他設定。", ["sklearn_cv_txt", "sklearn_pitfalls_txt"])
claim("c10", "concept", "只讓其中一格多試並挑最佳seed，會引入額外模型選擇差異。", "共同評估段『大模型多選了幾次最好種子』",
      [("cawley", "§4.1 pp2086–2087; §5 pp2094–2095", "有限樣本選最好評分受variance及selection bias影響")],
      "這是一般選擇偏差原理在seed候選上的應用；不宣稱每次多試必然改變數字，也不從單seed估計置信區間。", ["cawley2010_txt"])
claim("c11", "software", "正式資料為80個object短文，按完整物體編號切成64訓練、8驗證、8test，16組取訓練前16。", "『把計畫落成實驗』段",
      [("formal_text", "run_text_foundation L279–299", "80 pool及train[:count]"),
       ("formal_common", "split_records L50–70", "family先分組和seed42 split")],
      "唯一object family的ASCII規則短文；16訓練為64子集；留出object不代表所有color/shape組合未見。",
      ["audit_result_json", "train_jsonl", "validation_jsonl", "test_jsonl"],
      verify("80=64+8+8；三split不重疊，16為64子集；各SHA吻合正式manifest", "全部相符", "重新生成完整pool，split_records執行；序列化檔案全hash一致。"))
claim("c12", "empirical", "正式四组一層、width16/32、seed42，各更新150步、每批4篇。", "正式實驗段及表前設定",
      [("formal_text", "L287–299", "layers1、steps150、batch4"), ("formal_common", "new_lm L45–48; fit_lm L122–203", "seed42、私有sampler與訓練配置"),
       ("execution", "original_environment/evaluations.original_training", "完整正式結果与公開checkpoint config核對")],
      "正式報告為step_scale1 completed單次L4 GPU實驗；未在本審閱重訓，CPU僅載入正式導出權重再評估。",
      ["text_foundation_raw_json", "audit_result_json", "export_manifest_json", "formal_common_py", "formal_text_py"],
      verify("width16/32 layers1,seed42,150steps,batch4", "原始報告及CPU載入config符合；抽樣每格600曝光", "完整原結果、正式來源hash與checkpoint provenance互核。",
             denominators={"seed": 42, "updates_per_cell": 150, "batch_records": 4, "cells": 4, "formal_step_scale": 1.0}))
claim("c13", "numeric", "正式width16、32的一層模型參數為13744及33632。", "實驗段參數句",
      [("model", "config/description", "總格數規則"), ("calculation", "12w²+667w", "完整參數算式")],
      "vocab264,max_length128、未綁輸出權重、預設GELU/LayerNorm/heads1的一層配置。",
      ["audit_result_json"], verify("13744及33632", "13744及33632", "三寬度CPU新建逐named parameter加總；正式checkpoint config一致。"))

for index, (name, result) in enumerate(audit["evaluations"].items(), start=14):
    width = result["config"]["width"]
    count = result["original_training"]["records"]
    for split in ("train", "validation", "test"):
        value = result["recomputed"][split]
        identifier = f"c{index}_{split}"
        n = value["cpu"]["effective_tokens"]
        claim(identifier, "empirical", f"width{width}、{count}訓練短文的{split}平均下一單位NLL為{value['five_decimal_table']}。",
              f"表格width{width}/n{count}行、{split}代價欄",
              [("execution", f"evaluations.{name}.recomputed.{split}", "親讀完整正式結果並用其導出權重CPU重算全集"),
               ("formal_common", "_nll L106–119,fit_lm L176–202", "最後重算全集sum/count；不是最後訓練batch loss")],
              f"單seed42正式GPU權重，{split}有效目標{n}；有效位置加權平均自然log CE（nats），包括EOS，並非完整短文生成成功率。",
              ["audit_result_json", "text_foundation_raw_json", "export_manifest_json", "download_manifest_json"],
              verify(value["five_decimal_table"], f"CPU {value['cpu']['nll']:.12f}，原GPU {value['original_cuda_nll']:.12f}",
                     "_nll迭代全部資料，每批loss_sum再累加，labels為long，padding=-100，float32 logits，sum/count；資料與權重SHA鏈完整。",
                     "CPU/GPU NLL abs差<1e-6；表顯示5位小數，四捨五入誤差≤5e-6。",
                     {"seed": 42, "updates": 150, "training_records": count, "evaluation_records": count if split == "train" else 8,
                      "effective_evaluation_targets": n, "evaluation_batch_size": 16, "device_original": "NVIDIA L4 cuda", "device_recheck": "cpu"}))
for count, expected in ((16, 20765), (64, 20623)):
    claim(f"c18_n{count}", "empirical", f"{count}篇訓練配置兩種寬度的累計有效目標均為{expected}。", f"四格表n{count}的『累計有效目標』",
          [("execution", f"sampling.{count}", "實際重播正式seed42抽樣和pad mask計數"),
           ("formal_common", "fit_lm L129–149", "獨立random.Random seed42，choices replacement，計labels!=IGNORE")],
          "這是重複曝光的有效目標總量，非不同文字數；每種資料量600抽樣，width不影響私有sampler。",
          ["audit_result_json", "text_foundation_raw_json"],
          verify(str(expected), str(audit["sampling"][str(count)]["effective_targets"]), "親讀原報告並用CPU重播150×4的全部抽樣，逐labels count累加。",
                 denominators={"seed": 42, "updates": 150, "batch_records": 4, "draws": 600, "unique_training_records": count,
                               "targets_per_draw_range": "33–35，含EOS；padding忽略"}))
claim("c19", "empirical", "本次兩寬度下64篇的驗證代價均低於16篇，訓練代價卻較高。", "表後第1句",
      [("execution", "evaluations.*.recomputed", "四格完整數字的直接比較")],
      "僅seed42與150更新配置；訓練平均是在不同材料、不同分母上，不能當共同新題評估。test未被用來調參。",
      ["audit_result_json", "text_foundation_raw_json"],
      verify("validation下降且train上升，兩寬度均如此", "w16 val .49288→.31105/train .27166→.33032；w32 val .40927→.21914/train .14502→.20107",
             "比較原始與CPU相符的四格數字；train平均有效目標553 vs2200，共同val274。",
             denominators={"seeds": [42], "widths": [16, 32], "train_records": [16, 64], "train_eval_targets": [553, 2200], "validation_targets": 274}))
claim("c20", "software", "相同步數與每批筆數仍有不等有效目標，因實際短文長度與抽樣不同；16篇反覆抽更多。", "表後預算限制段",
      [("formal_common", "fit_lm sampler/labels L129–149", "資料抽樣與長度決定累計count"), ("execution", "sampling/targets_per_record/draw_counts", "重播計數與每篇抽到次數")],
      "每格600 record曝光平均16篇37.5次、64篇9.375次，個别頻次不同；不宣稱match FLOPs、時間或監督總量。",
      ["audit_result_json"], verify("20765 vs20623差142；16篇平均重複較多", "差142，頻次與實際長度33–35均保存", "同seed sampler對長度序列加總，不因width不同改抽樣。"))
claim("c21", "concept", "scaling law是規模與表現的經驗擬合；本節单seed四個小設定不建立大型模型普遍規律。", "最後研究與限制段",
      [("chinchilla", "§3開頭/§3.3 Eq(2),§5 Discussion & Conclusion pp15–16", "多配置經驗損失函數及資料/epoch/外推限制")],
      "本資料為規則短文且多epoch replacement；不將Chinchilla少於一epoch、大模型係數直接外推到四格。", ["chinchilla_v1_txt", "audit_result_json"])
claim("c22", "numeric", "每步4筆且每筆4有效答案，要3200目標應跑200步；100步只有1600。", "結尾練習",
      [("calculation", "3200/(4×4),100×4×4", "逐步單位換算"), ("execution", "exercise", "CPU整數算式")],
      "每筆長度4是練習假設，正式資料須量count替代。", ["audit_result_json"],
      verify("200步、1600答案", "200步、1600答案", "CPU執行3200//(4*4)及100*4*4；整除且精確。"))
claim("c23", "numeric", "抽象例『貓』預測『看』，再『貓看』預測『狗』共2個有效位置。", "第2段『就是兩個有效位置』",
      [("calculation", "抽象前文/下一答案的兩對", "逐對計數，不以問題數當分母")],
      "示例將貓/看/狗指定為3個抽象單位；非ByteTokenizer中文byte計數，也未列入EOS/BOS。", [],
      {"method": "hand_calculation", "expected": "2個有效位置", "observed": "2個有效位置", "tolerance": "精確整數",
       "details": "列出兩對(貓,看)、(貓看,狗)，每對有一個非忽略目標，1+1=2。"})

numeric_ids = [c["id"] for c in claims if c["kind"] in ("numeric", "empirical")]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "5.13", "source": "course/chapters/05.md#5.13",
    "reviewer_task": "/root/integration_technical_coordinator/fact_v2_05_13", "reviewer_context": "fresh",
    "source_sha256": audit["source_sha256"], "figure_sha256": {}, "verdict": "pass",
    "prerequisites_read": ["course/chapters/04.md#4.8", "course/chapters/05.md#5.2", "course/chapters/05.md#5.9",
                           "course/chapters/05.md#5.10", "course/chapters/05.md#5.15"],
    "claims": claims, "sources": sources, "artifacts": artifacts, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "逐核概念、唯一原碼及正式結果。2×2計畫與另外四格正式實驗分開，未以參數推出品質；實際ByteTokenizer對ASCII每字元1byte，另有EOS目標。", "claim_ids": [c["id"] for c in claims]},
        "numeric_verification": {"status": "pass", "details": "原碼CPU執行；完整參數拆帳6104/13744/33632，预算3200/200/1600與2DH128/512核對。完整正式資料SHA一致；12個全集CPU NLL與GPU報告最大差5.53e-8，所有表5位小數相符；抽樣復算20765/20623。", "claim_ids": numeric_ids},
        "figure_consistency": {"status": "not_applicable", "details": "sections() 擷取5.13至下一##的全部原文沒有任何圖片或SVG引用；指定前置對本節沒有需另核的圖。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "親讀Chinchilla v1、Cawley JMLR原文、scikit-learn1.9.1官方快照、與torch2.14.1 git吻合的官方Linear/CE原碼；正式程式git快照SHA匹配完整GPU報告。NIST下載403未被当證據；無概念依賴其來源。", "claim_ids": [c["id"] for c in claims]},
        "limitations": {"status": "pass", "details": "正文恰當限制單seed、規則短文、更新/batch match而非target/compute match、訓練評估材料不同、有限四點無大型scaling外推。2DH為記帳近似；CPU複算品質不支持GPU速度，也不宣稱已重訓。", "claim_ids": [c["id"] for c in claims]},
    },
    "inspection_notes": ["指定前置完整親讀；無教材修改，未猜author_tasks。",
                         "現行common僅生成prefix字元邊界改動（ASCII無變），現行text僅別的tokenizer metadata變動；正式訓練、_nll與scaling配置相同。",
                         "新Python只作來源保存、CPU驗證及報告組裝，Ruff含import檢查pass；未付費或長訓練、未修改共用env。"],
}
(ROOT / "docs/technical-reviews/5.13.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": "docs/technical-reviews/5.13.json", "claims": len(claims), "verdict": "pass", "source_sha256": audit["source_sha256"]}))
