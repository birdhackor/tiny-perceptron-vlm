"""Seal only the independently inspected R.4 backfill review and small receipts."""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
from scripts.check_technical_reviews import _validate

BASE = "docs/technical-reviews/artifacts/revision-20261016/backfill/"
OUT = ROOT / BASE
TASK = "/root/backfill_tech_r_4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return path


freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-14.json").read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "course")
body = lesson_slices((ROOT / page["snapshot"]).read_text(encoding="utf-8"))["R.4"]
(OUT / "R.4-source.txt").write_text(body, encoding="utf-8")
execution = json.loads((OUT / "R.4-execution.json").read_text())

links = []
for ref in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body):
    target, _, anchor = ref.partition("#")
    target_path = (ROOT / "course" / target).resolve()
    assert target_path.is_file(), ref
    text = target_path.read_text()
    if anchor:
        assert re.search(r"^## " + re.escape(anchor) + r" ", text, re.M), ref
    links.append({"reference": ref, "exists": True, "anchor_exists": bool(anchor)})
license_text = (ROOT / "LICENSE").read_text()
assert license_text.startswith("MIT License") and "associated documentation files" in license_text
assets = (ROOT / "assets/training/README.md").read_text()
assert "各來源的實際轉換實作" in assets and "不能統一當成此repo的MIT內容" in assets
training_source = (ROOT / "tiny_perceptron/lesson_products/training.py").read_text()
assert all(s in training_source for s in ["loss.backward()", "optimizer.step()", "optimizer.zero_grad("])
product_receipts = {}
for product in ("b", "c"):
    path = f"docs/lesson-products/results/initial-{product}-execution.json"
    rec = json.loads((ROOT / path).read_text())
    assert rec["status"] == "completed" and rec["training"]["random_initialization"]
    assert rec["training"]["weights_changed"] and rec["job"]["resume"] is None
    product_receipts[product] = {"path": path, "sha256": digest(ROOT / path), "run_id": rec["run_id"], "revision": rec["code_revision"], "steps": rec["training"]["last_step"], "random_initialization": True, "weights_changed": True, "environment": rec["training"]["environment"]}
final_origins = {}
final_results = json.loads((ROOT / "docs/selftrained/results/v2-final-public-results.json").read_text())
for architecture, directory in (("moe", "moe-native"), ("dense", "dense-weighted")):
    runner_path = f"docs/selftrained/results/training-raw/{directory}/receipt.json"
    train_path = f"docs/selftrained/results/training-raw/{directory}/raw/train-receipt.json"
    runner = json.loads((ROOT / runner_path).read_text())
    train = json.loads((ROOT / train_path).read_text())
    assert runner["run_id"] == final_results["architectures"][architecture]["selected_source"]["run_id"]
    assert train["origin"]["kind"] == "all-neural-weights-random"
    final_origins[architecture] = {"run_id": runner["run_id"], "origin_kind": train["origin"]["kind"], "seed": train["origin"]["seed"], "stage_history": [{"stage": s["stage"], "step": s["step"], "checkpoint_sha256": s["checkpoint_sha256"]} for s in train["stage_history"]], "source_sha256": {p: digest(ROOT / p) for p in (runner_path, train_path)}}
static = write("R.4-scope-execution.json", {"command": ".venv/bin/python " + BASE + "R.4-seal.py", "environment": {"python": platform.python_version(), "device": "cpu", "mode": "read-only metadata and text checks"}, "result": "All R.4 local links and numbered anchors exist; MIT license and separate upstream-license notice verified; PyTorch backward/step calls inspected; original B/C GPU runner receipt fields and final v2 selected training origins extracted without rerunning training.", "links": links, "product_primary_receipts": product_receipts, "final_v2_training_origins": final_origins, "pins": {p: digest(ROOT / p) for p in ["LICENSE", "assets/training/README.md", "tiny_perceptron/lesson_products/training.py", "docs/selftrained/V2-DATA-NOTICE.md"]}})

sources = []
receipts = []


def original(identifier, kind, title, url, version, original_file, read_file, locator, note, quote):
    original_path = Path(original_file)
    read_path = Path(read_file)
    read_text = read_path.read_text()
    assert quote in read_text, identifier
    receipt = {"id": identifier, "url": url, "version": version, "accessed_on": "2026-10-16", "original_local_path": str(original_path), "original_sha256": digest(original_path), "readable_fulltext_path": str(read_path), "readable_fulltext_sha256": digest(read_path), "locator": locator, "inspection_note": note, "short_original_excerpt": quote, "checked_original": True}
    receipts.append(receipt)
    sources.append({"id": identifier, "kind": kind, "title": title, "url": url, "version": version, "verified": True, "checked_original": True, "accessed_on": "2026-10-16", "authority_reason": "原作者論文／維護者官方文件的一手全文；支持範圍限於本節的角色及機制概括。", "inspection_note": note + " 原檔與讀取文字SHA、定位及短摘錄封存於R.4-original-source-receipts.json。"})


FT = "/workspace/work/curriculum-20261016/fulltexts/"
OWN = "/workspace/work/curriculum-20261016/R.4-fulltexts/"
original("attention", "paper", "Attention Is All You Need", "https://arxiv.org/html/1706.03762v7", "arXiv v7, 2023-08-02", FT + "14.11-transformer.source", FT + "14.11-transformer.txt", "§3.2, §3.2.1 Eq.(1), §3.4", "親讀query/key相容分數、softmax與value加權和，以及embedding和輸出線性層；這只支持機制示意，未把譬喻當成人類認知或品質證據。", "The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.")
original("optimization", "official_source", "PyTorch Learn the Basics: Optimizing Model Parameters", "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/optimization_tutorial.py", "pytorch/tutorials main as read 2026-10-16; source SHA ae54687a9b959fbc64375974d933089e5d705492baed6235018e2dc6899fc28a", OWN + "pytorch-optimization.py", OWN + "pytorch-optimization.py", "intro; Optimization Loop; Optimizer; train_loop and test_loop", "親讀訓練反覆算loss/導數/調整參數、backward/step、no_grad測試；官方渲染URL回403，改讀官方GitHub同一篇完整原始教學，未執行其下載或訓練。", "# - Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the backward pass.")
original("holdout", "official_docs", "scikit-learn: Common pitfalls and recommended practices", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/common_pitfalls.rst", "scikit-learn 1.7.2", FT + "19.20-backfill/pitfalls.txt", FT + "19.20-backfill/pitfalls.txt", "Data leakage; How to avoid data leakage", "親讀test/train分開及test不得fit或用於模型選擇；支持留題檢查能力的必要性，單次留出結果不構成通用能力保證。", "Test data should never be used to make choices about the model.")
original("switch", "paper", "Switch Transformers", "https://arxiv.org/html/2101.03961v3", "arXiv v3, 2022-06-16", FT + "14.11-switch.source", FT + "14.11-switch.txt", "Abstract; §1", "親讀Dense每例同參數與MoE選部分參數的分工及稀疏目的；不採其大型模型速度數字為本課成果。", "parameters for each incoming example.")
original("ppo", "paper", "Proximal Policy Optimization Algorithms", "https://arxiv.org/pdf/1707.06347v2", "arXiv v2, 2017-08-28", FT + "chapter-13-ppo.pdf", FT + "chapter-13-ppo.pdf.txt", "Abstract; §1; §2.1", "親讀PPO是policy-gradient強化學習、取樣與surrogate目標交替的角色；不宣稱本課已替所有產品跑PPO。", "proximal policy optimization (PPO)")
original("dpo", "paper", "Direct Preference Optimization", "https://arxiv.org/html/2305.18290v3", "arXiv v3, 2024-07-29", FT + "chapter-13-dpo.html", FT + "chapter-13-dpo.txt", "Abstract; §1; Figure 1 caption", "親讀偏好資料、直接分類loss與PPO-based RLHF比較；支持可替代的訓練安排，不把PPO與DPO說成每次都要串行執行。", "DPO optimizes for human preferences while avoiding reinforcement learning.")
original("mtp", "paper", "Better & Faster Large Language Models via Multi-token Prediction", "https://arxiv.org/html/2404.19737v1", "arXiv v1, 2024-04-30", OWN + "mtp.html", OWN + "mtp.txt", "Abstract; §1; §2 opening", "親讀同一trunk多個未來token輸出頭的輔助學習任務，以及推論可僅用下一token頭；支持其角色與選擇性，不借用論文效益數字。", "Considering multi-token prediction as an auxiliary training task")
original("quant", "official_docs", "Hugging Face Transformers Quantization Overview", "https://huggingface.co/docs/transformers/v4.57.1/quantization/overview", "Transformers v4.57.1", FT + "G.4/quant.source", FT + "G.4/quant.txt", "Overview opening paragraph", "親讀較低精度儲存權重以降低記憶體、不同方法有取捨；本節沒有低位元精確度或速度主張。", "Quantization lowers the memory requirements of loading and using a model by storing the weights in a lower precision while trying to preserve as much accuracy as possible.")
original("distill", "paper", "Distilling the Knowledge in a Neural Network", "https://arxiv.org/pdf/1503.02531v1", "arXiv v1, 2015-03-09", FT + "chapter-18-hinton.pdf", FT + "chapter-18-hinton.txt", "Abstract; §1; §2", "親讀以teacher類別機率為soft targets轉移到較小部署模型的角色；不將所有學習都叫蒸餾，也不宣稱本課模型已借入成熟權重。", "soft targets")
original("jev", "official_docs", "TypeSafe Jev Introduction", "https://docs.typesafe.ai/introduction", "live documentation, original snapshot read 2026-10-16", FT + "jev-intro.source", FT + "jev-intro.txt", "Introduction; TypeSafe primitives", "親讀state與typed questions輸入、Choice/Score/Noul輸出契約；只支持Jev式介面啟發，不推論未公開商業架構或配方。", "Jev evaluates typed questions against a state and returns structured results directly.")
write("R.4-original-source-receipts.json", {"reviewer_task": TASK, "sources": receipts, "retention": "外部全文只留在/workspace/work，Git僅保存URL、版本、SHA及短摘錄。"})


def repo(identifier, path, note):
    sources.append({"id": identifier, "kind": "repository_code", "title": path, "path": path, "sha256": digest(ROOT / path), "version": "workspace bytes independently inspected 2026-10-16; pinned SHA-256", "verified": True, "inspection_note": note})


repo("v2-model", "tiny_perceptron/selftrained/model.py", "實讀SelftrainedConfig、MaskedBlock、SelftrainedLanguageModel、VisionEncoder/OCREncoder/AudioEncoder、LimitedAssistant._embeddings與forward/generate的相應路線；核對感知projection與共享文字核心。")
repo("v2-train", "scripts/selftrained/train.py", "實讀main的720–781行：train/validation取樣、僅接受本管線checkpoint、LimitedAssistant建構、origin random與AdamW；未執行訓練。")
repo("bc-model", "tiny_perceptron/lesson_products/model.py", "實讀Tokenizer.encode、LoopedSolver、CandidateScorer、candidate_batch、grouped_scores；state/question/候選意義真正進輸入，各產品各自建構。")
repo("bc-inference", "tiny_perceptron/lesson_products/inference.py", "實讀Product載入、solve、judge、rule_depth和auto_solve；softmax/temperature回傳候選機率，Auto以C判斷後呼叫B。")
repo("bc-train", "tiny_perceptron/lesson_products/train.py", "實讀無resume時獨立建構模型、train資料更新與training receipt記錄；未親跑訓練。")
repo("bc-training", "tiny_perceptron/lesson_products/training.py", "實讀step的zero_grad、CE、backward、梯度裁切和optimizer.step，predict用inference_mode；與既有GPU primary receipts對應。")
repo("v2-results", "docs/selftrained/results/v2-final-public-results.json", "核對兩architecture的capacity/config/initialization、selected_source、safe_weights、final_test完成數/門檻/工具/語音/OCR、data_scope與comparison_limit；未沿用檔內獨立review判斷。")
repo("license", "LICENSE", "逐字讀MIT License的授權與associated documentation條款。")
repo("assets", "assets/training/README.md", "讀資料入口的首批資料與v2範圍及授權分列；各資料包不得統稱本repo MIT，非商用另包。未把資料來源摘要代替上游能力證據。")
repo("route-20", "docs/course-publish-20261008/freeze/current/sources/20.1.md", "讀凍結20.1的前兩段：Whisper轉寫後接Qwen圖文核心。只用來核對R.4所指的成熟模型延伸路線，未重審20.1的程式或圖。")
sources.append({"id": "run", "kind": "execution", "title": "R.4 bounded CPU and existing primary receipts inspection", "artifact_id": "run", "verified": True})
sources.append({"id": "static", "kind": "execution", "title": "R.4 navigation, license and primary GPU receipt field checks", "artifact_id": "static", "verified": True})


def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


claims = []


def claim(identifier, kind, statement, location, scope, ev, observed=None, denominator=None):
    c = {"id": identifier, "kind": kind, "statement": statement, "location": location, "status": "verified", "scope": scope, "evidence": ev, "artifact_ids": []}
    if kind in ("software", "empirical"):
        c["artifact_ids"] = ["run", "static"]
        c["verification"] = {"method": "executed", "expected": "原文所述介面／來源／有限成果可由指定實作與primary receipt核對；不要求重訓。", "observed": observed, "details": "本人親跑R.4-check.py的有限CPU/匿名metadata查核，以及R.4-seal.py的檔案/來源收據查核；正式GPU訓練和最後測試是已讀的既有primary receipts，非本人重跑。"}
        if denominator:
            c["verification"]["denominators"] = denominator
    else:
        c["artifact_ids"] = ["originals"]
    claims.append(c)


claim("c1", "concept", "查表成向量、依相容分數算比例及混合內容，再形成下一字分數，可作追蹤機制的小例；查筆記是譬喻。", "第1–2段", "只核對embedding/attention/output的機制概括；原文未把小例或譬喻當成訓練品質或人類讀書的證明。", [evidence("attention", "§3.2–3.2.1 Eq.(1), §3.4", "embedding、softmax權重與value加權和及next-token輸出；非完整人類認知模型。")])
claim("c2", "concept", "權重是訓練時調整的模型數值；真正學習反覆更新，能力檢查保留不參與更新的題目。", "第1段及第4段首句", "是一般訓練/評估分工；保留題目仍須代表所宣稱的任務，不能由此推出通用能力。", [evidence("optimization", "intro; Optimizer; train_loop/test_loop", "反覆更新模型參數與不算梯度的測試。"), evidence("holdout", "Data leakage", "test不得fit或用來選模型。")])
claim("c3", "software", "既有有限助理用文字核心生成，圖片/聲音經自行實作入口及projection轉成核心能接收的向量。", "第3段第2句", "只核對路線與張量介面；隨機小配置forward不是已學會看圖/聽音的證明。", [evidence("v2-model", "SelftrainedLanguageModel; VisionEncoder; AudioEncoder; LimitedAssistant._embeddings/forward", "感知tokens代入共用embedding序列，再用同一lm輸出logits。")], "MoE與Dense小配置均得logits[1,3,32]、image tokens[2,16]、audio tokens[16,16]。")
claim("c4", "software", "新增Looped產品是獨立載入的有限文字解題器，以共享block多輪處理事件並讀出答案類別。", "第3段第3句及第6段新增成品入口", "有限固定語法；一次成功不表示一般算術/聊天，未聲稱每多一輪必改善。", [evidence("bc-model", "Tokenizer.encode; LoopedSolver.forward", "文字編碼、shared block重用與answer分類頭。"), evidence("bc-inference", "Product.__init__; solve", "B獨立權重及完整有限題目輸入。")], "固定本機B權重SHA與產品manifest相符；A=2,B=5，A設1後交換，查B的3輪輸出answer=1。")
claim("c5", "software", "Jev式判斷服務以狀態、問題及候選意義為輸入，回傳候選機率，且可獨立於B載入。", "第3段第4句", "本課C只受商業Jev介面啟發；不等同商業配方、校準保證或答對機率。", [evidence("jev", "Introduction; TypeSafe primitives", "state/typed-question與結構化機率輸出的介面概念。"), evidence("bc-model", "Tokenizer.encode; CandidateScorer.forward", "state+question+candidate meaning實際組成token輸入。"), evidence("bc-inference", "Product.__init__; judge", "獨立C模型、temperature與候選集合softmax。")], "本機C權重獨立載入，回六候選機率、總和與1差<1e-6，選boxes；Auto另呼叫choice/info後選B三輪。")
claim("c6", "software", "主線從程式初始權重及本課資料學起，未借用成熟神經權重；MoE助理另保留Dense對照。", "第4段第2–4句", "針對本節既有v2與本課B/C；成熟Qwen/Whisper另外列為第20章延伸。MoE/Dense並非等FLOP或效果因果對照。", [evidence("v2-train", "main 720–781行", "無來源時直接建構本課模型、random origin；續接只接受本管線checkpoint。"), evidence("v2-results", "architectures.*.capacity.initialization/config; comparison_limit", "兩版初始化與比較邊界。"), evidence("bc-train", "train無resume分支及receipt", "B/C各自從隨機模型建立。")], "MoE/Dense pretrain primary receipt均記all-neural-weights-random；B/C原GPU runner receipts均random_initialization=true、resume=null、weights_changed=true；CPU分支各為PaddingSafeMoE/DenseFFN。")
claim("c7", "empirical", "兩種v2已完成訓練、公開推論權重與固定最後測試，仍未過全部原判準；任務範圍有限。", "第5段第1–2句", "依固定v2原成績作描述；不比較架構優劣、不替語音或工具失敗加分、不把有限字卡/商品/意圖說成開放世界能力。", [evidence("v2-results", "architectures.moe/dense.final_test; capability_acceptance_passed; data_scope", "兩版完成、原門檻和有限資料範圍。"), evidence("run", "v2.architectures; public_metadata", "本人讀primary訓練與最後測試receipt、核對門檻，匿名GET公開配置與HEAD權重。")], "兩版primary test journal各complete=3734；MoE工具完整往返0/276，低於0.95；兩版capability_acceptance_passed=false。固定HF revision 979cdf…的MoE/Dense配置與本地結果相同，權重HEAD均200。", {"architectures": "2: MoE and Dense", "final_test_per_architecture": "3734 rows; no rerun", "tool_roundtrip": "276 prompts per architecture", "moe_voice_qa": "90 prompts / 30 source recordings", "train_validation_test_rows": "28876 / 2435 / 3734", "scope": "3 image classes, 12 OCR glyphs, 3 voice intents; recordings are not speaker counts"})
claim("c8", "concept", "MoE是架構與稀疏分工的選擇，並非完成小助理必套的一個訓練階段。", "第6段第1–3句中的MoE", "只驗證方法角色與按需求選用；不把Switch的效益數字移植到本課。", [evidence("switch", "Abstract; §1", "選部分參數的架構安排，與訓練訊號/交付方法不同。")])
claim("c9", "concept", "PPO與DPO是可依訓練需求選擇、部分互為替代的安排，不需在每個小助理串行套用。", "第6段第1–3句中的PPO/DPO與互替", "互替指偏好優化的相關安排，不宣稱DPO適用所有RL環境或兩者在任何資料條件都等效。", [evidence("ppo", "Abstract; §2.1", "policy-gradient取樣及優化角色。"), evidence("dpo", "Abstract; §1; Figure 1", "直接偏好分類loss可避開PPO-based RLHF流程。")])
claim("c10", "concept", "MTP是預測多個未來token的輔助訓練設計，是否使用取決於任務。", "第6段第1–3句中的MTP", "不宣稱本課三產品皆有MTP或都因它改善；只支持方法可選與角色。", [evidence("mtp", "Abstract; §1; §2 opening", "共享trunk上多預測頭與輔助訓練任務。")])
claim("c11", "concept", "量化回應權重表示精度與部署記憶體等交付取捨，可按交付需要選用。", "第6段第1–3句中的量化", "不推出任何CPU/GPU加速保證或完全不掉精度。", [evidence("quant", "Overview opening paragraph", "較低精度儲存與準確率/方法取捨。")])
claim("c12", "concept", "蒸餾透過教師輸出分布等訊號轉移到學生模型，是可選的訓練安排。", "第6段第1–3句中的蒸餾", "不將本課隨機初始化說成載入教師權重；不表示所有學生或任務都必須蒸餾。", [evidence("distill", "§1–2", "以soft targets轉移映射到較易部署模型。")])
claim("c13", "software", "既有助理、B/C與Auto、舊合成實驗及成熟Qwen/Whisper延伸有各自可定位的閱讀/操作入口。", "第3、5、6段各入口連結", "只查本節導覽與任務分路，不重新驗收連結章節所有主張、圖或成熟模型的能力。", [evidence("static", "links", "所有本節repo相對連結存在，19.x編號anchor可定位。"), evidence("run", "route", "freeze-14對應19.2/6/11/12/13/17/19/20與20.1的snapshot與標題核對。"), evidence("route-20", "第1–2段", "成熟Whisper轉寫及Qwen圖文核心的分路。")], "全部本節本地連結及帶編號錨點核對成功；必要來源19.13區分A/B/C，20.1前文明列Qwen與Whisper分路；未重跑成熟模型。")
claim("c14", "software", "本課使用PyTorch做表與向量計算、梯度及參數更新。", "details「程式、資料與授權」第1句", "本人CPU只親跑推論與forward；梯度更新依已讀程式及既有GPU primary receipts，不把靜態檢查說成本人重訓。", [evidence("bc-model", "Tokenizer/nn.Embedding/nn.Linear", "PyTorch張量與查表計算。"), evidence("bc-training", "step", "loss.backward與optimizer.step更新。"), evidence("optimization", "Optimizer; train_loop", "官方API使用意義。")], "CPU載入/forward/softmax通過；static artifact抽取B/C既有GPU訓練環境與weights_changed=true，並確認backward/step呼叫；沒有重啟訓練。")
claim("c15", "software", "本repo程式/文字/自製圖以MIT發布；資料、字型與上游模型依各自來源授權處理，閱讀不需先下載全部材料。", "details「程式、資料與授權」", "核對repo授權宣告、資料入口分列及讀正文無下載前置；未對所有上游資產作全套法律合規審計。", [evidence("license", "MIT License permission grant", "software與associated documentation的MIT授權。"), evidence("assets", "開頭資料表；第19章v2資料段；第20章資料段", "不同來源許可與按需要選資料，並非全體MIT。"), evidence("static", "pins and result", "實際檔案、來源與授權文本檢查。")], "LICENSE首行MIT且含associated documentation；資料入口明說不得統稱repo MIT並分列錄音/字型來源；本節正文與連結可離線讀取，下載指令只在另選操作流程。")


def artifact(identifier, kind, filename, description, **extra):
    path = BASE + filename
    return {"id": identifier, "kind": kind, "path": path, "sha256": digest(ROOT / path), "description": description, **extra}


artifacts = [
    artifact("body", "source_snapshot", "R.4-source.txt", "freeze-14對應course snapshot經lesson_slices取得的R.4原UTF8，沒有正規化空白或重抄。"),
    artifact("run", "execution", "R.4-execution.json", "本人有限CPU、既有primary receipt讀取與匿名HF metadata查核；不是重訓或正式測試重跑。", command=execution["command"], result="全部bounded assertions成功；B答案1，C選boxes，Auto成功；兩版公開配置與權重HEAD成功；v2兩份test journal各3734且未全過門檻。", environment=execution["environment"]),
    artifact("static", "execution", "R.4-scope-execution.json", "本人導覽、授權、PyTorch更新程式與既有GPU primary receipt欄位查核。", command=".venv/bin/python " + BASE + "R.4-seal.py", result=static.read_text()[:0] + "全部本節相對連結、編號錨點、MIT與各來源授權分列通過；B/C原始GPU receipt隨機初始化與權重更新欄位吻合。", environment={"python": platform.python_version(), "device": "cpu", "mode": "read-only metadata and text checks"}),
    artifact("originals", "source_snapshot", "R.4-original-source-receipts.json", "親讀原始論文/官方文件的URL、版本、原檔SHA、定位、支持範圍與小段摘錄；全文留在work。"),
    artifact("code", "code", "R.4-check.py", "有限CPU及公開metadata/原始receipt查核可重做程式。"),
    artifact("seal", "code", "R.4-seal.py", "本節來源收據、導覽/授權查核及報告封存程式。"),
]
report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "R.4", "source": "course/README.md#R.4", "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": ["/root"], "source_sha256": hashlib.sha256(body.encode()).hexdigest(), "figure_sha256": {}, "verdict": "pass", "scope_note": "只獨立技術審R.4及其必要來源/分路；未讀作者plan或他人review。未看本節實際網站桌機/手機版面，這份技術報告不宣稱閱讀/版面驗收。R.4本身無圖，必要證據為來源與介面/成品收據。", "frozen_source": page, "claims": claims, "sources": sources, "artifacts": artifacts, "issues": [], "checks": {
    "factual_accuracy": {"status": "pass", "details": "逐項核對機制、訓練/留出分工、A/B/C角色、從零來源、方法選擇與導覽/授權；未見與一手來源矛盾的實質主張。", "claim_ids": [c["id"] for c in claims]},
    "numeric_verification": {"status": "pass", "details": "R.4沒有手算或速度數字；對兩版v2已完成但未全過的實測概括，親讀原始test journal各3734並核對0/276工具往返低於原0.95門檻、兩版acceptance=false。CPU小例只證明介面，未外推品質。", "claim_ids": ["c7"]},
    "figure_consistency": {"status": "not_applicable", "details": "R.4原文沒有SVG/點陣圖引用；未將連結來源章節的圖計作本節已完成的圖審。", "claim_ids": []},
    "source_verification": {"status": "pass", "details": "真正讀Transformer、Switch、PPO、DPO、MTP、Hinton原文及PyTorch/scikit-learn/HF/Jev官方原文；各自有版本、定位、SHA與支持範圍。程式、primary訓練/評測receipt與公開metadata另保留實際查核。", "claim_ids": [c["id"] for c in claims]},
    "limitations": {"status": "pass", "details": "原文明確分小例與正式訓練、三產品任務、有限v2與成熟模型，並保留未過原能力判準。報告將一次CPU介面、既有GPU正式收據、方法一般性、候選機率與通用能力分開；沒有由一次跑通推出跨平台或一般泛化。", "claim_ids": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15"]}
}}
path = ROOT / "docs/course-revision-20261016/reviews/technical-R.4-backfill-original-01.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
errors = _validate(ROOT, ROOT / "course/README.md", "R.4", body, report, {TASK: ["R.4"]}, set())
write("R.4-validation.json", {"command": ".venv/bin/python " + BASE + "R.4-seal.py", "report": str(path.relative_to(ROOT)), "report_sha256": digest(path), "errors": errors, "result": "format/hash check only; substantive verification recorded in report"})
if errors:
    raise SystemExit(json.dumps(errors, ensure_ascii=False))
print(json.dumps({"report": str(path.relative_to(ROOT)), "sha256": digest(path), "verdict": report["verdict"], "claims": len(claims), "sources": len(sources), "artifacts": len(artifacts), "validation_errors": errors}, ensure_ascii=False))
