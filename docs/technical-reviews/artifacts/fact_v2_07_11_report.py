"""Assemble the independent review from personally inspected sources and saved execution evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_07_11_"
audit = json.loads((OUT / (PREFIX + "audit.json")).read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def artifact(identifier, kind, path, description, **extra):
    return {"id": identifier, "kind": kind, "path": path, "sha256": sha(path),
            "description": description, **extra}


def own(name):
    return "docs/technical-reviews/artifacts/" + PREFIX + name


artifacts = [
    artifact("a_cpu", "execution", own("audit.json"),
             "本 reviewer 真 CPU 執行原文、補充 loss/梯度/權重核對、兩個短 CLI 通路、逐題與分母重算；未重訓 GPU。",
             command=".venv/bin/python -m docs.technical-reviews.artifacts.fact_v2_07_11_audit",
             result="exit 0；原文 batch [2,10]、四目標、loss 5.337442874908447；17 組梯度有限且非零；backward 未改權重；CLI 真更新17組參數；60筆原始生成及全部三階段有效目標核對通過。",
             environment=audit["environment"]),
    artifact("a_code", "code", own("audit.py"), "可重做的 bounded CPU 核對程式；實際 Ruff E4/E7/E9/F/I/UP 通過。"),
    artifact("a_section", "source_snapshot", own("section.txt"),
             "sections() 抽取的本節全部原文，含標題至下一 ## 前所有 LF，沒有正規化。"),
    artifact("a_original_code", "code", own("original_code.txt"), "本節原樣程式；由 audit.py compile/exec 真執行。"),
    artifact("a_original_stdout", "source_snapshot", own("original_stdout.txt"), "原樣程式實際標準輸出；執行環境與結果同 a_cpu。"),
    artifact("a_dataset", "source_snapshot", own("reconstructed_dataset.json"),
             "依與當次相同產生器重建三側全部60題及45篇續寫材料；每側 JSONL hash 與原實報完全相等。"),
    artifact("a_raw", "source_snapshot", own("all_sft_raw_samples.json"),
             "四個評估階段的全部60份原始問答、預期答案、完整生成 ID、EOS與匹配；全部親讀並重新判分。"),
    artifact("a_sft", "source_snapshot", "docs/course-experiments/results/sft.json",
             "原始完整 L4 實報，revision a253d1262bf5f361f9ac4e19232ae752f0ecc7a3；保留配置、訓練歷史、有效目標、完整留出生成和產物指紋。"),
    artifact("a_ablation", "source_snapshot", "docs/course-experiments/results/sft_ablation.json",
             "原始錯標/回放實報；本節只取錯示範限制的 clean/noisy 配方、四筆污染與兩側全部生成作旁證。"),
    artifact("a_text_foundation", "source_snapshot", "docs/course-experiments/results/text_foundation.json",
             "明連 T.4 的獨立600步文字實報；確認它與本節45篇250步屬性文字階段不是同一支線。"),
    artifact("a_formal_run", "code", own("formal_run.txt"),
             "親讀的當次 Git revision 原始 runner 全文；SHA 與 sft.json 的 code_sha256 完全相等。"),
    artifact("a_formal_generator", "code", own("formal_prepare_data.txt"),
             "親讀的當次原始產生器，Git show 精確取回；與目前產生器 bytes 相等。"),
    artifact("a_instructgpt", "source_snapshot", own("instructgpt_original_excerpts.txt"),
             "本 reviewer 用 pdftotext 直接取原 PDF 頁1、6–10、18–19；頁1自證 arXiv:2203.02155v1。"),
    artifact("a_lima", "source_snapshot", own("lima_original_excerpts.txt"),
             "本 reviewer 用 pdftotext 直接取原 PDF 頁1–7、10；讀方法、假說、品質消融與 Discussion 限制，頁1自證v1。"),
    artifact("a_ce", "source_snapshot", own("torch_cross_entropy.txt"),
             "親讀目前安裝 PyTorch 2.14.1+cpu 的 F.cross_entropy 原始函式全文、原檔行號與全檔SHA。"),
    artifact("a_loss", "source_snapshot", own("torch_loss.txt"),
             "親讀目前版本官方 CrossEntropyLoss 原始類別全文；含不忽略目標的公式與正確有效分母。"),
    artifact("a_backward", "source_snapshot", own("torch_backward.txt"),
             "親讀目前版本 Tensor.backward 全文；直接描述 .grad 的累加而非 optimizer 更新。"),
]

external_common = {"verified": True, "checked_original": True, "accessed_on": "2026-10-04"}
sources = [
    {"id": "s_instructgpt", "kind": "paper", "title": "Training language models to follow instructions with human feedback",
     "url": "https://arxiv.org/abs/2203.02155v1", "version": "arXiv:2203.02155v1, 4 March 2022; PDF SHA-256 c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98",
     **external_common, "authority_reason": "Ouyang 等作者的原始方法論文，不是來源庫摘要。",
     "inspection_note": "親讀原PDF直接抽取及原文§1、§3.1 Step1、§3.2、§3.5 SFT、§3.6、§5.2–5.3。SFT從已有GPT-3以示範繼續更新；完整 InstructGPT 還有RM/PPO，不能把其RLHF效果全歸SFT。§3.6將同來源留出提示、人類判準與舊NLP能力分開；限制是特定標註者/資料分佈且模型仍會犯錯。"},
    {"id": "s_lima", "kind": "paper", "title": "LIMA: Less Is More for Alignment",
     "url": "https://arxiv.org/abs/2305.11206v1", "version": "arXiv:2305.11206v1, 18 May 2023; PDF SHA-256 759ddebaa24a03ee772a70d3a0ecc546f53310f209de0f8f6052a3dddb289144",
     **external_common, "authority_reason": "Zhou 等作者的原始指令微調研究。",
     "inspection_note": "親讀原PDF頁1–7、10，§1 next-token pretraining、§2 Superficial Alignment Hypothesis、§3 Training LIMA、§4相同prompt的生成/判準、§5品質消融與§7 Discussion。65B LLaMa加1000份精選示範的有限實驗支持『可能調動已學能力』；§5另用7B和GPT3.5代理評分，未過濾資料品質與風格因素不能混成純錯標實驗。作者稱其hypothesis並明言不如產品級模型穩健，不能外推到本節隨機TinyLM或宣布SFT不會學新知識。"},
    {"id": "s_official_ce", "kind": "official_source", "title": "PyTorch F.cross_entropy implementation and docstring",
     "url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py",
     "version": "Installed PyTorch 2.14.1+cpu; git 5c4886908584029761b579af026dcfb627c84070; lines3478–3578",
     **external_common, "authority_reason": "執行中官方 PyTorch 套件的原始函式，與 runtime git_version 固定定位。",
     "inspection_note": "inspect.getsourcelines 親讀全部cross_entropy：未正規化logits、int64類別targets、ignore_index=-100、reduction sum、交給C++CE算子。本節先reshape再求sum/count，沒有先softmax或把-100放輸入embedding。完整source snapshot及原檔SHA在a_ce。"},
    {"id": "s_official_loss", "kind": "official_source", "title": "PyTorch CrossEntropyLoss class and indexed-target formula",
     "url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/loss.py",
     "version": "Installed PyTorch 2.14.1+cpu; git 5c4886908584029761b579af026dcfb627c84070; CrossEntropyLoss starts line1200",
     **external_common, "authority_reason": "官方源碼內完整損失契約及數學式。",
     "inspection_note": "親讀class全文：未加weight、未加label smoothing的類別target loss為-log softmax(z)_target；ignore_index位置不貢獻直接gradient，平均分母只算未忽略targets。它只以給定target計算，沒有真偽檢查；數值例也獨立log_softmax/gather驗算。"},
    {"id": "s_official_backward", "kind": "official_source", "title": "PyTorch Tensor.backward implementation and leaf-gradient contract",
     "url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_tensor.py",
     "version": "Installed PyTorch 2.14.1+cpu; git 5c4886908584029761b579af026dcfb627c84070; Tensor.backward starts line566",
     **external_common, "authority_reason": "正在執行版本的官方Tensor接口源碼。",
     "inspection_note": "親讀全部Tensor.backward：對graph leaves按chain rule求導並累加.grad；scalar可省gradient；沒有optimizer.step。a_cpu核對梯度存在、所有權重保持逐位相等，不能因此聲稱已微調。"},
]


def repo_source(identifier, path, note):
    sources.append({"id": identifier, "kind": "repository_code", "title": path,
                    "path": path, "sha256": sha(path), "verified": True,
                    "version": "Current Git " + head + "; pinned by whole-file SHA-256",
                    "inspection_note": note})


repo_source("s_data", "tiny_perceptron/data.py",
            "親讀全檔。ByteTokenizer每byte+8、vocab264與EOS2；render_chat把assistant內容及EOS放targets、右移一次；pad_batch右補齊並分開valid/IGNORE；toy_conversations按a外圈b內圈排列。")
repo_source("s_model", "tiny_perceptron/model.py",
            "親讀全檔，ModelConfig預設vocab264、TinyLM保持同架構、forward輸出[B,T,V]；loss_sum用CE sum及labels!=-100 count、masked_loss除有效分母；generate的greedy/EOS/上下文上限。與sft原實報記錄的全檔hash相等。")
repo_source("s_attention", "tiny_perceptron/attention.py",
            "親讀全檔，attention_mask causal AND valid key；manual_attention對無有效key行明確零輸出。右補齊維持真實位置ID；a_cpu額外不等長batch prefix差2.384185791015625e-7。")
repo_source("s_training", "tiny_perceptron/training.py",
            "親讀全檔，save/load契約保存config/model/optimizer/step/RNG/metadata/tokenizer資訊；restore_rng與optimizer重建分開。vocab大小不等於任意tokenizer語義相容。")
repo_source("s_cli", "scripts/train.py",
            "親讀全檔，parser任務/train/checkpoint/resume；prepare_examples SFT資料；main載入基模、byte264大小核對；新階段建新AdamW；--resume才恢復optimizer/RNG並拒絕改task；--train才step且保存。真CPU以可信臨時base跑dry與一個SFT更新並核對參數；非264詞表實際被拒绝。")
repo_source("s_common", "scripts/course_experiments/common.py",
            "親讀new_lm、split_records、text_examples、fit_lm、_nll、evaluate_lm完整實作。new_lm用ctx.seed；fit預設batch16/lr0.003、新AdamW/Random42抽樣；有效目標逐batch累加；evaluate在全留出上按raw ID/greedy32/EOS分開。與当次原實報code_sha256相等。")
repo_source("s_text", "scripts/course_experiments/text.py",
            "親讀_steps/_save_splits/_evaluations及run_sft/run_sft_ablation。run_sft兩次new_lm(width64,layers2)，直接900；另一支只將train45題問題+答案串文字250後同model再SFT900。沒有story權重依賴。fit會先保存pretrain.pt再繼續同一個記憶體model，並非另做disk reload。與当次原實報code_sha256相等。")
repo_source("s_prepare", "scripts/prepare_data.py",
            "親讀generate_records attributes-sft與main內整家族切分；3色×2形狀×2音高，每家族5題，全部60題。正式入口另用common.split_records整家族分45/5/10。當次git產生器bytes與目前相同，重建正式入口每側JSONL與原始實報SHA完全相同。")
sources.extend([
    {"id": "s_cpu", "kind": "execution", "title": "本 reviewer bounded CPU audit", "verified": True, "artifact_id": "a_cpu"},
    {"id": "s_ids", "kind": "derivation", "title": "byte ID、shift長度與有效目標推導", "verified": True,
     "details": "0+0=?有5個ASCII bytes，回答0有1個；未shift序列長1 BOS+1 user+5問題+1 EOS+1 assistant+1答案+1 EOS=11，x/y右移後各10。assistant只監督答案byte及EOS，每筆2，兩筆4；ASCII0=48、1=49、3=51，加8後ID56/57/59；EOS固定2，vocab=256 bytes+8控制ID=264。"},
    {"id": "s_budget", "kind": "derivation", "title": "兩支線的總更新與有效目標預算", "verified": True,
     "details": "直接SFT總更新900；另一條250文字+900SFT=1150。兩條SFT各101091目標，額外文字191742，所以總目標直接101091、兩階段191742+101091=292833。對話階段相同不代表总预算相同；原文明确需另匹配预算，沒有把这轮成績說成形式本身的因果效果。"},
    {"id": "s_wrong", "kind": "derivation", "title": "給定錯label的交叉熵更新方向", "verified": True,
     "details": "未加權且無label smoothing時L=-log p_y，對logits z_j有∂L/∂z_j=p_j-1[j=y]。若把正確0 ID56改成錯3 ID59，監督的負項便移至59；在獨立logit梯度下降的小步更新，p59-1<0會增大z59。這說明目標機制接受錯label，並不保證共享神經網路每一步都單調增加所有錯例概率；LIMA資料品質與本專案clean/noisy原始結果另提供有限旁證。"},
])


def evidence(source_id, locator, supports):
    return {"source_id": source_id, "locator": locator, "supports": supports}


claims = []


def claim(identifier, kind, statement, location, refs, scope, artifact_ids=None, verification=None):
    entry = {"id": identifier, "kind": kind, "statement": statement, "location": location,
             "status": "verified", "evidence": refs, "artifact_ids": artifact_ids or [], "scope": scope}
    if verification is not None:
        entry["verification"] = verification
    claims.append(entry)


def executed(expected, observed, details, **extra):
    return {"method": "executed", "expected": expected, "observed": observed, "details": details, **extra}


claim("c1", "concept", "預訓練語言模型以一般文字的下一token預測學續寫。", "07.md:332，首句前半",
      [evidence("s_lima", "§1 Introduction, p1", "作者明確寫pretrained to predict next token並描述大規模原文階段。")],
      "這是自回歸LM的訓練目的；不保證TinyLM小世界具有一般知識。", ["a_lima"])
claim("c2", "concept", "SFT是在既有模型上用示範更新參數；僅加『請回答』前文不構成這個微調過程。", "07.md:332/334",
      [evidence("s_instructgpt", "§3.1 Step1, p6; §3.5 SFT, p8", "作者從GPT-3以labeler demonstrations繼續supervised training，與GPT prompted基線分開。"),
       evidence("s_lima", "§3 Training LIMA, p4", "由既有LLaMa用示範、AdamW和epochs實施微調。")],
      "純prompt可改推論行為；本文只界定沒有optimizer參數更新就不是此SFT，沒有聲稱prompt無效。", ["a_instructgpt", "a_lima"])
claim("c3", "software", "對話資料使用同一TinyLM架構與CE，角色/答案mask與valid分別提供監督和讀取許可。", "07.md:332–345/357",
      [evidence("s_data", "render_chat lines51–66; pad_batch lines69–86", "產生一次shift的long X/Y和bool valid，user保留輸入而非直接target。"),
       evidence("s_model", "TinyLM lines53–86; loss_sum/masked_loss lines92–105", "同forward及CE實作，不因SFT換掉模型類別。"),
       evidence("s_attention", "attention_mask lines10–18; forward lines51–76", "valid與causal合併key讀取許可。")],
      "CPU FP32、預設byte264、width8、一層、128上下文。原樣兩筆等長全valid；另外不等長測試只支援本次padding機制。",
      ["a_cpu", "a_code"], executed("有效回答target和合法输入分離；valid可阻止PAD key讀取。",
                                  "X/Y int64、valid bool；補齊prefix最大差2.384185791015625e-7。",
                                  "原樣程式與額外不等長右padding逐位置比較；按有效mask以log_softmax/gather重算loss。"))
claim("c4", "numeric", "toy前兩題是0+0→0、0+1→1；各答案一byte加EOS，共四個有效目標。", "07.md:351，前兩句",
      [evidence("s_data", "toy_conversations lines117–122; ByteTokenizer lines15–23; render_chat lines51–66", "產生順序、byte偏移和EOS監督。"),
       evidence("s_ids", "1byte答案+1EOS，2筆×2=4", "完整计数推導。")], "僅前兩筆ASCII玩具問答，不按中文字數外推。",
      ["a_cpu"], executed("4有效目標，[56,2,57,2]。", "4，[56,2,57,2]。", "讀首兩筆消息、逐Y位置列出並按Y!=-100計數。", tolerance="整數精確相等"))
claim("c5", "numeric", "兩筆問答的X/Y batch形狀為[2,10]。", "07.md:351，『兩筆每筆10位置』",
      [evidence("s_ids", "11序列項右移後10；batch2", "含所有結構標記的完整長度推導。")], "未截斷的兩筆相同長度示範。",
      ["a_cpu"], executed("X/Y/valid都[2,10]。", "三者均[2,10]。", "原樣pad_batch输出shape逐項assert。", tolerance="shape整數精確相等"))
claim("c6", "numeric", "模型在每個位置輸出264個byte/結構候選。", "07.md:351，『每位置264』",
      [evidence("s_model", "ModelConfig lines14–16; TinyLM.output lines64/85", "預設字表264和width到字表输出層。"),
       evidence("s_ids", "256+8=264", "控制ID與普通byte候選總數。")], "本節預設ByteTokenizer與匹配ModelConfig，不能通用套到別的詞表。",
      ["a_cpu"], executed("logits [2,10,264]。", "[2,10,264]，torch.float32。", "真forward的最后候選軸及dtype核對。", tolerance="shape精確相等"))
claim("c7", "numeric", "本次接口CE loss為正，並按四個有效目標平均。", "07.md:345/348/351",
      [evidence("s_official_ce", "F.cross_entropy lines3478起，ignore_index/reduction", "接受logits與類別target，忽略-100。"),
       evidence("s_model", "loss_sum/masked_loss lines92–105", "先CE sum再除有效count。")], "初始化seed42的CPU FP32數值；此loss不代表模型已學會問答。",
      ["a_cpu", "a_ce"], executed("有限且>0；独立-log_softmax有效target均值应等於masked_loss。", "loss=5.337442874908447，独立均值相符。",
                                               "输入、label、dtype及全部有效target见a_cpu.interface；只用四位置gather，不把user/PAD列入分母。", tolerance="loss對重算值绝對誤差≤1e-6，rtol=0；正值判斷严格>0"))
claim("c8", "software", "loss.backward產生梯度；沒有optimizer.step，所以這段不更新權重。", "07.md:346/351",
      [evidence("s_official_backward", "Tensor.backward lines566起", "對graph leaves累加.grad，没有參數step。"),
       evidence("s_cpu", "audit.json interface.gradient_maxima/weights_unchanged_after_backward", "实际有限非零梯度及同seed初始化權重逐項比較。")],
      "此無dropout初始化接口；未把gradient當成訓練品質或微調完成。", ["a_cpu", "a_backward"],
      executed("非零且有限gradient，backward之後權重不变。", "17組gradient全有限且非零；output.weight最大0.7800482511520386；權重全部torch.equal。",
               "精確執行本節未改碼，再重建seed42同配置起點逐parameter/state比較。"))
claim("c9", "software", "--task sft --checkpoint BASE --train會載入基模並對問答目標啓用更新。", "07.md:353，flags範例",
      [evidence("s_cli", "parser lines38–45; prepare_examples lines88–104; main lines204–228/298–413", "任務選擇、checkpoint载入、AdamW與train才step。")],
      "真CLI仅一個CPU更新；BASE為本reviewer可信自建byte264文字checkpoint；正式步數/data/output須依明連T.4配方，不将此步解讀為對話能力。",
      ["a_cpu"], executed("dry-run不保存更新檔；加--train保存SFT checkpoint並改基模權重。", "dry無輸出checkpoint；train一步17組參數改變，saved task=sft，width=8，step=1，有效target4。",
                           "两条实际subprocess argv/stdout/stderr全存a_cpu.cli；loaded config由base取得。"))
claim("c10", "software", "續接需匹配tokenizer/模型ID；同階段resume和新階段載基模要按checkpoint契約區分。", "07.md:353/357",
      [evidence("s_training", "save_checkpoint lines31–67; load_checkpoint lines80–105", "保存model/config、optimizer、RNG及tokenizer資料，重建模型。"),
       evidence("s_cli", "main lines227–239/298–317", "CLI使用byte264、非264拒絕；新階段新optimizer，resume才帶回歷史並核task。")],
      "詞表大小相同仍不能證明ID語義相同，本文把映射責任明示給使用者。新SFT階段可新建optimizer，不能把此段讀成強制恢復舊text optimizer。",
      ["a_cpu"], executed("已保存匹配基模可更新，不匹配字表CLI拒絕。", "264-ID checkpoint真載入；5-ID checkpoint以非零exit及『byte tokenizer…詞表不相容』拒绝。",
                           "用可信临时checkpoint测试，实际错误输出与成功payload在a_cpu.cli；親讀T.4末段关于不加resume的新阶段说明。"))
claim("c11", "concept", "微調比較應保留同一批題與生成設置，將內容、格式及舊能力分別記錄。", "07.md:355，第一句",
      [evidence("s_instructgpt", "§3.2用户ID留出，§3.6 Evaluation p10", "相同来源留出分佈的policy比較，分開人類metadata和舊NLP任务评估，不以單次loss替代。"),
       evidence("s_lima", "§3/§4 Human Evaluation", "held-out dev選模型；對每個test prompt比較各模型回答，perplexity不等於生成品質。")],
      "这是比较程序要求；本節CPU样例沒有测旧能力，正式属性报告也不是所有语言能力考卷。", ["a_instructgpt", "a_lima", "a_sft"])
claim("c12", "concept", "SFT可能讓已有知識/能力更容易被調動。", "07.md:355，第二句前半",
      [evidence("s_lima", "§1/§2 Superficial Alignment Hypothesis，§7 Discussion", "原論文直接提出示範教格式、調動pretraining已有能力，且限定hypothesis及稳健性。")],
      "原文只說『可能』。LIMA65B有限研究支持此可能性，不证明TinyLM具有那些知識，也不证明SFT只调格式而永不学新內容。", ["a_lima"])
claim("c13", "concept", "以錯誤示範作SFT target會把錯內容送入學習目標，資料格式不会自动保證答案正確。", "07.md:355，第二句後半",
      [evidence("s_official_loss", "CrossEntropyLoss indexed-target公式，class starts1200", "只根据给定label計-log p_target，沒有真偽判斷。"),
       evidence("s_lima", "§2 Alignment Data，§5資料品質消融", "需要主动精选回答/控制质量，並非僅含角色格式。"),
       evidence("s_wrong", "∂L/∂z_j=p_j-1[j=y]", "错误label决定目标方向；明确独立logit推導范围。")],
      "不保證共享参数模型每次更新都完全記住错例；本项目clean/noisy仅单seed、小世界旁证，test NLL與完整匹配还会方向不一致。",
      ["a_loss", "a_lima", "a_ablation"])
claim("c14", "empirical", "正式直接對話路線由隨機模型在45題屬性資料上訓練900步，配置width64/layers2/seed42/batch16/lr0.003。", "07.md:359，第一條與共同配置",
      [evidence("s_text", "run_sft lines556–566", "直接new_lm再fit_lm，沒有预训练checkpoint依赖。"),
       evidence("s_common", "new_lm lines45–47; fit_lm lines122–197", "seed、batch16/lr0.003默认、更新循环與有效目標累加。")],
      "审计既有完整NVIDIA L4、Py3.13.3、Torch2.14.1+cu126原報；未重跑GPU。60题12家族仅45/5/10，正式结果不外推一般聊天。",
      ["a_cpu", "a_sft", "a_dataset", "a_raw", "a_formal_run", "a_formal_generator"],
      executed("完整900步、45训练题、同配置、全留出评估。", "原实报steps900/45records/141568parameters；重建三侧hash全相等，SFT目标101091；after验证1/5、test5/10，EOS全结束。",
               "亲读原report所有4阶段共60samples；按原raw ID重新计算exact/EOS/NLL分母；用同seed sampler重算全部900×16曝光，无模型长训练。",
               denominators={"seed":42,"train_records":45,"train_families":9,"updates":900,"batch_size":16,"supervised_targets":101091,
                             "validation_records":5,"validation_targets":36,"test_records":10,"test_targets":69,"generation_max_new_tokens":32,"temperature":0}))
claim("c15", "empirical", "另一支線只把相同45題問題+答案串文字更新250步，保存后再同配置做900步SFT，没有沿用故事模型。", "07.md:359，第二條與來源界線",
      [evidence("s_text", "run_sft lines567–590", "text_records仅parts.train问题+答案、再次new_lm、250/900明确调用。"),
       evidence("s_common", "fit_lm lines159–197", "文字阶段返回前保存pretrain.pt，接下來同一model继续，而不另引入story。")],
      "两阶段有新optimizer；同一已保存model在記憶體继续，不声称实际另做disk重載。额外文本仍是属性训练材料，不是广泛知识；正式L4记录是历史证据而非本reviewer GPU重跑。",
      ["a_cpu", "a_sft", "a_dataset", "a_raw", "a_text_foundation", "a_formal_run"],
      executed("45相同属性训练题文字化，250→900，固定留出题。", "文字records SHA22460ba6…、SFT SHA cf22a7eb…与重建全相等；250文字目标191742、900SFT目标101091；after验证3/5、test7/10；同题同32greedy。",
               "重建全部45文字材料；核对原报告全SHA、所有训练history与complete_run/step_scale1；所有阶段5+10题逐一对照原始生成ID、预期答案與分母。",
               denominators={"seed":42,"pretrain_records":45,"pretrain_updates":250,"pretrain_targets":191742,"sft_updates":900,
                             "sft_targets":101091,"batch_size_each_stage":16,"validation_records":5,"validation_targets":36,
                             "test_records":10,"test_targets":69,"generation_max_new_tokens":32,"temperature":0}))
claim("c16", "numeric", "第二條比直接SFT多250次文字更新，總更新1150而非900，總有效目標也不同。", "07.md:359，最後一句的預算限制",
      [evidence("s_budget", "900+250=1150；191742+101091=292833", "透明加法及不同計分位置暴露；明示不是同總更新/目标预算对比。")],
      "第二条总budget较大，观测结果支持该有限配方比较；没有隔离预训练形式本身的因果效果。",
      ["a_cpu", "a_sft"], executed("direct总900/101091；两阶段1150/292833。", "逐sampler重算得到900、250+900、101091、191742+101091。",
                               "先分别按shift/text与assistant/SFT目标计数，再加总，不把两种目标计分定义混同。", tolerance="整数精确相等"))
claim("c17", "numeric", "练习复制首笔assistant答案0改3后，有效ID由[56,2]变[59,2]，问题和角色不变。", "07.md:361，練習",
      [evidence("s_ids", "ASCII48+8=56，ASCII51+8=59，EOS2", "ID精确计算。"),
       evidence("s_data", "ByteTokenizer.encode与render_chat", "不靠扫描内容判角色，EOS沿同控制ID。")],
      "复制消息操作用于演示错答案，本reviewer没有修改正式数据；原toy仍0。",
      ["a_cpu"], executed("[56,2]→[59,2]；问题及角色结构不变。", "两份有效targets完全相符，X除末尾答案byte外逐项相等，原toy答案仍0。",
                           "deepcopy首例后只改assistant.content，再render逐项比较；EOS不作为普通byte编码。", tolerance="ID与X前缀精确相等"))

prerequisites = []
context_parts = []
for path, ids in (("course/chapters/04.md", ["4.6"]),
                  ("course/chapters/07.md", ["7.1", "7.3", "7.7"]),
                  ("course/chapters/05.md", ["5.7"]),
                  ("course/training.md", ["T.4"])):
    all_sections = dict(sections(ROOT / path))
    for identifier in ids:
        section = all_sections[identifier]
        context_parts.append(path + "#" + identifier + "\n" + section)
        prerequisites.append({"source": path + "#" + identifier,
                              "source_sha256": hashlib.sha256(section.encode()).hexdigest(),
                              "inspection_note": "本reviewer完整亲读必要前置；未沿用其先前审阅结论。"})
(OUT / (PREFIX + "prerequisites.txt")).write_text("\n".join(context_parts), encoding="utf-8")
artifacts.append(artifact("a_prerequisites", "source_snapshot", own("prerequisites.txt"),
                          "明示必要前置与训练配方T.4的完整原文快照；所有recipe和resume/new-stage说明完整亲读。"))
for label in ("ruff", "checker"):
    receipt_path = own(label + "_receipt.json")
    if (ROOT / receipt_path).exists():
        receipt = json.loads((ROOT / receipt_path).read_text())
        artifacts.append(artifact("a_" + label, "execution", receipt_path,
                                  "本reviewer真執行的" + label + "命令及原始輸出；checker只核對格式/身份/版本。",
                                  command=receipt["command"],
                                  result="exit " + str(receipt["returncode"]) + "; " + receipt["stdout"].strip(),
                                  environment=receipt["environment"]))

report = {
    "schema_version": 1,
    "review_stage": "technical",
    "lesson_id": "7.11",
    "source": "course/chapters/07.md#7.11",
    "reviewer_task": "/root/integration_technical_coordinator/fact_v2_07_11",
    "reviewer_context": "fresh",
    "source_sha256": audit["source_sha256"],
    "figure_sha256": {},
    "verdict": "pass",
    "claims": claims,
    "sources": sources,
    "artifacts": artifacts,
    "issues": [],
    "prerequisites_read": prerequisites,
    "review_note": "只审7.11，无教材编辑、无代理委派、无author_tasks猜测、未读旧审阅结论。候选原文亲读才登记；新CPU执行与历史GPU原始结果分别标明，checker仅格式/身份/版本核对。",
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "定义与软件机制逐项对照原论文/官方版本原码及本项目源码；实际CPU接口与CLI都执行。没有把随机backward叫微调或把本节属性阶段当故事模型。", "claim_ids": ["c1","c2","c3","c8","c9","c10","c11","c12","c13","c14","c15"]},
        "numeric_verification": {"status": "pass", "details": "ASCII偏移、shift后10格、4有效目标、264候选、正loss、练习59/EOS2全核对；CE按四目标独立重算。历史三阶段目标与所有60份生成重新计数，900对1150预算明列。", "claim_ids": ["c4","c5","c6","c7","c14","c15","c16","c17"]},
        "figure_consistency": {"status": "not_applicable", "details": "7.11没有SVG或其他图引用；本审阅未用前置4.6的图证明任何主张，因此没有需要渲染核对的本节图。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "InstructGPT与LIMA实际原PDF自证v1，读取方法/定位/限制并直接pdftotext留存；PyTorch2.14.1+cpu按git5c488690固定原码和行号。实报配置/逐题/完整hash均核对，历史runner另从Git精确取回，没有用候选摘要当权威。", "claim_ids": [entry["id"] for entry in claims]},
        "limitations": {"status": "pass", "details": "接口不验证品质；正式单seed、小世界、1+2留出家族与额外文本预算不能外推一般聊天或因果预训练效果。LIMA能力调动只是有限可能性；tokenizer相同大小不等于语义匹配；新阶段与resume区别依明连T.4读法明确。", "claim_ids": ["c1","c2","c3","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16"]},
    },
}
assert hashlib.sha256(dict(sections(ROOT / "course/chapters/07.md"))["7.11"].encode()).hexdigest() == report["source_sha256"]
(ROOT / "docs/technical-reviews/7.11.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print(json.dumps({"lesson": "7.11", "verdict": report["verdict"], "claims": len(claims),
                  "source_sha256": report["source_sha256"], "sources": len(sources),
                  "artifacts": len(artifacts)}, ensure_ascii=False))
