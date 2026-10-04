"""Package independently inspected primary sources and the fresh 13.11 CPU audit."""

import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = "docs/technical-reviews/artifacts/fact_v2_13_11_"
AUDIT = json.loads((ROOT / (BASE + "audit.json")).read_text())
ARTIFACTS = []


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def artifact(identifier, path, kind, description, command=None, result=None):
    item = {"id": identifier, "kind": kind, "path": path, "sha256": digest(path), "description": description}
    if kind == "execution":
        item.update(command=command, result=result, environment=AUDIT["environment"])
    ARTIFACTS.append(item)
    return identifier


def original(identifier, kind, title, url, version, note, evidence_ids):
    return {
        "id": identifier, "kind": kind, "title": title, "url": url, "version": version,
        "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
        "authority_reason": "原作者發表的論文原本／專案維護者官方 GitHub 原碼，直接定義所核對的方法或 API。",
        "inspection_note": note, "artifact_ids": evidence_ids,
    }


def repository(identifier, title, path, note):
    return {
        "id": identifier, "kind": "repository_code", "title": title, "path": path,
        "sha256": digest(path), "version": f"2026-10-04 current file sha256={digest(path)}",
        "verified": True, "inspection_note": note,
    }


def reference(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, evidence, artifact_ids, scope, verification=None):
    item = {
        "id": identifier, "kind": kind, "statement": statement, "location": location,
        "status": "verified", "evidence": evidence, "artifact_ids": artifact_ids, "scope": scope,
    }
    if verification is not None:
        item["verification"] = verification
    return item


def main():
    command = AUDIT["command"]
    artifact("a_audit", BASE + "audit.json", "execution", "真 CPU 執行：本節及練習、梯度隔離、所有原始資料／checkpoint／逐題推論與 rollout 優勢核對。", command, AUDIT["result"])
    artifact("a_code", BASE + "audit.py", "code", "上述實際執行的審閱程式；沒有訓練、環境安裝或教材修改。")
    artifact("a_derivation", BASE + "derivation.txt", "derivation", "單步終止條件、手算、baseline 抵消與 MSE 條件均值、log-probability 梯度方向及限制。")
    artifact("a_inspection", BASE + "source_inspection.txt", "source_snapshot", "本人對原論文／官方原碼／現有完整實驗的讀取定位與條件紀錄。")
    artifact("a_prerequisites", BASE + "prerequisites.json", "source_snapshot", "完整讀取指定五節前置的原文與 sections SHA；包含親讀修訂後 13.10。")
    artifact("a_section", BASE + "section.txt", "source_snapshot", "依 scripts.check_technical_reviews.sections 擷取的 13.11 未正規化完整原文。")
    artifact("a_retrieval", BASE + "retrieval.json", "source_snapshot", "固定 commit 的 OpenAI／PyTorch 官方原碼及 contextual-bandit 原 PDF HTTPS 下載結果與 SHA。")
    artifact("a_frozen", BASE + "frozen_evaluations.json", "execution", "全部 165 題的原始 row、所有候選 reward／機率／選擇與分母，當次僅由已核對原checkpoint的portable JSON重建再做真CPU推論；與保存結果逐欄完全相同。", command, "132+15+18 題全部重現；逐欄數字最大絕對誤差 0。")
    for name in ("original", "exercise", "prerequisite13_10", "prerequisite13_10_exercise"):
        execution = AUDIT["snippet_executions"][name]
        artifact("a_" + name, BASE + name + "_stdout.txt", "execution", f"實際执行 {name} 的完整 stdout；同名 code.txt 保存當次原碼。", execution["command"], f"exit_code=0; stdout={execution['stdout'].strip()}")
        artifact("a_" + name + "_code", BASE + name + "_code.txt", "code", f"{name} 當次逐字擷取／練習替換後的程式內容。")
    for name in ("ppo", "instructgpt", "dpo"):
        artifact("a_" + name + "_pdf", BASE + name + "-original.pdf", "source_snapshot", f"實際讀取的 {name} 原論文 PDF 的逐byte複本；與先前原檔SHA完全相同，原檔保留。")
        artifact("a_" + name + "_text", BASE + name + "-original_pdftotext.txt", "source_snapshot", "本人從上述原 PDF 新執行 pdftotext -layout 的完整原文抽取。")
    artifact("a_bandit_pdf", BASE + "contextual_bandit_v2.pdf", "source_snapshot", "直接 HTTPS 取得的 Li 等人 contextual-bandit 原論文 v2。")
    artifact("a_bandit_text", BASE + "contextual_bandit_v2.txt", "source_snapshot", "原論文 contextual-bandit 定義 §2.1 的全文文字抽取。")
    artifact("a_openai", BASE + "openai_train_policy.txt", "source_snapshot", "固定 OpenAI 原始 commit 的 train_policy.py 完整 bytes，與候選庫原碼一致。")
    artifact("a_torch", BASE + "torch_tensor.txt", "source_snapshot", "對應目前安裝 torch git commit 的官方 torch/_tensor.py 完整原碼。")
    artifact("a_records", BASE + "records.json", "source_snapshot", "既有實驗完整 train／validation／test 原始 165 題資料的逐byte複本，SHA不變；重建及所有 family／欄位核對。")
    artifact("a_training_report", "docs/course-experiments/results/posttraining.json", "source_snapshot", "原始完整實驗結果、環境／配置／code SHA／rollout trace／逐題結果與計時。")
    for checkpoint in AUDIT["checkpoint_sha256"]:
        name = checkpoint["name"]
        artifact("a_checkpoint_" + name, checkpoint["path"], "source_snapshot", "親讀原checkpoint後導出的portable JSON；保存原binary全檔SHA／metadata及所有tensor shape、dtype、bytes SHA、values。逐tensor roundtrip完全相同；當次凍結重驗僅由此JSON重建，原binary保留。")
    export_command = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_11_portable.py --export"
    portable_command = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_11_portable.py"
    artifact("a_portable_code", BASE + "portable.py", "code", "一次原檔讀取與portable匯出、僅JSON重建重驗的真程式；重驗模式禁止torch.load與outputs目錄讀取。")
    artifact("a_portable_export", BASE + "portable_export_receipt.json", "execution", "本人的原檔讀取／byte-copy／五份checkpoint metadata與21個tensor／原模型165題輸出保存紀錄。", export_command, "原binary SHA與報告一致；五份metadata及21個tensor JSON roundtrip逐byte相同；PDF／records複本SHA不變。")
    artifact("a_portable_verify", BASE + "portable_verify_receipt.json", "execution", "僅用portable JSON重建五模型的CPU驗證收據；torch.load与outputs讀取被禁止。", portable_command, "五模型全部165題輸出與原binary參考精確相等；完整逐題evaluation精確重現，最大誤差0。")
    artifact("a_native_outputs", BASE + "native_five_model_reference.json", "execution", "親讀五份原binary時保存的全部165題原模型logits／RM／critic輸出，用於等價核對。", export_command, "保存132+15+18題與五模型全輸出，無訓練。")
    artifact("a_portable_outputs", BASE + "portable_five_model_outputs.json", "execution", "僅從JSON重建後的同一五模型全部165題輸出。", portable_command, "與native_five_model_reference逐字典／數字相同。")
    artifact("a_closure_history", BASE + "closure_history.json", "source_snapshot", "本次可發布closure前之本人report／code／audit／checks收據的完整SHA與留存位置。")
    history = json.loads((ROOT / (BASE + "closure_history.json")).read_text())
    for index, item in enumerate(history):
        artifact("a_before_closure_" + str(index), item["archived_path"], "source_snapshot", "本人上一輪實際原檔審閱證據的原bytes歷史；不作當前ignored路徑依賴。")
    artifact("a_builder", BASE + "build_report.py", "code", "將本人主張與已核對證據封裝成本報告的程式；不產生或替代技術判斷。")
    sources = [
        original("s_ppo", "paper", "Proximal Policy Optimization Algorithms", "https://arxiv.org/pdf/1707.06347v2", "arXiv:1707.06347v2, 28 Aug 2017; header in actual original PDF", "親讀標題／摘要、§2.1 式1–2、§3 式6–7、§5 式9–12與 Algorithm 1。核固定 advantage、stochastic policy、舊策略比值、自己的 value MSE；單步末端 V=0 才能縮成 reward−old_value。不是所有 PPO 配方相同或每步機率保證改善。", ["a_ppo_pdf", "a_ppo_text", "a_inspection"]),
        original("s_instructgpt", "paper", "Training language models to follow instructions with human feedback", "https://arxiv.org/pdf/2203.02155v1", "arXiv:2203.02155v1, 4 Mar 2022; actual PDF pp. 8–9 and Appendix C.4 p. 23", "親讀 §3.1 的 scalar RM、reward shift invariance／mean-zero normalization、prompt-response bandit environment、per-token KL與式2；附錄 C.4 的 GAE。文中 episode 層次稱 bandit，不代表 token PPO 沒有多個位置；沒有引用該文品質結果證明教材小模型。", ["a_instructgpt_pdf", "a_instructgpt_text", "a_inspection"]),
        original("s_dpo", "paper", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "https://arxiv.org/pdf/2305.18290v3", "arXiv:2305.18290v3, 29 Jul 2024; original PDF §3 pp. 3–4", "親讀 §3 式1–2 的相對 reward 模型，式3與其後 shaped reward 段落。支持 reward 零點不校準正確率及可加入固定 reference KL 代價；沒有把 DPO 結論當成 PPO 品質證據。", ["a_dpo_pdf", "a_dpo_text", "a_inspection"]),
        original("s_bandit", "paper", "A Contextual-Bandit Approach to Personalized News Article Recommendation", "https://arxiv.org/pdf/1003.0146v2", "arXiv:1003.0146v2, 1 Mar 2012; actual PDF §2.1 p. 2", "下載並親讀 §2.1 steps 1–3：observe context、select one arm、receive payoff、improve strategy；下一 trial 沒有本題行動造成的多步回饋鏈。僅用此任務定義，未引 LinUCB 或 Yahoo 實驗結論。", ["a_bandit_pdf", "a_bandit_text"]),
        original("s_openai", "official_source", "OpenAI lm-human-preferences train_policy.py", "https://raw.githubusercontent.com/openai/lm-human-preferences/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py", "commit cbfd210bb8b08f6bc5c26878c10984b90f516c66; snapshot sha256 f0fc233815d0f20530202747b16ec3e0b71d82fa6f4381977b7d77c9e20ed626", "直接下載官方 pinned 原碼；親讀 compute_rewards lines149–156 與 loss lines319–357，看到 per-token KL、terminal score、反序 GAE、terminal nextvalue=0、whiten和stop_gradient、自己的value loss及new/old ratio。未執行 TensorFlow 專案，不主張教材 unwhitened 優勢／獨立網路完全等同此配方。", ["a_openai", "a_retrieval", "a_inspection"]),
        original("s_torch", "official_source", "PyTorch Tensor.detach official source", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_tensor.py", "installed torch 2.14.1+cpu; git commit 5c4886908584029761b579af026dcfb627c84070", "下載對應安裝 commit 官方原碼，親讀 lines798–815 Tensor.detach docstr：result detached from graph、never require gradient、forward AD removed、shares storage。實際 CPU probe 補驗圖隔離，沒有跨 dtype／device／版本外推。", ["a_torch", "a_retrieval", "a_audit"]),
        repository("s_module", "Finite-card posttraining helpers and networks", "tiny_perceptron/posttraining.py", "完整讀取原碼；bandit_advantage lines24–28 驗形狀且detach差值；ppo_clipped_objective lines31–50 固定old/advantage；FiniteResponsePolicy lines53–61／RewardModel lines64–75／ValueModel lines78–86 輸入與輸出。"),
        repository("s_experiment", "Fixed finite-card CPU experiment", "scripts/course_experiments/posttraining.py", "完整讀取原碼：build_records lines63–106 預寫回答與規則、split_records lines109–121 家族切分；_evaluate lines187–253 保存完整逐題；rollout lines335–355 固定舊value與RM reward，另優化value MSE／policy KL；輸出配置／時間scope lines448–535。原始code SHA與目前檔一致；親讀原checkpoint且保存等價JSON後，全部165題僅由JSON重建凍結推論。"),
        {"id": "s_math", "kind": "derivation", "title": "Terminal-bandit advantage and exact decimal calculation", "verified": True, "details": (ROOT / (BASE + "derivation.txt")).read_text()},
        {"id": "s_execution", "kind": "execution", "title": "Fresh CPU original/exercise/gradient and frozen full-row audit", "verified": True, "artifact_id": "a_audit"},
    ]
    r = reference
    common = ["a_audit", "a_code", "a_derivation"]
    claims = [
        claim("c1", "concept", "收到評分本身不會改寫下一次選擇；需要優化策略參數。", "第1段「光收到這個數字」", [r("s_ppo", "Abstract and §5 Algorithm 1", "先採樣與收回饋，再以所建目標優化；兩個步驟不同。"), r("s_experiment", "lines335–355", "reward收集與policy optimizer更新分开。")], ["a_inspection"], "此章無跨回合內部狀態的小策略；不是說所有有記憶系統的回應永遠固定。"),
        claim("c2", "concept", "policy 是依問題情境分配回答卡選擇機率的規則。", "第2段 policy 定義與整篇預寫回答限定", [r("s_ppo", "§2.1 equations(1)–(2)", "pi_theta(a|s) 是stochastic policy。"), r("s_bandit", "§2.1 steps1–2", "情境特徵決定選擇一個action。"), r("s_module", "FiniteResponsePolicy lines53–61", "四特徵對四張卡的logits。")], ["a_frozen"], "action 是預寫整篇回答的編號；不會由此生成文字token或學算術。"),
        claim("c3", "concept", "本章基準值估計更新前策略在情境下的預期分數，單步 advantage 估計為 sampled reward−old value。", "第2段 baseline／advantage及第7段 contextual bandit", [r("s_ppo", "§5 equations(10)–(12), terminal V(next)=0", "只剩一個reward且已終止時多步式縮成R−V_old(s)。"), r("s_math", "terminal reduction, baseline cancellation and conditional MSE mean", "分清樣本估計與真A=E[R|s,a]−V(s)，基準不依所選card。"), r("s_experiment", "lines339–345, 353–355", "收樣時old_values固定；critic用context估計RM分數。")], ["a_derivation", "a_audit"], "是單一步驟、終止後future value=0；基準可能不準，不是多步任務的普遍R−V公式。此章critic不把另加的KL當target。"),
        claim("c4", "numeric", "人工 rewards=[1,0] 與 old_values=[0.4,0.4] 得到 [0.6,−0.4]；reward=baseline=1 時沒有正優勢。", "第1段假設与唯一程式後的輸出說明", [r("s_math", "original decimal arithmetic", "1−0.4=0.6；0−0.4=−0.4；1−1=0。"), r("s_execution", "snippet_executions.original, detach_probes[0], reward_equals_baseline_probe", "逐字原碼真CPU float32與打印核對。")], ["a_original", "a_original_code"] + common, "人工兩個數，不是實跑RM的一般分數或品質改善。", {"method": "executed", "expected": "打印 [0.6, -0.4]；理想實數減法 [0.6,-0.4]；1−1=0。", "observed": str(AUDIT["detach_probes"][0]["advantage"]) + "; original stdout完全相同；reward=oldvalue=1 的實際CPU輸出為0。", "tolerance": "round(x,2) 的打印字串完全相等；float32對手算值絕對誤差≤1e-7、rtol=0。", "details": "兩個有效樣本，shape=[2]，dtype=float32，無mask；實際輸入R=[1,0],V=[0.4,0.4]。手算與执行分開記錄於derivation／audit。"}),
        claim("c5", "software", "bandit_advantage 將差值 detach，所以 requires_grad=False；策略 loss 不沿這份差值更新reward或舊基準。", "程式 old_values.requires_grad=True 與最後 False 的説明", [r("s_module", "bandit_advantage lines24–28", "return(rewards-old_values).detach()，不是只靠no_grad context。"), r("s_torch", "torch/_tensor.py lines798–815 Tensor.detach", "detach結果不要求梯度。"), r("s_execution", "detach_probes both baseline settings", "固定advantage的PPO loss有policy logits梯度，而reward/value皆無該路徑梯度。")], ["a_original"] + common, "在本輪固定採樣資料及獨立value網路下隔離；不主張所有共用底座都不會透過別的loss更新。", {"method": "executed", "expected": "advantage.requires_grad=False、grad_fn=None；policy backward 不給reward/value梯度。", "observed": "兩個基準設定都False/None；policy logits有預期梯度，reward/value在policy backward後皆grad=None；另做value MSE才得到value gradient。", "details": "R與V都另設requires_grad=True；兩筆雙候選softmax，new=old ratio=1，以mean分母2的PPO surrogate求導，隔離不是因value本來不可求導。"}),
        claim("c6", "concept", "基準網路用自己的分數預測代價學習，與策略的固定優勢評語是不同梯度來源。", "程式後第1段最後兩句，連13.14", [r("s_ppo", "§5 equation(9) and following value-error definition", "自己的squared value error與policy surrogate分开定义，可共享參數但不交换职责。"), r("s_experiment", "lines326–332, 353–355", "此章另建value network/optimizer，以RM sampled reward為MSE目標。"), r("s_openai", "loss lines339–357", "stop_gradient advantage和另一个value loss。")], ["a_derivation", "a_audit", "a_prerequisites"], "此章用独立小网络；一般PPO可共享参數，且不同实现value target／KL配方可不同。13.14讀到的MSE説明一致。"),
        claim("c7", "concept", "固定 advantage 為正／負時，該所選動作的政策梯度貢獻分別鼓勵提高／降低它的機率。", "第6段「更新的直覺」及C.7連結", [r("s_ppo", "§2.1 equations(1)–(2); §3 eq(7) around ratio=1", "advantage加權log policy；new=old附近有同一局部方向。"), r("s_math", "fixed-advantage two-logit gradient", "推得−A*(indicator−p)；正負號相反。")], ["a_derivation", "a_audit", "a_prerequisites"], "是局部單項梯度直覺；共享參數／batch其他項／KL／clip可影響實際整輪結果。沒有承諾每項單調增加或品質改善。"),
        claim("c8", "concept", "PPO 即 proximal policy optimization，是比較新舊policy機率並控制過大更新的優化方法。", "第6段PPO名称與更新方法說明", [r("s_ppo", "title; §3 equations(6)–(7); §5 Algorithm1", "名稱、probability ratio和clipped surrogate，適用一般policy並非LM架構。"), r("s_openai", "loss lines350–355", "exp(new logprob−old logprob)與clipped policy objective。")], ["a_ppo_text", "a_openai"], "本節只引入思想；clip不是硬性保證所有機率／KL在固定界內，也不是新的語言模型架構。"),
        claim("c9", "concept", "一次依情境選一張卡、拿分就終止的本章任務屬 contextual bandit。", "第7段contextual bandit定义", [r("s_bandit", "§2.1 p.2 steps1–3", "context→one chosen arm→payoff的正式定義。"), r("s_instructgpt", "§3.1 p.9 bandit environment", "prompt/completion後拿RM分並結束episode。"), r("s_experiment", "build_records and rollout lines335–345", "每次只采一card編號并取得同題RM分。")], ["a_bandit_text", "a_audit", "a_records"], "训练有许多独立试次；一题内没有transition或后续回报。不是把多次更新误叫一条多步episode。"),
        claim("c10", "concept", "逐token生成會改變后續前文，需另外處理多步回饋與位置優勢；不能把兩個數的減法當完整LLM PPO。", "第7段后半", [r("s_ppo", "§5 equations(10)–(12)", "非終止步包含後續reward／bootstrap value及GAE加總。"), r("s_openai", "loss lines319–339", "responses长度為位置軸，reverse GAE，最後一步才nextvalue=0。"), r("s_instructgpt", "§3.1 p.9, Appendix C.4 p.23", "bandit episode层面的回答仍有per-token penalty和GAE訓練。")], ["a_openai", "a_inspection"], "支持本節對逐位置自回歸優化的限制；沒有宣稱所有sequence-level估計器都一定用GAE或value critic。"),
        claim("c11", "concept", "Reward model 的scalar不一定是0/1或非負，RLHF可另加偏離固定reference的代價，其定義與尺度需要明確。", "第8段reward尺度與參考代價", [r("s_dpo", "§3 equations(1)–(3) and p.4 shaped reward", "比較只依分數差，reward可任意平移；標準配方含−beta log policy/reference。"), r("s_instructgpt", "§3.1 pp.8–9 normalization and eq(2)", "mean-zero RM與per-token KL penalty。"), r("s_openai", "compute_rewards lines149–156", "KL項與終止score组成reward。")], ["a_dpo_text", "a_inspection"], "『可能』表示不同配方。此章明確把exact reference KL另加policy loss，critic只估normalized RM score，不能混称所有reward目标一样。"),
        claim("c12", "software", "本節 [1,0] 為人工例子；本章實跑的固定RM分經明確歸一化，不要求每次回0/1。", "第8段最后一句及第7段本章任務範圍", [r("s_experiment", "_normalized_scores lines160–162; rollout lines337–345; result lines490–505", "减去同題四卡平均、除以仅train所得固定尺度；选中的RM分给advantage，不是correctness binary。"), r("s_execution", "all split_audit rows; first_rollout_reward_range", "先前亲读原checkpoint後，當次僅用其portable JSON重建真推論所有165題，并逐 draw 檢查已保存的64個reward／old_value／fixed advantage。")], ["a_audit", "a_frozen", "a_training_report", "a_records"], "仅软件行为和固定seed证据；没重训，没有质量／速度／泛化比较。记录旧计时scope不当新测量。", {"method": "executed", "expected": "学到的RM输出连续归一化分，三次reuse均用reward−固定oldvalue；不是人工[1,0]。", "observed": "全部165题逐栏最大误差0；first rollout 64分全部非0/1，范围[-1.1883575916290283,1.3835170269012451]；192个reuse条目 advantage 减法精确重现。", "details": "原code SHA及五checkpoint原binary SHA／metadata／21個tensor SHA核對；SFT/PPO/DPO/RM／value之portable JSON重建輸出與原binary參考全165題精確相同；完整evaluation与旧逐题资料逐栏比较。归一化scale=5.6094536781311035，effective_tokens=0。历史 PPO 秒数只涵盖旧训练loop，未当GPU或重复速度实证。", "denominators": {"seed": 42, "train_contexts": 132, "validation_contexts": 15, "test_contexts": 18, "candidate_count": 4, "effective_tokens": 0, "first_rollout_actions": 64, "reuse_epochs": 3, "historical_rollouts": 120, "historical_policy_updates": 360, "historical_value_updates": 360, "historical_sampled_actions": 7680, "historical_reused_draws": 23040}}),
        claim("c13", "numeric", "只把兩個old_values改成0.9，advantage變[0.1,−0.9]，reward不變。", "末段練習", [r("s_math", "exercise decimal arithmetic", "1−0.9=0.1；0−0.9=−0.9。"), r("s_execution", "snippet_executions.exercise and detach_probes[1]", "原程式只換舊基準後，CPU輸出如預期。")], ["a_exercise", "a_exercise_code"] + common, "人工基準变化显示相对期望，与答题评分本身变化不同；不代表已经更新网络。", {"method": "executed", "expected": "打印[0.1,-0.9]，最后仍False。", "observed": str(AUDIT["detach_probes"][1]["advantage"]) + "; round(x,2)打印完全吻合。", "tolerance": "打印字串精确相等；float32对理想十进制减法绝对误差≤1e-7、rtol=0。", "details": "原始rewards=[1,0]不变，两个old_values只从0.4换为0.9；dtype=float32,shape=[2]，两笔有效动作、无mask。"}),
        claim("c14", "concept", "advantage正负不能直接当答案对错。", "最后练习要求及第1段同样reward的相对基准", [r("s_ppo", "§2.1 eq(1), §5 eq(12)", "估计相对回报，用作更新方向，不是truth标签。"), r("s_dpo", "§3 equations(1)–(2)", "RM仅比较差距，没有标定真值正误的绝对零点。"), r("s_math", "advantage-sign counterexamples", "不同正确性可同为zero；误差基准／任意RM分还可使相同reward有不同正负。")], ["a_derivation"], "对不同评分目标与近似旧基准成立；若特定二元reward及准确baseline有更强关系，仍不能把一般advantage当正确率或标签。"),
    ]
    for item in claims:
        if item["id"] in {"c6", "c12"}:
            item["artifact_ids"] += ["a_portable_export", "a_portable_verify", "a_native_outputs", "a_portable_outputs"]
    report = {
        "schema_version": 1, "review_stage": "technical", "lesson_id": "13.11",
        "source": "course/chapters/13.md#13.11",
        "reviewer_task": "/root/integration_technical_coordinator/fact_v2_13_11",
        "reviewer_context": "fresh", "source_sha256": AUDIT["section_sha256"]["13.11"],
        "figure_sha256": {}, "verdict": "pass", "claims": claims, "sources": sources,
        "artifacts": ARTIFACTS, "issues": [],
        "prerequisite_sha256": {key: value for key, value in AUDIT["section_sha256"].items() if key != "13.11"},
        "recheck_history": [{"reason": "可發布證據closure；正文/資料未改，親讀原checkpoint後本人真核JSON等價，沒有重訓。",
                             "previous_report_path": history[0]["archived_path"],
                             "previous_report_sha256": history[0]["sha256"],
                             "reviewer_task": "/root/integration_technical_coordinator/fact_v2_13_11",
                             "portable_verification_artifact_id": "a_portable_verify", "verdict": "pass"}],
        "checks": {
            "factual_accuracy": {"status": "pass", "details": "14項逐一核對：固定策略更新、policy、單步終止advantage、圖隔離、自己的critic MSE、正負局部梯度、PPO名稱與ratio、contextual bandit、多步token限制、reward尺度及正誤分離，與原始公式／官方原碼及本章配方相符。", "claim_ids": [c["id"] for c in claims]},
            "numeric_verification": {"status": "pass", "details": "手算0.6/−0.4及0.1/−0.9并真CPU执行逐字原码/練習，打印精确匹配；float32误差≤1e-7。另做梯度隔离及全部165题冻结推論、64动作×3reuse优势核對。本节无品质或速度比较，不以旧run计时支持外推。", "claim_ids": ["c4", "c5", "c12", "c13"]},
            "figure_consistency": {"status": "not_applicable", "details": "13.11完整section無SVG或其他圖片引用。讀到前置的圖不被當成本節的證據，不需要用图hash假装视觉核對。", "claim_ids": []},
            "source_verification": {"status": "pass", "details": "实际原PDF核到PPOv2、InstructGPTv1、DPOv3及Li contextual-banditv2，保存新提取全文；OpenAI与PyTorch官方固定commit原码直接下载并亲读相关行。未采用來源庫摘要／旧review结论，版本、原始URL、定位、原文件SHA和执行证据完整；原PDF/records逐byte複本及五checkpoint等價JSON使當前直接來源可收錄Git，原檔與本人舊證據保留。", "claim_ids": [c["id"] for c in claims]},
            "limitations": {"status": "pass", "details": "明确限单步四卡bandit、人工数字、固定旧基准、近似baseline、局部单项梯度及本章独立critic配方。归一化RM与另加exact KL不混称通用token PPO，没把单seed冻结推论当语言生成、学习算术、质量普遍保证或GPU加速。", "claim_ids": ["c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14"]},
        },
    }
    path = ROOT / "docs/technical-reviews/13.11.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)} with {len(claims)} claims, {len(sources)} sources, {len(ARTIFACTS)} artifacts")


if __name__ == "__main__":
    main()
