"""Write only this reviewer's new report; never read the old report."""
from pathlib import Path
import hashlib
import importlib.util
import json

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_19_4"
PIN = "1df335318bda03fd771807f66976953231d5a00b"
TORCH = "5c4886908584029761b579af026dcfb627c84070"
BASE = ART.relative_to(ROOT).as_posix()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):
    return path.relative_to(ROOT).as_posix()
cpu = json.loads((ART / "cpu-results.json").read_bytes())
env = {key: value for key, value in cpu["environment"].items() if isinstance(value, str)}
artifacts = []
def artifact(identifier, path, kind, description, **extra):
    entry = {"id": identifier, "path": relative(ART / path), "sha256": sha(ART / path), "kind": kind, "description": description}
    entry.update(extra)
    artifacts.append(entry)
    return identifier
artifact("section_raw", "inputs/section.md", "source_snapshot", "親讀本節原始UTF-8 bytes，包含標題且不正規化換行。")
artifact("frozen_chapter", "inputs/frozen-19.md", "source_snapshot", "首次讀取時的全章frozen input；不是目前整章版本宣稱。")
artifact("context_intro", "inputs/chapter-intro.md", "source_snapshot", "必要上下文：新成品尚未訓練／驗收，舊合成模型單獨界定。")
artifact("context_19_1", "inputs/context-19.1.md", "source_snapshot", "實際親讀19.1，僅用主線計畫與舊合成任務的區分。")
artifact("freeze", "freeze-manifest.json", "source_snapshot", "原件與永久副本完整SHA一致及首次raw來源指紋。")
artifact("inspection", "inspection.md", "derivation", "本人實際讀scope、具名JSON pointers、來源定位、數學分母、範圍與工具預期修正。")
artifact("commands", "commands.json", "source_snapshot", "實際命令與exit status，保留算例首次失敗和修正後成功；沒有宣稱執行訓練。")
artifact("report_builder", "write_report.py", "code", "只寫本人新報告、不讀舊報告；重核當前section bytes及真實canonical reviewer_task。")
artifact("fence", "fence-1.py", "code", "原教材Python fence，未修補、逐byte保存。")
artifact("cpu_code", "verify_cpu.py", "code", "原fence、短CPU變體及336筆既有原始記錄的核對程式；沒有optimizer或完整模型評測。")
artifact("cpu_run", "cpu-stdout.txt", "execution", "原fence逐字符輸出、mask／byte變體與既有記錄的重新彙總。", command=f".venv/bin/python {BASE}/verify_cpu.py > {BASE}/cpu-stdout.txt 2> {BASE}/cpu-stderr.txt", result="exit_code=0; ALL CHECKS PASSED; 43/10有效目標；四站0/42/75/71於同一84題；336原紀錄逐筆重算。", environment=env)
artifact("cpu_stderr", "cpu-stderr.txt", "source_snapshot", "成功CPU查核stderr，空檔。")
artifact("cpu_environment", "cpu-environment.json", "source_snapshot", "真實Python／PyTorch CPU版本、裝置、實際匯入程式SHA。")
artifact("cpu_results", "cpu-results.json", "derivation", "可機讀實際CPU查核結果、答案分母、label變體與三段完整父SHA。")
artifact("objective_code", "objective_accounting.py", "code", "短CPU計算：偏好序列目標與DPO／路由auxiliary的數學；不訓練。")
artifact("objective_run", "objective-stdout.txt", "execution", "chosen/rejected/replay=10/28/10；DPO等margin得log2；路由surrogate=1/4。", command=f".venv/bin/python {BASE}/objective_accounting.py > {BASE}/objective-stdout.txt 2> {BASE}/objective-stderr.txt", result="exit_code=0; all numeric assertions passed after correcting reviewer-side byte expectation.", environment=env)
artifact("objective_stderr", "objective-stderr.txt", "source_snapshot", "成功偏好數學查核stderr，空檔。")
artifact("first_objective_code", "first-attempt-objective_accounting.py", "code", "保留本輪第一版算例錯估附加字串byte數的真實原碼。")
artifact("first_objective_error", "first-attempt-objective-stderr.txt", "source_snapshot", "保留本輪錯誤預期導致的assert失敗，inspection說明已修正，非教材錯誤。")
artifact("fetch_code", "fetch_sources.py", "code", "對正文指定immutable版本及官方PyTorch來源的read-onlyHTTPS取得程式。")
artifact("fetch_run", "fetch-stdout.txt", "execution", "十三個HTTPS原來源成功取得，SHA逐byte對照一致。", command=f".venv/bin/python {BASE}/fetch_sources.py > {BASE}/fetch-stdout.txt 2> {BASE}/fetch-stderr.txt", result="exit_code=0; 13 HTTP 200 responses; all local/cached-original byte comparisons true.", environment={"python": env["python"], "device": "not applicable: read-only HTTPS source retrieval", "TLS": "Python urllib default certificate verification"})
artifact("fetch_results", "fetch-results.json", "source_snapshot", "實際原站URL、access日期、HTTP200、raw bytes／SHA和與本地對照。")
artifact("authority_provenance", "authority-provenance.json", "source_snapshot", "原論文first-page版本與原始source取得／副本完整SHA一致。")
for label in ("instructgpt", "switch", "dpo"):
    artifact(label + "_pdf", f"authority/{label}.pdf", "source_snapshot", "本人親讀原論文，非舊審閱正文；first-page版本與method定位見inspection。")
    artifact(label + "_text", f"authority/{label}.txt", "source_snapshot", "由本輪原PDF實際pdftotext -layout所得可定位原文。")
for label in ("torch-functional.py", "torch-autograd.md", "torch-random.py", "torch-tensor-docs.py", "torch-docs.py"):
    artifact(label.replace(".", "_").replace("-", "_"), f"authority/{label}", "source_snapshot", "與installed torch git revision一致的官方原始source／API documentation；本人親讀相關定位。")

sources = []
def external(identifier, kind, title, url, version, inspection, snapshot, authority):
    sources.append({"id": identifier, "kind": kind, "title": title, "url": url, "version": version, "accessed_on": "2026-10-05", "authority_reason": authority, "verified": True, "checked_original": True, "inspection_note": inspection, "snapshot_artifact_ids": snapshot})
external("instructgpt", "paper", "Training language models to follow instructions with human feedback", "https://arxiv.org/pdf/2203.02155v1", "arXiv:2203.02155v1, 2022-03-04", "親讀first-page OpenAI作者與arXiv標記；§3.1步驟1、§3.3 SFT和§5.5 next-word objective。只支持從pretrained model接示範訓練；不拿論文結果代替本課實測。", ["instructgpt_pdf", "instructgpt_text"], "此方法原研究論文，作者提出InstructGPT工作流程。")
external("switch", "paper", "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", "https://arxiv.org/pdf/2101.03961v3", "arXiv:2101.03961v3, 2022-06-16", "親讀Google作者／版本；§2 A Differentiable Load Balancing Loss式(4)–(6)、α=10^-2；repo top-2只使用同類目的，不聲稱等同top-1或保證均衡。", ["switch_pdf", "switch_text"], "提出Switch路由及load balancing的原研究論文。")
external("dpo", "paper", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "https://arxiv.org/pdf/2305.18290v3", "arXiv:2305.18290v3, 2024-07-29", "親讀Stanford作者／版本；§4式(7)、DPO outline和reference初始化。repo另加CE replay，不是純DPO。", ["dpo_pdf", "dpo_text"], "DPO原研究論文，提供chosen/rejected與reference policy的目標。")
for identifier, rel, title, locator, snapshot in [
    ("torch_ce", "torch/nn/functional.py", "PyTorch cross_entropy official source", "cross_entropy lines3478–3533: class-index targets, ignore_index, reduction=sum and shape", "torch_functional_py"),
    ("torch_freeze", "docs/source/notes/autograd.md", "PyTorch autograd: freezing parameters", "Setting requires_grad lines194–230: .requires_grad_(False) and nn.Module.requires_grad_", "torch_autograd_md"),
    ("torch_seed", "torch/random.py", "PyTorch manual_seed official source", "manual_seed lines49–76: sets seed for all devices", "torch_random_py"),
    ("torch_scalars", "torch/_torch_docs.py", "PyTorch sum and isfinite API source", "isfinite lines5744–5776; sum lines11263–11291", "torch_docs_py"),
    ("torch_tensor_api", "torch/_tensor_docs.py", "PyTorch Tensor APIs used by the fence", "Tensor.isfinite 2656–2672; Tensor.item 2784–2805; requires_grad_ 4121–4147; Tensor.sum 5023–5037", "torch_tensor_docs_py"),
]:
    external(identifier, "official_source", title, f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/{rel}", f"PyTorch2.14.1+cpu; immutable upstream commit {TORCH}", "親讀" + locator + "；installed torch.version.git_version與此commit相同，副本完整SHA核對。", [snapshot], "PyTorch維護者官方原始程式和API docstrings。")

for identifier, rel, inspected in [
    ("capstone", "tiny_perceptron/capstone.py", "AST先定位，親讀1–121、124–375、428–605、608–694；資料、編碼、batch、DPO、EOS、工具、loader與export的原方法。"),
    ("data", "tiny_perceptron/data.py", "親讀1–86；IGNORE=-100，ByteTokenizer每UTF-8 byte一token，特殊token與shift。"),
    ("model", "tiny_perceptron/model.py", "親讀1–105；Dense/MoE、logits[B,T,V]，loss_sum flatten後CE sum再除有效label格數。"),
    ("attention", "tiny_perceptron/attention.py", "親讀10–18、31–74；valid決定可見key，不用loss label mask刪上下文。"),
    ("modern", "tiny_perceptron/modern.py", "親讀35–84；Dense沒有expert分派，MoE top-2與N*sum(load*importance)auxiliary。"),
    ("alignment", "tiny_perceptron/alignment.py", "親讀29–42；sequence log probability是回答有效label格求和，DPO相對margin。"),
    ("stage_runner", "scripts/course_experiments/capstone.py", "AST先定位，親讀1–44、70–263、280–309；父檔與資料guard、CE/DPO+CE replay、optimizer更新、effective_tokens只累加replay labels、validation gate；沒有執行訓練。"),
]:
    a = artifact("code_" + identifier, "original/" + rel, "code", "本人親讀／使用的原碼永久副本，與實際原件SHA相同。")
    sources.append({"id": identifier, "kind": "repository_code", "title": rel, "path": rel, "sha256": sha(ROOT / rel), "version": "本輪原件完整SHA；capstone與stage_runner另以正文pin之HTTPS原件和/results/code_sha256核對。", "verified": True, "inspection_note": inspected, "snapshot_artifact_id": a})

stage_results = []
raw_results = []
configs = []
for stage, name in (("pretrain", "pretrain"), ("sft", "sft"), ("joint", "joint"), ("dpo", "preference")):
    rel = f"docs/course-experiments/results/capstone_{name}.json"
    d = json.loads((ART / "original" / rel).read_bytes())
    configs.append(d["results"]["parameters"]["config"])
    a = artifact("result_" + stage, "original/" + rel, "source_snapshot", "指定版本完整原始實報；只讀具名measurement/provenance pointers，保留原檔SHA。")
    external("result_" + stage, "official_source", f"本repo {stage} 原始實報", f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{PIN}/{rel}", f"publication commit {PIN}; run revision {d['revision']}", "root/results必要key/type之後只讀measurement與provenance pointers，詳inspection；原站HTTP200、raw SHA與本地完全一致，親讀不是沿用摘要。", [a], "本repo實驗作者在指定immutable commit發布的第一手執行紀錄。")
    stage_results.append("result_" + stage)
    rel = f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
    a = artifact("raw_" + stage, "original/" + rel, "source_snapshot", "84份原始generated token/action/EOS/runtime/final trace，不是review摘要；與實報/artifacts SHA相同。")
    external("raw_" + stage, "official_source", f"本repo {stage} 原始validation記錄", f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{PIN}/{rel}", f"publication commit {PIN}; raw artifact SHA matched report receipt", "先root/record/trace keys/types，再讀/records內原始樣本／答案／trace和/count,/by_task,/protocol；336記錄實際逐筆重算，沒有新模型reevaluation。", [a], "指定實驗的第一手生成記錄，完整SHA被原實報artifact receipt記錄。")
    raw_results.append("raw_" + stage)
assert all(config == configs[0] for config in configs)
assert configs[0]["experts"] == 4 and configs[0]["top_k"] == 2
sources.append({"id": "cpu", "kind": "execution", "title": "原fence+短CPU變體+336筆既有记录重聚合", "artifact_id": "cpu_run", "verified": True})
sources.append({"id": "objective", "kind": "execution", "title": "DPO額外評分計算與路由balance數學", "artifact_id": "objective_run", "verified": True})
sources.append({"id": "target_derivation", "kind": "derivation", "title": "Byte目標數與masked CE分母", "verified": True, "details": "本例user32 UTF-8 bytes + newline1 + answerDIRECT:29的9 bytes + EOS1=43 next-token targets；SFT9+EOS1=10。模型logits[B,T,V]與labels[B,T]同shift一次；有效格數而非中文字數作分母。兩個有效logit列手算CE為0.5032044053，容忍絕對誤差1e-7。"})

def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}
def verification(expected, observed, details, **extra):
    v = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    v.update(extra)
    return v
claims = [
    {"id": "c_plan", "kind": "concept", "statement": "同一核心接續示範與感知入口；凍結僅表示本階段不更新，仍須交代原權重；偏好／壓縮是按需分支。", "location": "course/chapters/19.md#19.4 第164–166、191行", "scope": "這是新成品的設計與provenance要求，非已驗收主線能力；導言／19.1明示新成品未訓練驗收。可先各自訓練感知元件，不等於每站重設共同核心。", "status": "verified", "evidence": [evidence("instructgpt", "§3.1 Step1；§3.3 SFT", "從pretrained language model接續demonstration fine-tuning的成熟流程。"), evidence("torch_freeze", "Setting requires_grad 194–230", "以requires_grad_(False)排除該參數的梯度與更新，非重新初始化權重。"), evidence("dpo", "§4 DPO outline/reference initialization", "偏好優化可從既有policy/reference接續，不提供本課新成品已成功證據。"), evidence("capstone", "load_capstone644–652；frozen_reference371–375", "原loader沿父模型config及state_dict載入，reference為原model複製後freeze。")], "artifact_ids": ["context_intro", "context_19_1", "inspection"]},
    {"id": "c_dense_fence", "kind": "software", "statement": "原fence選到train的style29例，建立局部Dense，印出兩種loss有限；沒有更新權重，也不是共同MoE權重。", "location": "course/chapters/19.md#19.4 第168、170–189行", "scope": "只查局部計分對齊；沒有checkpoint載入或optimizer更新，不支持任何已學會能力。普通torch API合組核對。", "status": "verified", "evidence": [evidence("capstone", "default_config34–48；build_dataset139–282；prepare_batch321–338", "dense=True令experts=0，固定seed資料首個style為29。"), evidence("model", "Block31–41；masked_loss92–105", "Dense選支與以有效token平均的loss。"), evidence("torch_seed", "manual_seed49–76", "torch.manual_seed(42)控制亂數。"), evidence("torch_scalars", "isfinite5744–5776；sum11263–11291", "sum數有效格；isfinite回布林有限性，並未判定內容正確。"), evidence("torch_tensor_api", "Tensor.sum5023–5037；Tensor.isfinite2656–2672", "原fence使用的tensor API。"), evidence("cpu", "EXACT ORIGINAL FENCE；parameters_unchanged", "原fence逐字符輸出与小變體中參數未改，僅forward與額外gradient示範。")], "artifact_ids": ["fence", "cpu_code", "cpu_run", "cpu_results"], "verification": verification("問題照抄數字29；DIRECT:29；Dense experts=0；兩loss finite；無更新。", "逐字输出完全吻合，Dense experts=0；兩loss finite=True；clone對照參數不變。", "沒有optimizer／step；CPU額外backward只驗證mask，沒有訓練。")},
    {"id": "c_targets", "kind": "numeric", "statement": "預訓練此例有效目標43；SFT僅答案与EOS，有效目標10。", "location": "course/chapters/19.md#19.4 第189行", "scope": "ByteTokenizer此一例的有效label格，不是中文字數或普遍固定長度。目標都shift一次，pretrain也包含EOS。", "status": "verified", "evidence": [evidence("data", "ByteTokenizer14–28；IGNORE10", "UTF-8每byte一token。"), evidence("capstone", "encode_record308–318", "pretrain對user+newline+answer+EOS計分；SFT prefix為IGNORE，suffix答案+EOS。"), evidence("target_derivation", "32+1+9+1=43；9+1=10", "以byte及EOS手算分母。"), evidence("cpu", "label_variants；manual_cross_entropy", "原43/10，問題加甲46/10，答案加9為44/11，manual CE分母2。")], "artifact_ids": ["cpu_run", "cpu_results", "inspection"], "verification": verification("精確43/10；長問題46/10；長答案44/11；两格CE0.5032044053。", "全部實際assert通過；手算与masked_loss在1e-7內一致。", "logits[B,T,V]flatten成有效格×V，labels[B,T]每有效格一目標；CE sum除有效格數。", tolerance="有效目標整數精確相等；手算CE絕對誤差≤1e-7。")},
    {"id": "c_ignore_scope", "kind": "concept", "statement": "IGNORE排除該格loss，不從上下文刪去問題與前文；两種不同目標的loss大小不能直接判教法好壞。", "location": "course/chapters/19.md#19.4 第189行", "scope": "ignore_index控制目标loss；valid是獨立attention可見mask。不同目標／角色上下文的單次未訓練loss不構成公平教法比較。", "status": "verified", "evidence": [evidence("torch_ce", "cross_entropy3478–3533", "ignore_index不貢獻loss／direct input gradient，sum和mean的目標範圍。"), evidence("capstone", "prepare_batch321–338；encode_record314–318", "ids及valid保留prefix，只label填IGNORE。"), evidence("attention", "attention_mask10–18；CausalAttention.forward47–74", "attention從valid決定可見key而非labels。"), evidence("cpu", "ignored_direct_logit_gradients_zero；prompt_change_answer_logit_max_abs_delta", "IGNORE格直接logit梯度為0，prefix全部valid；同labels下問題29→28改變回答logits約0.01707。"), evidence("target_derivation", "兩種loss的分母／目標集合43對10", "比較兩值同時改變上下文與評分目標，不能隔離教法效果。")], "artifact_ids": ["cpu_run", "cpu_results", "inspection"]},
    {"id": "c_lineage", "kind": "empirical", "statement": "歷史pretrain→SFT→joint→DPO三條父鏈的完整64字元SHA相等，首站parent為null；同架構和資料版本，來源連續與能力證據分開。", "location": "course/chapters/19.md#19.4 第164、191、196、211行", "scope": "核對指定版本runner記錄的原始provenance與原載入方法；沒有重新下載或讀取整份權重，也不把matching SHA當能力提升。", "status": "verified", "evidence": [evidence("stage_runner", "train_stage99–116、153–169；_run_context280–293", "按predecessor載入、禁止後站隨機開局、計原檔SHA。"), evidence("capstone", "load_capstone644–652；save_capstone617–626", "tokenizer/data_version guards，用原config和strict state_dict載入。"), *[evidence(s, "/results/parent_checkpoint_sha256；/results/inference_export/sha256；/results/data_manifest/sha256；/results/parameters/config", "逐站完整父SHA、資料指紋和同一MoE config。") for s in stage_results], evidence("cpu", "published_records_recalculated/*/{parent_checkpoint_sha256,inference_export_sha256}", "三完整SHA連結精確相等，第一站null；code和manifest SHA一致。")], "artifact_ids": ["cpu_run", "cpu_results", "fetch_results", *stage_results], "verification": verification("null首父；pretrain export=SFT parent，SFT export=joint parent，joint export=DPO parent。", "三條完整64字元SHA等值assert通過；四config精確相同，experts4/top_k2。", "SFT→joint的值c35427bce4f5a8cd11ac9b702cb8390fafe8248f577e246137246f0d3e2aaba7；不是只對檔名／前綴。", denominators={"stages": 4, "parent_links": 3, "sha256_characters_each": 64})},
    {"id": "c_scores", "kind": "empirical", "statement": "固定資料capstone-small-world-v2、seed42、L4；四站完成300/1400/600/100更新，同84題正確0/42/75/71。SFT文字42/42、模態0/42；joint文字42/42、模態33/42，九失敗全image_shape。", "location": "course/chapters/19.md#19.4 第196–203、209行", "scope": "只此單seed合成RGB／純音／模板驗證套件；GPU裝置來自原實報，本輪CPU重算既有紀錄，不是重新GPU訓練或模型score重評。", "status": "verified", "evidence": [*[evidence(s, "/device,/gpu,/seed；/results/data_version,/steps,/new_steps,/requested_steps,/schedule_completed,/validation_summary", "各站裝置、完成步數與原總／分項結果。") for s in stage_results], *[evidence(s, "/records/*；/count；/by_task", "同一84題ID順序，原始內容／EOS／工具／回答逐筆重算，不照抄summary。") for s in raw_results], evidence("cpu", "RECALCULATED PUBLISHED RECORDS；record_count_recalculated=336", "原答案ID與自行重建資料對照，全總分和各task桶精確一致。")], "artifact_ids": ["cpu_run", "cpu_results", *raw_results, *stage_results], "verification": verification("更新300/1400/600/100；84題score0/42/75/71；joint模態33/42，9shape錯。", "336原紀錄逐筆assert；四總分、分項、資料SHA／ID完全符合；DPO額外4個joint失敗。", "分母84=42text+42modality，每站同ID順序；post-stage觀察不能作DPO必然提升的結論。", denominators={"validation_per_stage": 84, "stages": 4, "raw_records": 336, "text_questions": 42, "modality_questions": 42, "joint_image_shape_failures": 9, "seed": 42, "completed_updates_by_stage": {"pretrain": 300, "sft": 1400, "joint": 600, "dpo": 100}})},
    {"id": "c_exact_scoring", "kind": "software", "statement": "整題正確需完整動作字串／內容／參數及生成EOS；工具題須實際runtime結果與第二次正確回答。0/84不表示所有文字續寫能力為零。", "location": "course/chapters/19.md#19.4 第207行", "scope": "這個原exact protocol：invalid special立即停且EOS=false，截斷不通過；0/84是這84題未達該對話判準，不是通用語言能力測量。", "status": "verified", "evidence": [evidence("capstone", "generate_traces428–496；evaluate_rows499–547；parse_action550–561；calculator_runtime564–574", "完整raw+EOS判action；工具結果再餵同一core，最終答案還要正確。"), *[evidence(s, "/records/*/{action_trace,expected_action,runtime,final_trace,answer,expected_final,action_correct,end_to_end_correct}；/protocol", "檢查原始token/EOS與每題精確判準。") for s in raw_results], evidence("cpu", "336筆逐筆重算與全部EOS=true樣本的最後token／special檢查", "判準獨立重算與record flags、summaries全部一致。"), evidence("instructgpt", "§3.1 Step1；§5.5 Broader impacts（text1158–1170：next-word objective）", "next-token訓練目標與指定prompt回答任務不同，本文類比不聲稱controlled因果證明。")], "artifact_ids": ["cpu_run", "cpu_code", *raw_results], "verification": verification("依完整raw+EOS、allowlisted runtime與第二回答重算旗標；與原records一致。", "336紀錄全通過重算；全部EOS=true trace最後為EOS且前方只有byte token；無被decode丟棄的特殊token誤過關。", "根據保存的生成紀錄核對評分，没有呼叫訓練後模型重跑題目。")},
    {"id": "c_balance_and_budget", "kind": "concept", "statement": "前三站CE之外加0.01路由平衡項；各站目標不同，DPO的CE replay有效目標數不含chosen/rejected政策及reference評分，因此不是等成本預算。", "location": "course/chapters/19.md#19.4 第205行", "scope": "原方法及計算口徑；平衡項鼓勵分派均衡，不保證結果或速度。DPO是此repo的合成偏好加示範replay分支，不推成人類偏好或等FLOPs比較。", "status": "verified", "evidence": [evidence("switch", "§2式(4)–(6)，alpha=10^-2", "輔助load/importance項鼓勵專家均衡的目的，非repo實測勝敗。"), evidence("modern", "MoEFFN56–84", "top-2分派load與router probability importance的auxiliary。"), evidence("stage_runner", "train_stage198–218", "前三站ce+0.01aux；DPO preference+0.2ce+0.01aux；計数只加labels!=IGNORE。"), evidence("capstone", "preference_loss354–368", "policy/reference各做chosen/rejected兩次評分，另於CE batch之外。"), evidence("dpo", "§4式(7)", "chosen/rejected與reference log probabilities是DPO目標的不同評分計算。"), evidence("objective", "preference_scored_target_lengths；four_expert_balance_surrogate", "10/28偏好回答labels另於10 replay targets之外；uniform1集中4，係數0.01。")], "artifact_ids": ["objective_code", "objective_run", "inspection"]},
    {"id": "c_holdout_scope", "kind": "empirical", "statement": "四站只用逐站validation回饋，當時90題test尚未評估；最後更新檔案不必然最好，DPO此輪71低於joint75。", "location": "course/chapters/19.md#19.4 第209、213行", "scope": "此已發布固定版本，非所有DPO或成品的排名結論。新主線成品尚待後續工程／訓練／驗收；未追驗19.12其他成績。", "status": "verified", "evidence": [*[evidence(s, "/results/test_evaluated；/results/data_manifest/counts/test；/results/validation_summary/end_to_end_correct", "四站test_evaluated=false，預定test90；joint75與DPO71。") for s in stage_results], evidence("stage_runner", "train_stage241–244（validation gate）", "此流程只生成validation；本節未執行test recipe。"), evidence("cpu", "counts；published_records_recalculated", "重新建立train552/validation84/test90，四站test旗標assert false與score一致。")], "artifact_ids": ["cpu_run", "cpu_results", "context_intro", *stage_results], "verification": verification("四站test_evaluated=false；test90；joint75>DPO71。", "全部精確一致，數值差4題；末檔優劣是本validation範圍內觀察。", "本輪重算原validation紀錄，不重啟test、不下載資料或模型。", denominators={"training_rows": 552, "validation_rows": 84, "reserved_test_rows": 90, "test_runs_at_these_stages": 0, "validation_score_difference": 4})},
]
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
current, _, _ = facts.original_section(ROOT / "course/chapters/19.md", "19.4")
assert current == (ART / "inputs/section.md").read_bytes()
report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "19.4", "source": "course/chapters/19.md#19.4", "source_sha256": hashlib.sha256(current).hexdigest(), "reviewer_task": TASK, "reviewer_context": "fresh", "verdict": "pass", "figure_sha256": {}, "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [], "checks": {
    "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "逐項核對原fence與成熟方法／原runner／既有生成紀錄；43/10、三父SHA、0/42/75/71全部相符；沒有需要正文修正的實質錯誤。"},
    "numeric_verification": {"status": "pass", "claim_ids": ["c_targets", "c_lineage", "c_scores", "c_holdout_scope"], "details": "UTF-8 bytes與EOS分母精確重算；四站336 raw samples，84=42+42，同ID／SHA；完成步數與reserved test90核實；chosen10/rejected28/replay10，非等成本。"},
    "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "19.4沒有圖／SVG引用。本節計分及階段證據已由原fence和表格可查，不依賴想像圖片素材／空间位置；未宣稱做本節不存在的圖render。必要context19.1只用文字範圍聲明。"},
    "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "親讀三原論文指定arXiv版本與matching installed PyTorch官方source；正文pin的四實報／四raw validation／兩原程式全HTTP200與本地完整SHA相同；永久snapshot、JSON pointers與inspection完整保存。"},
    "limitations": {"status": "pass", "claim_ids": ["c_plan", "c_dense_fence", "c_ignore_scope", "c_scores", "c_exact_scoring", "c_balance_and_budget", "c_holdout_scope"], "details": "明確區分局部未更新Dense、單seed合成歷史實測、mature method與新成品計畫。0/84不泛化為全語言能力；matching父SHA不當能力證據；不同目標／額外DPO計算不當等成本比較。未做GPU、完整訓練、模型重評或資料準備。"}},
    "read_scope": {"instruction_sources": ["docs/review-tools/factual-reviewer-instructions.md", "scripts/check_technical_reviews.py", "docs/review-tools/section_facts.py", ".agents/skills/clear-tutorial/SKILL.md", ".agents/skills/clear-tutorial/references/review-protocol.md"], "lesson": "19.4全部原文與原fence", "necessary_context": "首次frozen版本章導言與19.1全原稿；僅用計畫／歷史區分，沒有驗收其他小節。", "full_frozen_input": {"path": relative(ART / "inputs/frozen-19.md"), "sha256": sha(ART / "inputs/frozen-19.md"), "meaning": "exact original whole-chapter bytes at initial read only"}, "detailed_inspection_artifact": "inspection", "old_reports_read": False, "author_result_interpretation_or_correction_summaries_read": False},
    "verification_summary": "短CPU原fence及變體+既有336記錄重新聚合通过；正文及圖未改，無實質未定。"
}
assert report["reviewer_task"] == TASK
(ROOT / "docs/technical-reviews/19.4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"reviewer_task": TASK, "verdict": report["verdict"], "source_sha256": report["source_sha256"], "report_sha256": sha(ROOT / "docs/technical-reviews/19.4.json"), "claims": len(claims), "artifacts": len(artifacts)}, ensure_ascii=False))
