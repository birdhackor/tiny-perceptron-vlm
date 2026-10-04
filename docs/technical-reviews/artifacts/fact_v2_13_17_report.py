"""Persist the independent 13.17 review with identities and current evidence hashes."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
DIR = "docs/technical-reviews/artifacts/"
PREFIX = DIR + "fact_v2_13_17_"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def ref(source_id, locator, supports):
    return {"source_id": source_id, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, evidence, artifacts, scope, verification=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "status": "verified", "evidence": evidence, "artifact_ids": artifacts, "scope": scope}
    if verification is not None:
        item["verification"] = verification
    return item


def verification(expected, observed, details, denominators=None, tolerance=None):
    value = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if denominators is not None:
        value["denominators"] = denominators
    if tolerance is not None:
        value["tolerance"] = tolerance
    return value


def artifact(identifier, kind, path, description, command=None, result=None, environment=None):
    value = {"id": identifier, "kind": kind, "path": path, "sha256": digest(path), "description": description}
    if kind == "execution":
        value.update(command=command, result=result, environment=environment)
    return value


def repository(identifier, path, note):
    return {"id": identifier, "kind": "repository_code", "title": path, "path": path, "sha256": digest(path), "version": "2026-10-04 current full-file SHA-256, recorded in sha256", "verified": True, "inspection_note": note}


def main():
    bodies = dict(sections(ROOT / "course/chapters/13.md"))
    body = bodies["13.17"]
    audit = json.loads((ROOT / (PREFIX + "audit_result.json")).read_text())
    assert audit["section_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    required = [("course/chapters/07.md", "7.18"), *(('course/chapters/13.md', n) for n in ['13.5', '13.12', '13.14', '13.3', '13.16']), ("course/chapters/19.md", "19.8")]
    prereqs = {path + "#" + lesson: dict(sections(ROOT / path))[lesson] for path, lesson in required}
    current_capstone = prereqs["course/chapters/19.md#19.8"]
    assert audit["linked_capstone_example"]["section_sha256"] == hashlib.sha256(current_capstone.encode()).hexdigest()
    (ROOT / (PREFIX + "section.txt")).write_text(body, encoding="utf-8")
    (ROOT / (PREFIX + "read_prerequisites.txt")).write_text("\n".join(prereqs.values()), encoding="utf-8")
    env = audit["environment"]
    claims = [
        claim("c1", "concept", "DPO以偏好對直接優化policy，不需先訓練獨立reward model，仍依賴偏好標準。", "首段與『沒有評分網路參與』", [ref("s-dpo", "§4 pp.4–5; DPO outline; §5.1", "change of variables avoids explicit standalone RM; offline labeled preferences remain the training input")], ["a-dpo"], "Bradley–Terry偏好建模與可用的固定reference；省略獨立RM不等於標籤必然可靠或無需驗收。"),
        claim("c2", "concept", "DPO使用policy chosen/rejected log差減fixed-reference同一差，再以正beta縮放，最小化−log sigmoid；這是相對差距目標。", "第二段公式及『不是強迫每篇回答只增加不減少』", [ref("s-dpo", "§3 Eq.(3); §4 Eqs.(5)–(7) and gradient p.5", "KL-regularized reward objective reparameterizes into a pairwise relative log-ratio loss; shared model gradients affect probabilities jointly")], ["a-dpo", "a-theory"], "同一prompt下的完整回答log機率；beta>0，reference保留不更新；有限卡片算例使用card probability。不能由下降loss推導每篇回答絕對機率都增加。"),
        claim("c3", "numeric", "reference為[0.5,0.5]、beta=1時，policy[0.5,0.5]/[0.7,0.3]/練習[0.3,0.7]的loss分別約0.6931/0.3567/1.2040。", "原Python區塊、算例段落與末段練習", [ref("s-math", "two-card normalized distribution derivation", "sigmoid(log(p/(1-p)))=p, so loss=-ln p"), ref("s-audit", "numeric_results and original_snippet_stdout", "original fenced code and reversed exercise executed on CPU")], ["a-audit-run", "a-audit-code"], "只有兩张卡且機率和為1、固定參考各0.5、beta=1，不能套作正式beta=0.1的loss。", verification("0.69314718056, 0.35667494394, 1.20397280433", "float32 CPU: 0.69314718246, 0.35667493939, 1.20397281647; displayed 0.6931/0.3567/1.2040", "完整原碼執行；獨立math.log解析值與torch dpo_loss比较。", tolerance="absolute error <1e-6; rounded four-decimal output exact")),
        claim("c4", "software", "本節呼叫的dpo_loss計算相對margin、detach reference差、使用−F.logsigmoid的batch mean，沒有reward-model呼叫。", "Python區塊 import與dpo_loss呼叫", [ref("s-alignment", "dpo_loss lines 38–42", "beta guard, reference detach, relative margin and mean negative logsigmoid"), ref("s-dpo", "Appendix B p.20 PyTorch dpo_loss", "original pseudocode computes beta*(pi_logratios-ref_logratios)")], ["a-audit-run", "a-theory"], "目前PyTorch2.14.1+cpu與repo API；原節batch只有一對，所以mean不改數值。不外推其他API版本。", verification("原例能執行；reference不接收梯度，beta=0.1相同起點時chosen/rejected scalar梯度為−0.05/+0.05", "原例輸出吻合，chosen grad −0.05000000075，rejected +0.05000000075，兩個reference leaf grad均None", "audit執行原節Python與另外四個leaf標量的梯度probe。")),
        claim("c5", "concept", "DPO的推導不保證任何PPO實作會得到同一個模型，有限資料/模型/更新配方仍須獨立驗證。", "DPO原論文連結段與方法外推限制", [ref("s-dpo", "§4 Eqs.(3)–(7); Appendix A.6 Theorem1; §7 limitations", "theoretical reward-policy mapping assumes preference model, beta>0 and positive reference support; OOD generalization needs further study"), ref("s-ppo", "§3 Eqs.(6)–(7), §5 Algorithm1", "PPO uses sampled advantages and finite repeated optimization, rather than a guarantee of exact KL-reward optimum")], ["a-dpo", "a-theory", "a-limits", "a-ppo"], "推導成立於指定KL目標與偏好模型；模型參數化、標註誤差和有限更新不由等價推導消失。"),
        claim("c6", "software", "正式有限候選CPU實驗保存SFT checkpoint，PPO與DPO從完全相同SFT state開始，並保留同一個全程不動的reference。", "圖前後與『逐項相同的權重開始』", [ref("s-runner", "run_posttraining lines 289–299, 326–328, 444–445", "saved deepcopy reference, two deepcopy policies, initial and final state digest assertions"), ref("s-cpu", "results.ppo/dpo.initial_state_sha256, reference_state_sha256_before/after", "fresh actual run verifies all four digests equal")], ["a-cpu-run", "a-audit-run", "a-published"], "僅這次seed42有限card-policyCPU執行；不同方法最終權重不要求相同，也不是先PPO再DPO。", verification("兩支initial state和reference前後均一致", "全部為f6ebe6766d91b324a2af12d628af5e7df9cfd144a1961e5ce551f46f2467aacc；全程固定reference assertion通過", "新完整實驗執行與公開報告的hash交叉核對，不只核對模型名字。")),
        claim("c7", "numeric", "示範起點的有限回答策略只有148個可學參數。", "正式CPU比較段『148參數』", [ref("s-networks", "FiniteResponsePolicy lines 54–61", "Linear(4,16), tanh, Linear(16,4)"), ref("s-math", "parameter count derivation", "4*16+16+16*4+4=148")], ["a-cpu-run", "a-audit-run"], "148是card policy，非reward241或value97，也非語言模型參數數量。", verification("148", "fresh sft.pt model state has148 elements; published/fresh policy count148", "實際載入fresh checkpoint求numel並對照架構。", tolerance="exact integer equality")),
        claim("c8", "empirical", "PPO與DPO各360次policy更新；DPO batch64累計23,040 pair draws，PPO另300次RM更新、360次critic更新及7,680 sampled actions，因此更新次數不是相同資料/計算預算。", "正式CPU比較段步數與預算", [ref("s-runner", "CONFIG lines 34–55; reward/PPO/DPO loops lines 302–313, 334–390, 416–431", "separate stage schedules, 120 rollouts reused3epochs and DPO offline batch64"), ref("s-cpu", "results.config, reward, ppo, dpo", "actual completed schedules and processed draw counts")], ["a-published", "a-raw", "a-cpu-run", "a-audit-run"], "抽用是有放回曝光，非不同標註對；train132contexts/44families/660uniquepairs，token分母0。未建立等資料、等時間或等統計預算比較。", verification("PPO360=120*3，DPO360，DPO23040=360*64，RM300，critic360，sampled7680=120*64", "所有計數exact吻合；RM19200pair draws，PPO重用23040action draws，DPO23040pair draws", "公開報告與新完整CPUrun配置、完成狀態、分母和loop實際輸出核對。", {"seed":42,"train_families":44,"train_contexts":132,"unique_train_pairs":660,"dpo_batch_pairs":64,"dpo_updates":360,"dpo_pair_draws":23040,"ppo_rollouts":120,"ppo_rollout_size":64,"ppo_epochs_per_rollout":3,"ppo_sampled_actions":7680,"ppo_reused_action_draws":23040,"rm_updates":300,"critic_updates":360,"effective_tokens":0})),
        claim("c9", "empirical", "最後18題PPO與DPO都12題滿足完整要求，分項同為數字6/6、說明0/6、求補資訊6/6；SFT起點僅6/18。", "正式CPU比較段及13.16分項連結", [ref("s-runner", "build_records, split_records, _evaluate lines 63–123, 188–253", "independent mode rules, held-out family split, greedy argmax and full raw rows"), ref("s-audit", "full_row_audits.published/fresh_cpu.all_rows.test", "recomputed every saved prediction and response against independent rule")], ["a-published", "a-cpu-run", "a-cpu-records", "a-audit-run"], "成功只表示從Python預寫4張卡選中符合要求的卡；不是生成18個算術答案。一個seed、一個split；說明條件train側也0/44；SFT只教number，偏好新增條件，非等資料SFT優勝證據。", verification("兩支12/18，分項6/6、0/6、6/6；SFT6/18", "published與fresh逐題均完全重現；三支train/validation/test全部165rows、495policy decisions、825RM比較均核對", "每列重建原始題與candidate，驗算probability normalization、argmax、response與mode要求；各家族split不重疊。", {"seed":42,"test_families":6,"test_contexts":18,"test_per_mode":6,"test_rm_pairs":90,"validation_contexts":15,"train_contexts":132,"candidate_actions":4,"effective_tokens":0})),
        claim("c10", "empirical", "報告這次CPU段落時間約PPO0.66秒、DPO0.34秒、另RM0.27秒，只支持本機小網路時間。", "計時與大型LLM限制段", [ref("s-runner", "stage_started/perf_counter intervals lines 302–313, 334–390, 416–431", "each stage times its own loop; checkpoint saves, evaluation and setup are outside stage timing"), ref("s-audit", "full_row_audits.*.stage_seconds", "published exact timings independently extracted; fresh run comparable scope")], ["a-published", "a-raw", "a-cpu-run", "a-audit-run"], "原次CPU2threads、單次完整段落、無暖機或重複benchmark，PPO包括rollout與policy/critic迴圈，RM另計；非CUDA/LLM效能或普遍速度比。", verification("published PPO0.657001449, DPO0.343355792, RM0.270443069 seconds", "原raw與published結果exact一致；fresh PPO0.676798125, DPO0.338443819, RM0.272190439seconds，新的wall time不是原值重現保證", "閱讀perf_counter範圍並跑全配置；確認0.66/0.34/0.27是原報告四捨五入，未把新CPUrun当成GPU加速證據。", {"archived_runs":1,"fresh_runs":1,"warmup_runs":0,"cpu_threads":2,"seed":42,"policy_parameters":148,"rm_parameters":241,"value_parameters":97,"ppo_updates":360,"dpo_updates":360,"rm_updates":300})),
        claim("c11", "concept", "真人比較訓練RM再PPO更新可以組成典型RLHF；本章固定規則程式標籤不能宣稱真人回饋研究。", "倒數第二段RLHF來源區分", [ref("s-instructgpt", "§3.1 Steps1–3 p.6", "trained human labelers supply demonstrations/comparisons; RM trained then policy updated with PPO"), ref("s-runner", "build_records lines 63–106, result.label_source", "course author fixed ranking templates applied by Python, no recruited raters")], ["a-instructgpt", "a-published"], "RLHF表示human feedback來源與流程；PPO是更新算法，DPO也可用真人或合成偏好，算法名稱本身不決定標籤來源。"),
        claim("c12", "software", "有限選卡PPO是概念比較支線；19.8成品的偏好stage使用自身token-model DPO，不載入這支card-policyPPO權重。", "倒數第二段19.8連結與模型權重範圍", [ref("s-capstone-pinned", "EXPERIMENTS, train_stage DPO objective lines 37–43, 119–133, 205–212; _run_context parent dependency lines 280–289; run_preference lines 308–309", "capstone stage dispatch has DPO from its own parent stage, with DPO+0.2CE+0.01auxiliary; no finite-policy/PPO checkpoint input"), ref("s-capstone-model", "preference_loss/frozen_reference lines354–375", "complete token-sequence scores and frozen CapstoneModel reference, a distinct model class"), ref("s-token-runner", "_dpo_train/run_dpo lines578–620,710–760", "13.3–13.6 tokenLM branch uses complete answer scores, independent of the new finite-card comparison")], ["a-audit-run", "a-token", "a-pinned-code", "a-pinned-link"], "本審閱只驗成品DPO分支的軟體來源區分與linked原例forward；不重跑GPU續訓、不對19.8全部品質矩陣作獨立判定。19.8固定commit新URL匿名HTTP200且與local source逐byte相等。", verification("capstone例子能以自身模型與reference計算DPO，與card model分開", "19.8原例CPU輸出same-start loss0.6931、reference requires_grad=False；pinned capstone.py與local source SHA一致", "當前19.8重新完整讀取；執行其原fenced example，並讀DPO階段dispatch/parent/混合目標；tokenLM舊GPU報告亦核beta0.1與250步/9448answer-token分母，未重新訓練。")),
        claim("c13", "concept", "圖以共同SFT checkpoint分叉PPO/DPO，再共同驗收，未畫成兩法依序串行，並提示相同步數仍有成本差。", "ppo_dpo_routes.svg及圖說段", [ref("s-dpo", "§4 DPO outline; §3 RL fine-tuning phase", "two alternative preference-optimization paths"), ref("s-runner", "run_posttraining two deepcopy branches and evaluations", "implementation supplies same-start branch topology and common evaluation")], ["a-figure", "a-audit-run"], "820x520示意布局不是計算成本比例尺；XML text/path與Inkscape實際render親看，文字完整可讀、箭頭從起點向两branch再匯向驗收。"),
    ]
    sources = [
        {"id":"s-dpo","kind":"paper","title":"Rafailov et al., Direct Preference Optimization","url":"https://arxiv.org/pdf/2305.18290v3","version":"arXiv:2305.18290v3, 29 Jul 2024","verified":True,"checked_original":True,"accessed_on":"2026-10-04","authority_reason":"DPO作者原論文，直接給KL目標、reward-policy重參數化與公式7。","inspection_note":"實讀原PDF轉文本pp3–5、10、18–20：Eqs1–7、梯度、offline pipeline、Theorem1與beta>0/reference support前提、AppendixB實作與generalization限制。原PDF hash與候選sources.json吻合，版本由PDF首頁重新抽取確認；未用任何technical-sources摘要當權威。"},
        {"id":"s-ppo","kind":"paper","title":"Schulman et al., Proximal Policy Optimization Algorithms","url":"https://arxiv.org/pdf/1707.06347v2","version":"arXiv:1707.06347v2, 28 Aug 2017","verified":True,"checked_original":True,"accessed_on":"2026-10-04","authority_reason":"PPO作者原論文，定义old-policy ratio與clip surrogate及取樣/重用順序。","inspection_note":"實讀pp3–5：§3 Eqs6–7、§5 Algorithm1先sample再Kepochs再替换old policy；區分此old policy和RLHF全程reference。此論文一般PPO不強制使用人類回饋或獨立RM。原PDF版本/hash重新核對。"},
        {"id":"s-instructgpt","kind":"paper","title":"Ouyang et al., Training language models to follow instructions with human feedback","url":"https://arxiv.org/pdf/2203.02155v1","version":"arXiv:2203.02155v1, 4 Mar 2022","verified":True,"checked_original":True,"accessed_on":"2026-10-04","authority_reason":"InstructGPT作者原論文直接記錄human demonstration/comparison、RM與PPO流程。","inspection_note":"實讀§3.1 p6 Steps1–3：真人labelers先示範，再比較train RM，再用scalar reward PPO更新；不是把任意程式標籤叫human feedback。PDF首頁版本與候選manifest hash重新確認。"},
        repository("s-alignment", "tiny_perceptron/alignment.py", "完整讀81行，重點sequence_log_probability sum answer mask、dpo_loss beta guard/relative margin/reference detach/mean logsigmoid。"),
        repository("s-runner", "scripts/course_experiments/posttraining.py", "完整讀575行：family split、四種預寫卡、作者規則、SFT/RM/PPO/critic/DPO各loop、clone/freeze、state fingerprints、all-row evaluation與stage timing；三code hashes與公開run exact吻合。"),
        repository("s-networks", "tiny_perceptron/posttraining.py", "完整讀86行：policy4->16->4、reward context+candidate identity與critic，精確KL與clipped surrogate；不處理中文或token生成。"),
        repository("s-token-runner", "scripts/course_experiments/behavior.py", "讀_pair_examples/_dpo_train/_preference_evaluate/run_dpo相关函数：答案masked序列sum、EOS、固定reference、250update beta0.1/1.0；與有限選卡實驗分开。"),
        repository("s-capstone-model", "tiny_perceptron/capstone.py", "讀preference_pairs/preference_loss/frozen_reference：CapstoneModel兩側完整sequence logps，共同reference no_grad/freezing，非finite-card PPO參數。"),
        {"id":"s-capstone-pinned","kind":"official_source","title":"Repository fixed capstone training source","url":"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py","version":"Git commit 1df335318bda03fd771807f66976953231d5a00b; SHA256 63fcc9d1d13503fa3b2ce28d6ac7fa6040d659f51fd8865ed18fabbe3d0fa5cf","verified":True,"checked_original":True,"accessed_on":"2026-10-04","authority_reason":"課程repository固定commit原始訓練入口，是成品分支來源的直接契約。","inspection_note":"按19.8更新連結匿名讀raw HTTP200，全檔與local current source逐byte相同；實讀stage dispatch、parent checkpoint、reference、DPO+CE+auxiliary loop，無finite-card PPO入口。不將來源庫摘要或link文字當執行證據。"},
        {"id":"s-math","kind":"derivation","title":"Two-card DPO and parameter/budget arithmetic","verified":True,"details":"reference odds=0.5/0.5=1；margin=ln(p/(1-p))。beta1時sigmoid(margin)=1/[1+(1-p)/p]=p；loss=-ln(p)。p0.5→0.69314718056、p0.7→0.35667494394、p0.3→1.20397280433。148=(4*16+16)+(16*4+4)；PPO360=120*3，DPOpairdraws23040=360*64，sampledactions7680=120*64。已在CPU獨立比較解析與API結果，abs error<1e-6。"},
        {"id":"s-audit","kind":"execution","title":"Independent original-code/numeric/full-row audit","verified":True,"artifact_id":"a-audit-run"},
        {"id":"s-cpu","kind":"execution","title":"Fresh fixed full CPU posttraining experiment","verified":True,"artifact_id":"a-cpu-run"},
    ]
    artifacts = [
        artifact("a-audit-run", "execution", PREFIX+"audit_result.json", "原例、反向練習、梯度/遮罩契約、全部raw rows/分母以及19.8 linked原例與SVGXML核對結果", ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_17_audit.py", audit["result"], env),
        artifact("a-audit-code", "code", PREFIX+"audit.py", "可重現的独立audit原碼，包含所有assertions"),
        artifact("a-cpu-run", "execution", PREFIX+"cpu_run/result.json", "本審閱新完整固定CPUrun，配置、stage時間、state SHA與全逐題結果", ".venv/bin/python -m scripts.course_experiments.posttraining --output docs/technical-reviews/artifacts/fact_v2_13_17_cpu_run", "Completed2.104s; SFT6/18, PPO12/18, DPO12/18;完整固定schedule無縮減", {"python":"3.13.5","torch":"2.14.1+cpu","device":"cpu","threads":"2","seed":"42"}),
        artifact("a-cpu-records", "source_snapshot", PREFIX+"cpu_run/records.json", "新run完整train/validation/test原始題、人工ranking、候選卡與family切分"),
        artifact("a-published", "source_snapshot", "docs/course-experiments/results/posttraining.json", "本節引用的原公開CPU實報，核all165rows與stage timing"),
        artifact("a-raw", "source_snapshot", PREFIX+"published_raw_experiment.json", "原CPU實驗raw結果的完整bytes副本；原ignored輸出仍保留，兩者全檔SHA完全相同；內容與published.results exact一致"),
        artifact("a-raw-records", "source_snapshot", PREFIX+"published_raw_records.json", "原ignored records.json完整bytes副本，供Git checkout後audit重建原始資料分母"),
        artifact("a-sft-state", "source_snapshot", PREFIX+"fresh_sft_state.json", "先前CPUrun的148個float32參數與形狀/metadata；JSON反解tensor逐項exact相等且stateSHA相同，無重訓"),
        artifact("a-closure", "execution", PREFIX+"closure_copy_receipt.json", "原證據bytes/SHA、原檔保留、歷史報告與lossless tensor snapshot等價核對", ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_17_closure_verify.py", "原raw/records副本byte-for-byte相同；所有既有reviewed source SHA不變、SFT148tensor exact；no training", env),
        artifact("a-closure-code", "code", PREFIX+"closure_verify.py", "closure當次親核原檔與可发布snapshot等價的完整assertions"),
        artifact("a-token", "source_snapshot", "docs/course-experiments/results/dpo.json", "獨立tokenLM GPU DPO報告，只核本節涉及beta/scope與全preference rows，未重新GPU訓練"),
        artifact("a-figure", "figure_render", PREFIX+"ppo_dpo_routes.png", "實際Inkscape1.4渲染820x520親看；兩路分叉、common驗收、成本限制文字與正文一致；command: inkscape course/figures/ppo_dpo_routes.svg --export-type=png --export-filename=docs/technical-reviews/artifacts/fact_v2_13_17_ppo_dpo_routes.png"),
        artifact("a-dpo", "source_snapshot", PREFIX+"dpo_original_excerpt.txt", "從原DPO PDF用pdftotext -f3 -l5重新抽取，Eqs1–7與DPO outline"),
        artifact("a-theory", "source_snapshot", PREFIX+"dpo_original_theory_code.txt", "原PDFpp18–20：Theorem1/Proposition1 support與beta前提、原PyTorchDPO loss"),
        artifact("a-limits", "source_snapshot", PREFIX+"dpo_original_limits.txt", "原PDFp10 §7：OOD與reward overoptimization/scale/evaluation仍待研究"),
        artifact("a-ppo", "source_snapshot", PREFIX+"ppo_original_excerpt.txt", "原PPO PDFpp3–5 ratio/clip/Algorithm1，非其他代理筆記"),
        artifact("a-instructgpt", "source_snapshot", PREFIX+"instructgpt_original_excerpt.txt", "原InstructGPT PDFp6 §3.1 three steps與human labelers"),
        artifact("a-provenance", "source_snapshot", PREFIX+"original_pdf_provenance.json", "三份原PDF全檔SHA與重新抽取首頁含arxiv revision/date，對照候選sources.json成功"),
        artifact("a-pinned-code", "code", PREFIX+"pinned_capstone.py", "19.8新固定commit URL匿名取得原code，与local逐byte相同"),
        artifact("a-pinned-link", "source_snapshot", PREFIX+"pinned_link_receipt.json", "新URLHTTP200、固定版本/hash、local equality和本次實際讀取scope"),
        artifact("a-section", "source_snapshot", PREFIX+"section.txt", "本次真正完整審閱13.17正文，包含未正規化空白與換行"),
        artifact("a-prereqs", "source_snapshot", PREFIX+"read_prerequisites.txt", "全部明示必要前置正文完整讀取；19.8pin變更後重新讀目前全文"),
    ]
    report = {
        "schema_version":1,"review_stage":"technical","lesson_id":"13.17","source":"course/chapters/13.md#13.17",
        "reviewer_task":"/root/integration_technical_coordinator/fact_v2_13_17","reviewer_context":"fresh",
        "source_sha256":hashlib.sha256(body.encode()).hexdigest(),"figure_sha256":{"course/figures/ppo_dpo_routes.svg":digest("course/figures/ppo_dpo_routes.svg")},
        "verdict":"pass","claims":claims,"sources":sources,"artifacts":artifacts,"issues":[],
        "review_scope":{"guide":"docs/technical-review-guide.md","guide_sha256":digest("docs/technical-review-guide.md"),"read_prerequisite_sha256":{path:hashlib.sha256(text.encode()).hexdigest() for path,text in prereqs.items()},"independence":"本節獨立fresh task；未讀舊technical/reader reports或作者歷史，不推測author_tasks；無子代理。","rechecks":"19.8 evidence URL pin變更後讀目前全文與變更清單，匿名核固定capstone.py source；13.17正文未變。","environment":"現有.venv與Inkscape足夠，不安裝、不改env、不付費、不跑長訓練、不改教材。"},
        "checks":{
            "factual_accuracy":{"status":"pass","details":"13個拆分主張均核實。DPO無獨立RM、相對log-ratio與PPO/RLHF流程對照原論文；有限實驗/成品tokenLM分支清楚分開。","claim_ids":["c1","c2","c4","c5","c6","c11","c12"]},
            "numeric_verification":{"status":"pass","details":"原碼、反向練習、解析log值、梯度與148參數/各更新draw分母核對；CPU完整重跑與所有raw rows重算通過，原報告秒數按原timing scope確認。","claim_ids":["c3","c7","c8","c9","c10"]},
            "figure_consistency":{"status":"pass","details":"已讀SVG完整XML text/path與820x520 layout；Inkscape1.4真正render親看，shared checkpoint分叉兩路再共同驗收，無串行暗示、無數字或縮放矛盾、字不截斷。","claim_ids":["c6","c13"]},
            "source_verification":{"status":"pass","details":"原DPOv3/PPOv2/InstructGPTv1 PDF版本、公式/Algorithm定位、全PDF SHA均核對；repo原碼full-file hash與實際執行結果可追溯；19.8固定commit原碼URLHTTP200且bytes一致。","claim_ids":["c1","c2","c4","c5","c11","c12"]},
            "limitations":{"status":"pass","details":"BT/reference-support/beta前提、toy beta1與formal0.1、單seed四卡非生成、step不等預算、SFT無新條件、各CPU段計時非LLM性能、程序標籤非humanRLHF、cardPPO非成品權重皆明示且未越界。","claim_ids":["c2","c3","c5","c8","c9","c10","c11","c12","c13"]},
        },
    }
    prior = json.loads((ROOT / (PREFIX+"before_closure_report.json")).read_text())
    report["recheck_history"] = prior.get("recheck_history", []) + [{
        "stage":"release_evidence_closure",
        "reviewer_task":report["reviewer_task"],
        "previous_report_path":PREFIX+"before_closure_report.json",
        "previous_report_sha256":digest(PREFIX+"before_closure_report.json"),
        "previous_checks_path":PREFIX+"before_closure_checks.json",
        "previous_checks_sha256":digest(PREFIX+"before_closure_checks.json"),
        "checkout_receipt_path":PREFIX+"checkout_closure.json",
        "checkout_receipt_sha256":digest(PREFIX+"checkout_closure.json"),
        "details":"完整讀原report和raw；將被ignore的原experiment/records實bytes複製入docs自己的prefix，核SHA/bytes相同，原檔/前report/前helper/前執行紀錄保留。新audit改讀發布snapshot與lossless148tensorJSON，不重訓。正文、圖、原數據、repository source與前置SHA全不變，保留pass判定。",
    }]
    report["review_scope"]["rechecks"] = prior["review_scope"]["rechecks"] + " 追加發布closure：原ignored raw實bytes完整複製保存並核等價；全部22direct artifacts與5repository sources不ignore，temporary Git index實際checkout後hash/audit/checker通過，无ignored outputs或pt依賴，shared index不變。"
    for identifier in ("c7",):
        item = next(value for value in report["claims"] if value["id"] == identifier)
        item["artifact_ids"].append("a-sft-state")
        item["verification"]["details"] += " 發布closure保存既有148float32 tensor的小JSON快照，原checkpoint與反解tensor exact相等、stateSHA相同；audit可由Git證據重跑。"
    path=ROOT/"docs/technical-reviews/13.17.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}: {len(claims)} verified claims, {len(sources)} sources, {len(artifacts)} artifacts; source {report['source_sha256']}")


if __name__ == "__main__":
    main()
