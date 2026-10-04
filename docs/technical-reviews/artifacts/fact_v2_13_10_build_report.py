"""Write this reviewer's independently verified lesson 13.10 report."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
BASE = "docs/technical-reviews/artifacts/"
PREFIX = "fact_v2_13_10"
AUDIT = json.loads((ROOT / BASE / f"{PREFIX}_audit_output.json").read_text())
ENVIRONMENT = AUDIT["environment"]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def artifact(identifier, kind, filename, description, command=None, result=None, environment=None):
    path = BASE + PREFIX + filename
    record = {"id": identifier, "kind": kind, "path": path, "sha256": digest(path), "description": description}
    if command:
        record.update(command=command, result=result, environment=environment or ENVIRONMENT)
    return record


def evidence(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, support, artifacts, scope, verification=None, status="verified"):
    record = {"id": identifier, "kind": kind, "statement": statement, "location": location,
              "status": status, "evidence": support, "artifact_ids": artifacts, "scope": scope}
    if verification:
        record["verification"] = verification
    return record


def verification(expected, observed, details, tolerance=None, denominators=None):
    record = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance:
        record["tolerance"] = tolerance
    if denominators:
        record["denominators"] = denominators
    return record


def main():
    checkpoint_snapshot = json.loads((ROOT / BASE / f"{PREFIX}_reward_checkpoint.json").read_text())
    export_receipt = json.loads((ROOT / BASE / f"{PREFIX}_checkpoint_export_receipt.json").read_text())
    verify_receipt = json.loads((ROOT / BASE / f"{PREFIX}_checkpoint_verify_receipt.json").read_text())
    assert digest(BASE + PREFIX + "_reward_checkpoint.json") == export_receipt["export_sha256"] == verify_receipt["export_sha256"]
    assert digest(BASE + PREFIX + "_checkpoint_portable.py") == export_receipt["verification_code_sha256"] == verify_receipt["verification_code_sha256"]
    assert checkpoint_snapshot["source_checkpoint"]["sha256"] == checkpoint_snapshot["original_evidence"]["prior_checkpoint_artifact_record"]["sha256"]
    assert verify_receipt["checks"]["binary_checkpoint_read"] is False
    assert all(item["all_raw_scores_exactly_equal"] for item in verify_receipt["checks"]["evaluations"].values())
    source = ROOT / "course/chapters/13.md"
    body = dict(sections(source))["13.10"]
    assert hashlib.sha256(body.encode()).hexdigest() == AUDIT["lesson_sha256"], "Must rerun CPU audit for changed lesson"
    original = "print(\"差距\", preferred_score,"
    label_is_wrong = original in body
    issues = [{"claim_id": "c11", "status": "unresolved",
               "details": "依練習只將rejected改1，print仍以preferred_score印差距0/2，而實際chosen-rejected為-1/1；機率與loss正確。需打印实际分數差。"}] if label_is_wrong else [{
                   "claim_id": "c11", "status": "resolved",
                   "details": "舊版依練習只改rejected=1後，把preferred_score误印成差距0/2，实际-1/1。旧版原文及实际输出保存为before_gap_fix artifacts。",
                   "resolution": "完整親讀当前13.10/13.1/13.2/13.15/1.8後，確認gap=chosen-rejected並打印gap.item()。新版原例真CPU仍为差距0/2，rejected=1时真实打印-1/1，P0.2689/0.7311与L1.3133/0.3133正确，兩側共同加10仍相同；literal stdout assertions及全165題/825對重核通过。"
               }]
    issues.append({
        "claim_id": "c14", "status": "resolved", "issue_type": "evidence_publication",
        "details": "原报告直接引用Git忽略的reward.pt；原checkpoint确已亲讀，原全檔hash和旧审计结果保留，没有否认或删除历史验证。",
        "resolution": "重新亲讀原reward.pt，并保存可提交JSON：全檔SHA05a995e38fdec977d9ba82aaf7f0b3f93e2d1172d298690faef1877f4bb0d17b、原metadata、5个float32 tensor的shape/dtype/逐字节SHA和全部257数值。JSON往返逐tensor bytes与原checkpoint完全相同，aggregate stateSHA6aab984375c8b5d340a72eff51670d3101a2f64048a45ebc9bc0b0d91528ef83保持。再以不存在的checkpoint路径执行JSON-only模式，全165题660个scalar分数exact匹配，凍結RM下游反傳无grad且参数不变。binary留本机；报告使用等价JSON/code/真实export与verify receipts，不再直接依赖binary，未重训。"
    })
    unresolved = any(issue["status"] != "resolved" for issue in issues)
    claims = [
        claim("c1", "concept", "同題只回數字的1+2題，3比含算式的長答更符合格式要求。", "第1段：3與1+2等於3的比較",
              [evidence("s_math", "derivation: same-prompt comparison and 1+2=3", "內容同為3，格式條件要求沒有額外文字。"), evidence("s_dpo", "v3 p3 §3 same-prompt pairs and Eq1 p(y1>y2|x)", "比較以共同prompt x為條件，勝者標籤來自當次任務偏好，論文不替教材決定格式標籤。")], ["a_math", "a_figure"], "這次格式偏好；不把含解釋的答案普遍判為較差，1+2=3與格式符合由本題明示規則直接核對。"),
        claim("c2", "concept", "獎勵模型以問題和回答為條件輸出一個分數，與負責作答的策略模型不同。", "第1–2段：另一個輸出分數的模型與reward model定義",
              [evidence("s_dpo", "v3 p3 §3 Reward Modelling Phase and Eq2; RL phase", "r_phi(x,y)是scalar reward，之後回饋给pi_theta。"), evidence("s_instructgpt", "v1 p8 §3.5 Reward modeling", "原文以prompt和response為輸入，輸出scalar reward。")], ["a_dpo_p3", "a_instructgpt_p8_9"], "一般語言RM的概念；本節的玩具RM另以結構特徵實現，沒有把中文當模型輸入。"),
        claim("c3", "concept", "同題兩篇回答的分數差可用sigmoid轉成較佳者勝出的偏好模型機率。", "第2段与程式win_probability",
              [evidence("s_dpo", "v3 p3 §3 Eq1", "exp(c)/(exp(c)+exp(d))=sigmoid(c-d)，BT是對偏好分布的建模假設。")], ["a_dpo_p3", "a_math"], "BT/logistic模型假設下的估計機率；不是所有真實偏好無條件符合BT，更不是事實正確率。"),
        claim("c4", "concept", "以chosen為勝者的負自然log偏好代價，會隨chosen相對rejected的分數提高而降低。", "程式後第1段：提高相對分數降低代價",
              [evidence("s_dpo", "v3 p3 Eq2", "負對數似然為-log sigma(c-d)。"), evidence("s_math", "derivation: dL/dc=sigma(delta)-1", "有限實數分數下對chosen的導數為負；對rejected為正。")], ["a_math", "a_audit"], "對固定單項分數差的單調性；不保證一次共享網路參數更新改善所有樣本。"),
        claim("c5", "numeric", "差距0的勝出機率0.5，代價四位小數0.6931。", "第2段與程式輸出第1行",
              [evidence("s_math", "derivation numeric table row delta=0", "1/(1+exp(0))=1/2；ln2=0.6931471806。")], ["a_math", "a_audit"], "一對人工float32分數[0]和[0]；不是實測RM分數。",
              verification("0.5 and 0.6931", AUDIT["examples"]["original"].splitlines()[0], "執行完整本節原程式；形狀[1]、dtype float32、CPU、mean分母1；另用float64/math復算。", "四位小數literal匹配；float64解析值絕對誤差≤1e-15。")),
        claim("c6", "numeric", "差距2的勝出機率約0.881，四位小數0.8808，代價0.1269。", "第2段與程式輸出第2行",
              [evidence("s_math", "derivation numeric table row delta=2", "1/(1+e^-2)=0.8807970780；ln(1+e^-2)=0.1269280110。")], ["a_math", "a_audit"], "一對人工float32分數[2]和[0]，印四位小數。",
              verification("0.8808 and 0.1269", AUDIT["examples"]["original"].splitlines()[1], "執行完整原程式並獨立解析代入；同題、形狀[1]、mean分母1。", "四位小數literal匹配；float64解析值絕對誤差≤1e-15。")),
        claim("c7", "software", "preference_loss對同形狀非空chosen/rejected張量取負logsigmoid分數差後，平均所有比較。", "程式呼叫及程式後第1段preference_loss定義",
              [evidence("s_posttraining", "preference_loss lines8–12", "檢查同形狀/非空後-return F.logsigmoid(chosen-rejected).mean()。"), evidence("s_torch", "installed functional.py line2049 and expit docstring", "logsigmoid與sigmoid的元素公式及C綁定。")], ["a_torch", "a_audit"], "本專案當前原碼，已執行Python3.13.5/PyTorch2.14.1+cpu；其他軟體版本不由這次CPU執行保證。",
              verification("one-pair and four-pair means match analytic negative-log likelihood; invalid shape/empty inputs reject", f"four-pair mean {AUDIT['batch_mean_loss']}; invalid inputs {AUDIT['invalid_input_checks']}; extreme stable mean {AUDIT['extreme_stable_loss']}", "檢查shape、dtype、1與4比較的mean分母、梯度符號及穩定極端值；原碼是logsigmoid，沒有先以sigmoid再log導致underflow。")),
        claim("c8", "concept", "同題所有分數共同加10，不改分數差、勝出機率或偏好loss。", "程式後第2段與練習最後兩句",
              [evidence("s_dpo", "v3 p5 §5.1 Definition1 / Lemma1", "相差prompt-only函數的reward導出相同偏好分布。"), evidence("s_math", "derivation: common offset cancellation", "(c+10)-(d+10)=c-d。")], ["a_dpo_p5_6", "a_math", "a_audit"], "實數公式恒等；toy整數+10在此浮點精確可表示。任意浮點分數巨大共同偏移可能有rounding，未宣稱任意bitwise不變。"),
        claim("c9", "concept", "原始獎勵分數不是校準過的正確率，也不表示理解程度倍數；評分可能學到偏好代理特徵。", "程式後第2段：3不是三成/三倍與長答、措辭、禮貌",
              [evidence("s_dpo", "v3 p5 Definition1/Lemma1; p10 §6.4; p11 limitations", "score可平移；GPT4偏好評估受長度及評估prompt影響，不能當普遍品質定理。"), evidence("s_instructgpt", "v1 p17 before §5; p19 §5.3", "作者懷疑labeler獎勵謙遜造成過度hedging的RM代理學習，並說標注群與模型有限。"), evidence("s_math", "derivation: score shift and absence of correctness target", "loss只收到比較標籤；沒有绝对truth label，score偏移仍同樣loss。")], ["a_math", "a_dpo_p5_6", "a_dpo_limits", "a_instructgpt_limits"], "可能性與應檢查的失敗模式；沒有宣稱本玩具RM實測偏好長度或礼貌，亦不拿GPT4評審偏差直接當所有RM實驗結果。"),
        claim("c10", "numeric", "把rejected改1，兩個差距為-1/1，勝出機率0.2689/0.7311，loss1.3133/0.3133；共同+10仍同值。", "最後段練習中的全部四位數值",
              [evidence("s_math", "derivation numeric rows delta=-1,+1 and common offset", "直接代入logistic與負自然log。")], ["a_math", "a_audit"], "只替換指示的分數；新版差距標籤與實際分數差同步，舊問題保留在resolved c11。",
              verification("P=0.2689/0.7311, loss=1.3133/0.3133, +10 unchanged", AUDIT["examples"]["rejected_1"] + AUDIT["examples"]["rejected_1_shift_10"], "逐字執行新版原程式的rejected替換及共同+10替換；float32打印與math/float64復算；輸出標籤-1/1 literal斷言通过。", "四位小數literal匹配；float64解析值絕對誤差≤1e-15；toy+10結果精確相同。")),
        claim("c11", "software", "依練習改rejected為1後，程式標示的差距應對應chosen-rejected=-1/1。", "程式print行與最後段練習",
              [evidence("s_math", "derivation numeric delta=c-d", "改rejected後chosen本身0/2不是差距。")], ["a_audit"], "練習實際指示只改rejected；不是原始rejected=0的算例錯誤。",
              verification("printed gap -1.0 and 1.0; original0.0/2.0", AUDIT["examples"]["rejected_1"], "親讀gap=chosen-rejected、gap.item()後執行当前完整程式/rejected=1及共同+10，保存所有stdout；对每行差距/机率/loss均以literal assertions检查。"), status="contradicted" if label_is_wrong else "verified"),
        claim("c12", "software", "實驗使用四張程式預先計算的卡，候選包括短答、算式句、錯數字與請求補資訊。", "程式後第3段：四張回答卡與候選文字",
              [evidence("s_experiment", "build_records lines64–112", "Python計算a+b及a+b+1，建立四個字串卡，未由網路生成。")], ["a_audit", "a_run", "a_formal"], "所有165題候選逐項核對；只選卡，不證明模型加法能力。",
              verification("exactly four Python-template candidates per context, sums predefined", "165/165 public and fresh context candidate lists equal independently regenerated records; card0=a+b; card2=a+b+1", "逐題檢查全train/validation/test原文及候選字串，非只檢查摘要。")),
        claim("c13", "software", "三種條件依預定規則決定偏好；購買數量未给時，請補數量勝过三種無依據總價。", "程式後第3段：三種條件與購買數量",
              [evidence("s_experiment", "build_records lines67–95; rule_best_action", "number/explain完整排序各6對，missing只3勝其他3對；單價正值排除零價例外。")], ["a_audit", "a_formal"], "a在1–10為正；沒有捏造missing错误候選互相排序，也不推至所有自然語言價格場景。",
              verification("number best0, explain best1, missing best3; missing preferences exactly [3,0],[3,1],[3,2]", "165/165 context rules and all825 preference pairs match; 55 missing contexts have3 clarification wins each", "独立重建優先卡及各條件排名，驗證每條原始pair。")),
        claim("c14", "software", "玩具評分網路讀取問題條件與候選編號數字特徵，不讀中文，也未學算術。", "程式後第3段：評分網路輸入限定",
              [evidence("s_posttraining", "FiniteRewardModel forward lines64–77", "四個context feature與四維candidate_identity one-hot拼接；network輸入8維，輸出每張卡scalar。"), evidence("s_experiment", "_features and build_records features", "只有[a/10,b/10,explain_flag,missing_flag]傳入。")], ["a_audit", "a_reward_checkpoint", "a_checkpoint_export", "a_checkpoint_verify"], "候選意義固定，數學答案預計算；沒有語言token target。",
              verification("float32 input[N,4], per-context per-card output[N,4], one-hot IDs", str(AUDIT["frozen_rm"]) + "; JSON-only reconstruction exactly matches all165 contexts/660 raw scores", "原审计真实載入reward.pt并確認全18測試score。發布補核重新加载原binary检查全部tensor/metadata，JSON恢复同5tensor state，另无binary执行全165题660score与原结果精确匹配；真实export与verify receipt均保存。")),
        claim("c15", "software", "偏好標籤由程式套用寫定規則，本實驗無招募真人標注，不是人類回饋RLHF實證。", "程式後第3段最後兩句",
              [evidence("s_experiment", "build_records preference_pairs / label_source and result label_source", "training標籤直接由ranking/missing规则产生；沒有human-rater資料讀取路径。")], ["a_audit", "a_formal", "a_run"], "只驗證此實驗資料生成和訓練通路；不否定一般RLHF研究，可由原論文區分真human labels。",
              verification("all labels come from fixed Python rules", "all825 pairs in165 contexts equal generated rule pairs; public/fresh label_source explicitly no human-rater study", "親讀資料生成、檔案和實際執行記錄，不把名字PPO/RM自動當RLHF實證。")),
        claim("c16", "software", "以canonical加數家族整組切分，交換加數不能跨訓練與最後測試家族。", "程式後第4段：1+2/2+1家族",
              [evidence("s_experiment", "build_records a≤b; split_records lines116–132", "只建立a≤b canonical family；44/5/6家族互斥，pair:1:2預留test。")], ["a_audit", "a_formal"], "實際資料沒有另建逆序題；因此不是對逆序問題的泛化實測。不同模式整組留同split。",
              verification("55 unique canonical families; no split overlap; all modes same family stay together", "44/5/6 families,132/15/18 contexts; generated,original records,fresh records and public row IDs exact match; family sets disjoint", "親讀全部family IDs並用set交集與逐題配對檢查，SHA對應各split原始record。")),
        claim("c17", "software", "實際先由訓練家族偏好更新RM，再將RM凍結用於PPO選擇模型。", "程式後第4段第1句與13.15明示使用",
              [evidence("s_experiment", "run_posttraining reward loop lines309–339; PPO rollout rewards", "train-only pair tensor抽64對更新300次，後requires_grad_(False).eval()再作PPO reward。"), evidence("s_posttraining", "FiniteRewardModel and preference_loss", "相同網路與loss的直接實作。")], ["a_audit", "a_run", "a_checkpoint_export", "a_checkpoint_verify"], "同seed固定CPU合成選卡實驗；critic目标是normalized RM reward，token數0。",
              verification("train preference updates precede frozen downstream use; no RM grads or parameter changes", "original full fixed300-step reward run completed; original and JSON-restored checkpoint scores match all165 rows; frozen downstream backward updates policy while all241 RM params stay unchanged with no grad", "原验证亲讀順序/optimizer并完整CPU重跑、加载checkpoint反传。此次仅发布证据补核，无重训；原binary往返全部tensor exact，JSON-only还原后再次实际下游反传确认RM无grad且参数不变。")),
        claim("c18", "empirical", "評分員的訓練代價與未見家族排序表現分別保存。", "程式後第4段：訓練代價與未見家族排名各保存",
              [evidence("s_experiment", "reward.history save and _evaluate pairwise_accuracy/rows; final evaluations", "loss_before_update為各被抽訓練minibatch，评估另計全split嚴格胜出。")], ["a_audit", "a_run", "a_formal"], "單seed42，合成有限卡；heldout90/90只是此6家族18題之90比較，不能外推自然聊天。Loss曲線點来自不同有放回minibatch，不是同整train数据loss。保存時間是單次CPUtraining/evaluation/local checkpoint saves，非環境安装或跨硬體性能比較。",
              verification("separate train loss history and exhaustive held-out pair rankings preserved", "public/fresh histories step1 loss0.7149028182→step300 minibatch loss0.0029645436; all saved rows verify RM660/660 train,75/75 validation,90/90 test; fresh state and all165 rows match original exactly", "核對完整config、全部165逐題及825比较、目標單位、checkpoint hashes、紀錄時間scope；未以摘要accuracy替代逐題檢查。",
                           denominators={"seed": 42, "train_families": 44, "validation_families": 5, "test_families": 6,
                                         "train_contexts": 132, "validation_contexts": 15, "test_contexts": 18,
                                         "train_unique_pairs": 660, "validation_pairs": 75, "test_pairs": 90,
                                         "reward_updates": 300, "pair_draws_per_update": 64, "processed_pair_draws": 19200,
                                         "effective_tokens": 0, "effective_target_unit": "one binary preference comparison per sampled pair",
                                         "timing_runs": 1, "timing_warmups": 0, "device": "cpu", "cpu_threads": 2})),
    ]
    artifacts = [
        artifact("a_math", "derivation", "_derivation.md", "逐步推導BT概率、负log、導數、共同平移、全部数值和比较分母。"),
        artifact("a_audit_code", "code", "_audit.py", "本reviewer独立CPU审计程序，不改教材；读取全部原始逐題与参数。"),
        artifact("a_builder_code", "code", "_build_report.py", "本reviewer报告构造原码，防止未重跑时只换正文SHA，并保存旧问题resolved记录。"),
        artifact("a_audit", "execution", "_audit_output.json", "原文程序与练习、数值/梯度/分母、全部逐题比较、checkpoint和凍結的实际执行结果。", AUDIT["command"], AUDIT["result"]),
        artifact("a_run", "execution", "_cpu_run/result.json", "固定原始实验全部步骤fresh CPU重跑结果，包含完整配置/逐題/時間。", ".venv/bin/python -m scripts.course_experiments.posttraining --output docs/technical-reviews/artifacts/fact_v2_13_10_cpu_run", "Exit0; elapsed2.094126017s; RM300×64 draws and whole held-out evaluation completed; all saved state fingerprints match prior formal run."),
        artifact("a_run_stdout", "execution", "_training_stdout.txt", "完整实验实际stdout。", ".venv/bin/python -m scripts.course_experiments.posttraining --output docs/technical-reviews/artifacts/fact_v2_13_10_cpu_run > docs/technical-reviews/artifacts/fact_v2_13_10_training_stdout.txt 2> docs/technical-reviews/artifacts/fact_v2_13_10_training_stderr.txt", "Exit0; saved result and printed SFT6/18, PPO12/18, DPO12/18; these policy counts are validation of the run, not language ability claims."),
        artifact("a_formal", "source_snapshot", "_formal_snapshot.json", "实际亲核的公开实测原文件完整byte副本，原文件SHA在audit记载。"),
        artifact("a_inspection", "source_snapshot", "_inspection.json", "原始PDF、当前完整前置节与XML/实际PNG的亲读定位及hash。"),
        artifact("a_torch", "source_snapshot", "_torch_original.txt", "实际installed官方PyTorch2.14.1源码绑定与sigmoid/logsigmoid/expit docstrings。"),
        artifact("a_dpo_p3", "source_snapshot", "_dpo_page3.txt", "从实际DPO v3原始PDF用pdftotext提取page3 eq1–2。"),
        artifact("a_dpo_render", "figure_render", "_dpo_page3.png", "actual原始PDFpage3渲染；reviewer已view_image亲看公式及建模假设。"),
        artifact("a_dpo_p5_6", "source_snapshot", "_dpo_pages5_6.txt", "actual DPOv3 §5.1 common-offset Definition1 / Lemma1原文。"),
        artifact("a_dpo_version", "source_snapshot", "_dpo_version.txt", "actual PDF首页arXiv2305.18290v3/29Jul2024，非候选摘要。"),
        artifact("a_dpo_limits", "source_snapshot", "_dpo_pages10_11.txt", "actual DPOv3 §6.4自动评审长度偏差和 §7 limitations。"),
        artifact("a_instructgpt_p8_9", "source_snapshot", "_instructgpt_pages8_9.txt", "actual InstructGPTv1 §3.5 scalar RM、loss、共同平移原文。"),
        artifact("a_instructgpt_version", "source_snapshot", "_instructgpt_version.txt", "actual PDF首页arXiv2203.02155v1/4Mar2022。"),
        artifact("a_instructgpt_limits", "source_snapshot", "_instructgpt_page17.txt", "actual InstructGPT p17作者关于RM學到hedging偏好的明示怀疑；不是因果证明。"),
        artifact("a_instructgpt_limitations", "source_snapshot", "_instructgpt_page19.txt", "actual InstructGPT §5.3标注者不代表所有人/模型仍会犯错之限制。"),
        artifact("a_figure", "figure_render", "_prereq_pair.png", "必要13.1前置SVG实际Inkscape渲染并亲看：共享问题两箭头，正确数字/格式A通过B未通过；13.10无图。"),
        artifact("a_svg_log", "source_snapshot", "_svg_render_stderr.txt", "成功Inkscape rendering非fatal警告记录。"),
        artifact("a_lesson", "source_snapshot", "_lesson.md", "本次实际执行的完整13.10 UTF8原文含全部空白换行。"),
        artifact("a_old_lesson", "source_snapshot", "_before_gap_fix_lesson.md", "亲核旧版完整原文，打印差距preferred_score的问题依据，未换成新正文hash。"),
        artifact("a_old_audit", "source_snapshot", "_before_gap_fix_audit_output.json", "亲核旧版实际CPU输出：修改rejected后仍误标差距0/2；供resolved issue追溯。"),
        artifact("a_records", "source_snapshot", "_cpu_run/records.json", "fresh CPU全部165原始context和825比较对。"),
        artifact("a_reward_checkpoint", "source_snapshot", "_reward_checkpoint.json", "已亲讀原reward.pt之可提交等价JSON：原full-file hash/metadata、逐tensor shape dtype raw-byte hash及完整257数值；真实往返bytes完全相同，原binary仍留本机。"),
        artifact("a_checkpoint_code", "code", "_checkpoint_portable.py", "可重跑export亲讀原binary并核对既有证据；verify只由持久JSON还原state，检查全部165题分数及freeze，无重训。"),
        artifact("a_checkpoint_export", "execution", "_checkpoint_export_receipt.json", "本次实际亲讀原checkpoint/完整metadata和5tensor导出与逐bytes往返核查receipt。", export_receipt["command"], export_receipt["result"], export_receipt["environment"]),
        artifact("a_checkpoint_verify", "execution", "_checkpoint_verify_receipt.json", "实际指定不存在checkpoint路径的JSON-only核查；全部165题660分数、stateSHA及凍結反傳均通过。", verify_receipt["command"], verify_receipt["result"], verify_receipt["environment"]),
        artifact("a_checkpoint_prior_review", "source_snapshot", "_before_portable_review.json", "证据替代前本reviewer报告原bytes，保留直接binary引用及历史hash/issue/resolution，不能据此声称未读checkpoint。"),
        artifact("a_ruff", "execution", "_ruff.txt", "三个审查Python实际Ruff检查输出。", ".venv/bin/ruff check --no-force-exclude docs/technical-reviews/artifacts/fact_v2_13_10_audit.py docs/technical-reviews/artifacts/fact_v2_13_10_build_report.py docs/technical-reviews/artifacts/fact_v2_13_10_checkpoint_portable.py", "Exit0: All checks passed!"),
        artifact("a_checker", "execution", "_checker_snapshot.txt", "当节review checker实际范围/版本/证据格式检查stdout保存副本；不自动证明论述真伪。", ".venv/bin/python scripts/check_technical_reviews.py --lesson 13.10", "Exit0; 1/1节来源与证据版本一致，实际真伪判断由此独立review记录负责。"),
    ]
    sources = [
        {"id": "s_dpo", "kind": "paper", "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
         "url": "https://arxiv.org/pdf/2305.18290v3", "version": "arXiv:2305.18290v3,29July2024 (actual downloaded PDF header)",
         "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
         "authority_reason": "原方法作者之原始论文；本节直接引用其BT reward model公式。",
         "inspection_note": "亲讀实际v3 PDF p3 §3式1–2并渲染亲看；p5–6 §5.1 Definition1/Lemma1共同offset；p10 §6.4长答评审bias；p11 limitations。BT是建模假设，MLE不自动保证unseen/human preference或truth calibration；DPO避免explicit standalone RM但本節介紹其前置RM。原PDF完整SHA与新extracts均保存。"},
        {"id": "s_instructgpt", "kind": "paper", "title": "Training language models to follow instructions with human feedback",
         "url": "https://arxiv.org/pdf/2203.02155v1", "version": "arXiv:2203.02155v1,4March2022 (actual downloaded PDF header)",
         "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
         "authority_reason": "原始RLHF方法与实证作者，明述scalar reward/同题比較及RM代理失敗與限制。",
         "inspection_note": "亲读actual PDF p8 §3.5 equation1、p9共同shift與RL阶段、p17對过度hedging起因之作者怀疑、p19 §5.3标注及模型限制。其多响应排名按prompt组打包的loss权重和本toy sampling不完全相同；只核对scalar/BT基礎及可能代理失敗，不声称toy复现其人類回饋品質。"},
        {"id": "s_torch", "kind": "official_source", "title": "PyTorch installed official functional.py and sigmoid/expit docstrings",
         "url": "https://github.com/pytorch/pytorch/blob/v2.14.1/torch/nn/functional.py", "version": "installed torch2.14.1+cpu; functional.py SHA95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b",
         "verified": True, "checked_original": True, "accessed_on": "2026-10-04",
         "authority_reason": "当前使用的官方PyTorch package原始模块及其原始C binding/docstrings，直接说明所执行API。",
         "inspection_note": "实际读取installed functional.py line2049 logsigmoid=_add_docstr(torch._C._nn.log_sigmoid,...); torch.sigmoid原始docstring为special.expit alias，special.expit docstring写1/(1+exp(-x))。URL定位发布代码，实际检查的是同版本已安装原码/原docstrings，不声称下载过网页。随审计实际CPU执行一对/四对/极端差与梯度，版本不外推。"},
        {"id": "s_posttraining", "kind": "repository_code", "title": "Current finite preference and reward model implementation",
         "path": "tiny_perceptron/posttraining.py", "sha256": digest("tiny_perceptron/posttraining.py"), "version": "current working tree inspected2026-10-04",
         "verified": True, "inspection_note": "亲读全文件。preference_loss line8–12同形状非空、负logsigmoid后mean；FiniteRewardModel以4feature+4onehot输入8维输出card scalar；freeze及梯度行为通过执行核对。"},
        {"id": "s_experiment", "kind": "repository_code", "title": "Current fixed finite-candidate CPU experiment",
         "path": "scripts/course_experiments/posttraining.py", "sha256": digest("scripts/course_experiments/posttraining.py"), "version": "current working tree inspected2026-10-04; hashes exactly match saved formal report",
         "verified": True, "inspection_note": "亲读配置、build_records、rule_best_action、split_records、_features/_pairs、_evaluate以及完整run_posttraining。查看所有165原始題與825比較，300×64 train draw，之后freeze再PPO，final whole split eval与loss/时间保存；完整fresh重跑且所有165逐題值与正式报告完全同。canonical a≤b不另测交換順序；每卡score并未读中文字。"},
        {"id": "s_math", "kind": "derivation", "title": "Independent BT probability/NLL/offset and toy-unit derivation", "verified": True,
         "details": (ROOT / BASE / f"{PREFIX}_derivation.md").read_text()},
        {"id": "s_execution", "kind": "execution", "title": "Fresh exact lesson, exercise and whole-result CPU audit", "verified": True, "artifact_id": "a_audit"},
        {"id": "s_run", "kind": "execution", "title": "Fresh complete fixed CPU posttraining run", "verified": True, "artifact_id": "a_run"},
    ]
    report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "13.10",
              "source": "course/chapters/13.md#13.10", "reviewer_task": "/root/integration_technical_coordinator/fact_v2_13_10",
              "reviewer_context": "fresh", "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
              "figure_sha256": {"course/figures/multimodal_preference_pair.svg": digest("course/figures/multimodal_preference_pair.svg")},
              "verdict": "revise" if unresolved else "pass", "claims": claims, "sources": sources,
              "artifacts": artifacts, "issues": issues,
              "checks": {
                  "factual_accuracy": {"status": "revise" if unresolved else "pass", "details": "BT/scalar奖励、NLL與相对分数含义经原论文核实；四卡/输入/标签/切分/freeze逐源码及实际CPU核对。練習改分数后差距打印仍错，详c11。" if unresolved else "BT与RM定义、数字、程式及实验实际行为均核实。旧print错误已以当前新版完整正文及CPU原例/rejected=1/共同+10重核解决；issue保留具体resolution。", "claim_ids": ["c1", "c2", "c3", "c4", "c7", "c9", "c11", "c12", "c13", "c14", "c15", "c16", "c17"]},
                  "numeric_verification": {"status": "pass", "details": "全部正文/练习概率loss逐步手算并真CPU执行；float32四位小数吻合，float64解析误差≤1e-15，mean分母及梯度确认，共同+10 toy精确相等。练习打印标签错误另属c11，不混作数值公式错。", "claim_ids": ["c5", "c6", "c8", "c10"]},
                  "figure_consistency": {"status": "pass", "details": "13.10没有本节图。必要13.1图XML与Inkscape真实PNG均亲看，900×360、共同提问箭头分至A/B、格式结论與同题偏好规则一致；纸本公式page3亦实际渲染亲看。", "claim_ids": ["c1", "c3"]},
                  "source_verification": {"status": "pass", "details": "亲读DPO v3原PDF eq1–2/共同offset/限制；InstructGPT v1原PDFscalar/shift/代理失敗限制；当前官方installed PyTorch源及docstrings。候选summary不是权威；全部repository/执行证据hash定位保存。", "claim_ids": ["c2", "c3", "c4", "c7", "c8", "c9", "c12", "c14", "c17", "c18"]},
                  "limitations": {"status": "pass", "details": "区分BT假设和真实偏好、相对RM score与truth calibration、实数offset与浮點近似；固定seed合成四卡structured features/0token targets/程序标签不证明中文算术生成或人类RLHF质量。全165題/825比較与family不重疊核实；代价历史为被抽minibatch，时间只对应单次CPU训练/评估/保存。", "claim_ids": ["c3", "c4", "c8", "c9", "c14", "c15", "c16", "c17", "c18"]},
              }}
    target = ROOT / "docs/technical-reviews/13.10.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target.relative_to(ROOT)}: {report['verdict']}, {len(claims)} claims, {len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
