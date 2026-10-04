"""Persist this reviewer's independently checked claims and evidence hashes."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_07_18_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    body = dict(sections(ROOT / "course/chapters/07.md"))["7.18"]
    audit = json.loads((OUT / f"{PREFIX}audit.json").read_text())
    fetch = json.loads((OUT / f"{PREFIX}fetch.json").read_text())
    sources = []
    artifacts = []

    def artifact(identifier, name, kind, description, execution=None):
        path = OUT / f"{PREFIX}{name}"
        item = {"id": identifier, "kind": kind, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "description": description}
        if execution:
            item.update({key: execution[key] for key in ("command", "result", "environment")})
        artifacts.append(item)

    artifact("a_section", "source_7.18.txt", "source_snapshot", "使用 checker.sections 保留 7.18 全部原始 LF 至 EOF 的 UTF-8 切片。")
    for lesson in ("7.1", "5.8", "13.3", "C.6"):
        artifact(f"a_pre_{lesson}", f"source_{lesson}.txt", "source_snapshot", f"完整閱讀的明示前置 {lesson}；提供概念上下文，沒有沿用前置審查結論。")
    artifact("a_audit", "audit.json", "execution", "原節程式精確輸出、實際資料/答案遮罩與 DPO 梯度/固定參考的 CPU 核對，以及原 SVG XML 與渲染紀錄。", audit)
    artifact("a_fetch", "fetch.json", "execution", "版本固定原論文 HTTPS 取得、PDF 與 pdftotext 雙 SHA/實際命令及環境。", fetch)
    artifact("a_stdout", "stdout.txt", "source_snapshot", "精確原節 Python fence 的三行 stdout。")
    artifact("a_snippet", "snippet.py", "code", "從 checker.sections 原節取出的原始 Python fence，實際執行且輸出精確相等。")
    artifact("a_audit_code", "audit.py", "code", "當次 CPU/API/字典與 SVG 核對程式。")
    artifact("a_fetch_code", "fetch.py", "code", "當次版本固定原文取得與 PDF 轉文字程式。")
    artifact("a_builder", "build_report.py", "code", "本獨立 reviewer 逐主張判斷與報告持久化程式。")
    artifact("a_notes", "inspection.txt", "derivation", "親讀原文位置/公式、逐步算術與格式推導、圖像親看紀錄及證據範圍。")
    artifact("a_render", "posttrain_signals.png", "figure_render", "以 /usr/bin/chromium + Playwright 將原 SVG XML 渲染到 820x490，已親看所有箭頭/文字/數字。")
    ruff_execution = {
        "command": ".venv/bin/ruff check --no-force-exclude docs/technical-reviews/artifacts/fact_v2_07_18_audit.py docs/technical-reviews/artifacts/fact_v2_07_18_fetch.py docs/technical-reviews/artifacts/fact_v2_07_18_snippet.py docs/technical-reviews/artifacts/fact_v2_07_18_build_report.py > docs/technical-reviews/artifacts/fact_v2_07_18_ruff.txt",
        "result": (OUT / f"{PREFIX}ruff.txt").read_text().strip(),
        "environment": {"ruff": "0.16.9", "python": audit["environment"]["python"], "device": "CPU"},
    }
    artifact("a_ruff", "ruff.txt", "execution", "明確對所有新增 Python 執行 Ruff（包含 import 排序）的通過輸出。", ruff_execution)
    paper_info = {
        "instructgpt": ("s_instructgpt", "Training language models to follow instructions with human feedback", "2203.02155v1, 2022-03-04", "OpenAI 原作者發表的 InstructGPT 方法論。", "親讀 3.1/Fig.2、3.5/Eqs.(1)-(2)、5.2-5.3/Fig.9；示範 SFT、人工比較 RM、scalar reward/PPO，含 KL、ptx 與偏好人口/錯答限制。"),
        "deepseek_r1": ("s_r1", "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", "2501.12948v1, 2025-01-22", "DeepSeek-AI 原作者對其 R1-Zero/R1 配方的版本固定論文。", "親讀 2.1-2.3 全流程，2.2.1/Eqs.(1)-(3) GRPO、2.2.2 rule accuracy/format、2.3.1 cold-start、2.3.2-2.3.4 後續階段；亦讀 2.2.4/Table2/Fig2、3 完整評估設置以限定範圍。未以候選庫 v2 替 v1 引用，未宣稱有完整原訓練配置/逐題資料。"),
        "dpo": ("s_dpo", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "2305.18290v1, 2023-05-29", "Rafailov 等人發表 DPO 的原論文。", "親讀 3-4/Eqs.(1)-(7) 與 DPO outline：BT 偏好、KL reference、partition cancellation、direct logistic loss、given reference/SFT 初始化；只支持原始 reference-based DPO，不涵蓋所有後續變體或保證資料品質。"),
        "ppo": ("s_ppo", "Proximal Policy Optimization Algorithms", "1707.06347v1, 2017-07-20", "Schulman 等人定義 PPO 的原論文。", "親讀 2-3/Eqs.(1),(7)、5/Eq.(9)/Algorithm1：取樣、advantage、old-policy ratio、多輪更新；PPO 不限定回饋出自人，old policy 不等於固定 SFT-reference。"),
    }
    for fetched in fetch["originals"]:
        name = fetched["name"]
        identifier, title, version, reason, note = paper_info[name]
        sources.append({"id": identifier, "kind": "paper", "title": title, "url": fetched["url"], "version": version, "verified": True, "checked_original": True, "accessed_on": "2026-10-04", "authority_reason": reason, "inspection_note": note, "artifact_ids": [f"a_{name}_pdf", f"a_{name}_txt"]})
        artifact(f"a_{name}_pdf", f"{name}.pdf", "source_snapshot", f"實際取得並親讀定位的原始 {version} PDF。")
        artifact(f"a_{name}_txt", f"{name}.txt", "source_snapshot", "對同版本 PDF 執行 pdftotext -layout 的完整文字；供定位查核，非摘要。")
    code_info = [
        ("s_data", "tiny_perceptron/data.py", "ByteTokenizer/render_chat，lines14-28,54-70；按 role 建立 assistant byte+EOS targets，prompt/角色 -100。"),
        ("s_model", "tiny_perceptron/model.py", "loss_sum/masked_loss，lines92-105；cross_entropy sum 除有效 target count。"),
        ("s_alignment", "tiny_perceptron/alignment.py", "sequence_log_probability/dpo_loss，lines28-41；答案 sum、policy/reference margin 與 reference detach。"),
        ("s_postrunner", "scripts/course_experiments/posttraining.py", "lines289-290 固定 SFT copy，413-445 DPO loop/no_grad reference/hash invariant；只讀配方上下文，未重跑正式實驗。"),
    ]
    for identifier, path_name, note in code_info:
        path = ROOT / path_name
        snapshot_name = f"frozen_{identifier}.txt"
        (OUT / f"{PREFIX}{snapshot_name}").write_bytes(path.read_bytes())
        artifact(f"a_{identifier}", snapshot_name, "source_snapshot", f"親讀的目前 repo 原碼完整快照：{path_name}。")
        sources.append({"id": identifier, "kind": "repository_code", "title": path_name, "path": path_name, "sha256": sha(path), "version": "working tree inspected 2026-10-04", "verified": True, "inspection_note": note})
    sources.extend([
        {"id": "s_snippet", "kind": "repository_code", "title": "7.18 原 Python 資料卡範例", "path": f"docs/technical-reviews/artifacts/{PREFIX}snippet.py", "sha256": sha(OUT / f"{PREFIX}snippet.py"), "version": hashlib.sha256(body.encode()).hexdigest(), "verified": True, "inspection_note": "逐行親讀且從完整 section slice 精確抽出；只有 literal dict/print，沒有抽樣、模型或更新。"},
        {"id": "s_derivation", "kind": "derivation", "title": "1+2、只回數字格式與新任務判準推導", "verified": True, "details": "1+2=3；4≠3。3 是純十進位數字；答案是3喔！雖表達同結果，含非數字，因此不符原格式。以本題正確給1、錯誤給0則4得0，但原碼只是填入literal reward。新任務要求解釋合併一與二的過程，兩份旧字串都未解釋，僅交换偏好標籤不充分。詳細逐步推導見 a_notes。"},
        {"id": "s_execution", "kind": "execution", "title": "原節與直接 CPU 軟體執行", "verified": True, "artifact_id": "a_audit"},
    ])
    claims = []

    def evidence(source_id, locator, supports):
        return {"source_id": source_id, "locator": locator, "supports": supports}

    def claim(identifier, kind, statement, location, refs, scope, artifact_ids=None, verification=None):
        item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "status": "verified", "evidence": refs, "artifact_ids": artifact_ids or [], "scope": scope}
        if verification:
            item["verification"] = verification
        claims.append(item)

    claim("c1", "concept", "示範資料提供期望模型學習的回答。", "第1-2段；demonstration 卡", [evidence("s_instructgpt", "3.1 Step1/Fig2", "人工示範 desired behavior 用於 supervised policy。")], "資料是監督目標，格式本身不證明能力。")
    claim("c2", "concept", "偏好資料比較同題回答並標示較佳與較差。", "第2段及 preference 卡", [evidence("s_dpo", "3, Eqs.(1)-(2); 4 DPO outline", "x 下 yw≻yl 比較與 offline labeled pairs。")], "偏好依題目與標註判準，不等同數學真值或跨任務的絕對品質。")
    claim("c3", "concept", "Reward 是更新用的數字訊號，本身不保證真實品質。", "第2段 reward 定義與標準提醒", [evidence("s_r1", "2.2.1 Eqs.(1)-(3); 2.2.2", "reward/advantage 決定 RL 更新方向，規則有 accuracy/format 限制。"), evidence("s_instructgpt", "3.5 Eq.(2); 5.2-5.3/Fig9", "proxy preference、特定回饋人群與錯答限制。")], "不是情緒；信號支持範圍取決於評分器/標註規則。")
    claim("c4", "software", "範例實際印出示範3、3勝過附加句，以及字面錯答4的reward0，並未真正抽樣模型。", "Python fence 及其後輸出說明", [evidence("s_snippet", "lines1-6", "逐行 literal dict 和 print；不存在 sampling API。"), evidence("s_execution", "audit.json expected_stdout/observed_stdout", "精確執行相符。")], "Python3.13.5；只是三份人工寫的字典，不是可運作的訓練器或 reward 評分程式。", ["a_audit", "a_snippet", "a_stdout"], {"method": "executed", "expected": audit["expected_stdout"], "observed": audit["observed_stdout"], "details": "按原節完整 UTF-8 section 抽取唯一 Python fence，compile/exec 後捕获 stdout 精確比較。"})
    claim("c5", "numeric", "1+2的正確數字為3；兩篇偏好答文都表達3但附加句違反只回數字；4是錯答。", "第1段；資料卡算例及其後解释", [evidence("s_derivation", "inspection.txt Arithmetic and format derivation", "逐步加法、格式與0/1正確性規則。")], "單一透明算例；reward0已填字典，不能由此推論通用評分器。", ["a_notes", "a_audit"], {"method": "hand_calculation", "expected": "1+2=3；4≠3；3符合純數字，附加句不符合。", "observed": "逐步推導得到3，格式逐字检查與原文一致；CPU assert也獨立核對1+2及4≠3。", "tolerance": "整數及字串判準精確相等，不需浮點容差。", "details": "把1與2合併得到3；比較3、4；按『只回數字』檢查整串字元，不以抽取其中的3代替完整格式。"})
    claim("c6", "concept", "SFT 以理想示範回答繼續監督式微調。", "SFT 定義段；圖 SFT 框", [evidence("s_instructgpt", "3.1 Step1; 3.5 SFT", "在 pretrained GPT-3 上 supervised demonstrations。"), evidence("s_dpo", "3 SFT phase", "maximum likelihood on task demonstration dataset。")], "提高示範回答機率是最佳化目標，不是每次更新或每個例子都必然改善的保證。")
    claim("c7", "concept", "原始 DPO 可直接用偏好對最佳化回答策略，省去另行訓練獎勵模型。", "DPO 定義段", [evidence("s_dpo", "4, Eqs.(4)-(7) and DPO outline", "reparameterization取消partition並將 preference loss 寫成policy/ref ratios；無另行RM。")], "原始reference-based DPO；仍需可用比較標註，不等於沒有implicit reward。")
    claim("c8", "concept", "本節原始 DPO 的固定參考是偏好訓練起點保留的不更新模型。", "固定參考粗體定義", [evidence("s_dpo", "3 Eq.(3); 4 Eq.(7), DPO outline", "given πref、SFT初始化與policy/ref log ratios。"), evidence("s_postrunner", "lines289-290,413-445", "repo fixed copy/no_grad/ref hash assertion。")], "指原始配方；SFT reference不可與PPO每輪old policy混為一談；DPO不要求全部後續變體都使用此實體副本。", ["a_audit", "a_s_postrunner"])
    claim("c9", "concept", "偏好對可先訓練一個獎勵模型以估計更合適的回答，之後讓策略取樣得到評分並更新。", "獎勵模型與RLHF段", [evidence("s_instructgpt", "3.1 Steps2-3; 3.5 Eqs.(1)-(2)", "比較資料擬合 scalar RM，sampled response在bandit環境受評分並以PPO更新。")], "RM預測給定偏好人群/規則，不是保證每次判斷正確的裁判。")
    claim("c10", "concept", "RLHF 是 reinforcement learning from human feedback，標明人類回饋来源與流程。", "RLHF 全名與來源說明", [evidence("s_instructgpt", "Abstract; 3.1/Fig2", "原文human demonstrations/rankings -> RLHF。")], "通常可由RM轉述人類比較；不是每次RL取樣都直接由真人打分。")
    claim("c11", "concept", "PPO 是近端策略最佳化的更新方法，與 RLHF 名詞意義不同。", "PPO 定義與兩者區別", [evidence("s_ppo", "title; 2-3 Eq.(7); 5 Algorithm1", "PPO以old-policy samples與advantages更新，不限定人類回饋。"), evidence("s_instructgpt", "3.1 Step3", "PPO是該RLHF流程使用的策略更新方法。")], "本節未教完整PPO公式；PPO不是所有RLHF唯一方法，亦不代表reward來源。")
    claim("c12", "concept", "程式驗算產生的可驗證回饋，不會僅因使用 PPO 就變為人類回饋。", "C.6 連結句；圖來源標籤", [evidence("s_r1", "2.2.2 Accuracy/Format rewards", "deterministic math/compiler test cases以rule-based feedback評分。"), evidence("s_ppo", "5 Algorithm1", "update algorithm與reward來源分離。")], "只在可可靠驗證任務成立；R1所用為GRPO，本節沒有聲稱R1使用PPO。")
    claim("c13", "concept", "SFT、PPO、DPO 是可選與可組合的路線，不要求每個模型按序全部跑過。", "圖前段及圖末標籤", [evidence("s_dpo", "4 Eq.(7)/DPO outline", "直接偏好訓練取代explicit RM/RL。"), evidence("s_r1", "2.1-2.3", "without-SFT R1-Zero與cold-start R1不同路線。")], "典型個別配方有順序和前提；不是聲稱各步驟可任意交換而結果相同。")
    claim("c14", "concept", "後訓練的目標可包含解題步驟或答案正誤，不只語氣；成功仍須新題檢查。", "後訓練能力條件段首三句", [evidence("s_r1", "2.2.2; 2.3.1-2.3.2; 3 Evaluation Setup", "rule correctness、long-CoT demonstrations和對新benchmark問題的另行評估。"), evidence("s_instructgpt", "5.3 Models/Fig9", "最佳化後仍可錯答，方法/資料不等於品質保證。")], "條件性目標/可能性說明，沒有聲稱本地測得能力提升或重現R1；原論文aggregate分數不可替代完整配置與逐題本地證據。", ["a_notes"])
    claim("c15", "concept", "R1-Zero 的配方直接從已預訓練底座以回饋做 RL，所以 SFT 並非所有 RL 必須的前一步。", "DeepSeek-R1 兩條路線中的第一條及末句", [evidence("s_r1", "2.1; 2.2.1 Eqs.(1)-(3); 2.2.2", "R1-Zero/V3-Base without any SFT data、GRPO及rule rewards。")], "指該大模型特定配方；不表示scratch model、不保證任意微型模型可省SFT達同能力。")
    claim("c16", "concept", "R1 的另一條路線先用少量冷啟動理想解題示範微調，再接後續訓練以改善起始可讀性。", "冷啟動資料定義與第二條路線", [evidence("s_r1", "2.3.1 Cold Start; 2.3.2-2.3.4", "thousands long-CoT examples起始SFT；其後reasoning RL/rejection SFT/general RL。")], "少量相對large-scale recipe；不全由真人寫，不是只含SFT→RL兩步的完整R1 pipeline。")
    claim("c17", "concept", "SVG 的示範、偏好、回饋三張卡及偏好→RM分支與正文方法區分一致。", "posttrain_signals.svg 全圖", [evidence("s_instructgpt", "3.1/Fig2", "demo SFT、compare RM、reward update。"), evidence("s_dpo", "4 Eq.(7)", "direct preference route without standaloneRM。"), evidence("s_ppo", "5 Algorithm1", "reward-guided update method。")], "方向與標籤是資料/方法示意，沒有比率軸、效能數據或強制執行先後；SFT概率說明為最佳化目標。", ["a_render", "a_audit", "a_notes"])
    claim("c18", "concept", "改成一句話解釋加法後，應重新寫示範與判偏好；僅交換原兩答標籤不保證符合新任務。", "末段練習", [evidence("s_instructgpt", "3.1 Step1", "示範是prompt distribution上desired behavior。"), evidence("s_dpo", "3 Eqs.(1)-(2)", "偏好以同一prompt x為條件。"), evidence("s_derivation", "inspection.txt Arithmetic and format derivation", "原答文3/答案是3喔皆沒有加法說明。")], "依明示新任務判準；真人/程式/模型打分要按實際來源辨別，不能由缩寫推定。", ["a_audit", "a_notes"])
    claim("c19", "concept", "真人比較、程式驗算、另一模型評分的直接回饋來源有別。", "末段練習最後一句", [evidence("s_instructgpt", "3.1 Steps2-3", "真人標註可轉為RM估計的回饋。"), evidence("s_r1", "2.2.2; 2.3.3 Reasoning data", "規則驗算與feeding ground-truth/model predictions into DeepSeek-V3 for judgment皆有明記。")], "說清楚直接評分者與其上游標註來源；模型評分可間接承接人類偏好，不能把所有model reward斷言為與人無關。")
    conceptual = [item["id"] for item in claims if item["kind"] == "concept"]
    report = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "7.18",
        "source": "course/chapters/07.md#7.18",
        "reviewer_task": "/root/integration_technical_coordinator/fact_v2_07_18",
        "reviewer_context": "fresh",
        "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "figure_sha256": {"course/figures/posttrain_signals.svg": sha(ROOT / "course/figures/posttrain_signals.svg")},
        "verdict": "pass",
        "claim_classification": {"concept": "資料訊號/原始方法/條件性可用訓練目標", "numeric": "透明加法與格式判準", "software": "實際原節字典與print行為", "empirical": "本節沒有測得品質/速度/記憶體比較；不把論文配方或條件性能力目標冒稱為本地實測。原始論文逐題/完整訓練配置缺失僅限制外推，不用其aggregate結果驗證未主張的本地效果。"},
        "prerequisites_read": audit["sources"],
        "claims": claims,
        "sources": sources,
        "artifacts": artifacts,
        "issues": [],
        "checks": {
            "factual_accuracy": {"status": "pass", "details": "親讀 InstructGPT/DPO/PPO/R1 的版本固定原文公式與方法段落，逐主張區分資料來源、直接偏好與RM/RL路線；R1明確是GRPO。原節CPU程式精確執行。", "claim_ids": [item["id"] for item in claims]},
            "numeric_verification": {"status": "pass", "details": "1+2=3與4錯答手算精確相符；附加句內容對而格式不符合。字典0是人工給值，沒有假稱模型/評分器實驗；答案遮罩/分母/shape/dtype也以真CPU核對。", "claim_ids": ["c4", "c5"]},
            "figure_consistency": {"status": "pass", "details": "親讀完整XML並以Chromium/Playwright渲染親看820x490 PNG；三主箭頭和偏好卡RM分支方向、數字3、來源與可選路線標籤均與正文相符，沒有隱含axis或數字比較。", "claim_ids": ["c17"]},
            "source_verification": {"status": "pass", "details": "四原論文版本固定HTTPS原PDF已下載/轉字/親讀，記錄原版修訂ID、日期、公式及限制；候選庫R1 v2未作教材v1權威來源。原码與完整原节/前置均有SHA与持久快照。", "claim_ids": conceptual + ["c4", "c5"]},
            "limitations": {"status": "pass", "details": "資料卡不抽樣/訓練；reward非品質保證；標準DPO需固定參考而PPO old-policy另有用途；R1冷啟動只是多階段一部分。無本地empirical能力/速度結論，CPU API/梯度核對不支持R1能力或方法優勝外推。", "claim_ids": [item["id"] for item in claims]},
        },
    }
    (ROOT / "docs/technical-reviews/7.18.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lesson_id": "7.18", "verdict": report["verdict"], "claims": len(claims), "source_sha256": report["source_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
