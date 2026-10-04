"""Write only the personally reviewed G.4 snapshot and its original-source evidence."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PREFIX = "fact_finish_g_4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


SOURCE_DATA = [
    (
        "switch",
        "paper",
        "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        "arXiv:2101.03961v3",
        "Fedus/Zoph/Shazeer original sparse Transformer design",
        "親讀Introduction、§2、§2.1式(1)-(2)、Figure2圖說及§2.2開頭，original extracted-text lines114-405。Dense FFN replaced by per-token top-k expert FFNs; y=sum_{i in T}p_i E_i(x). §2仍保留各expert獨立參數並分散存放；sparsity指每token啟動權重，非刪除全部未啟動權重，也不是所有歷史MoE都稀疏。Router/通訊/容量皆有成本。",
    ),
    (
        "cache",
        "official_docs",
        "Hugging Face Transformers: Caching",
        "Transformers v4.57.1 tag, original cache_explanation.md",
        "Transformers maintainers' original causal KV-cache documentation",
        "本人親讀全份原Markdown。Caching、Attention matrices、Cache class、Cache position：每層保留前文K/V，append新K/V，當前Query仍讀完整cache。固定權重/前文/因果位置時未來不改舊特徵；mask與位置要對齊。不是免除新token工作或保證延遲。文中示意mask乘法不是本審數學依據。",
    ),
    (
        "flash",
        "paper",
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "arXiv:2205.14135v2",
        "Dao et al. original algorithm and IO complexity analysis",
        "親讀§2.1-2.2、§3.1完整tiling/recomputation、Algorithm1、Theorem1，以及§3.2 Theorem2/Proposition3，text172-395。Block Q/K/V由HBM載SRAM，max/normalizer重縮放避免完整S/P中間表寫HBM；仍O(N²d)算術、O(N)額外記憶。IO定理條件d≤M≤Nd，實際收益需適配kernel/hardware；CPU Python分塊不能證GPU速度，exact目標不保證浮點bitwise一致。",
    ),
    (
        "quant",
        "paper",
        "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference",
        "arXiv:1712.05877v1",
        "Jacob et al. original affine integer quantization formulation",
        "親讀Introduction storage/runtime區分、§2.1完整scheme與QuantizedBuffer、§2.2式(1)-(6)，text35-180。r=S(q−Z)，q是B-bit integer，S/Z另需存。只有真正低位元/打包表示才省儲存，FP32內只round不等於壓縮；intro說weights-only主要省storage而部分位移法在現有hardware little benefit。精度/範圍可能受損，不保證品質或速度。",
    ),
    (
        "distill",
        "paper",
        "Distilling the Knowledge in a Neural Network",
        "arXiv:1503.02531v1, 9 March 2015",
        "Hinton/Vinyals/Dean original knowledge-distillation paper",
        "親讀§1、§2及式(1)、soft-target CE/T² discussion，text35-164。大型教師或ensemble提供soft probabilities給較小學生，亦可混真實hard labels；學生容量可能無法完全擬合。此文支持distribution訊號，教師生成文字另以MiniLLM原文核。蒸餾不是能力或教師正確性保證。",
    ),
    (
        "minillm",
        "paper",
        "MiniLLM: On-Policy Distillation of Large Language Models",
        "arXiv:2306.08543v6, 31 January 2026",
        "Gu et al. original LLM distillation study defining accessible teacher signals",
        "親讀abstract、完整§1、§2/§2.1開頭，text1-175；§1明确black-box只取teacher-generated texts、white-box可取output distribution/intermediate hidden states。仅用作G.4答案或分布的原始定義依據；不把MiniLLM reverse-KL/on-policy方法冒稱本章forward-KL實作。",
    ),
    (
        "ppo",
        "paper",
        "Proximal Policy Optimization Algorithms",
        "arXiv:1707.06347v2, 28 August 2017",
        "Schulman et al. original PPO paper and actor-critic algorithm",
        "親讀text1-272：§2.1式(1)-(2)、§2.2、§3式(6)-(7)、§4、§5式(9)-(12)、Algorithm1。r=πθ(a|s)/πold(a|s)；Lclip=min(rA,clip(r,1−ε,1+ε)A)。採樣資料old與advantage固定供K epochs；V(s)用作advantage基準。多步GAE比R−V复杂，裁切不硬锁全部ratio；原文另有KL-penalty PPO，本節具體教PPO-Clip。",
    ),
    (
        "instructgpt",
        "paper",
        "Training language models to follow instructions with human feedback",
        "arXiv:2203.02155v1",
        "Ouyang et al. original InstructGPT human-feedback pipeline",
        "親讀§3.1 Steps1-3 text294-321、§3.5 SFT/RM/RL與式(1)-(2) text413-516。Pretrained LM supervised on示範，真人labelers比較，scalar RM學score differences，PPO由RM reward及相對SFT KL更新策略。支持human來源/更新法與RM角色；只是典型配方，不主張所有RLHF必须PPO或另設RM。",
    ),
    (
        "dpo",
        "paper",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "arXiv:2305.18290v3",
        "Rafailov et al. original DPO derivation and preference loss",
        "親讀§3-4 text133-262，並重新讀182-245閉式變換。BT式(1)-(2)；reward−βKL(πθ||πref)式(3)，πref為初始SFT；policy/reward map式(4)-(5)、偏好式(6)、DPO直接policy loss式(7)。扣固定reference同題chosen/rejected log-prob差，不需另擬合顯式RM。品質/支持/capacity/optimization条件仍存在，非任意PPO都產生同權重。",
    ),
    (
        "rag",
        "paper",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "arXiv:2005.11401v4, 12 April 2021",
        "Lewis et al. original retrieval-conditioned generation paper",
        "親讀§1及§2-2.3 text39-190；pη(z|x)找top-K passages，pθ(yi|x,z,y<i)依文件生成，§2.3 concatenate input/retrieved text。原文sequence/token latent marginalization與joint training不是G.4全部實作宣稱；只核檢索→內容條件→生成。检索不等於正確/可信作答或永久學入權重。",
    ),
    (
        "function_calling",
        "official_docs",
        "OpenAI API: Function calling",
        "Live unversioned official guide retrieved 2026-10-04; original HTML SHA retained",
        "OpenAI original API guide defines model requests and application execution",
        "親讀text1-119 How it works及five-step flow、2015-2050 schema best practices、3640-3694 strict mode。Model提出tool/name/input，application execute後回填output；請求文字不等於執行。Enums/object schema限制invalid states，strict结构也不證參數符合問題、授權或工具結果可信。API默認等細節只限此頁版本，G.4沒有該細節主張。",
    ),
    (
        "model_spec",
        "official_docs",
        "OpenAI Model Spec",
        "Dated official 2025-12-18 HTML, retrieved 2026-10-04",
        "Provider's original behavior specification and explicit model-training target",
        "親讀Overview text178-209（明确train to align with spec）、Assume best intentions720-739、uncertainty/clarification2153-2248、illicit behavior/refusal1382-1440、tool-output trust795-827。行为目标含有用協助、適當澄清、拒絕邊界；仅支持本節實用概括，非完整alignment理論或toy部署安全實證，原頁自己說production models未完全遵循。",
    ),
]

CLAIM_DATA = [
    (
        "dense",
        "Dense在此表的FFN基準中每token用同一完整計算零件。",
        "68 Dense半句",
        [("switch", "Figure2 caption; §2.1", "Dense FFN replaced by independently routed expert FFNs.")],
        "Dense在linked15.1指非sparse-expert routing的共享FFN，非所有Dense架构相同或所有位置只算一次。",
    ),
    (
        "moe",
        "本節稀疏MoE讓每token只選少數expert處理。",
        "68 MoE半句",
        [("switch", "§2.1 equations(1)-(2)", "Top-k indices T select gate-weighted E_i(x).")],
        "針對linked15.4的sparse top-k MoE；不泛稱所有歷史MoE均稀疏或跳過expert自動加速。",
    ),
    (
        "kv",
        "生成下一token保存已處理前文的每層K/V，減少舊位置重算。",
        "69 KV cache",
        [("cache", "Caching; Attention matrices; Cache class", "Cached past KV is reused and new KV appended.")],
        "固定因果前文/權重/位置/mask；仍需新token計算和對過去attention，cache占記憶。下一字按linked16.3為下一token。",
    ),
    (
        "flash",
        "FlashAttention分塊減少大型中間表儲存與HBM搬移。",
        "70 Flash Attention",
        [
            (
                "flash",
                "§3.1 Algorithm1/Theorem1; §3.2 Theorem2",
                "Tiling/online statistics/recomputation avoid full S/P in HBM and reduce IO.",
            )
        ],
        "完整attention仍二次算術；GPU kernel/hardware條件影響收益，CPU分塊不證GPU速度，exact非bitwise保證。",
    ),
    (
        "quant",
        "量化映到較少離散格子，低位元表示可省儲存。",
        "71 quantization",
        [("quant", "§2.1 equation(1), QuantizedBuffer", "r=S(q−Z), B-bit integer q plus metadata.")],
        "需實際低位元/打包及保存scale/zero-point；FP32內round不省bytes，有限格子有精度/範圍損失。",
    ),
    (
        "distill",
        "較小學生可從教師答案或輸出分布學習。",
        "72 distillation",
        [
            (
                "distill",
                "§1; §2 equation(1) and soft-target CE",
                "Distribution distillation transfers teacher probabilities to smaller model.",
            ),
            ("minillm", "§1 first paragraph", "Defines teacher-generated texts and white-box output distributions."),
        ],
        "較小是此表壓縮目的而非所有蒸餾必要定义；學生容量有限、教師可錯，分布對欄需詞表對齊，無能力保證。",
    ),
    (
        "rlhf",
        "RLHF用人類判斷提供回饋指引RL，PPO描述可選更新方法。",
        "73 RLHF及85來源/方法句",
        [
            ("instructgpt", "§3.1 Steps2-3; §3.5 RM/RL", "Actual human comparisons train RM and PPO updates policy."),
            ("ppo", "§2-3; Algorithm1", "Policy optimizer is independent of who supplied reward."),
        ],
        "不把任何PPO或程式驗算叫RLHF；典型RM+PPO不等於所有RLHF唯一配方，人類偏好非自動真實品質。",
    ),
    (
        "ppo",
        "本節PPO以採樣時舊機率對照目前策略，用裁切目標更新。",
        "74 PPO",
        [
            (
                "ppo",
                "§3 equations(6)-(7); §5 Algorithm1",
                "r=πθ/πold and min of clipped/unclipped advantage objectives after sampling.",
            )
        ],
        "本節linked教PPO-Clip；另有KL-penalty變體。裁切不是硬鎖所有機率或更新品質保證；逐token多步LLM需另算advantage。",
    ),
    (
        "reward",
        "偏好RM先學回答比較，再估問題/回答標量分數。",
        "75 reward model",
        [
            (
                "instructgpt",
                "§3.5 Reward modeling equation(1)",
                "Scalar rθ(x,y) fitted to preferred/dispreferred comparison.",
            ),
            ("dpo", "§3 equations(1)-(2)", "BT models comparison from score differences."),
        ],
        "本章偏好RM而非所有reward定義；分数平移/尺度與排序不等於校準正確率。",
    ),
    (
        "policy",
        "策略是條件動作或回答的機率選擇規則。",
        "76 policy",
        [("ppo", "§2.1 equation(1)", "πθ(a|s) is the stochastic policy.")],
        "有限選卡action是整篇預寫回答；自回歸action可为token，不混稱兩個空間。",
    ),
    (
        "critic",
        "價值模型估預期得分，可作比較此次表現的baseline。",
        "77 value/critic/baseline",
        [
            (
                "ppo",
                "§5 equations(9)-(12), Algorithm1",
                "V(s) is value baseline in advantage estimation with a separate value target.",
            )
        ],
        "本章one-step bandit估RM預期分，多步則估後續return。baseline可不用網路，critic與評已選回答的RM不同。",
    ),
    (
        "reference",
        "偏好/RL後訓練保留起點參考策略衡量回答傾向偏移。",
        "78 fixed reference",
        [
            (
                "dpo",
                "§3 equation(3) and πref definition; §4 equation(7)",
                "Initial SFT reference enters KL and direct preference log ratios.",
            ),
            ("instructgpt", "§3.5 equation(2)", "RL is constrained relative to SFT policy."),
        ],
        "起点按linked13.12/13.14為偏好更新的SFT副本，不指整個預訓練起点；old每輪變而reference固定，KL非語意可靠性。",
    ),
    (
        "safety",
        "安全對齊以辨認可完成、需澄清、需拒絕情境為教學目標。",
        "79 safety alignment",
        [
            (
                "model_spec",
                "Overview; Assume best intentions; Consider uncertainty/clarification; Do not facilitate illicit behavior",
                "Original model-training target includes useful completion, appropriate clarification and refusal boundaries.",
            )
        ],
        "實用行為目標不是完整AI alignment理論，邊界由任務定義而成效需測；不聲稱toy模型部署安全，澄清非預設阻擋。",
    ),
    (
        "rag",
        "RAG先找外部資料，再依檢索內容條件作答。",
        "80 RAG",
        [
            (
                "rag",
                "§2 opening; §2.1; §2.3",
                "Retriever obtains passages and generator conditions on query plus retrieved text.",
            )
        ],
        "概括流程而非聲稱完整original latent-document算法；資料/生成仍可能錯，不是永久學入權重。",
    ),
    (
        "tool",
        "Tool call由模型提出工具要求，程式按規則檢查再執行。",
        "81 tool call",
        [
            (
                "function_calling",
                "How it works; Tool calling flow; schema best practices; Strict mode",
                "Model requests named tool/input; application executes and returns output, schemas constrain structure.",
            )
        ],
        "本章本地add/multiply入口；請求文字非執行，schema非正確參數/授權/結果可信證據；執行也可由远端服務程式完成。",
    ),
    (
        "speed",
        "檔案變小不保證所有硬體都變快。",
        "83 檔案變小句",
        [
            (
                "quant",
                "Introduction weights-only storage versus runtime discussion",
                "Weights-only concerns storage and some low-bit arithmetic has little hardware benefit.",
            ),
            ("flash", "§2.1 Hardware Performance", "Runtime depends on bandwidth, arithmetic intensity and hardware."),
        ],
        "限制一般推論，无實測速度數字；CPU流程不证明GPU加速。",
    ),
    (
        "expert_storage",
        "少啟動expert不代表全部未啟動權重可省略儲存。",
        "83 expert儲存句",
        [
            (
                "switch",
                "§2 final paragraph; §2.1 equations(1)-(2); Figure3 terminology",
                "Distinct expert weights persist across distributed devices while routing activates subsets.",
            )
        ],
        "可在不同GPU/CPU/磁碟offload，非要求同GPU全駐；刪除權重改變模型而非只少執行。",
    ),
    (
        "dpo",
        "DPO可直接由同題偏好對更新，保留起點固定參考。",
        "85 DPO句及明示G.2入口",
        [
            (
                "dpo",
                "§4 equations(4)-(7)",
                "Direct policy preference loss uses chosen/rejected policy/reference log-probabilities without separately fitted explicit RM.",
            )
        ],
        "原始DPO可靠資料/reference支持/capacity之條件；不泛稱所有偏好訓練為DPO或PPO/DPO必須先後跑。",
    ),
]


def main():
    read_path = BASE / f"{PREFIX}_read_G.4.md"
    saved = read_path.read_bytes()
    assert saved == dict(sections(ROOT / "course/glossary.md"))["G.4"].encode()
    receipts = json.loads((BASE / f"{PREFIX}_fetch_receipts.json").read_text())
    audit = json.loads((BASE / f"{PREFIX}_cpu_audit.json").read_text())
    sources = []
    for identifier, kind, title, version, authority, note in SOURCE_DATA:
        receipt = next(item for item in receipts if item["name"] == identifier)
        assert sha(BASE / receipt["path"]) == receipt["sha256"]
        sources.append(
            {
                "id": identifier,
                "kind": kind,
                "title": title,
                "url": receipt["requested_url"],
                "version": version,
                "verified": True,
                "checked_original": True,
                "accessed_on": "2026-10-04",
                "authority_reason": authority,
                "inspection_note": note,
                "snapshot_path": f"docs/technical-reviews/artifacts/{receipt['path']}",
                "snapshot_sha256": receipt["sha256"],
            }
        )
    for identifier, relative, note in [
        (
            "repo_core",
            "tiny_perceptron/posttraining.py",
            "本人親讀全86行；8-49 preference/R−V/exact-KL/PPO式，53-86有限policy/RM/value網路；structured features輸出4個candidate logits，沒有詞表decoder。",
        ),
        (
            "repo_run",
            "scripts/course_experiments/posttraining.py",
            "本人分段親讀全部575行；58-106预写候選/偏好，188-253逐candidate評估lookup，256-358 SFT/RM/current採樣/PPO/critic，449-538完整scope和輸出；沒有真人annotation loader或token generator。",
        ),
    ]:
        sources.append(
            {
                "id": identifier,
                "kind": "repository_code",
                "title": relative,
                "path": relative,
                "sha256": audit["source_sha256_before_and_after"][relative],
                "version": "本人2026-10-04讀取與CPU前後SHA一致的working-tree版本",
                "verified": True,
                "inspection_note": note,
            }
        )
    sources.append(
        {
            "id": "cpu",
            "kind": "execution",
            "title": "Fresh CPU finite-card workflow and all-record audit",
            "verified": True,
            "artifact_id": "cpu_execution",
        }
    )
    claims = []
    for identifier, statement, location, evidence, scope in CLAIM_DATA:
        claims.append(
            {
                "id": identifier,
                "kind": "concept",
                "statement": statement,
                "location": "course/glossary.md:" + location,
                "status": "verified",
                "evidence": [
                    {"source_id": source, "locator": locator, "supports": support}
                    for source, locator, support in evidence
                ],
                "artifact_ids": ["original_inspections"],
                "scope": scope,
            }
        )
    claims.append(
        {
            "id": "local_ppo",
            "kind": "software",
            "statement": "第13章新增PPO只選整張預寫回答卡，標籤由程式規則提供，沒有真人評分及逐字生成。",
            "location": "course/glossary.md:85 第13章PPO句",
            "status": "verified",
            "evidence": [
                {
                    "source_id": "repo_run",
                    "locator": "lines58-106, 188-253, 256-358",
                    "supports": "All answer templates and rule preferences preexist training; actions are indices and responses lookup.",
                },
                {
                    "source_id": "repo_core",
                    "locator": "FiniteResponsePolicy53-61; FiniteRewardModel64-75",
                    "supports": "Structured numeric features output candidate logits/scores without vocabulary/text decoder.",
                },
                {
                    "source_id": "cpu",
                    "locator": "cpu_audit verified/all_record_keys_checked; cpu_records and cpu_result first_rollout_trace",
                    "supports": "Fresh run finishes; every record/output audited; old log probabilities/reference remain fixed while actions are card indices.",
                },
            ],
            "artifact_ids": ["cpu_execution", "cpu_result", "cpu_records", "replay_code", "code_core", "code_run"],
            "verification": {
                "method": "executed",
                "expected": "輸入候選和偏好由程式模板/規則生成，策略只取候選索引，再lookup回應。",
                "observed": "Exit0；165完整records通過重構；132/15/18題各策略回應都等於candidates[chosen_action]，rollout action全0..3，effective_tokens=0，reference前後SHA一致。",
                "details": "完整親讀來源且逐筆audit全部三側records、候選算式、偏好rules、所有策略回應及三次reuse的old SHA。保存full config/records/result/stdout/stderr/replay；不單靠scope字串或exit判定。",
            },
            "scope": "只核本輪seed42短CPU有限選卡軟體流程，不证明人類RLHF/中文理解/加法/逐字LLM能力/方法品質優勝或GPU速度。",
        }
    )
    numeric_note = "本人逐句檢查完整G.4 snapshot：表中13.11等只是章節定位，没有數值算例、公式預期值或量測品質/時間等實測主張；CPU數字用於software核對，不冒稱本節numeric/empirical結果。"
    figure_note = "本人完整讀G.4 snapshot並檢查Markdown圖像引用：本節沒有任何圖片或SVG；必要前置圖並未用作本節證據。"
    inspections = {
        "reviewer_task": "/root/fact_finish_g_4",
        "source": "course/glossary.md#G.4",
        "source_snapshot_sha256": sha(read_path),
        "first_section": False,
        "intro_applicability": "首個##為G.1，G.4不是首節，不需intro_sha256/intro_summary。",
        "reading_scope": "只評G.4及明示links的必要前置。首次cat glossary也將G.1/G.3輸出到工具，但未用作本節判定；G.2是明示DPO入口。未讀其他review、作者歷史或預期結論。",
        "source_readings": [
            {
                "id": item["id"],
                "version": item["version"],
                "note": item["inspection_note"],
                "snapshot_sha256": item.get("snapshot_sha256"),
            }
            for item in sources
            if item["kind"] in {"paper", "official_docs"}
        ],
        "claim_inventory": [item["id"] for item in claims],
        "numeric_inspection": numeric_note,
        "figure_inspection": figure_note,
        "not_expanded_claims": "G.4未單獨定義pre-training/SFT或寫KL公式。原paper/前置中的相關定義仅用于核准reference/PPO，不增造正文待審主張。",
        "issues_found": [],
    }
    write_json(BASE / f"{PREFIX}_original_inspections.json", inspections)
    special = {
        f"{PREFIX}_original_inspections.json": (
            "original_inspections",
            "source_snapshot",
            "本人逐原始來源閱讀定位/條件與圖數值NA判斷",
        ),
        f"{PREFIX}_cpu_audit.json": (
            "cpu_execution",
            "execution",
            "實際CPU流程與完整records及輸出核對，含真command/result/environment",
        ),
        f"{PREFIX}_cpu_result.json": (
            "cpu_result",
            "source_snapshot",
            "本輪CPU完整配置、history、rollout與全部evaluation",
        ),
        f"{PREFIX}_cpu_records.json": ("cpu_records", "source_snapshot", "本輪CPU完整三側候選/偏好records，未刪行"),
        f"{PREFIX}_replay.py": ("replay_code", "code", "本人可重跑固定短CPU流程與逐記錄audit程式"),
        f"{PREFIX}_code_tiny_perceptron_posttraining.py.txt": (
            "code_core",
            "source_snapshot",
            "本人親讀且CPU前後SHA一致的完整核心原碼",
        ),
        f"{PREFIX}_code_scripts_course_experiments_posttraining.py.txt": (
            "code_run",
            "source_snapshot",
            "本人親讀且CPU前後SHA一致的完整實驗原碼",
        ),
    }
    artifacts = []
    for path in sorted(BASE.glob(f"{PREFIX}_*")):
        if path.name.endswith(("_ruff.json", "_checker.json")):
            continue
        identifier, kind, description = special.get(
            path.name,
            (
                path.name.removeprefix(PREFIX + "_"),
                "code" if path.suffix == ".py" else "source_snapshot",
                "本人下載/閱讀快照或當次日志：" + path.name,
            ),
        )
        item = {
            "id": identifier,
            "kind": kind,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path),
            "description": description,
        }
        if kind == "execution":
            item.update(
                {
                    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_g_4_replay.py",
                    "result": "Exit0; fixed CPU recipe and all 165 record/response audits pass. Inner exact command/result retained in JSON.",
                    "environment": audit["environment"],
                }
            )
        artifacts.append(item)
    ids = [item["id"] for item in claims]
    report = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "G.4",
        "source": "course/glossary.md#G.4",
        "reviewer_task": "/root/fact_finish_g_4",
        "reviewer_context": "fresh",
        "source_sha256": hashlib.sha256(saved).hexdigest(),
        "figure_sha256": {},
        "verdict": "pass",
        "claims": claims,
        "sources": sources,
        "artifacts": artifacts,
        "issues": [],
        "checks": {
            "factual_accuracy": {
                "status": "pass",
                "details": "所有定義逐概念有本人原論文/官方資料定位與條件；本地選卡PPO另外親讀完整原碼並CPU核對。",
                "claim_ids": ids,
            },
            "numeric_verification": {"status": "not_applicable", "details": numeric_note, "claim_ids": []},
            "figure_consistency": {"status": "not_applicable", "details": figure_note, "claim_ids": []},
            "source_verification": {
                "status": "pass",
                "details": "十二份原論文/官方资料本輪新下載原bytes、逐定義親讀，保存真version/URL/hash/定位；不採來源摘要或repo概念實作當一般定義authority。repo兩原碼只核本地software並附真執行。",
                "claim_ids": ids,
            },
            "limitations": {
                "status": "pass",
                "details": "已區分儲存/耗時、active expert/全權重、human來源/PPO方法、old/reference、RM/critic、one-card/逐token、teacher訊號/正確性、檢索/回答可信、schema/參數語意；未以CPU證GPU，無品質/安全普遍保證。",
                "claim_ids": ids,
            },
        },
    }
    write_json(ROOT / "docs/technical-reviews/G.4.json", report)
    print(
        json.dumps(
            {
                "lesson": "G.4",
                "verdict": report["verdict"],
                "claims": len(claims),
                "sources": len(sources),
                "artifacts": len(artifacts),
                "source_sha256": report["source_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
