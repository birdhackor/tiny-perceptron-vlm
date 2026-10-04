"""Assemble the review from this reviewer's already-inspected evidence."""

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
COMMAND = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_12/audit.py"
FINISH = ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_12/finish_checks.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = read(OUT / "row-audit.json")
    cpu = read(OUT / "cpu-audit.json")
    final = read(OUT / "finish-checks.json")
    env = {k: str(v) for k, v in cpu["environment"].items()}
    artifacts, artifact_paths = [], {}
    execution = {
        "row-audit.json": (
            COMMAND,
            "獨立由原始生成ID、EOS、完整文字、請求參數、runtime和最後回答重算12×90 test、6×84 validation及36+18 swaps；全部逐題欄位與聚合一致。",
        ),
        "cpu-audit.json": (
            COMMAND,
            "11份固定HF commit公開模型以token=False取得且指紋/bytes符合清單，與原實驗權重逐tensor完全一致；加未訓練基線共12×90 CPU重新生成，所有原始ID與GPU檔一致。",
        ),
        "audit.log": (
            COMMAND,
            "最終退出0；每個版本分別報首段、整題及原GPU生成ID差異，12版本差異均0。初次helper錯選joint loader已修正，舊失敗log另存，沒有更改教材或模型。",
        ),
        "finish-checks.json": (
            FINISH,
            "五個教材固定GitHub目標、兩份官方PyTorch來源與本機bytes一致；原實驗核心函式AST與現在相同；拒絕子字串、常數基線、唯一CE/KD成敗差異及中位數重算均符合。",
        ),
        "finish-checks.log": (FINISH, "退出0；來源身份和逐項補充限制核對完成。"),
        "lesson-count.log": (
            "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_12/lesson-count.py",
            "親跑教材唯一程式塊，列12種task的分母並合計90；不生成回答。",
        ),
        "pytest.log": (
            ".venv/bin/python -m pytest -q tests/test_capstone.py tests/test_capstone_release.py",
            "70 passed in 7.91s；涵蓋家族切分、答案/EOS遮罩、真素材、cache、工具runtime、錯讀回不被取代、checkpoint及PTQ重載。",
        ),
    }
    for file in sorted(OUT.rglob("*")):
        if not file.is_file() or file.name in {
            "previous-report-unread.json.txt",
            "audit-initial-loader-error.log",
            "checker.log",
        }:
            continue
        if "__pycache__" in file.parts:
            continue
        relative = file.relative_to(OUT).as_posix()
        identifier = "a" + str(len(artifacts) + 1)
        item = {
            "id": identifier,
            "kind": "code" if file.suffix == ".py" else "source_snapshot",
            "path": file.relative_to(ROOT).as_posix(),
            "sha256": sha(file),
            "description": "本輪保存的完整原始UTF-8/原bytes快照；親讀/計算定位見來源及主張：" + relative,
        }
        if relative in execution:
            command, result = execution[relative]
            item.update(kind="execution", command=command, result=result, environment=env, description=result)
        elif relative.startswith("cpu-test-") or relative in {
            "cpu-image-swaps.json",
            "cpu-audio-swaps.json",
            "cpu-cache-consistency.json",
        }:
            item.update(
                kind="execution",
                command=COMMAND,
                environment=env,
                result="本輪CPU FP32實際模型逐題生成；完整ID、EOS、解析、runtime、最後答案與聚合保留。cache檔另保留12題每步logits比較；沒有GPU時間或記憶體量測。",
                description="親跑完整CPU結果：" + relative,
            )
        artifacts.append(item)
        artifact_paths[relative] = identifier

    def aid(path):
        return artifact_paths[path]

    sources = [
        {
            "id": "selection-paper",
            "kind": "paper",
            "title": "Cawley & Talbot, On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation",
            "url": "https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf",
            "version": "JMLR 11 (2010), 2079–2107, published July 2010",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "作者發表的原始研究，直接分析選擇過程對最終評估的偏差。",
            "inspection_note": "親讀原PDF abstract、§1與§5/5.1（pp.2094–2095；本輪pdftotext行1080–1118），訓練與選擇都屬擬合流程；独立最終評估不能用於本輪選擇。不是以該論文替本專案證明實際歷史。",
            "artifact_ids": [aid("cawley2010.pdf"), aid("cawley2010.txt")],
        },
        {
            "id": "averaging-doc",
            "kind": "official_docs",
            "title": "scikit-learn model evaluation: averaging metrics",
            "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/model_evaluation.rst",
            "version": "scikit-learn tag 1.7.2, doc/modules/model_evaluation.rst",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "維護者的版本固定官方文件，明示macro、weighted和micro分母。",
            "inspection_note": "親讀行563–584：macro每類等權、weighted按樣本數、micro累加分子分母。本節以任務準確率作能力矩陣；只借平均原理，不把其分數稱成multiclass F1。",
            "artifact_ids": [aid("sklearn-model-evaluation.rst")],
        },
        {
            "id": "holdout-doc",
            "kind": "official_docs",
            "title": "scikit-learn cross-validation: training, validation, final test",
            "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst",
            "version": "scikit-learn tag 1.7.2, doc/modules/cross_validation.rst",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "版本固定的官方評估流程文件。",
            "inspection_note": "親讀行60–70：test調超參數會讓test知識洩漏；training更新、validation選擇、test最終評估。本輪合成圖音與正常問法並未覆蓋一般照片、語音或安全，不能由固定小世界推到那些未測分佈。",
            "artifact_ids": [aid("sklearn-cross-validation.rst")],
        },
        {
            "id": "torch-load",
            "kind": "official_source",
            "title": "PyTorch torch.load actual-version source",
            "url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/serialization.py",
            "version": "PyTorch 2.14.1, commit 5c4886908584029761b579af026dcfb627c84070; current 2.14.1+cpu",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "PyTorch官方實際安裝版本的原始碼，下載bytes與本機serialization.py一致。",
            "inspection_note": "親讀load行1309–1378：CPU初次反序列化、map_location='cpu'指定全張量裝置、weights_only載入tensor/primitive/dictionary；本輪真的載入FP32與PTQ後再生成，没有把CPU品質核對說成GPU加速。",
            "artifact_ids": [aid("torch-serialization.py.txt"), aid("finish-checks.json")],
        },
        {
            "id": "torch-sync",
            "kind": "official_source",
            "title": "PyTorch torch.cuda.synchronize actual-version source",
            "url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/cuda/__init__.py",
            "version": "PyTorch 2.14.1 official commit 5c4886908584029761b579af026dcfb627c84070; original GPU report 2.14.1+cu126",
            "verified": True,
            "checked_original": True,
            "accessed_on": "2026-10-04",
            "authority_reason": "官方CUDA介面原始碼與本機相同，明確定義等待GPU工作的範圍。",
            "inspection_note": "親讀行1271–1281：等待指定CUDA裝置所有stream的kernel完成；實驗benchmark_generation在每次生成前後同步。只支持所保存L4 FP32短句wall time的定義，未重跑GPU也沒有測生成記憶體峰值。",
            "artifact_ids": [aid("torch-cuda-init.py.txt")],
        },
    ]
    repo_sources = {
        "core": (
            "tiny_perceptron/capstone.py",
            "親讀CapstoneModel/default_config、build_dataset、modality_tensors、ByteTokenizer流程、generate_trace/generate_traces、evaluate_rows、parse_action、calculator_runtime、run_assistant、expected_final。原實驗核心SHA d02d1c…與此快照相同；完整字串+EOS計分，工具算完仍由同一模型回答。",
        ),
        "train": (
            "scripts/course_experiments/capstone.py",
            "親讀完整train_stage和run_deployment：固定seed與家族split；4階段300/1400/600/100步，僅validation；首次deployment才評test。不同訓練stage是順序接續，不能把其數字當只差一個方法的隨機化對比。",
        ),
        "deploy": (
            "scripts/course_experiments/capstone_deployment.py",
            "親讀image_counterfactuals、counterfactual_pairs、cache_consistency、benchmark_generation與run：推薦joint固定；保存重載PTQ；先選每task第一個validation row；3暖機/10測量，CUDA同步，未量生成峰值。",
        ),
        "student": (
            "scripts/course_experiments/capstone_student.py",
            "親讀全文：DPO教師固定不更新；隨機Dense width48起點深拷貝CE/KD各350步，seed+5000抽樣；只有validation/test及KD重載PTQ，沒有學生換圖/換聲對照。沒有逐GPU批次實錄的聲稱。",
        ),
        "quant": (
            "tiny_perceptron/capstone_quantization.py",
            "親讀quantize_capstone/restore_quantized_payload/load_quantized_capstone：非router Linear列刻度、真正int4打包，載入還原float32；實際重載和逐tensorhash驗證，儲存縮小不是執行加速或載入後低位元記憶體證明。",
        ),
        "data": (
            "tiny_perceptron/data.py",
            "親讀ByteTokenizer：8個special IDs、256 byte IDs，decode略過special但保留原始生成ID另核EOS；與原始JSON逐項比對。",
        ),
        "modal": (
            "tiny_perceptron/multimodal.py",
            "親讀scene、tone、log_mel原始生成和频谱程式；教材素材確為合成RGB/純音，沒有一般照片/語音模型或搜尋工具。",
        ),
        "cli": (
            "scripts/capstone.py",
            "親讀main全部命令分支；evaluate真正load/重載模型並生成，infer實際run_assistant；count程式只呼叫build_dataset，不是此評分命令。",
        ),
        "tests": (
            "tests/test_capstone.py",
            "親讀切分、答案遮罩、真素材、cache、action/runtime與錯讀回測試；實跑70項含release tests，不以單一exit0代替實際70項結果。",
        ),
        "release-tests": (
            "tests/test_capstone_release.py",
            "親讀PTQ重載、image_counterfactual、cache選題和generation benchmark測試；與tests/test_capstone.py共70項親跑通過。",
        ),
    }
    for identifier, (file, note) in repo_sources.items():
        snapshot = "code/" + file.replace("/", "__") + ".txt"
        path = OUT / snapshot
        sources.append(
            {
                "id": identifier,
                "kind": "repository_code",
                "title": file + " (本人保存的原bytes快照)",
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha(path),
                "version": "本輪2026-10-04親讀原檔SHA256=" + sha(path),
                "verified": True,
                "inspection_note": note,
                "artifact_ids": [aid(snapshot)],
            }
        )
    for identifier, path, title in [
        ("audit", "row-audit.json", "本輪独立重算全部原始記錄"),
        ("cpu", "cpu-audit.json", "本人固定公開权重的全題CPU重新生成与逐tensor原权重驗證"),
        ("finish", "finish-checks.json", "來源身份、例子、限制与中位數补充核對"),
        ("count-run", "lesson-count.log", "教材分母程式親跑"),
        ("test-run", "pytest.log", "70項真軟體檢查親跑"),
    ]:
        sources.append(
            {"id": identifier, "kind": "execution", "title": title, "verified": True, "artifact_id": aid(path)}
        )
    sources.append(
        {
            "id": "averaging-derivation",
            "kind": "derivation",
            "title": "題數平均與任務等權平均的透明公式",
            "verified": True,
            "details": "令task j有n_j題、c_j正確：所有題平均=sum(c_j)/sum(n_j)=78/90=0.8666666666666667；等權任務平均=(sum(c_j/n_j))/12=0.8888888888888888。calculator 10/12、tool_return 5/6、image_shape 0/9，其餘九task均1；(10/12+5/6+0+9)/12=8/9。原文開頭80分是比喻，不當作實測。",
        }
    )
    claims = []

    def evidence(source, locator, supports):
        return {"source_id": source, "locator": locator, "supports": supports}

    def add(
        kind,
        statement,
        location,
        scope,
        expected=None,
        observed=None,
        details=None,
        refs=None,
        files=None,
        denominators=None,
        tolerance=None,
    ):
        claim = {
            "id": "c" + str(len(claims) + 1),
            "kind": kind,
            "statement": statement,
            "location": location,
            "status": "verified",
            "evidence": refs or [],
            "artifact_ids": [aid(path) for path in files or []],
            "scope": scope,
        }
        if kind != "concept":
            claim["verification"] = {
                "method": "executed",
                "expected": expected,
                "observed": observed,
                "details": details,
            }
            if kind == "numeric":
                claim["verification"]["tolerance"] = tolerance or "精確整數相等。"
            if kind == "empirical":
                claim["verification"]["denominators"] = denominators
        claims.append(claim)

    add(
        "concept",
        "題數不等時全題平均讓題多的能力權重較高，任務等權平均是不同指標。",
        "開頭及Counter塊後平均方式段",
        "此處是task準確率的分子/分母，不是多類F1；80分開頭是比喻。",
        refs=[
            evidence("averaging-doc", "model_evaluation.rst:563–584", "等權macro與按頻數/全分母聚合的差別。"),
            evidence(
                "averaging-derivation",
                "78/90及(10/12+5/6+0+9)/12",
                "本考卷可直接得到86.6667%與88.8889%，總分不等於所有能力都可靠。",
            ),
        ],
        files=["row-audit.json"],
    )
    add(
        "concept",
        "最後權重應依預先說明的validation規則選定，不以test挑漂亮分數。",
        "前置段及DPO比較表後段",
        "是獨立評估方法；本輪實際選擇紀錄另逐項核，沒有把同一test稱為再開發後的新證據。",
        refs=[
            evidence("selection-paper", "§5/5.1, pp.2094–2095", "選擇亦屬擬合，評估須含選擇偏差。"),
            evidence("holdout-doc", "cross_validation.rst:60–70", "validation選設定，test保留最終評估。"),
        ],
        files=["raw/capstone-selection.json"],
    )
    add(
        "software",
        "Counter程式只建立固定test並列每task count，不生成模型回答。",
        "唯一Python程式塊及其後解釋",
        "只給題庫分母；真正能力評分另讀模型trace。",
        "12種task分母、合計90；沒有模型/生成呼叫。",
        "audio6/calculator12/concept6/image_color9/image_shape9/joint18/missing3/rag3/safety3/style3/tool_return6/unavailable12；合計90。",
        "親跑原程式塊，build_dataset只用固定seed/素材規格組資料；另對照保存data全部splits与manifest。",
        refs=[
            evidence("core", "build_dataset:139–282", "建資料，不建模型。"),
            evidence("count-run", "lesson-count.log全文", "實際每項分母。"),
        ],
        files=["lesson-count.py", "lesson-count.log", "row-audit.json"],
    )
    add(
        "numeric",
        "固定最後考卷有90題、12種task；各表格分母對應這些實題。",
        "最後90題段及三張能力/版本/學生矩陣",
        "6個數字家族、1個視覺家族、2個音高家族、3個context家族；不是90個獨立素材家族。",
        "90題；12種task；所有版本用同90 id。",
        "90題、12種；12份test逐筆id序列均與原資料一致。",
        "Counter、保存data、12份完整逐題檔皆親自解析；家族先分train/validation/test，無家族交集。",
        refs=[evidence("audit", "row-audit.json:data/evaluations", "全版本有效題數與ID。")],
        files=["row-audit.json", "lesson-count.log"],
    )
    add(
        "software",
        "action_correct要求首段完整expected_action字串一致且正常EOS，不只是DIRECT/ASK/TOOL類別。",
        "第一個分數欄定義段",
        "精確字串含TOOL名稱/順序參數與DIRECT/ASK內容；合法解析但內容錯仍為false。",
        "EOS真且raw==row.answer；square/circle、1/0的錯誤皆false。",
        "90題逐項獨立按ID解碼並核EOS；9形狀與tool_return的DIRECT:0合法但action_correct false。",
        "獨立byte解碼（IDs>=8減8）、第一停止ID/EOS、完整字串、regex parser與保存欄位逐項assert一致。",
        refs=[
            evidence("core", "evaluate_rows:499–545、parse_action:550–561", "實際計分與解析不同。"),
            evidence("audit", "evaluations.joint.records", "內容錯誤與EOS逐項結果。"),
        ],
        files=["row-audit.json", "cpu-test-joint.json", "pytest.log"],
    )
    add(
        "software",
        "end_to_end_correct還要求最後答案；工具執行結果只成為第二次生成輸入，直接回答無第二次生成。",
        "第二分數欄定義段",
        "工具請求正確/運算正確不能代替助理最後答對；不同於B.7只選卡分類。",
        "action_correct AND answer==expected_final；TOOL成功後再次模型生成，DIRECT/ASK不再生成。",
        "所有12×90紀錄的runtime、followup prompt、final trace及答案均與獨立重算相同；錯讀回保留錯誤。",
        "逐題按parsed_action分支，allowlist/available與加法runtime獨立重算；成功工具後新prompt無圖音；最終需正常EOS的DIRECT。",
        refs=[
            evidence("core", "evaluate_rows:517–535/run_assistant:577–598", "真runtime及第二段模型回答。"),
            evidence(
                "tests",
                "test_runtime_return_does_not_overwrite_a_wrong_model_final_answer:180–197",
                "工具3不覆蓋模型錯答9。",
            ),
        ],
        files=["row-audit.json", "cpu-audit.json", "pytest.log"],
    )

    scopes = {
        "calculator": "12個完整加法流程、固定窄句型，數字0–9留出組合；不代表任意工具。",
        "unavailable": "12題固定ASK:計算器未開模板；runtime未執行，無任意求助能力保證。",
        "tool_return": "6題直接讀取已提供的計算器回報，無這一題新的工具執行。",
        "concept": "6題答案固定DIRECT:把兩個數合起來；常數基準就能全對。",
        "image_color": "9個原圖答案全green；常數green就能全對，主joint另有换色對照。",
        "image_shape": "留出green square的9種明暗/位移，全部正常EOS答circle；不是所有形狀泛化研究。",
        "joint": "18題只問color/pitch，顏色green常數、low/high各9；不包含shape。",
        "audio": "low/high各3題，兩群合成純音；不是語音辨識。",
        "rag": "3題資料已在user文字，桌子/書櫃/抽屜各1；沒有新增搜尋/檢索。",
        "missing": "3題固定ASK:請提供數量，規則模板小世界。",
        "safety": "3題固定密碼拒絕句；不支持通用安全或置信校準。",
        "style": "3題指定數字與短回答的照抄；不是一般文體能力。",
    }
    labels = {
        "calculator": "計算器完整迴圈",
        "unavailable": "計算器不可用",
        "tool_return": "單獨回填工具結果",
        "concept": "加法概念說明",
        "image_color": "原始圖片顏色",
        "image_shape": "原始圖片形狀",
        "joint": "顏色與音高聯合",
        "audio": "單獨音高",
        "rag": "已提供短資料問答",
        "missing": "缺數量求助",
        "safety": "密碼拒絕示範",
        "style": "簡短照抄",
    }
    desired = {
        "calculator": (12, 12, 10),
        "unavailable": (12, 12, 12),
        "tool_return": (6, 5, 5),
        "concept": (6, 6, 6),
        "image_color": (9, 9, 9),
        "image_shape": (9, 0, 0),
        "joint": (18, 18, 18),
        "audio": (6, 6, 6),
        "rag": (3, 3, 3),
        "missing": (3, 3, 3),
        "safety": (3, 3, 3),
        "style": (3, 3, 3),
    }
    for task, (count, action, end) in desired.items():
        actual = audit["evaluations"]["joint"]["summary"]["by_task"][task]
        assert actual == {"count": count, "action_correct": action, "end_to_end_correct": end}
        for field, score, name in [
            ("action_correct", action, "首段完整輸出吻合"),
            ("end_to_end_correct", end, "整題正確"),
        ]:
            add(
                "empirical",
                f"推薦joint的{labels[task]}：{name}{score}/{count}。",
                f"第一矩陣「{labels[task]}」列「{name}」欄",
                scopes[task],
                f"{score}/{count}",
                f"原GPU逐題獨立重算與本人CPU均{score}/{count}。",
                f"逐題原ID/EOS→parsed_action→runtime→final_answer→{field}重算；CPU固定公開权重重生成，原始生成ID与原GPU完全一致。",
                refs=[
                    evidence("audit", f"evaluations.joint.summary.by_task.{task} / records", "分子分母与判讀。"),
                    evidence("cpu", f"models.joint / cpu-test-joint.json task={task}", "本人真模型全題生成核對。"),
                ],
                files=["row-audit.json", "cpu-test-joint.json", "finish-checks.json"],
                denominators={
                    "test_rows_for_task": count,
                    "test_total": 90,
                    "seed": 42,
                    "max_new_tokens": 64,
                    "batch_size": 24,
                    "original_device": "NVIDIA L4 FP32",
                    "reverification_device": "CPU FP32",
                },
            )
    for field, score in [("action_correct", 80), ("end_to_end_correct", 78)]:
        add(
            "empirical",
            f"推薦joint全題直接合計{field}為{score}/90。",
            "第一矩陣合計列；第二矩陣joint列",
            "全題加總，非每task等權平均；不能蓋過shape 0/9與回填失敗。",
            f"{score}/90",
            f"GPU原記錄重算与本輪CPU皆{score}/90。",
            "12個task分子分母相加；不是拿教材合計當真值；同時核每個trace和所有task欄。",
            refs=[
                evidence("audit", "evaluations.joint.summary / aggregation", "逐項合計。"),
                evidence("cpu", "models.joint.cpu_summary", "本輪重現。"),
            ],
            files=["row-audit.json", "cpu-test-joint.json"],
            denominators={"test_rows": 90, "tasks": 12, "seed": 42},
        )
    add(
        "empirical",
        "首段80/90与整題78/90差兩題，皆工具回1而模型讀回錯答。",
        "第一矩陣前解釋段及calculator範圍欄",
        "兩個方向0+1和1+0；另一个独立tool_return错误不属于這兩個首段成功例。",
        "兩個action_correct=true/end=false，runtime result=1。",
        "6452…的0+1最後0；246ec…的1+0最後111；两者tool action精確且EOS正常。",
        "檢查兩個原id的runtime與final_trace；原runtime結果不覆蓋模型答0/111；tool_return 27c…另答0。",
        refs=[
            evidence(
                "audit",
                "evaluations.joint.records id=6452e1d5197f5dc7987c、246ec17075366e4bea5f、27c797992da6c52a10ee",
                "錯誤層與真生成。",
            )
        ],
        files=["row-audit.json", "cpu-test-joint.json"],
        denominators={"calculator": 12, "first_action_pass_final_fail": 2, "independent_tool_return": 6},
    )
    for metric, expected in [
        ("nonrefusal", "87題其他任務中，首段/最後trace均沒有拒絕句；0/87。"),
        ("correct", "非拒絕題整題75/87。"),
    ]:
        add(
            "empirical",
            expected,
            "第一矩陣後拒絕子集段",
            "只排除這87题每題都產生該固定拒絕句的常數策略；沒有測完所有正常問法。",
            expected,
            "逐題排除task=safety的3題，餘87題完整生成拒絕子字串0次，end_to_end correct加總75。",
            "除了精確answer equality，也親檢首段及最終raw中該拒絕句的子字串。",
            refs=[
                evidence("audit", "refusal", "有效非拒絕分母與正确数。"),
                evidence(
                    "finish", "limits.nonrefusal_substring_count_across_all_first_final_traces", "完整raw不存在拒絕句。"
                ),
            ],
            files=["row-audit.json", "finish-checks.json"],
            denominators={"non_safety_test_rows": 87, "safety_rows": 3},
        )
    add(
        "software",
        "圖片成對成功需原素材與換後答案都正確；原圖錯、換後對不能算成功。",
        "圖片對照段最後一句",
        "不把形狀原圖全部答circle、換circle後巧合正确當成功。",
        "both_end_to_end_correct=original_ok AND swapped_ok。",
        "36組逐id重算；image_shape原0/9、換9/9，所以pair 0/9。",
        "深拷貝素材規格後真生成畫素，重新標期待答案，兩端各生成再取AND。",
        refs=[evidence("deploy", "image_counterfactuals:35–56 / counterfactual_pairs:59–86", "實際對照與AND規則。")],
        files=["row-audit.json", "cpu-image-swaps.json"],
    )
    for task, score, count in [("image_color", 9, 9), ("image_shape", 0, 9), ("joint", 18, 18), ("all", 27, 36)]:
        add(
            "empirical",
            f"圖片{task}成對兩端皆正確{score}/{count}對。",
            "圖片成對對照摘要與19.6連結",
            "顏色green→blue；形狀square→circle；joint只換顏色保持音高。shape兩端皆circle，未辨識原square。",
            f"{score}/{count}對",
            f"逐題原/換AND与本輪CPU相同，{score}/{count}對。",
            "保存原90題+換36題+pair檔；按去掉-image-swap後的id對齊。本人CPU重新生成swap的原始ID全部一致。",
            refs=[
                evidence("audit", "image_pairs.by_task / image_pairs.pairs / image_swaps", "每種對照兩端的分母和AND。"),
                evidence("cpu", "joint_counterfactuals", "CPU圖像對照重現。"),
            ],
            files=["row-audit.json", "cpu-image-swaps.json"],
            denominators={"paired_rows": count, "original_plus_swapped_evaluations": count * 2, "joint_stage": "joint"},
        )
    add(
        "empirical",
        "保持圖片、聯合題low/high互換後，18/18對兩端都正確。",
        "換聲18對段与19.6連結",
        "只有joint task的合成兩群純音，沒測一般音訊/語音。",
        "18/18對",
        "原joint18題皆正確、swap18题皆正確，CPU全部原始ID重現。",
        "重新生成變換pitch的16帶平均log-mel，再按green與新pitch標答案；用原id與-audio-swap對齊。",
        refs=[
            evidence("audit", "audio_swaps / evaluations.joint task=joint", "兩端完整內容。"),
            evidence("cpu", "joint_counterfactuals.audio_changed_ids=[]", "真CPU素材生成。"),
        ],
        files=["row-audit.json", "cpu-audio-swaps.json"],
        denominators={"pairs": 18, "generations": 36, "low_high_balanced": "9+9"},
    )

    total_expect = {
        "untrained": 0,
        "pretrain": 0,
        "sft": 45,
        "dpo": 78,
        "joint-int4": 78,
        "joint-int8": 78,
        "dpo-int4": 78,
        "dpo-int8": 78,
        "student-ce": 62,
        "student-kd": 61,
        "student-kd-int4": 61,
    }
    for stage, count in total_expect.items():
        actual = audit["evaluations"][stage]["summary"]["end_to_end_correct"]
        assert actual == count
        add(
            "empirical",
            f"{stage}在同份固定最後題目整題正確{count}/90。",
            f"第二版本矩陣「{stage}」對應列",
            "阶段配方/架構不都相同：pretrain300、SFT1400、joint600、DPO100是接續权重；學生CE/KD各350與較小Dense；同分只表示此考卷精確評分，非逐token/普遍能力相同。",
            f"{count}/90",
            f"原GPU記錄逐題重算与本輪CPU均{count}/90；CPU原始生成ID差0。",
            "每版本90个id一致；逐第一段EOS/內容與runtime後最終回答重算，並親跑各模型CPU；PTQ是存檔下載後重載还原FP32。",
            refs=[
                evidence("audit", f"evaluations.{stage}", "全题而非只摘要。"),
                evidence("cpu", f"models.{stage}", "指紋、逐tensor和全題重新生成。"),
            ],
            files=["row-audit.json", "cpu-audit.json", f"cpu-test-{stage}.json"],
            denominators={
                "test_rows": 90,
                "test_seed": 42,
                "max_new_tokens": 64,
                "batch_size": 24,
                "one_fixed_recipe": True,
            },
        )
    add(
        "empirical",
        "DPO雖test與joint同為78/90，固定validation由joint75/84退至DPO71/84。",
        "DPO比較表後段",
        "驗證是同84題，joint task退4題；test同分不抹去已觀察validation下降，也不是所有DPO會退步。",
        "joint75/84、DPO71/84；test各78/90。",
        "6份validation亲逐题解碼重算，其中joint75/84/DPO71/84；联合题18/18→14/18。",
        "对比原joint/dpo validation全部id，曾正确变錯的4个joint回答誤色为red，純音部分依然正确。",
        refs=[evidence("audit", "evaluations.joint-validation / evaluations.dpo-validation", "同题validation與test。")],
        files=["row-audit.json", "raw/joint/validation.json", "raw/dpo/validation.json"],
        denominators={"validation_rows_per_stage": 84, "joint_validation_rows": 18, "test_rows_per_stage": 90},
    )
    add(
        "software",
        "推薦joint的選擇文件在test前記錄validation規則，DPO保留作比較及學生教師。",
        "測試前選擇紀錄固定GitHub連結段",
        "只核保存的pre-test文件、規則、指紋、原实验入口和各阶段未評test紀錄；不聲稱可證明不存在未保存試驗。",
        "selected_stage=joint、selected_before_test_generation=true；75>71，数据/權重未按test改。",
        "固定1df335…公開原bytes與本機選擇文件相同；75/84、71/84皆由完整原始validation核實；選擇checkpointsha與原權重匹配。",
        "原選擇UTC2026-10-04T01:51:49.261544+00:00；deploy函式權重固定joint後才生成test；train_report test_evaluated=false；公開權重逐tensor核原來源。",
        refs=[
            evidence("finish", "source_identity / experiment_function_identity", "固定公開來源与本機一致。"),
            evidence("train", "train_stage report / run_deployment", "test推遲。"),
            evidence("deploy", "run recommended_stage='joint'", "推薦与後續評分分離。"),
        ],
        files=["raw/capstone-selection.json", "row-audit.json", "cpu-audit.json", "finish-checks.json"],
    )
    for name, changed in [("joint-int4", 1), ("joint-int8", 0), ("dpo-int4", 2), ("dpo-int8", 0)]:
        add(
            "empirical",
            f"{name}與各自FP32同分78/90，但原始生成ID改變{changed}/90題。",
            "量化同分却有少數生成改变段与19.10連結",
            "只核90题首段与最後完整ID序列；不能稱所有输入或所有數字相同，也不表示原错題被修复。",
            f"同分78/90；{changed}/90生成不同。",
            f"逐題ID對齊比較得{changed}題變化；生成改變者前后都錯，CPU重現各版本原ID。",
            "同時比較action_trace和final_trace的generated_ids，非只看解碼文字或總分。",
            refs=[evidence("audit", f"quantization_changes.{name}", "逐題列出生成變動与正确状态。")],
            files=["row-audit.json", "cpu-audit.json"],
            denominators={
                "paired_test_rows": 90,
                "traces_checked": "action and final if present",
                "versions": "own FP32 vs reloaded PTQ",
            },
        )
    add(
        "empirical",
        "快取的12種預選validation輸入，full/cache原始生成ID一致。",
        "效率段首句及19.9連結",
        "每task第一题，最多16新ID，可能截斷；機制对照而非完整能力评分，未測生成記憶體峰值。",
        "12/12完整原始生成ID及EOS/停止理由一致。",
        "原GPU记录每对ID一致，本輪CPU12題全部一致；same-history CPU每步logits也close。",
        "選題按固定validation顺序每task第一题，sorted task名，与原cache records的id逐项相同；实CPU full/cache生成及logits比較。",
        refs=[
            evidence("deploy", "cache_consistency:90–152 / first_by_task in run", "先选题，再比較完整生成。"),
            evidence("audit", "gpu_cache_record_audit", "原记录全部12题。"),
            evidence("cpu", "cpu_cache", "CPU独立机制核對。"),
        ],
        files=["row-audit.json", "cpu-cache-consistency.json"],
        denominators={"validation_inputs": 12, "tasks": 12, "paths_per_input": 2, "max_new_tokens": 16},
    )
    add(
        "empirical",
        "L4同一短照抄題full/cache中位生成時間為68.054/59.978毫秒。",
        "效率段",
        "同joint FP32、照抄數字15，3次暖機+10次測量/路；包括前文/傳輸/貪婪生成/解碼和CUDA同步，排除載入/启动/HF傳輸；沒有一般GPU加速或記憶體峰值結論。",
        "full68.054ms；cache59.978ms，取10次中位數保留3位。",
        "原始秒值排序第5/6均值得68.05368149999858ms與59.97750799999935ms，四捨五入符合。",
        "親读全部13轮/路输出；均DIRECT:15+EOS共10新ID；benchmark预選首validation style行，原报告gpu=NVIDIA L4；我只重算原L4数据，CPU未复测GPU时间。",
        refs=[
            evidence("finish", "median_derivation", "原10次全样本透明中位数。"),
            evidence("deploy", "benchmark_generation:207–284", "计时包围范围。"),
            evidence("torch-sync", "cuda/__init__.py:1271–1281", "前後同步等待GPU工作。"),
        ],
        files=[
            "finish-checks.json",
            "row-audit.json",
            "raw/deployment/generation-benchmark.json",
            "raw/capstone_deployment.json",
        ],
        denominators={
            "warmup_iterations_per_path": 3,
            "measured_iterations_per_path": 10,
            "paths": 2,
            "prompts": 1,
            "generated_ids_including_eos_per_run": 10,
            "device": "NVIDIA L4",
            "dtype": "FP32",
            "seed": 42,
        },
    )
    add(
        "numeric",
        "小Dense學生共有79,920個總參數，比328,128參數MoE教師小。",
        "小Dense段及縮小確實完成結論",
        "總參數与MoE logical-active proxy分开，Dense沒有router/expert；不能從參數/檔案縮小推速度或生成記憶體数字。",
        "學生79920；教師328128。",
        "親載入CE/KD/KD-int4並按parameters numel计总数，全为79920，DPO教师328128；完整逐tensor shape/dtype/numel/hash保存。",
        "CapstoneModel.description直接sum(parameters.numel)，不把activated proxy當总数；CE/KD均2层width48、experts0；PTQ載入FP32結構不變。",
        refs=[
            evidence("core", "CapstoneModel.description:110–121", "總数与logical_active字段分开。"),
            evidence(
                "cpu", "models.student-ce/student-kd/student-kd-int4.state_tensors及description", "實際完整張量與配置。"
            ),
        ],
        files=["cpu-audit.json", "finish-checks.json"],
    )
    add(
        "software",
        "學生教師事先固定为DPO分支，不是推薦joint的蒸餾版；4-bit是存檔重載版本。",
        "小Dense支線段",
        "仅该CE/KD实验和公開Dense别名；學生是獨立Dense支線，教師固定DPO，推薦主模型仍為joint。",
        "DPO教师sha=b2428be8…；CE/KD隨機Dense起點；KD4-bit重載来源明确。",
        "CE/KD/KD-int4公开metadata均绑定完整b2428be8adfdbc60dff589ec8ca9270f3377e1688da863169c75d40326396b1f；與DPO原权重sha一致，與joint8f7e…不同。",
        "亲读学生runner固定输入capstone_preference/model.pt、requires_grad_(False)/eval；manifest与原题一致；匿名取得的11公開模型与原权重逐tensor同一且全部重新load。",
        refs=[
            evidence("student", "run:169–243", "教師固定DPO，学生另造Dense，PTQ存檔後重載。"),
            evidence("quant", "quantize/load_quantized_capstone", "真存檔重載PTQ。"),
            evidence("cpu", "models.student-*.public_metadata/original_metadata", "教師與來源完整指紋。"),
        ],
        files=["cpu-audit.json", "raw/student/student-report.json", "pytest.log"],
    )

    ce_scores = {
        "calculator": 1,
        "unavailable": 12,
        "tool_return": 1,
        "concept": 6,
        "image_color": 9,
        "image_shape": 0,
        "joint": 18,
        "audio": 6,
        "rag": 3,
        "missing": 3,
        "safety": 3,
        "style": 0,
    }
    for stage in ("student-ce", "student-kd", "student-kd-int4"):
        desired_scores = {**ce_scores, "calculator": 1 if stage == "student-ce" else 0}
        actual = audit["evaluations"][stage]["summary"]["by_task"]
        for task, count in [(task, values[0]) for task, values in desired.items()]:
            score = desired_scores[task]
            assert actual[task]["end_to_end_correct"] == score and actual[task]["count"] == count
            add(
                "empirical",
                f"{stage}的{labels[task]}整題正確{score}/{count}。",
                f"第三學生矩陣「{labels[task]}」對應列、{stage}欄",
                scopes[task] + " 學生未另跑主joint的成對swap；固定350步短分支，非所有Dense/KD结论。",
                f"{score}/{count}",
                f"完整原GPU記錄重算与本人CPU皆{score}/{count}；原始生成ID差0。",
                "同90题id；首段请求、EOS、runtime與final答案獨立重算；新建小Dense CE/KD、KD-int4儲存後重載，未重訓。",
                refs=[
                    evidence("audit", f"evaluations.{stage}.summary.by_task.{task} / records", "每项分子和有效分母。"),
                    evidence("cpu", f"models.{stage}", "親跑原始生成ID核對。"),
                ],
                files=["row-audit.json", f"cpu-test-{stage}.json"],
                denominators={
                    "test_rows_for_task": count,
                    "test_total": 90,
                    "updates_per_branch": 350,
                    "seed": 42,
                    "max_new_tokens": 64,
                    "batch_size": 24,
                },
            )
    add(
        "software",
        "主joint的換圖/換聲成對成績沒有替學生另跑，不能作學生证据。",
        "學生常數基準限制段",
        "學生image_color9/9全green，也可被常數green達成；只親核该正式学生支线，不推断所有未保存操作。",
        "學生正式runner只validation/test与KD PTQ，無student swaps。",
        "亲读runner全文与完整student-report/evidence目录：没有image/audio swaps生成；不同權重不能借主joint36/18对成绩。",
        "正式report evaluations列只有validation_ce/test_ce/validation_kd/test_kd/test_kd_ptq4；本輪也沒有幫學生另跑swap冒充原实验。",
        refs=[
            evidence("student", "run evaluations:204–239", "實際評估范围。"),
            evidence("finish", "limits.student_runner_does_not_run_swaps", "源码范围与色彩常数分布。"),
        ],
        files=["raw/student/student-report.json", "finish-checks.json"],
    )
    add(
        "empirical",
        "KD比CE少對的一題1+8，KD先用错參數1+9令工具回10，最后又答11。",
        "學生矩陣后KD差一題段",
        "唯一CE→KD成功状态差异；其他ID雖有生成變動均沒有增加成功，不將工具執行當成正确參數。",
        "CE TOOL:calculator:1+8→9→DIRECT:9；KD TOOL:calculator:1+9→10→DIRECT:11。",
        "ffe914bb5e9abc53f39f的首/最终ID、runtime与真值9逐项符合；CE整題true、KDfalse。",
        "独立重算generated tool args与runtime整数和；精确action要求原ordered args，最终另要求真值9；CPU完全重现。",
        refs=[
            evidence("finish", "student_only_success_change", "唯一整題成功變動。"),
            evidence(
                "audit", "student_ce_vs_kd / evaluations.student-kd.records id=ffe914bb5e9abc53f39f", "两个错误层。"
            ),
        ],
        files=["finish-checks.json", "cpu-test-student-ce.json", "cpu-test-student-kd.json"],
        denominators={"paired_test_rows": 90, "success_status_changed": 1, "calculator_rows": 12},
    )
    add(
        "empirical",
        "KD4-bit與KD同61/90，仍有6/90題首段或最後生成ID改變。",
        "學生量化同分段与19.10連結",
        "改變的六題原本都错、量化后仍错；不能宣稱逐token相同或所有能力保留。",
        "6/90個ID生成變動；整題分數各61/90。",
        "完整ID对比6題；六题before/after_correct全false；CPU重载兩版再次全部重現原ID。",
        "同时比较action_trace/final_trace，按同数据id对齐；PTQ元数据与逐tensor验证保存；不是只核總分。",
        refs=[
            evidence("audit", "quantization_changes.student-kd-int4", "完整六例及正确状态。"),
            evidence("finish", "changed_quantized_generations_all_wrong_before_and_after", "未改變總分的具體原因。"),
        ],
        files=["row-audit.json", "cpu-test-student-kd.json", "cpu-test-student-kd-int4.json"],
        denominators={"paired_test_rows": 90, "changed_generation_rows": 6},
    )
    add(
        "concept",
        "合成色塊/兩群純音/窄模板的高分不能直接证明一般照片、語音、搜尋、通用安全与校准；擴展分布需新增獨立标注素材与檢查。",
        "能力界線段及最後兩段擴展/練習",
        "是對已核實輸入分布与未量測项的限制，不聲稱已经证明模型對所有域外输入一定失败；沒有新工具搜尋执行。",
        refs=[
            evidence(
                "holdout-doc",
                "cross_validation.rst:60–70與最終獨立評估原则",
                "新问题不能由已用小世界分数提供独立成绩。",
            ),
            evidence("core", "build_dataset / modality_tensors / calculator_runtime", "实际仅合成素材和单一加法工具。"),
            evidence(
                "selection-paper", "abstract / §5", "有限评估與选择的范围必须明确，不能拿已观察题库再选择当新证据。"
            ),
        ],
        files=["row-audit.json", "finish-checks.json", "raw/deployment/data.json"],
    )
    substantive = [c["id"] for c in claims]
    numeric = [c["id"] for c in claims if c["kind"] in ("numeric", "empirical")]
    report = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "19.12",
        "source": "course/chapters/19.md#19.12",
        "reviewer_task": "/root/fact_finish_19_12",
        "reviewer_context": "fresh",
        "source_sha256": final["section"]["sha256"],
        "figure_sha256": {},
        "verdict": "pass",
        "claims": claims,
        "sources": sources,
        "artifacts": artifacts,
        "issues": [],
        "checks": {
            "factual_accuracy": {
                "status": "pass",
                "details": "全部三張矩陣按每cell核原ID、生成/EOS、解析、runtime和最后答案，未用总分或exit0代替逐题；12版1080题与36+18 swap本人CPU全原ID重现。",
                "claim_ids": substantive,
            },
            "numeric_verification": {
                "status": "pass",
                "details": "有效分母90/84、各task與pair全部重算，参数79920由完整加载张量核；L4各10次中位排序第5/6值重算68.054/59.978ms；没有CPU冒充GPU测量。",
                "claim_ids": numeric,
            },
            "figure_consistency": {
                "status": "not_applicable",
                "details": "本節没有图片/SVG引用。三张Markdown矩阵是正文数值，已逐cell验证；不需要图渲染。",
                "claim_ids": [],
            },
            "source_verification": {
                "status": "pass",
                "details": "亲读Cawley/Talbot原PDF与固定scikit-learn原文件、实际PyTorch2.14.1 commit官方load/synchronize源碼；SHA与安装源一致。教材五个固定1df335…目标匿名取得且bytes完全相同；HF固定33c689…11权重token=False，真实hash、metadata与逐tensor记录持久保存，binary权重未提交。",
                "claim_ids": substantive,
            },
            "limitations": {
                "status": "pass",
                "details": "区分常数green与色彩swap、纯音与语音、模型请求/真实工具/读回、总参数与启用代理、main joint与DPO教师学生、GPU时测/CPU品质、短分支配方与普遍方法、同分与生成相同、未测记忆体与学生swap；原shape/tool失败保留。",
                "claim_ids": substantive,
            },
        },
        "inspection": {
            "read_snapshot": final["section"],
            "prerequisites_actually_read": [x["source"] for x in read(OUT / "read-manifest.json")["sections"][1:]],
            "stage": "第二輪技術正確性；沒有閱讀reader/其他technical/作者歷史或coordinator檔案，也未開始第三輪或閱讀估時。",
            "previous_report": "若旧报存在，原bytes仅保存到previous-report-unread.json.txt；本人不讀其結論。",
            "original_training_not_rerun": "正式模型與教師在本輪0訓練更新。现有pytest另在臨時隨機小模型上做pretrain/SFT/joint各1次更新（共3次），仅核流程，未作能力分数。原正式训练配置、有效token和阶段指紋保存于raw。GPU原逐样本记录复核，亲跑CPU品质與cache；无GPU速度或全局记忆体外推。",
            "cpu_test_rows": 1080,
            "cpu_counterfactual_rows": 54,
            "cpu_cache_inputs": 12,
            "original_validation_rows_audited": 504,
            "all_cpu_generated_ids_match_original_gpu": True,
            "first_helper_error": "首次audit误把joint子串int判成量化，loader失败；已改为明确-int4/-int8后Ruff check/format再重跑全审计。旧失败log保留，最终audit退出0。",
        },
    }
    destination = ROOT / "docs/technical-reviews/19.12.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"{len(claims)} independently verified claims; {len(sources)} sources; {len(artifacts)} persistent artifacts; report SHA256 {sha(destination)}"
    )


if __name__ == "__main__":
    main()
