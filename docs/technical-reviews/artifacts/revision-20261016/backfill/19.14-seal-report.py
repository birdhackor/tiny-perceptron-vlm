"""Assemble the independently checked 19.14 original report and validate it."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).parent
TASK = "/root/backfill_tech_19_14"
REPORT = ROOT / "docs/course-revision-20261016/reviews/technical-19.14-backfill-original-01.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, references, scope, verification=None):
    result = {
        "id": identifier,
        "kind": kind,
        "statement": statement,
        "location": location,
        "status": "verified",
        "evidence": references,
        "artifact_ids": ["a_audit"] if verification else ["a_official"] if kind == "concept" else [],
        "scope": scope,
    }
    if verification:
        result["verification"] = verification
    return result


def executed(expected, observed, details):
    return {"method": "executed", "expected": expected, "observed": observed, "tolerance": "整數與SHA均精確相等。", "details": details}


def main():
    from scripts.reading_time import lesson_slices
    from scripts.check_technical_reviews import _validate

    audit = json.loads((BASE / "19.14-audit.json").read_bytes())
    receipts = json.loads((BASE / "19.14-official-source-receipts.json").read_bytes())
    source = ROOT / "course/chapters/19.md"
    body = lesson_slices(source.read_text())["19.14"]
    assert hashlib.sha256(body.encode()).hexdigest() == audit["source_sha256"]
    sources = []
    notes = {
        "s_group": "實際讀§3.1開頭對validation/test與超參數過擬合的說明，及§3.1.2.4、§3.1.2.4.1對dependent groups與heldout groups的條件。這支持家族隔離與評估單位，不證明本課模型有一般語義能力。",
        "s_leak": "實際讀§11.2與§11.2.1；test不能用於選擇模型，提前分資料及限制fit範圍。支持看錯題後再修訓練已將該批資料用作開發，不能維持未使用最後test的稱呼。",
        "s_cal": "實際讀§1.16.2：calibrator應使用獨立於classifier fitting的資料；§1.16.3.1也明述calibration方法的條件。僅用於支持獨立校準資料的角色，不將sigmoid公式當作本課temperature實作。",
        "s_hash": "實際讀Hash algorithms與hash.hexdigest()：sha256接受bytes、hexdigest回傳hex字串。範例在Python3.13.5 CPU親跑；文件是3.12.15版且所用SHA-256介面一致。",
    }
    titles = {"s_group": "Cross-validation: evaluating estimator performance", "s_leak": "Common pitfalls: Data leakage", "s_cal": "Probability calibration: Calibrating a classifier", "s_hash": "Python hashlib"}
    for entry in receipts["sources"]:
        ident = entry["source_id"]
        sources.append({"id": ident, "kind": "official_docs", "title": titles[ident], "url": entry["url"], "version": entry["version"], "verified": True, "checked_original": True, "accessed_on": entry["accessed_on"], "authority_reason": "官方維護者發布的版本化文件；直接說明該評估方法或標準庫API的契約。", "inspection_note": notes[ident], "fulltext_sha256": entry["fulltext_sha256"]})
    code_notes = {
        "s_frozen": ("docs/course-revision-20261016/freeze-10/sources/19.14.md", "完整讀本節；以lesson_slices比對工作樹與freeze-14指向的snapshot逐位元組一致，不將整章列為已查證。"),
        "s_data": ("tiny_perceptron/lesson_products/data.py", "讀family/split_of、render、arithmetic_worlds、choose_salt、box_worlds、service_worlds、c_rows、build_rows、observable_key、audit_rows與write_dataset。分清核心四分組和只由test衍生的shift測試。"),
        "s_model": ("tiny_perceptron/lesson_products/model.py", "讀Tokenizer建構／encode／save／load與audit_tokens；核對每字一個唯一ID和question/candidate分隔。神經網路部分僅為理解檔案上下文，本報告不驗收訓練品質。"),
        "s_eval": ("tiny_perceptron/lesson_products/evaluation.py", "讀auto_inputs、select_settings的calibration與policy資料選擇、freeze來源身份、frozen_guard、evaluate_frozen一次啟動、shift輸出與最終count產生方式；未執行任何模型推論。"),
        "s_train": ("tiny_perceptron/lesson_products/train.py", "讀129–205行train/policy資料流、更新、選best checkpoint與test不參與選擇的程式路徑。"),
        "s_manifest": ("docs/lesson-products/data-v2-manifest.json", "完整讀v2 config、B/C audit、family policy、核心切分統計及原始檔SHA/列數；親自重算原始JSONL而非只相信manifest。"),
        "s_token": ("docs/lesson-products/data-v2-token-audit.json", "完整讀字元token audit；結合Tokenizer的單射映射及原始text觀測重算核對。未重新跑40秒token audit。"),
        "s_pretest": ("docs/lesson-products/results/pretest-freeze-record-v1.json", "完整讀實際pretest receipt、資料來源與預先選擇。此為既有primary receipt，不是審閱者親跑訓練。"),
        "s_freeze": ("docs/lesson-products/results/settings-freeze-v1.json", "讀資料／source SHA、calibration的per-question溫度及policy search選擇記錄、test_outcomes_observed=false；親自核對freeze SHA與start/final記錄一致。"),
        "s_started": ("docs/lesson-products/results/settings-freeze-v1.test-started.json", "完整讀正式啟動marker，核對時間晚於pretest與Git凍結commit，且SHA一致；不把marker當作全世界不可繞過的鎖。"),
        "s_final": ("docs/lesson-products/results/final-report-v1.json", "檢查recorded_at、資料／settings SHA、B/C核心count、shifts種類與scope；未重新產生模型分數，也未以本節審閱覆蓋無關的latency/mechanism品質結論。"),
        "s_cal_receipt": ("docs/lesson-products/results/c-calibration.json", "完整讀per-question temperatures、selected_on與calibration分母；只核對選設定用途，不重新驗證其quality數值。"),
        "s_policy_receipt": ("docs/lesson-products/results/b-auto-policy.json", "完整讀policy-only規則、輪數、事件數門檻與grid_count；只核對用途與來源。"),
        "s_generation": ("docs/lesson-products/data-v2-generation-receipt.json", "完整讀生成器/tokenizer/manifest SHA與生成時間；對照當前來源。"),
    }
    for ident, (path, note) in code_notes.items():
        sources.append({"id": ident, "kind": "repository_code", "title": path, "path": path, "sha256": digest(ROOT / path), "version": "finite-worlds-v2 / existing formal v1 receipts; inspected at git " + audit["git_head"], "verified": True, "inspection_note": note})
    sources.append({"id": "s_audit", "kind": "execution", "title": "審閱者原始資料唯讀重算", "verified": True, "artifact_id": "a_audit"})
    sources.append({"id": "s_arithmetic", "kind": "derivation", "title": "加數交換的家族等價關係", "verified": True, "details": "在題目指定的『測未見加數組合』契約下，a+b與b+a代表同一無序multiset：sort([2,3])=sort([3,2])=[2,3]，且2+3=3+2=5。三數正式生成器用combinations_with_replacement生成排序triple，再把set(permutations(triple))的各排列掛同一fid。這是預先定義測試單位，不保證其他目標也必須同組。"})
    claims = [
        claim("c1", "concept", "要評估未見底層事件，相關改寫、不同問題與候選重排需以共同家族分組，避免同事件落在訓練與最後測試。", "frozen 19.14 lines 3–5", [evidence("s_group", "§3.1.2.4 and §3.1.2.4.1", "dependent samples要以group隔離，驗證家族不應出現在對應訓練資料"), evidence("s_leak", "§11.2", "test資訊參與建模會令novel-data評估樂觀")], "只防止定義好的家族跨核心集合；不自動防不同家族映成相同觀測或所有語義等價關係。"),
        claim("c2", "software", "範例以同一family_id計算SHA-256分桶，三個variant標籤的分組輸出相同。", "frozen lines 9–22 code", [evidence("s_hash", "Hash algorithms; hash.hexdigest()", "bytes到SHA-256 hex摘要的API"), evidence("s_audit", "snippet_stdout", "從凍結正文直接抽出程式並執行")], "範例是穩定分配；7/2/1分桶不保證有限資料精確比例，也不是正式v2的76/8/8/8配方。", executed("三行相同split", "原句 train；改問資訊是否充分 train；候選倒序 train", "直接exec凍結程式區塊，沒有改寫輸入、沒有建立或訓練模型。")),
        claim("c3", "software", "正式B/C的核心切分以家族共同分派，具體題目id與family_id分工不同；Auto開發與測試材料沿用相應B集合。", "frozen lines 5–7", [evidence("s_data", "family/split_of lines 51–57; c_rows 187–212; build_rows 215–252", "同底層世界繼承fid，B/C核心split由同一salt/fid決定；id由具體欄位計算"), evidence("s_eval", "auto_inputs lines 67–77; select_settings 220–224; evaluate_frozen 396–401", "路由材料直接由已分組B rows衍生，不另隨機切分")], "相同split指train/calibration/policy/test的核心partition。wording_shift/entity_shift是同一test家族的另外測試類別，不是額外訓練集合；long_shift家族不在核心集合。", executed("B/C跨核心家族重疊0，所有核心列符合共同hash split", "joint_core_family_overlap=0；B/C core_wrong_hash_split均0", "逐列重算salt+fid的SHA分桶並跨B/C建立family→core-split集合，未只檢查題目ID。")),
        claim("c4", "concept", "未見家族與大量題目列，不能單獨證明全新語義、一般中文理解或同樣多個獨立世界。", "frozen lines 22, 26, 28", [evidence("s_group", "§3.1.2.4 grouped data", "同來源樣本有依賴，group generalization的評估目標受group定義約束"), evidence("s_manifest", "audit.c scope; family_counts", "本課是固定語法，C同世界衍生多列並且存在跨家族同觀測"), evidence("s_final", "scope", "正式報告明限finite fixed grammar")], "正文沒有把家族數當統計獨立性證明，也沒有用固定語法高分推廣至open-domain；本節未聲稱或驗收模型一定採用哪種內部推理策略。"),
        claim("c5", "concept", "校準與策略選擇屬開發工作；最後test應在設定固定後才評分，已依test錯題修訓練的資料不能再當未使用的最後測試。", "frozen line 24", [evidence("s_group", "§3.1 opening, validation/test discussion", "反覆依test選超參數會令知識洩入模型；validation用於開發而test留到最後"), evidence("s_leak", "§11.2–11.2.1", "test資料不能用於模型選擇"), evidence("s_cal", "§1.16.2", "calibrator應fit於獨立於classifier training的資料")], "校準只在固定任務分布有對應解讀；留新test是恢復未參與開發評估的契約，不保證任意新資料可代表所有部署情境。"),
        claim("c6", "software", "本版C溫度只在calibration選，Auto輪數與事件門檻在policy選，正式test的既有記錄晚於設定凍結。", "frozen line 24", [evidence("s_eval", "select_settings 203–280; frozen_guard 302–327; evaluate_frozen 368–410", "calibration/policy分流、設定SHA與exclusive marker先於正式推論"), evidence("s_train", "129–205", "梯度使用train，checkpoint選擇使用policy"), evidence("s_pretest", "recorded_at, test_command, test_outcomes_observed", "實際測試前primary receipt"), evidence("s_started", "recorded_at, settings_freeze_sha256", "正式啟動身份"), evidence("s_cal_receipt", "selected_on", "calibration-only溫度"), evidence("s_policy_receipt", "selected_on", "policy-only輪數及門檻")], "審閱者親跑的是receipt身份／時間核對，不是再訓練或正式推論。一次性marker只限制同一freeze路徑，不能證明沒有人在其他機器看過複本。", executed("freeze identity一致且pretest < test start < report", "pretest 07:31:35.864691 < start 07:33:06.462608 < final 07:35:04.193466 UTC；SHA 1a221870…80c44一致；Git凍結commit為07:32:43Z", "讀既有正式primary receipts並核對SHA、資料manifest身份、時間序；不重啟test。")),
        claim("c7", "numeric", "正式v2原始資料共B 47,673列、C 420,110列，包含核心四組與變化測試。", "frozen line 26 first sentence", [evidence("s_manifest", "files.b.jsonl and files.c.jsonl", "固定SHA與總列數"), evidence("s_audit", "products.b.rows; products.c.rows; counts", "直接掃描原始JSONL重算")], "這是資料列數，包含同世界多個問題和test衍生shift，不能當獨立世界數或訓練樣本數。", executed("B=47673，C=420110", "B=47673，C=420110", "先驗證兩原始檔SHA等於固定manifest，再逐行json.loads計數，保留各split/task或question分母。")),
        claim("c8", "numeric", "核心test的B為2,998題（2,956追蹤、42算術）；C為33,577題（11,797 Choice、10,890 info、10,890 rubric）且554家族。", "frozen line 26 second sentence", [evidence("s_manifest", "audit.b.counts; audit.c.counts and family_counts.test", "核心測試分母"), evidence("s_audit", "products.*.counts and family_counts", "原始列重算相同"), evidence("s_final", "b.5.{boxes,sum}.count; c.{choice,info,rubric}.after.count", "正式報告使用同一核心test分母")], "554是C核心test家族；B核心test另有1488家族。措辭／實體測試是相同test底層世界的變體，未另作獨立世界累加。", executed("B 2956+42=2998；C 11797+10890+10890=33577；C family=554", "各項逐一相等，C test family=554", "依split==test精確過濾原始列，按task/question計數並用set(family_id)去重，對照正式報告count。")),
        claim("c9", "numeric", "B跨split相同輸入0；C有5,453種跨split相同觀測，涉及215,427列，沒有衝突gold標籤。", "frozen line 28", [evidence("s_data", "observable_key 271–274; audit_rows 277–311", "B以state；C以state、question、候選語義集合定義完整可觀測題目"), evidence("s_manifest", "audit.b and audit.c duplicate fields", "正式資料稽核"), evidence("s_token", "b and c", "實際token audit與text觀測一致"), evidence("s_model", "Tokenizer.encode; audit_tokens", "逐字唯一token映射與候選規範化"), evidence("s_audit", "products.{b,c}.core_* and all_split_*", "獨立從原始列重算重複集合、列數與gold衝突")], "5,453按完整觀測而非僅state文字計；215,427包含這些觀測在全部相關核心組的所有列，並非僅test列。額外掃描所有split仍得到相同跨split數量；不表示一般語義去重完備。", executed("B shared=0；C shared=5453，affected rows=215427，conflicts=0", "B 0；C 5453／215427／0，核心與所有split統計一致", "B key=state；C key=(state,question,sorted candidates)。每key收集split集合、gold集合及列數；shared要求split集合大小>1，conflict要求gold集合大小>1。")),
        claim("c10", "software", "正式結果分別列核心未見事件家族、實體更名、措辭變化與較長序列，這些測試的改變可以由生成器定位。", "frozen line 22", [evidence("s_data", "render 78–104; build_rows 215–252", "措辭只換任務prefix；實體A/B→P/Q；核心事件長1–4，長序列5–6"), evidence("s_eval", "evaluate_frozen 435–465", "分別產生wording/entity/long與C wording結果"), evidence("s_final", "shifts keys and subgroup counts", "正式報告實際含四類shift結果")], "更名與措辭變化仍是預先公開的有限字元表，不是開放中文；wording/entity只從core test衍生；long家族與所有core集合不重疊。", executed("報告包含wording_shift/entity_shift/long_shift/c_wording；shift家族不流入train/dev", "四keys齊全；wording/entity家族全屬core test；long 1600家族不在core", "直接檢查既有報告keys與原始資料所有family→split集合，不生成新分數。")),
        claim("c11", "software", "v2保存可追溯的生成器版本、seed／世界規則、家族及題目欄位、檔案SHA與切分統計。", "frozen line 30", [evidence("s_data", "REQUIRED/EVENTS; family; c_rows; build_rows config; write_dataset", "生成規則與欄位位置、主要seed6021與service seed1349、家族與題目id"), evidence("s_generation", "generator_sha256, tokenizer_implementation_sha256, data_manifest_sha256", "實際生成primary receipt pin版本"), evidence("s_manifest", "config, audit, files", "固定資料配方、統計與檔案指紋")], "驗收目前正式v2及其保留生成器；不聲稱所有未來版本都已存在，不把manifest當完整欄位教學或資料內容本身。", executed("生成收據pin住目前generator/tokenizer/manifest，原始資料符合manifest", "generator efcf4ca5…c202；tokenizer d779b641…cf4e；manifest dea64eac…0bb；B/C SHA逐檔吻合", "讀生成收據與實作、核對原始B/C檔全檔SHA，來源與小型執行收據在本報告保存。")),
        claim("c12", "concept", "若評估目標是未見加數組合，2+3與3+2應先以排序後加數合併同一家族。", "frozen lines 30–32 exercise", [evidence("s_group", "§3.1.2.4", "group定義依domain和未見群組的評估目標決定"), evidence("s_arithmetic", "完整sort及交換律推導", "兩式屬同一無序加數組合，不能分別落train/test仍稱新組合"), evidence("s_data", "arithmetic_worlds 129–133", "正式三加數資料的sorted triple與permutation共用fid")], "這是特定測試目標的group contract。若目標改測表示順序，可能選別種切法，須事前定義而非看成績後改。"),
    ]
    artifacts = [
        {"id": "a_audit", "kind": "execution", "path": (BASE / "19.14-audit.json").relative_to(ROOT).as_posix(), "sha256": digest(BASE / "19.14-audit.json"), "description": "親跑凍結範例、原始資料列數／家族／觀測重算及既有receipt身份核對；沒有模型訓練或正式推論。", "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/19.14-audit.py", "result": "exit 0；B/C raw SHA、各正文分母及重複觀測吻合；核心family overlap=0；正式receipt的freeze身份／時間相符。", "environment": audit["environment"]},
        {"id": "a_code", "kind": "code", "path": (BASE / "19.14-audit.py").relative_to(ROOT).as_posix(), "sha256": digest(BASE / "19.14-audit.py"), "description": "可重讀的唯讀資料稽核程式；直接掃描raw JSONL，不呼叫模型推論或GPU入口。"},
        {"id": "a_official", "kind": "source_snapshot", "path": (BASE / "19.14-official-source-receipts.json").relative_to(ROOT).as_posix(), "sha256": digest(BASE / "19.14-official-source-receipts.json"), "description": "實際讀取的官方原文URL、版本、完整下載檔SHA、小段支持引文與定位；全文在/workspace/work，未放Git。"},
    ]
    report = {
        "schema_version": 1, "review_stage": "technical", "lesson_id": "19.14", "source": "course/chapters/19.md#19.14", "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": ["/root"], "source_sha256": audit["source_sha256"], "figure_sha256": {}, "verdict": "pass", "claims": claims, "sources": sources, "artifacts": artifacts, "issues": [],
        "review_scope": "只審freeze-14所指19.14。原始JSONL掃描與範例為親跑；資料生成／訓練／校準／formal test是既有primary receipts，沒有重訓、改參數或重啟test。未讀作者plan、處置或其他／舊review結論。無圖，不宣稱頁面版面或真人初學者驗收。",
        "checks": {
            "factual_accuracy": {"status": "pass", "details": "逐項核對家族分組、觀測與獨立世界的區分、validation用途與有限語法限制；正式切分遵守核心家族partition，shift屬最後測試變化類別。", "claim_ids": ["c1", "c3", "c4", "c5", "c6", "c10", "c11", "c12"]},
            "numeric_verification": {"status": "pass", "details": "親自從兩個固定SHA原始JSONL重算47673/420110、2998/33577、554家族及5453/215427；整數精確相等。凍結範例三行均train。", "claim_ids": ["c2", "c7", "c8", "c9"]},
            "figure_consistency": {"status": "not_applicable", "details": "本節沒有SVG、img或其他必要圖引用；figure_sha256為空。本報告不把沒有圖的情況寫成已render。", "claim_ids": []},
            "source_verification": {"status": "pass", "details": "真正讀取scikit-learn 1.7.2與Python 3.12.15官方相關原文，保留URL/定位/全文SHA小收據；程式、凍結正文、manifest與formal receipts只pin實際核對檔案與範圍。", "claim_ids": [c["id"] for c in claims]},
            "limitations": {"status": "pass", "details": "核心family disjoint不能替代觀測去重或一般語義泛化；C duplicates是同標籤固定語法且215427不是test獨立題；554只指C test。一次啟動與時間收據不證明所有機器上的不可繞過性，未把資料CPU掃描當作親跑訓練或模型品質證據。", "claim_ids": ["c1", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c12"]},
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    errors = _validate(ROOT, source, "19.14", body, report, {TASK: ["19.14"]}, set())
    validation = {"report": REPORT.relative_to(ROOT).as_posix(), "report_sha256": digest(REPORT), "reviewer_task": TASK, "checker": "scripts.check_technical_reviews._validate(root,source,'19.14',body,report,{actualTask:['19.14']},set())", "errors": errors, "scope": "僅格式／身份／來源SHA與證據存在自查；實質查證見報告逐項記錄。"}
    (BASE / "19.14-format-validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    assert not errors


if __name__ == "__main__":
    main()
