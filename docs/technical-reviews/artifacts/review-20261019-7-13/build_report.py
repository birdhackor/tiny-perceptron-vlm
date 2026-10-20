"""組裝本次獨立審閱的原始判斷；只建立新報告。"""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).parent
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_7-13/technical-7-13-original-01.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


v = json.loads((A / "verification.json").read_text())
env = v["environment"]
sources = []
notes = {
    "python_json": ("Python 3.13.16", "json.loads、json.JSONDecoder 的 JSON→Python 對照表：object→dict、整數→int、true/false→True/False。"),
    "python_types": ("Python 3.13.16", "Boolean Type 的 bool/int 關係、dict.get 的預設值、字典迭代鍵與布林 and 的短路規則。"),
    "python_builtins": ("Python 3.13.16", "isinstance 包含子類別；type(object) 回傳實際類型；set 以 iterable 建構。"),
    "rfc_json": ("RFC 8259，December 2017", "Introduction、JSON Grammar、Objects、Values：JSON 是文字序列化格式，object 由名稱與值組成；語法不判定算術或安全內容。"),
    "cross_validation": ("scikit-learn 1.9.1", "Cross-validation 開頭的訓練／驗證／測試說明：反覆以測試成績挑選超參數會洩漏測試資訊；最終評估保留獨立測試集。"),
    "helm": ("Stanford CRFM，2022-11-17 官方研究說明，當日網頁版本", "Holistic evaluation 的 Multi-metric measurement、Standardization，及段末開放原始 prompts/predictions 的說明：分項指標揭露取捨，相同場景與受控適應策略才便於比較。只核對方法原則，未借用模型排名。"),
}
for item in json.loads((A / "source-excerpts.json").read_text())["sources"]:
    version, note = notes[item["id"]]
    sources.append({
        "id": item["id"], "kind": "official_docs", "title": item["title"], "url": item["url"],
        "version": version + "；讀取原檔 SHA-256 " + item["original_sha256"],
        "verified": True, "checked_original": True, "accessed_on": "2026-10-20",
        "authority_reason": "語言、規格、套件維護者或該研究團隊直接發布的官方原文。",
        "inspection_note": note + " HTTPS curl 收據、讀取日期與短摘錄見 source-excerpts.json。",
    })
t5 = json.loads((A / "metric-source.json").read_text())
sources.append({
    "id": "t5_metrics", "kind": "official_source", "title": "Google T5 evaluation metrics 原始碼",
    "url": t5["url"], "version": t5["version"], "verified": True, "checked_original": True,
    "accessed_on": "2026-10-20", "authority_reason": "Google Research 的 T5 原始評分實作。",
    "inspection_note": t5["inspection_note"] + " 原檔 SHA-256 " + t5["original_sha256"] + "；固定 commit 的收據見 metric-source.json。",
})
repo_specs = [
    ("text", "scripts/course_experiments/text.py", "run_sft、run_sft_ablation、_evaluations", "直接 SFT 從 new_lm 開始，900 次更新保存 model.pt；另一支線先 text 250 次再 SFT 900 次。後續實驗從 dependency('sft') 複製底座。text.py 的 SHA 與原始實驗紀錄一致。"),
    ("common", "scripts/course_experiments/common.py", "Context.dependency、new_lm、fit_lm、evaluate_lm", "dependency 預設 model.pt；new_lm 建立 TinyLM；evaluate_lm 比較 EOS 前原始 ID 與正解編碼，exact/eos 分別記錄。common.py 的 SHA 與原始實驗紀錄一致。"),
    ("runner", "scripts/course_experiments/run.py", "experiment_spec、execute、main", "核對 CLI、輸出根、sft 路由與計時區間。現行 runner SHA 與 sft.json 紀錄不同，且紀錄 revision 中無此路徑；不宣稱取得舊版 runner。計時範圍以原始 timing_scope 為主。"),
    ("fetch", "scripts/fetch_training_assets.py", "main", "--asset 選擇資料包；LFS pointer 才先執行 git lfs pull，隨後 unpack_asset。只讀程式與實跑 help，未下載。"),
    ("tokenizer", "tiny_perceptron/data.py", "ByteTokenizer", "UTF-8 byte 加 8，eos_id=2；以此重算原始樣本的 exact/eos。此檔 SHA 與實驗紀錄一致。"),
    ("model", "tiny_perceptron/model.py", "generate", "權重、輸入 ID、生成上限、temperature 與 eos_id 共同控制輸出；預設 temperature=0 使用 argmax。"),
    ("sft", "docs/course-experiments/results/sft.json", "SFT 原始實驗 JSON", "全文讀取並逐題重算 results.after.test：900 次、model.pt、5/10；另一支線 7/10。L4 與 31.657301130999997 秒為原始紀錄資訊，未重測。"),
    ("ablation", "docs/course-experiments/results/sft_ablation.json", "後續實驗原始 JSON", "results.before.A_attributes.test 整份物件包含逐題生成 ID，與 sft 的 results.after.test 相等。"),
    ("plan", "docs/course-experiments/plan.json", "實驗計畫的 sft 與 sft_ablation", "實際呼叫 experiment_spec/list_assets：sft 路由至 text.run_sft，資料包 ultrachat-sft。"),
    ("manifest", "assets/training/manifest.json", "ultrachat-sft 資料包定義", "以 --list-assets sft 核對 archive 路徑。"),
    ("recipe", "course/training.md", "fixed-sft-recipe", "讀取頁面明確連結的固定配方：兩條命令相同，區分 sft/model.pt 與 pretrain-sft.pt，預設產物位置一致。"),
    ("generator", "scripts/prepare_data.py", "generate_records", "本回合重新產生 attributes-sft，再依 seed=42 切分，逐題問題與正解均與紀錄一致。"),
]
for identifier, path, title, note in repo_specs:
    sources.append({"id": identifier, "kind": "repository_code", "title": title, "path": path,
                    "sha256": sha(ROOT / path), "version": "本回合工作樹原檔，版本由全檔 SHA-256 識別",
                    "verified": True, "inspection_note": note})
sources.extend([
    {"id": "local", "kind": "execution", "title": "示例、練習、CLI 與原始樣本的實際核對", "verified": True, "artifact_id": "verification"},
    {"id": "section", "kind": "execution", "title": "原文唯一 Python fence 的實跑", "verified": True, "artifact_id": "section_run"},
    {"id": "aggregation", "kind": "derivation", "title": "總分抵消的反例", "verified": True,
     "details": "分項依序為格式、內容。修改前 (0,1)，修改後 (1,0)，等權加總皆為1，內容卻由1降為0；保存分項仍可看見變動。verification.json 保存兩個向量與加總。這只證明可能性，不是本課模型品質的實測。"},
    {"id": "rubric", "kind": "derivation", "title": "字面、格式與內容的規則不同", "verified": True,
     "details": "1+1 的數值為2；字串『2』與『答案是2』不同，卻可表達同一數值。若格式限定整數欄位，仍需另判格式。分類合法標籤由任務先定義。合法 JSON 可以容納任何真假內容，因此解析成功不蘊含理由、安全邊界或事實正確。T5 多參考答案與 HELM 多維指標提供原始方法佐證；『可能需要分項審讀』是依任務需求的有限推論。"},
])

artifacts = []


def artifact(identifier, kind, name, description, **extra):
    path = A / name
    artifacts.append({"id": identifier, "kind": kind, "path": str(path.relative_to(ROOT)),
                      "sha256": sha(path), "description": description, **extra})


section_command = 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/07.md#7.13" --execute'
artifact("section_command", "execution", "section-command.txt", "指定工具命令的原始 stdout 收據。",
         command=section_command, result="exit_code=0，原文 SHA 相符，1 個 Python fence，沒有 SVG。", environment=env)
artifact("section_run", "execution", "section-execution.json", "從本回合工具目錄複製 extraction、execution、environment、stdout、stderr 與原始 fence；每份複本含原檔 SHA。",
         command=section_command, result="格式 True、內容 False；stderr 空，assert 成立，沒有 guard event。", environment=env)
artifact("source_code", "code", "collect_sources.py", "以 curl 讀取官方原文，全文只留在記憶體的程式。")
artifact("source_receipts", "source_snapshot", "source-excerpts.json", "六份官方原文的 HTTPS URL、版本、日期、原檔 SHA、curl 收據與短摘錄。")
artifact("verify_code", "code", "verify.py", "示例與練習變體、CLI、資料再生及原始實驗樣本的核對程式。")
artifact("verification", "execution", "verification.json", "包含命令、環境與結果的執行證據：逐題 ID 重算、題目再生、CLI 與原始程式核對。",
         command=v["command"], result=v["result"], environment=env)
artifact("metric_receipt", "source_snapshot", "metric-source.json", "Google T5 固定 commit 原始程式收據與短摘錄。")
artifact("report_builder", "code", "build_report.py", "以已核對的主張組裝原始判斷；只建立新報告。")

claims = []


def evidence(identifier, locator, supports):
    return {"source_id": identifier, "locator": locator, "supports": supports}


def concept(identifier, statement, location, refs, scope, artifact_ids=None):
    claims.append({"id": identifier, "kind": "concept", "statement": statement, "location": location,
                   "status": "verified", "evidence": [evidence(*ref) for ref in refs],
                   "artifact_ids": artifact_ids or [], "scope": scope})


def measured(identifier, kind, statement, location, refs, expected, observed, details, scope, artifact_ids=None, denominators=None):
    verification = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if kind == "numeric":
        verification["tolerance"] = "整數運算，精確相等。"
    if kind == "empirical":
        verification["denominators"] = denominators
    claims.append({"id": identifier, "kind": kind, "statement": statement, "location": location,
                   "status": "verified", "evidence": [evidence(*ref) for ref in refs],
                   "artifact_ids": artifact_ids or ["verification"], "scope": scope, "verification": verification})


def software(identifier, statement, location, refs, expected, observed, details, scope=None, artifact_ids=None):
    measured(identifier, "software", statement, location, refs, expected, observed, details,
             scope or "限本節固定字串與 Python 3.13.16；不宣稱這是任意模型輸出的完整驗證器。", artifact_ids)


concept("json_definition", "JSON（JavaScript Object Notation）以文字序列化資料；此例 object 含欄位名稱與值。", "JSON 定義段",
        [("rfc_json", "Introduction、JSON Grammar、Objects", "縮寫、文字格式與名稱／值。"), ("python_json", "模組開頭、JSONDecoder", "Python 的 JSON 定義。")],
        "只說明本例 object；JSON 頂層也可為其他值，未說所有 JSON 都是字典。", ["source_receipts"])
measured("arithmetic", "numeric", "1+1 的正確數值是2，所以答案3內容錯誤。", "開頭與內容檢查",
         [("local", "python_semantics.one_plus_one", "直接整數運算。")], "2", "2", "執行1+1並與固定正解2比較。", "普通整數加法。")
software("literal_reply", "reply 是示意字串，沒有呼叫模型生成。", "reply 指派與說明",
         [("section", "fence-1.py、stdout.txt", "原碼僅字串指定、解析、評分及assert。")], "字面 answer:3，無模型呼叫", "原始碼與實跑一致", "讀取並實跑原始 fence。", artifact_ids=["section_run"])
software("loads_object", "json.loads 把本例解析為 dict，answer 的值是整數3。", "json.loads 與解釋",
         [("python_json", "json.loads、JSONDecoder 轉換表", "object→dict、整數→int。"), ("section", "fence-1.py", "直接解析此字串。")],
         "dict、answer=3", "原始輸入answer:3，格式符合True", "實跑並核對官方型別對照表。", artifact_ids=["section_run", "verification", "source_receipts"])
software("dict_guard", "isinstance(parsed, dict) 判斷解析結果是不是字典。", "格式檢查與解釋",
         [("python_builtins", "isinstance", "實例及子類別判斷。")], "object條件通過、list不通過", "原例格式True；[]的格式條件False", "實跑原例與[]；後者獨立內容運算在get引發AttributeError。")
software("keys_guard", "set(parsed) 取字典全部鍵；等於{'answer'}要求鍵恰好只有answer。", "格式檢查與解釋",
         [("python_types", "Mapping Types — dict", "字典迭代鍵。"), ("python_builtins", "set", "從iterable建構集合。")],
         "answer唯一鍵可過，改名／額外鍵不過", "dict_keys=['answer']，改名／額外鍵／空字典均格式False", "執行字典轉集合與三種邊界。")
software("integer_guard", "type(value) is int 排除解碼的bool與float，只接受此解析器回傳的int。", "整數格式檢查與解釋",
         [("python_builtins", "type(object)", "實際類型。"), ("python_json", "JSONDecoder", "各JSON值的預設Python型別。")],
         "2/3可過，true/false/2.0不過", "整數原例可過；bool與float均格式False", "同一程式只替換reply執行；2.0內容True而格式False，屬不同指標。")
software("json_boolean", "JSON true、false 分別解碼成 Python True、False。", "為何不用isinstance的說明",
         [("python_json", "JSONDecoder 轉換表", "布林解碼。")], "bool、bool", "true_type=bool、false_type=bool", "直接執行json.loads('true')與json.loads('false')並取type。")
software("bool_subtype", "bool是int子類別，所以isinstance(True,int)為True。", "為何不用isinstance的說明",
         [("python_types", "Boolean Type — bool", "bool/int關係。"), ("python_builtins", "isinstance", "子類別也通過。")],
         "issubclass與isinstance均True", "兩者均True，但type(True)is int為False", "直接執行三個predicate對照。")
software("true_one", "Python的True與整數1比較相等。", "為何不用isinstance的說明",
         [("python_types", "Boolean Type — bool", "True/False數值行為對應1/0。")], "True", "true_equals_one=True", "直接執行True==1。")
software("get_missing", "dict.get未提供預設值時，缺answer回傳None，沒有KeyError。", "內容檢查解釋",
         [("python_types", "dict.get(key, default=None)", "缺鍵預設。")], "None、無KeyError", "missing_dict_get=null，result變體內容False", "執行{}.get('answer')與改名變體。",
         "限dict.get；parsed若為list，本例沒有處理，會在get引發AttributeError。")
software("original_scores", "原例印出格式True、內容False。", "print(scores)與輸出說明",
         [("section", "stdout.txt", "原文實跑輸出。")], "True、False", "格式符合=True；1+1內容正確=False", "原文與獨立重跑一致。", artifact_ids=["section_run", "verification"])
software("original_assert", "原assert要求格式通過且內容不通過，在原例中成功。", "assert與解釋",
         [("section", "execution.json、stderr.txt", "exit_code=0、沒有例外。")], "assert成立", "exit_code=0、exception=null", "核對and/not真值與實際執行。", artifact_ids=["section_run", "verification"])
software("two_scores", "reply的3改成2後，兩項分數皆True。", "練習",
         [("local", "example_variants.answer_two_original_assertion", "完整變體實跑。")], "True、True", "True、True", "只替換reply數字，保存stdout與scores。")
software("two_old_assert", "答案改2而保留原assert會失敗。", "練習",
         [("local", "example_variants.answer_two_original_assertion.exception", "捕捉AssertionError。")], "AssertionError", "AssertionError", "True and not True為False，實際執行確認。")
software("two_new_assert", "改成兩項都是True的assert後，答案2變體通過。", "練習",
         [("local", "example_variants.answer_two_both_true_assertion", "沒有例外。")], "assert成立", "exception=null", "移除內容條件的not，再執行答案2。")
software("renamed_key", "把answer改成result，即使值是2，格式與內容檢查都失敗。", "練習",
         [("local", "example_variants.renamed_result", "兩項False。"), ("python_types", "dict.get", "缺answer回傳None。")],
         "False、False", "False、False；舊assert失敗", "數字改2再只改鍵名；鍵集合不符，None也不等於2。")
concept("separate_dimensions", "格式與內容是可分開通過或失敗的評估面向。", "開頭與評估規則段",
        [("helm", "Multi-metric measurement", "分項量測與取捨。"), ("rfc_json", "JSON Grammar", "語法不定義算術正解。"), ("local", "example_variants", "直接例子。")],
        "未宣稱兩者在模型上統計獨立。", ["verification", "source_receipts"])
concept("task_rules", "比較前應明定任務規則，例如算術精確數值、分類合法標籤。", "評估規則段",
        [("t5_metrics", "sequence_accuracy、f1_score_with_invalid", "精確比對、非法標籤處理。"), ("rubric", "任務規則推導", "何謂正確取決於題目與規約。")],
        "這是規則設計例子，未說標籤合法就代表分類正確。", ["metric_receipt"])
concept("open_answer", "開放式回答可能需多個合理答案或分項審讀，不一定只比單一字串。", "評估規則段",
        [("t5_metrics", "squad、trivia_qa", "每題可有多份參考答案。"), ("helm", "Multi-metric measurement、場景分類", "依場景量測不同面向。"), ("rubric", "字面／內容反例", "依任務需求作出的有限方法推論。")],
        "多參考問答是可核對實例；分項審讀是方法建議，沒有通用自動評分保證。", ["metric_receipt", "source_receipts"])
concept("aggregation_tradeoff", "加總分數可能讓格式提升抵消事實退步；指標向量保留分項差異。", "指標向量段",
        [("helm", "Multi-metric measurement", "多指標揭露取捨。"), ("aggregation", "向量(0,1)與(1,0)", "總分相同但分項不同。")],
        "只證明可能性，沒有宣稱本課模型出現此品質變動。", ["verification"])
concept("preserve_baseline", "固定比較應保存完整題目、模型權重、tokenizer與生成設定，前後用同題重測。", "基準保存段",
        [("helm", "Standardization、開放raw prompts/predictions", "相同場景與受控策略、可追查原始回答。"), ("model", "generate", "由權重、輸入ID與生成參數共同控制輸出的資料流推論。"), ("tokenizer", "ByteTokenizer", "文字與ID的配對。")],
        "必要保存項目，不保證僅靠這些即可跨硬體／版本逐位元重現。", ["verification", "source_receipts"])
concept("validation_test", "反覆以基準挑設定使它扮演驗證資料；最終需未參與挑選的測試題。", "基準保存段",
        [("cross_validation", "開頭超參數調整、validation set與final evaluation段落", "直接說明測試洩漏與保留測試集。")],
        "針對資料與模型選擇流程，不表示只改名能恢復獨立性。", ["source_receipts"])
concept("eos_dimension", "是否正常結束要和內容成績分開記錄。", "基準保存段",
        [("helm", "Multi-metric measurement", "不同面向分項衡量。"), ("common", "evaluate_lm", "exact與eos分開定義。"), ("local", "run_record.test_recomputation", "10題EOS但只有5題正確。")],
        "EOS是本專案結束標記，不保證語義完整或正確。", ["verification"])
concept("json_limits", "缺資訊、開放建議或安全邊界的判斷，不能由JSON解析代替。", "基準保存段結尾",
        [("rfc_json", "JSON Grammar、Values", "僅定義語法與資料值。"), ("helm", "場景與多維指標", "準確、毒性等另設指標。"), ("rubric", "語法與任務推導", "合法object可包含任意真假內容。")],
        "沒有把固定JSON示例當成真實安全行為的實測。", ["source_receipts"])
software("random_sft_base", "所保留sft/model.pt從隨機參數開始，以SFT格式訓練900次。", "補充的實驗起點",
         [("text", "run_sft", "new_lm後直接fit_lm，mode=sft、steps=900、name=model。"), ("common", "new_lm、fit_lm", "模型建立與checkpoint保存。"), ("sft", "results.training、results.checkpoint", "900次與model.pt。")],
         "steps=900、model.pt、從new_lm開始", "900、model.pt、另存start.pt", "解析紀錄並核對與原始實驗相同SHA的text/common程式。", "核對既有配方與紀錄，未重跑訓練。")
measured("pretrain_comparison", "empirical", "這次pretrain-sft.pt支線的測試答對數較高，不能混用其成績與底座。", "補充的較佳支線說明",
         [("sft", "results.pretrain_then_sft.after_sft.test與results.after.test", "同次紀錄兩條支線。"), ("text", "run_sft", "分開建立模型與checkpoint。")],
         "7/10 > 5/10", "7/10 > 5/10", "實際讀取兩份test的matches與records。", "僅此seed與同一十題，不推論一般訓練效益。",
         denominators={"test_records": "10（各支線同一題組）", "seed": "42", "direct_sft_steps": "900", "pretrain_then_sft_steps": "250 text + 900 sft", "generation": "greedy，max_new_tokens=32"})
software("downstream_base", "後續錯標、遺忘與回放實驗載入sft/model.pt底座。", "補充的後續實驗引用",
         [("text", "run_sft_ablation", "load_lm(ctx.dependency('sft'))後複製base。"), ("common", "Context.dependency", "預設filename=model.pt。"), ("ablation", "results.before.A_attributes.test", "初始test完整物件與sft after test相同。")],
         "sft/model.pt；初始test相同", "預設model.pt；ablation_before_equals_sft_after=True", "解析程式AST，實際比較兩份test，包括全部生成ID。", "只核對載入路徑與保存起點，未重跑下游訓練。")
measured("baseline_accuracy", "empirical", "所保留底座在固定十題測試答對五題。", "補充的底座成績",
         [("sft", "results.after.test.samples、matches、records", "逐題原始ID與正解。"), ("common", "evaluate_lm", "EOS前原始ID完全匹配。"), ("local", "run_record.samples、test_recomputation", "本回合逐題重算。")],
         "5/10、0.5", "逐題重算5/10；10題都有EOS", "ByteTokenizer編碼正解，與生成ID的EOS前片段比較；問題與正解也和再生題組一致。", "只支持固定英文屬性提取題、seed=42與該checkpoint，不等同一般對話能力。",
         denominators={"test_records": "10", "test_families": "2", "training_records": "45", "training_families": "9", "validation_records": "5", "seed": "42", "steps": "900", "effective_training_tokens": "101091", "generation": "greedy，max_new_tokens=32，EOS=2"})
software("sample_fields", "results.after.test與samples保存答對數、問題、正解、生成ID、exact與eos。", "補充的JSON紀錄說明",
         [("sft", "results.after.test", "實讀十份samples。"), ("common", "evaluate_lm", "欄位與累計定義。")],
         "每題都有指定欄位；matches為exact合計", "欄位完整；sum(exact)=matches=5，sum(eos)=10", "逐題讀取、重算，不只核對exit code。", "messages保存輸入對話；expected另存正解。exact為EOS前ID完整匹配，與結束不同。")
software("fetch_command", "--asset ultrachat-sft是取資料包的CLI參數，也是sft配方列出的資料包。", "補充重跑命令的取資料步驟",
         [("fetch", "main", "解析--asset、選包、必要時拉LFS後解包。"), ("plan", "id=sft的assets", "指定ultrachat-sft。"), ("manifest", "id=ultrachat-sft", "archive路徑。")],
         "--asset被接受；sft列出ultrachat-sft", "help成功；list-assets輸出ultrachat-sft-v1.tar.gz", "實跑help、list-assets與experiment_spec/list_assets，另讀fetch實作。", "驗證命令契約及包名；未下載或解包，不保證遠端服務當下可用。")
software("runner_command", "--experiment sft --device cuda路由到text.run_sft，工具不自動退回CPU。", "補充重跑命令的實驗步驟",
         [("runner", "main、execute、experiment_spec", "選項、路由與CUDA不可用時的RuntimeError。"), ("plan", "id=sft", "text.run_sft。"), ("local", "commands、experiment_spec", "help及spec實跑。")],
         "接受sft/cuda；函式text.run_sft", "help列出cpu/cuda；spec為text.run_sft", "實跑CLI與spec；未執行GPU訓練。", "命令需可用CUDA與前置資料，不保證任意電腦可跑。")
software("output_directory", "實驗預設把結果與權重寫到outputs/course-experiments/course-v1/sft/。", "補充的輸出位置",
         [("runner", "main的--output預設、args.output/args.experiment、execute", "預設輸出根加實驗名稱，保存result.json。"), ("common", "fit_lm", "保存ctx.output/model.pt。"), ("sft", "artifacts", "原始產物model.pt等。")],
         "預設根outputs/course-experiments/course-v1，加子目錄sft", "main的預設與輸出拼接一致；紀錄包含model.pt", "實際解析CLI/函式AST與產物清單。", "核對預設路徑契約，本回合未建立該訓練目錄。")
measured("recorded_timing", "empirical", "紀錄中的這次執行用NVIDIA L4，elapsed_seconds約31.7秒，含訓練、評估與本地存檔，不含建置與啟動。", "補充末句",
         [("sft", "gpu、device、elapsed_seconds、timing_scope", "原始紀錄的硬體、時間與範圍。"), ("runner", "execute的started/elapsed、CUDA synchronize、timing_scope", "現行邊界與原始範圍相容，未取回舊版runner。"), ("local", "run_record.metadata、elapsed_rounded_to_tenth", "解析與四捨五入。")],
         "31.657301130999997秒→31.7，NVIDIA L4", "31.7、NVIDIA L4、cuda；scope排除image build/startup/HF uploads", "實際解析並round(value,1)，交叉讀現行計時實作。", "限『紀錄中的這次』。未重測GPU；現行runner SHA不同，紀錄revision無該路徑，未從該commit取得舊runner。這不是跨環境速度保證。",
         denominators={"recorded_runs": "1", "seed": "42", "main_sft_steps": "900", "other_branches": "pretrain 250 + continued SFT 900；UltraChat pilot 80", "recorded_device": "cuda / NVIDIA L4", "recorded_software": "Python 3.13.3；torch 2.14.1+cu126", "measurement": "完整實驗含各支線、評估與本地checkpoint；不是純900次SFT耗時"})

counts = dict(Counter(claim["kind"] for claim in claims))
ids = [claim["id"] for claim in claims]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "7.13", "source": "course/chapters/07.md#7.13",
    "reviewer_task": "/root/rv1019_technical_7-13", "reviewer_context": "fresh", "author_tasks": ["/root"],
    "source_sha256": "f5227eabfdbcc3cb426a9c86211861e218e2d7d9b5464c338a9c2c7766ae44a2",
    "source_freeze": {"path": "docs/course-review-20261019/freeze-04.json", "sha256": "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"},
    "figure_sha256": {}, "verdict": "pass", "claims": claims, "sources": sources, "artifacts": artifacts, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "逐項核對示例、練習、評估原則與補充實驗。未把字串說成模型輸出；底座、步數與後續依賴一致。", "claim_ids": ids},
        "numeric_verification": {"status": "pass", "details": "1+1=2；逐題ID重算5/10，另一支線7/10，31.657301130999997秒四捨五入31.7。實測只核對已保存紀錄，不把本回合CPU當GPU重測。", "claim_ids": [c["id"] for c in claims if c["kind"] in {"numeric", "empirical"}]},
        "figure_consistency": {"status": "not_applicable", "details": "凍結小節沒有圖；section_facts的svg_references=0，figure_sha256={}，無需PNG核對。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "以curl實讀Python 3.13.16、RFC 8259、scikit-learn 1.9.1、Stanford HELM作者官方說明與Google T5固定commit原始碼，保存原檔SHA及短摘錄。專案事實核對程式與原始實驗JSON，不使用其他審閱或候選庫摘要。", "claim_ids": ids},
        "limitations": {"status": "pass", "details": "示例不是任意模型輸出的完整驗證器；list／非法JSON的例外不是本節反例。2.0內容可過而格式不過，符合分項規則。實驗只有單seed與十題，不外推一般能力；時間限記錄的一次L4實驗，CLI未下載或訓練。runner與紀錄SHA不同，紀錄revision沒有該路徑，限制舊runner重現，不改變原始timing_scope明載的時間報告。", "claim_ids": ["get_missing", "integer_guard", "preserve_baseline", "validation_test", "baseline_accuracy", "pretrain_comparison", "fetch_command", "runner_command", "recorded_timing"]},
    },
    "review_scope": {"independence": "保存原始判斷前，未讀其他角色報告、既有本節審閱問題或非本回合工具執行的outputs，沒有參與撰寫。", "executed": "原文fence、練習變體、語言邊界、資料再生與樣本重算、CLI help與列包。", "not_reexecuted": "沒有下載權重、重新訓練、重測GPU或跨平台重現；沒有做讀者理解審閱。", "claim_counts": counts},
    "escalation_signals": {
        "A": "無 — 查核方式：本回合核對原文與freeze-04 SHA，讀取section_facts與checker，執行指定section_facts並保存收據；未發現派工現況宣稱被推翻。",
        "B": "無 — 查核方式：核對本派工全文，未隨附已駁回／不要重提清單，無可核對的編號區間。",
        "C": "無 — 查核方式：只新增指定報告與artifact檔案；section_facts使用派工明許的工具輸出例外。",
        "D": "無 — 查核方式：未修教材；讀取凍結小節及直接相關程式，本次新增檔位置沒有舊修復痕跡。",
    },
}
assert sha(ROOT / "docs/course-review-20261019/freeze-01/sources/7.13.md") == report["source_sha256"]
assert sha(ROOT / report["source_freeze"]["path"]) == report["source_freeze"]["sha256"]
with REPORT.open("x", encoding="utf-8") as target:
    json.dump(report, target, ensure_ascii=False, indent=2)
    target.write("\n")
print(json.dumps({"report": str(REPORT.relative_to(ROOT)), "sha256": sha(REPORT), "claim_counts": counts, "total": len(claims), "verdict": "pass"}, ensure_ascii=False))
