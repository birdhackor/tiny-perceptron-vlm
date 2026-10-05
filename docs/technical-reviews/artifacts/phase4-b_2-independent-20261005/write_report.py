"""Create only this reviewer's complete B.2 report; never load the legacy report."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
P = OUT.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_b_2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


body = (OUT / "section.md").read_bytes()
raw = (ROOT / "course/chapters/0B.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
index = next(i for i,h in enumerate(headers) if h[0].startswith(b"## B.2 "))
current = raw[headers[index].start():headers[index+1].start()]
assert current == body
env = {"python":"3.13.5", "torch":"2.14.1+cpu", "device":"cpu", "cuda_build":"None", "cuda_available":"False"}
commands = json.loads((OUT / "commands.json").read_text())["commands"]
special_ids = {
    "original-fence-stdout.txt":"original-fence-run", "verification-stdout.txt":"bounded-cpu-run",
    "regression-stdout.txt":"existing-regression-run", "render-stdout.txt":"browser-render-run",
    "verification-results.json":"bounded-cpu-details", "verify_b2.py":"bounded-cpu-code",
    "inputs/docs/course-experiments/results/tools.json":"original-tools-raw",
    "inspection.md":"inspection", "section.md":"section-frozen",
    "desktop.png":"desktop-render", "mobile.png":"mobile-render",
    "sources/official-inspected-excerpts.json":"official-excerpts",
}
executions = {"original-fence-run":commands[0], "bounded-cpu-run":commands[1],
    "existing-regression-run":commands[2], "browser-render-run":commands[4]}
results = {
    "original-fence-run":"exit 0; original fence prints 結果 5 and two expected rejections; stderr empty",
    "bounded-cpu-run":"exit 0; exact schema, strict JSON, true user_id=3 through actual _sample, EOS, [] and finite-overflow checks pass; seven original injections reproduced and twenty stored episodes checked",
    "existing-regression-run":"exit 0; 12 passed, 3 unrelated cases deselected",
    "browser-render-run":"exit 0; actual Chromium desktop/mobile render, expanded details; both PNGs viewed by reviewer; no referenced diagram",
}
artifacts=[]
for path in sorted(OUT.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(OUT).as_posix()
    identifier = special_ids.get(rel,"snapshot-"+rel.replace("/","-"))
    kind = "source_snapshot"
    if path.suffix == ".py": kind="code"
    if path.suffix == ".png": kind="figure_render"
    if rel == "inspection.md": kind="derivation"
    item={"id":identifier,"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),
        "kind":kind,"description":"Personally retained B.2 evidence: "+rel+"; frozen original bytes where under inputs/sources. Actual inspected ranges/pointers are recorded in inspection.md."}
    if identifier in executions:
        item.update(kind="execution",command=executions[identifier]["command"],result=results[identifier],environment=dict(env))
        if identifier == "browser-render-run":item["environment"].update(chromium="151.0.7922.173",markdown="3.11",playwright="1.63.0")
        if identifier == "existing-regression-run":item["environment"]["pytest"]="9.1.1"
    artifacts.append(item)


def official(identifier,name,url,version,authority,note):
    return {"id":identifier,"kind":"official_docs","title":name,"url":url,"version":version,
        "authority_reason":authority,"verified":True,"checked_original":True,"accessed_on":"2026-10-05","inspection_note":note}


def repo(identifier,name,path,note):
    target=OUT / "inputs" / path
    return {"id":identifier,"kind":"repository_code","title":name,"path":target.relative_to(ROOT).as_posix(),
        "sha256":sha(target),"version":"Frozen current original bytes on 2026-10-05; complete SHA identifies version. Data/retrieval/applications also match tools.json's recorded code_sha256.",
        "verified":True,"inspection_note":note}


sources = [
    official("rfc8259","RFC 8259, The JavaScript Object Notation (JSON) Data Interchange Format",
        "https://www.rfc-editor.org/rfc/rfc8259.txt","December 2017, Standards Track RFC 8259, obsoletes RFC 7159",
        "RFC Editor publishes the original IETF Standards Track JSON specification.",
        "Personally read sections 2–6, especially object names SHOULD be unique, array as JSON value, number grammar excludes NaN/Infinity and permits implementation range limits. Original HTTPS bytes and SHA retained in sources/rfc8259.txt and fetch-provenance.json."),
    official("python-json","Python json decoder and encoder official reference",
        "https://docs.python.org/release/3.13.5/library/json.html","Python 3.13.5 release documentation",
        "Python project's official version-pinned documentation, matching this CPU runtime.",
        "Read #json.load/#json.loads/#json.JSONDecoder hooks, #infinite-and-nan-number-values and #repeated-names-within-an-object. parse_constant/parse_float/object_pairs_hook enable the local stronger contract; defaults accept nonfinite constants and keep the last duplicate. Original page title explicitly identifies Python 3.13.5; snapshots and exact excerpts retained."),
    official("python-bool","Python built-in Boolean type official reference",
        "https://docs.python.org/release/3.13.5/library/stdtypes.html","Python 3.13.5 release documentation",
        "Python project's official language/library type contract.",
        "Read #boolean-type-bool: bool is a subclass of int and True/False behave like 1/0 in many numeric contexts. This supports the need for an exact type check in this local numeric schema, not a universal rule banning Boolean values from JSON."),
    official("python-math","Python math.isfinite/isnan official reference",
        "https://docs.python.org/release/3.13.5/library/math.html","Python 3.13.5 release documentation",
        "Python project's official definitions for the predicates used by the parser/controller.",
        "Read #math.isfinite and #math.isnan; finite excludes infinities and NaN, which means not a number. Actual 1e308/1e999/product behavior was separately executed on the CPU runtime."),
    repo("retrieval","Actual minimal add/multiply call_tool contract","tiny_perceptron/retrieval.py","AST located call_tool, read lines 23–34; exact outer keys, name allowlist, exact argument keys/type, and actual arithmetic operation."),
    repo("byte-tokenizer","Actual byte tokenizer and dedicated role/EOS ID contract","tiny_perceptron/data.py","AST located ByteTokenizer, read SPECIALS and lines 14–28: IDs 0–7 are special, user=3/EOS=2; ordinary UTF-8 bytes offset 8; decode skips special IDs. No assumption that visible word user or numeral 3 is a special token."),
    repo("applications","Actual sampler, strict JSON parser and tool episode controller","scripts/course_experiments/applications.py","AST located/read _sync 28–30, _prompt_ids 33–38, _sample 42–103, _json 458–459, _tool_records 462–493, _parse_json_action 496–524, _tool_episode 527–605 and run_tools 608–676. Read only method/protocol/data-generation contracts, not other experiments' result interpretations. Original code SHA matches tools.json."),
    repo("regression-code","Original relevant CPU protocol regression cases","tests/test_application_protocols.py","Read original relevant tests and injection helper through line 109, plus lines 112–118 while inspecting the precise slice. Executed only parser/nonfinite/overflow/valid tool/hidden token cases; unrelated RAG/reasoning cases deselected."),
    {"id":"original-execution","kind":"execution","title":"B.2 original fence, actual CPU execution","verified":True,"artifact_id":"original-fence-run"},
    {"id":"bounded-execution","kind":"execution","title":"Independent bounded CPU checks and original raw-record inspection","verified":True,"artifact_id":"bounded-cpu-run"},
    {"id":"regression-execution","kind":"execution","title":"Actual existing protocol regression run","verified":True,"artifact_id":"existing-regression-run"},
]


def ref(source_id,locator,supports):
    return {"source_id":source_id,"locator":locator,"supports":supports}


def verified(identifier,kind,statement,location,scope,evidence,artifact_ids,expected=None,observed=None,details=None,tolerance=None,denominators=None):
    item={"id":identifier,"kind":kind,"statement":statement,"location":location,"scope":scope,
        "status":"verified","evidence":evidence,"artifact_ids":artifact_ids}
    if kind != "concept":
        item["verification"]={"method":"executed","expected":expected,"observed":observed,"details":details}
        if tolerance is not None:item["verification"]["tolerance"]=tolerance
        if denominators is not None:item["verification"]["denominators"]=denominators
    return item


claims = [
    verified("json-contract","concept","解析JSON結構與核對允許內容是不同步驟；格式完整不保證工具名稱/參數合法，更不會自行執行。","B.2 開場與 allowlist/schema 段",
        "JSON一般資料結構由RFC支持；allowlist/schema在本例指本地手寫契約，未宣稱完整JSON Schema標準實作或通用執行安全。",
        [ref("rfc8259","§§2–6 (JSON grammar; object/array/number)","JSON是結構化資料表示；物件與陣列可為合法JSON資料。"),
         ref("retrieval","call_tool lines 23–34","先json.loads，再name/arguments/a/b檢查，最後才執行加/乘。")],
        ["section-frozen","inspection"]),
    verified("minimal-call-tool","software","本例只允許 add/multiply、name/arguments、数值a/b；原fence回5且拒未知工具/bool。精確型別檢查也拒字串\"2\"。","B.2 Python fence及其後第一段",
        "只支持此最小入口的明示欄位/型別契約；不把嚴格JSON/非有限值保證套給它。",
        [ref("retrieval","call_tool lines 23–34","完整本例方法契約與返回值。"),
         ref("python-bool","#boolean-type-bool","True屬int子類；type(True)不等於int。"),
         ref("original-execution","original-fence-stdout.txt; fence-1.py","原始三份輸入真正執行而得所述結果。"),
         ref("bounded-execution","verification-results.json#/minimal_schema","獨立核對multiply、字串/bool、額外欄位與錯參數變體。")],
        ["original-fence-run","bounded-cpu-run","bounded-cpu-details","bounded-cpu-code"],
        "原樣輸出5/未知工具拒絕/bool拒絕；multiply(2,3)=6；string及額外欄位被拒。",
        "原fence exit0，stdout精確符合；五項非法schema變體全拒絕。",
        "原fence使用section_facts CPU guard執行；另用本地實函式有界核對，並確認isinstance(True,int)=True但type(True)is int=False。"),
    verified("legal-wrong-parameters","numeric","2+3=5，但合法請求add(2,4)回6；合法格式不能證明參數符合原題。","B.2『原題2+3、請求2+4』段",
        "手工參數變體支持分開格式/參數/執行；未拿玩具變體冒充模型錯誤成績。",
        [ref("retrieval","call_tool line 34","直接使用請求a/b運算，不修正為題目答案。"),
         ref("bounded-execution","verification-results.json#/minimal_schema/valid_add and /legal_wrong_parameter_result","實函式回5及6。")],
        ["bounded-cpu-run","bounded-cpu-details"],"add(2,3)=5，add(2,4)=6。","觀察5及6，均為Python整數。","分母不適用；精確整數加法且本例沒有資料集/模型評分。",tolerance="exact integer equality; no rounding"),
    verified("strict-json-layer","software","外層在call_tool前拒NaN/±Infinity、浮點指數溢位1e999及任意層級重複欄位；最小入口沒有這些保證。","B.2 正文嚴格JSON段與補充第二段",
        "這是本地controller更強的數值/唯一欄位契約。RFC本身允許陣列並用SHOULD描述唯一名稱；不是所有JSON解碼器的預設保證。",
        [ref("python-json","#json.load/#json.loads/#json.JSONDecoder; #infinite-and-nan-number-values; #repeated-names-within-an-object","原始官方hook語義及Python預設寬鬆行為。"),
         ref("rfc8259","§4 unique names SHOULD; §6 number syntax/range","非有限常量非JSON數字；实现可設定範圍限制。"),
         ref("applications","_parse_json_action lines 496–524; _tool_episode lines 549–559","parse_constant/parse_float/object_pairs_hook與控制器先解析再執行順序。"),
         ref("bounded-execution","verification-results.json#/strict_parser and /minimal_not_strict","九項真拒絕包含±Infinity、±1e999、巢狀/escape後重複鍵；最小入口接受非有限且最後重複name勝出。")],
        ["bounded-cpu-run","bounded-cpu-details","bounded-cpu-code","official-excerpts"],
        "外層九項strict違例解析一次/工具零次；最小入口未設定hooks。",
        "九项均以正確finite/duplicate原因拒；parser=1/tool=0；最小入口NaN/Infinity/1e999接受，重複name產生multiply結果6。",
        "經實際_sample ID處理再進_controller；以更深object與\\u0061重複鍵作必要小變體。既有8項parser回歸亦通過。"),
    verified("control-ids-and-eos","software","專用user角色ID混入請求會在JSON解析前拒；EOS是正常停止標記。不能只看解碼後JSON是否完整。","B.2 補充『控制 token』新例與7.2/7.9引用",
        "已真正核驗本地ByteTokenizer user_id=3的新增例；是注入ID/controller測試，不是原七份JSON文字故障或新模型成績。字面'user'與普通byte數字3不是同一物。EOS缺失本身不是此controller的解析拒絕條件。",
        [ref("byte-tokenizer","SPECIALS lines 10–11; ByteTokenizer lines 14–28","user=3、EOS=2及decode跳過所有<8專用ID；不是把專用標記當JSON字詞。"),
         ref("applications","_sample lines 66–90; _tool_episode lines 533–555","先按EOS停止；保存generated_ids；分類raw<8；_tool_episode在_parse_json_action前檢查invalid_special_tokens。"),
         ref("bounded-execution","verification-results.json#/control_id_group (control_id=3); #/eos","真_sample用手寫logit/ID序列產生可見合法JSON但保留[3]，spy證實parser=0/tool=0；正常EOS、早停EOS、token-budget停止均核。")],
        ["bounded-cpu-run","bounded-cpu-details","bounded-cpu-code","desktop-render","mobile-render"],
        "混入user_id=3：可見JSON仍相同，invalid_special_tokens=[3]，解析0次/工具0次。末尾EOS retained IDs且停止，不列非法ID。",
        "user_id=3及其餘六個非EOS special ID同組全拒：parser=0/tool=0。末尾EOS後的user ID從未生成；早EOS截斷JSON導致parse拒；token上限亦可無EOS停止。",
        "直接使用真正_sample與_tool_episode，只提供確定性IdSequence logits而未載入模型權重；保存實際IDs/trace/call spies。"),
    verified("float-examples","numeric","1e308是有限float，1e999轉float為inf；有限1e308乘1e308得到inf。NaN與±Infinity皆非有限。","B.2 數值補充第一段及溢位例",
        "針對CPython3.13.5實際float算例；不宣稱十進位1e308在二進位float中精確表示，亦不宣稱任意精度整數運算溢位。",
        [ref("python-math","#math.isfinite/#math.isnan","有限謂詞與NaN定義。"),
         ref("rfc8259","§6 number exponent grammar and range limits","e科學記數與浮點實作範圍限制。"),
         ref("bounded-execution","verification-results.json#/float_values and /overflow","親算有限輸入、無限float與乘法結果表示。")],
        ["bounded-cpu-run","bounded-cpu-details"],"float('1e308')有限；float('1e999')無限；repr(1e308*1e308)='inf'。",
        "isfinite=True，isinf=True，product_repr='inf'；parsed1e308两參數均為有限值。",
        "科學記數1×10^308與1×10^999；產品數量級10^616超出tested float的有限範圍；比較有限分類與字串表示，不將近似float和實數作精確等同。",tolerance="exact predicate equality and repr equality; no approximate decimal-value comparison"),
    verified("overflow-trace","software","有限輸入造成工具結果溢位時保留executed=true、finite_result=false、字串inf，以invalid_tool_result停止，實際呼叫仍算一次且可嚴格序列化。","B.2 補充末段 finite-overflow分支",
        "由CPU注入和既有回歸支持此controller分支；不是宣稱原模型曾生成這份巨大輸入。",
        [ref("applications","_tool_episode lines 558–577, 595–603; _json lines 458–459","真正呼叫後記錄結果分類與repr、停止狀態及actual_tool_calls。"),
         ref("python-json","json.dumps allow_nan; #infinite-and-nan-number-values","allow_nan=False可阻止JSON數字NaN/Infinity；字串inf是正常JSON字串。"),
         ref("bounded-execution","verification-results.json#/overflow","真的call_tool一次；trace finite_result false、tool_result string inf；json.dumps allow_nan=False成功。"),
         ref("regression-execution","test_finite_arguments_overflow_records_actual_execution_and_serializable_evidence; regression-stdout.txt","原相關回歸案例在本CPU環境通過。")],
        ["bounded-cpu-run","bounded-cpu-details","existing-regression-run"],
        "一個實際tool call；執行true/有限false/結果字串inf/status invalid_tool_result；strictJSON序列化成功。",
        "全部精確符合，correct/required_tool_behavior不假冒成功；原overflow回歸及11項相關案例共12 passed。",
        "手指定巨大有限輸入經真_sample→parser→call_tool，沒有新模型能力評分；不把非有限float原樣寫成JSON數字。"),
    verified("original-faults-and-scope","empirical","原始報告保存七份人工JSON故障注入，[]為其中一份，全部拒絕，與20題模型樣本分開；该小數字模型紀錄沒有溢位。","B.2 補充『報告另外保存七份』及最後一句",
        "只核對指名原始版本之故障紀錄、模型樣本數/ID與有限結果。專用user ID不在原七份文字注入中，新增ID例由本輪CPU獨立支持；不重跑模型、不推算/更新成績。",
        [ref("applications","run_tools lines 616–636, 665–674; _tool_records lines 462–493","episodes與injected分開建立/保存；[]明列故障，輸入域0–9。"),
         ref("bounded-execution","permanent original-tools-raw: /results/injected_protocol_checks_not_model_scores/*/{input,rejected,error}; /results/test/examples; /results/samples/*/trace; verification-results.json#/existing_raw_measurements","逐項重算7原故障input/error；20個樣本ID處理與18次executed finite_result/tool_result核驗；當前方法SHA與原code_sha256一致。"),
         ref("bounded-execution","verification-results.json#/top_level_array","[]是合法JSON陣列，解析器以外層object契約拒；parser=1/tool=0。")],
        ["original-tools-raw","bounded-cpu-run","bounded-cpu-details","bounded-cpu-code","inspection"],
        "故障7/7拒；模型sample及test examples各20；executed結果無非有限值；[]外層object拒絕且工具零次。",
        "7/7原input/error逐一一致；20樣本，18次實調全部0–9有限參數/有限結果，invalid_tool_result=0；[] parser=1/tool=0。",
        "先keys/types→具名原測量指標，未讀accuracy/作者結果評語；永久完整原JSON copy與原件SHA相等。原revision910aebc6419c9fc6217279a27fde9851c5cfad30、seed42、Python3.13.3、torch2.14.1+cu126/devicecuda是舊原紀錄，與本轮CPUPython3.13.5检查区分。",
        denominators={"manual_json_fault_inputs":7,"stored_model_test_episodes":20,"stored_actual_tool_executions_inspected":18,"new_model_training_runs":0,"new_model_ability_scores":0}),
]

report={"schema_version":1,"review_stage":"technical","lesson_id":"B.2","source":"course/chapters/0B.md#B.2",
    "source_sha256":sha(OUT/"section.md"),"verdict":"pass","reviewer_task":TASK,"reviewer_context":"fresh",
    "figure_sha256":{},"artifacts":artifacts,"sources":sources,"claims":claims,"issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","details":"逐主張親讀現稿/原方法與官方來源。最小入口與外層契約、ID優先拒絕與EOS停止、JSON陣列和Python requests清單分清，無未解實質主張。","claim_ids":[x["id"] for x in claims]},
        "numeric_verification":{"status":"pass","details":"原fence回5；錯參數變體回6；float有限/inf與乘法溢位、故障7及樣本20/實調18分母均自行CPU核對，沒有重跑能力評分。","claim_ids":["legal-wrong-parameters","float-examples","original-faults-and-scope"]},
        "figure_consistency":{"status":"not_applicable","details":"B.2無引用圖。仍實際渲染并view自己凍結Markdown的桌面/手機PNG；無缺失的圖形素材/箭頭/數字需要核。此非production頁面或第一次讀者驗收。","claim_ids":["control-ids-and-eos"]},
        "source_verification":{"status":"pass","details":"直接親讀version-pinned Python3.13.5 official docs及RFC8259原文；保存URLs、versions、authority、locators、原bytes/SHA。原實作SHA與原實測記錄匹配，正式raw證據用永久副本。","claim_ids":[x["id"] for x in claims]},
        "limitations":{"status":"pass","details":"限add/multiply本地ByteTokenizer/固定小數字協議。新user ID例真核但為CPU注入，不納原7或20。EOS不構成泛化格式保證。未重訓/GPU/下載模型與訓練資料/成品工程/發布。","claim_ids":["minimal-call-tool","strict-json-layer","control-ids-and-eos","overflow-trace","original-faults-and-scope"]},
    },
    "actual_read_scope":"See permanent inspection.md for exact current教材範圍、AST方法行號、原JSON pointers與official locators; no prior review verdict or author correction/result summary was opened.",
    "frozen_input_policy":{"whole_file_snapshot":"inputs/course/chapters/0B.md","meaning":"Original complete file bytes as read and frozen during this review, not an assertion of the current whole chapter version. Hash in frozen-inputs.json matches extraction source_file_sha256.","section_hash_policy":"Original UTF-8 bytes from B.2 heading through before B.3; no strip/newline normalization."},
    "new_protocol_example":{"tested":True,"real_path":"hand-specified IDs through actual applications._sample -> _tool_episode -> preparse guard","user_id":3,"observed_parser_calls":0,"observed_tool_calls":0,"historical_seven_fault_inputs":False,"scope":"Local CPU protocol mechanism, never historical/model ability evidence."},
}
target=ROOT / "docs/technical-reviews/B.2.json"
target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding="utf-8")
assert json.loads(target.read_text())["reviewer_task"] == TASK
print(json.dumps({"report":target.relative_to(ROOT).as_posix(),"reviewer_task":TASK,"source_sha256":report["source_sha256"],"verdict":report["verdict"],"claims":len(claims),"artifacts":len(artifacts)},ensure_ascii=False))
