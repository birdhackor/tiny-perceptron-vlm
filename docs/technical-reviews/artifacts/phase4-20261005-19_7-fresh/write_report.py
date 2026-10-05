import hashlib
import json
import shlex
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_19_7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


artifacts = []
env = json.loads((ART / "verification-results.json").read_bytes())["environment"]


def artifact(identifier, filename, kind, description, **kwargs):
    path = ART / filename
    artifacts.append({"id": identifier, "path": rel(path), "sha256": sha(path), "kind": kind, "description": description, **kwargs})


artifact("frozen-section", "inputs/section.md", "source_snapshot", "本人實際讀取的19.7原始UTF-8 bytes；無換行正規化。")
artifact("frozen-chapter", "inputs/19-frozen.md", "source_snapshot", "初讀當時完整章節frozen input；只作輸入版本定位，未聲稱讀完全章或代表目前全章。")
artifact("input-manifest", "input-manifest.json", "source_snapshot", "原稿/圖/審閱規約保存來源、SHA及實際讀取範圍。")
artifact("inspection", "inspection-record.json", "derivation", "本人原始來源、JSON pointers、AST方法、圖的實際查看與獨立性記錄。")
artifact("original-fence-code", "inputs/fence-1.py", "code", "教材原始fence，不修改人工請求或工具執行程式。")
artifact("verification-code", "verify_19_7.py", "code", "有限CPU原始trace解碼、判分重算、家族分割、解析/allowlist/範圍邊界和scripted控制流檢查；不載權重。")
artifact("lineage-code", "verify_lineage.py", "code", "JSON原始SHA/階段譜系核對，不開啟權重。")
artifact("render-code", "render_figure.py", "code", "使用實際凍結SVG及Chromium產生兩個寬度的render。")
artifact("report-authoring-code", "write_report.py", "code", "本人完整新報告的authoring code；未讀舊canonical報告。")
artifact("commands", "commands-run.json", "execution", "本次原fence、CPU檢查與render實際argv/cwd/exit/環境/輸出SHA。", command=".venv/bin/python: subprocess.run original-fence, cpu-verification, figure-render; argv saved in this file", result="三項exit_code=0；每項timeout=45秒。", environment={"python": env["python"], "device": "CPU; no CUDA; render uses headless Chromium"})
commands = {r["name"]: r for r in json.loads((ART / "commands-run.json").read_bytes())}
artifact("fence-execution", "original-fence-stdout.txt", "execution", "原fence真正執行的stdout。", command=shlex.join(commands["original-fence"]["command_argv"]), result="解析calculator,a=1,b=2；開回3；關回calculator_unavailable；明示未做模型生成。exit0。", environment=env)
artifact("cpu-execution", "cpu-verification-stdout.txt", "execution", "本人短CPU核對stdout；逐題重算12/12工具請求、10/12最終回答及兩筆失敗。", command=shlex.join(commands["cpu-verification"]["command_argv"]), result="所有assert通過；既有SFT/joint/DPO分組與最終joint工具紀錄均一致；exit0。", environment=env)
artifact("raw-recalculation", "verification-results.json", "derivation", "精确計算、原始read pointers、逐題解碼摘要與scripted控制流witness；不是新模型得分。")
artifact("lineage-execution", "lineage-stdout.txt", "execution", "本人原始checkpoint雜湊譜系比較stdout。", command=shlex.join(json.loads((ART / "lineage-command.json").read_bytes())["command_argv"]), result="SFT→joint→DPO parent hash、final test joint hash與原訓練code hash全部吻合；exit0。", environment={"python": env["python"], "device": "CPU; JSON/SHA only"})
artifact("lineage-command", "lineage-command.json", "execution", "譜系比較命令、裝置、實際exit0及輸出SHA。", command=".venv/bin/python verify_lineage.py (exact argv saved in this file)", result="exit_code=0", environment={"python": env["python"], "device": "CPU; JSON/SHA only"})
artifact("provenance-inspection", "raw-provenance-inspection.json", "derivation", "僅指定原始provenance pointers；stage/parent/export hash/objective，不取作者validation_summary。")
artifact("trainer-inspection", "trainer-inspection.json", "derivation", "AST精讀階段資料選取及訓練目標分支的行號；未執行訓練。")
artifact("pinned-sources", "pinned-sources.json", "source_snapshot", "原稿指定commit與當前必需原檔的byte相等比較。")
artifact("external-fetch", "external-fetch.json", "source_snapshot", "本人HTTPS取得官方原碼及DPO原論文；URL、HTTP200、日期、SHA。")
artifact("model-spec-snapshot", "external/openai-model-spec.md", "source_snapshot", "OpenAI官方model_spec指定commit原始bytes；親核Tool定義。")
artifact("tool-role-snapshot", "external/chat_completion_tool_message_param.py", "source_snapshot", "OpenAI官方SDK v1.109.1原始schema；親核tool角色與必填tool_call_id。")
artifact("dpo-paper", "external/dpo-v3.pdf", "source_snapshot", "原始arXiv:2305.18290v3 PDF；本人核首面版本/作者及Section4 Eq7。")
artifact("dpo-text", "external/dpo-v3.txt", "source_snapshot", "上述PDF的pdftotext -layout可定位文字；保持原PDF作權威。")
artifact("svg-snapshot", "inputs/rewrite-19-tool-roundtrip.svg", "source_snapshot", "本人核對的實際SVG bytes。")
artifact("figure-intrinsic", "figure-intrinsic.png", "figure_render", "640px實際SVG render；本人透過view_image查看。")
artifact("figure-mobile", "figure-mobile.png", "figure_render", "390px實際SVG render；本人透過view_image查看。")
artifact("render-environment", "figure-render-environment.json", "source_snapshot", "Chromium151.0.7922.173、Playwright版本及兩個render寬度。")
artifact("render-execution", "figure-render-stdout.txt", "execution", "真render命令stdout。", command=shlex.join(commands["figure-render"]["command_argv"]), result="640與390px render完成，exit0；隨後本人view_image實際查圖。", environment={"python": env["python"], "browser": "Chromium 151.0.7922.173", "device": "headless CPU renderer"})
for name in ("original-fence-stderr.txt", "cpu-verification-stderr.txt", "figure-render-stderr.txt", "lineage-stderr.txt"):
    artifact("stderr-" + name.split("-stderr")[0], name, "source_snapshot", "保留本次實際空stderr；不能省略失敗輸出。")

local_sources = [
    ("capstone", "tiny_perceptron/capstone.py", "原始工具策略、計算器與同模型回填實作", "AST先定位；親讀1-48,124-136,139-282,299-318,341-368,379-496,499-605；一般方法註解可讀，無舊review結果。"),
    ("tokenizer", "tiny_perceptron/data.py", "原始ByteTokenizer契約", "親讀SPECIALS行10-11及ByteTokenizer行14-28；原始UTF-8 bytes+8與EOS=2。"),
    ("trainer", "scripts/course_experiments/capstone.py", "原始階段資料選取與偏好/共同訓練規約", "AST先定位；只讀train_stage行97-122,194-226及run_*行300-309；未讀report assignment行245-275。"),
    ("data", "docs/course-experiments/capstone-evidence/deployment/data.json", "指定commit的原始合成資料", "先核頂層key/types；manifest/version,seed,split_unit,families,sha256,counts,split_policy及splits numeric rows原始欄位；按id查問句，重算split hash與家族交集。"),
    ("sft-records", "docs/course-experiments/capstone-evidence/sft/validation.json", "原始SFT逐題驗證trace", "count/protocol/by_task calculator,unavailable,tool_return；只讀相關records原始trace/runtime/answer/判準欄位，逐token解碼及重建prompt；詳见verification-results read_pointers。"),
    ("joint-records", "docs/course-experiments/capstone-evidence/joint/validation.json", "原始joint逐題驗證trace", "與SFT相同具名原始欄位與逐token/prompt核對；未讀任何舊技術報告。"),
    ("dpo-records", "docs/course-experiments/capstone-evidence/dpo/validation.json", "原始DPO分支逐題驗證trace", "與SFT相同具名原始欄位與逐token/prompt核對；只支持這些固定模板。"),
    ("test-records", "docs/course-experiments/capstone-evidence/deployment/test-joint.json", "原始最後joint逐題工具紀錄", "count/protocol/checkpoint_sha256/by_task三工具組及其records，親解碼generated_ids、確認EOS、按id查data問句及重建回填prompt；records18/20為正文失敗。"),
    ("sft-provenance", "docs/course-experiments/capstone-evidence/sft/train-report.json", "SFT原始訓練provenance", "Schema先讀；僅/data_version,/stage,/parent_checkpoint_sha256,/code_sha256,/checkpoint,/training_checkpoint,/inference_export,/objective；不讀validation_summary。"),
    ("joint-provenance", "docs/course-experiments/capstone-evidence/joint/train-report.json", "joint原始訓練provenance", "同上具名原始provenance pointers；只核parent/export/code hashes及方法objective。"),
    ("dpo-provenance", "docs/course-experiments/capstone-evidence/dpo/train-report.json", "DPO分支原始訓練provenance", "同上具名原始provenance pointers；只核joint parent及方法objective。"),
]
sources = []
for identifier, path, title, note in local_sources:
    snapshot = ART / "originals" / path
    artifact(identifier + "-snapshot", "originals/" + path, "source_snapshot", "原始完整檔，保留SHA與未讀欄位；具體scope記於inspection。")
    sources.append({"id": identifier, "kind": "repository_code", "title": title, "path": rel(snapshot), "sha256": sha(snapshot), "version": "1df335318bda03fd771807f66976953231d5a00b; independently confirmed pinned/current/saved bytes", "verified": True, "inspection_note": note})
sources += [
    {"id": "tool-loop", "kind": "official_source", "title": "OpenAI Model Spec: Definitions / Tool", "url": "https://raw.githubusercontent.com/openai/model_spec/7f1cf79fcb656c07f77c8d95b6fbc78dc7fac5b6/model_spec.md", "version": "7f1cf79fcb656c07f77c8d95b6fbc78dc7fac5b6", "authority_reason": "OpenAI官方model_spec repository的原始規約；本輪重新HTTPS取得。", "accessed_on": "2026-10-05", "verified": True, "checked_original": True, "inspection_note": "Definitions lines120-154，尤其line153 Tool：tool response追加role=tool且assistant再次invoked；不把工具輸出等同助理最終回答。", "artifact_id": "model-spec-snapshot"},
    {"id": "tool-role", "kind": "official_source", "title": "OpenAI Python SDK ChatCompletionToolMessageParam", "url": "https://raw.githubusercontent.com/openai/openai-python/a1493f92a7cd4399d57046aadc943aeadda5b8e7/src/openai/types/chat/chat_completion_tool_message_param.py", "version": "v1.109.1, tag independently resolved by git ls-remote to a1493f92a7cd4399d57046aadc943aeadda5b8e7", "authority_reason": "OpenAI官方API SDK的OpenAPI生成schema；非第三方教學摘要。", "accessed_on": "2026-10-05", "verified": True, "checked_original": True, "inspection_note": "全21行原始檔，class lines13-21：content、Literal[tool] role及Required[str] tool_call_id。支持可用角色/呼叫對應，沒有證明本repo舊協定已完成。", "artifact_id": "tool-role-snapshot"},
    {"id": "dpo", "kind": "paper", "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "url": "https://arxiv.org/pdf/2305.18290v3", "version": "arXiv:2305.18290v3, 29 Jul 2024; original first-page identifier verified", "authority_reason": "Rafailov等作者的原始DPO論文；本輪從arXiv取得並親核首面與方法段落。", "accessed_on": "2026-10-05", "verified": True, "checked_original": True, "inspection_note": "首面題名/作者/版本，摘要與引言，Section4 pp4-5 Eq7及DPO outline；偏好資料(x,yw,yl)直接優化policy。只支持DPO方法定義，不替repo能力評分。", "artifact_id": "dpo-paper"},
    {"id": "cpu-run", "kind": "execution", "title": "本輪CPU trace/判分/邊界核對", "verified": True, "artifact_id": "cpu-execution"},
    {"id": "fence-run", "kind": "execution", "title": "原始fence執行", "verified": True, "artifact_id": "fence-execution"},
    {"id": "lineage-run", "kind": "execution", "title": "原始階段/最終測試hash譜系比較", "verified": True, "artifact_id": "lineage-execution"},
    {"id": "arithmetic-and-criteria", "kind": "derivation", "title": "本人核對的算術、分母及讀回測試支持範圍", "verified": True, "details": "1+2=3；0+1=1；1+0=1；4+4=8。完全匹配TOOL:calculator:a+b且第一生成EOS才算正確請求；真runtime=a+b；第二生成DIRECT:<真結果>且EOS才能取answer。工具成功12題，最終成功10題，分母12而非12+6獨立回填題。替換runtime回值只作反事實讀回依賴檢查，必須標記為測試回放且不得計入正常真結果答對率；scripted generator只核control flow。"},
]


def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def claim(identifier, kind, statement, location, scope, evidence, ids, verification=None):
    value = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope, "status": "verified", "evidence": evidence, "artifact_ids": ids}
    if verification:
        value["verification"] = verification
    return value


def executed(expected, observed, details, denominators=None):
    out = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if denominators:
        out["denominators"] = denominators
    return out


claims = [
    claim("c1", "concept", "完整工具往返須區分模型請求、真工具執行、回填後模型再答；直接顯示工具結果不能驗收模型讀回。", "19.7開頭與圖說", "標準工具迴圈與本課驗收要分開的層次；同一模型在本repo由c3核對。", [ev("tool-loop", "Definitions Tool, line153", "工具由assistant選用，工具訊息回填後再次invoke assistant。"), ev("arithmetic-and-criteria", "inspection-record及逐題工具/最終回答分列", "最終內容需要額外判分，不能用外部結果冒充。")], ["model-spec-snapshot", "inspection"]),
    claim("c2", "software", "原fence是人工請求+真runtime示範：解析calculator,a=1,b=2，可用回3，關閉回錯誤；unknown可解析但不得執行，不eval生成文字。", "19.7唯一Python fence及其後段落", "只核短格式/解析/允許執行；沒有第一或第二次模型生成，沒有更新參數。", [ev("capstone", "parse_action550-561; calculator_runtime564-574", "EOS必須有；regex允許名稱但runtime僅allowlist calculator及0..999整數，直接a+b。"), ev("fence-run", "原fence stdout四行", "原文預期輸出精確出現。"), ev("cpu-run", "boundaries及boolean_argument", "unknown、未結束、越界、負数/尾接程式片段被對應關卡拒絕。")], ["original-fence-code", "fence-execution", "verification-code", "cpu-execution", "raw-recalculation"], executed("人工1+2請求回3/關閉error；unknown parse=tool但tool_not_allowlisted。", "原fenceexit0且完全吻合；999+999=1998，1000參數/尾接字串/負數不能解析，bool手工參數不能執行。", "實際用CPU呼叫原函式，不替模型生成；普通API合組核對。")),
    claim("c3", "software", "現有run_assistant在成功runtime後，把原題、計算器回報、請回答做成新user前文，用同一model再生成；工具錯誤不當成成功answer。", "19.7共同成品段落", "舊局部實作的真契約；測試使用scripted generator，只驗控制流不驗訓練模型讀回能力。", [ev("capstone", "run_assistant577-598; prompt_ids299-305", "兩次generate_trace使用相同model；followup=user文字且無tool token/呼叫ID；錯誤保留runtime，answer None。"), ev("cpu-run", "scripted_control_flow_witnesses", "同model identity、真返回3/回放9對應第二次前文、關閉只呼叫一次。")], ["capstone-snapshot", "verification-code", "cpu-execution", "raw-recalculation"], executed("成功同model兩次、工具關閉一次；回填值進user前文。", "3與測試9各保留第二次生成回答；關閉runtime=error、final_trace/answer=None。", "patch generate_trace僅供控制流witness；未開模型权重，未取得新score。")),
    claim("c4", "concept", "新協定可用tool角色及呼叫對應；共同成品的完整保存與角色化協定是計畫，舊短格式不能稱已實现。", "19.7共同成品段落", "核對所提協定可行性與目前支持範圍；不宣稱第5階段成品已驗收。", [ev("tool-role", "ChatCompletionToolMessageParam13-21", "tool角色和tool_call_id是API原始契約。"), ev("tool-loop", "Definitions Tool153", "工具回填後模型再呼叫的預期流程。"), ev("capstone", "prompt_ids299-305及run_assistant577-598", "目前只有新user前文，没有完整tool角色/呼叫對應。")], ["tool-role-snapshot", "capstone-snapshot", "inspection"]),
    claim("c5", "concept", "判分要分別檢查請求參數、正常真結果及模型回答；工具關閉須未執行。替換返回值是另外的讀回依賴診斷，不能加入正常答對率。", "19.7判分段落", "驗收與回放設計的支持範圍；沒有聲稱現有模型通過反事實讀回測試。", [ev("tool-loop", "Definitions Tool153", "外部工具和模型續答為不同環節。"), ev("capstone", "evaluate_rows499-547; run_assistant577-598", "action與end_to_end分列；真runtime後才續答，關閉error不生成最終回答。"), ev("arithmetic-and-criteria", "正常runtime/反事實runtime区分推導", "替代返回值不等於真計算器結果，需要單獨的診斷分母。")], ["raw-recalculation", "inspection"]),
    claim("c6", "empirical", "SFT的4+4驗證真請求TOOL:calculator:4+4、runtime8、同模型續答DIRECT:8，兩生成EOS；完整工具/關閉/獨立回填分組。", "19.7details第1-2段", "原始SFT固定模板验证紀錄，非任意自然語言或本輪重新評測。", [ev("sft-records", "/records/0及/by_task/calculator,/unavailable,/tool_return", "4+4整條trace與分組原始測量。"), ev("data", "/splits/validation/0 id=c6e56b84cc6d19c2df86", "按id對上原問題4+4等於多少。"), ev("capstone", "evaluate_rows499-547與generate_traces428-496", "runtime輸入相同model的第二生成，而非用runtime取代answer。"), ev("cpu-run", "selected_records/sft及recomputed_tool_summaries/sft", "生成IDs解碼、初始/回填prompt重建、EOS與比較重算全部一致。")], ["sft-records-snapshot", "data-snapshot", "cpu-execution", "raw-recalculation"], executed("4+4：請求4+4、工具8、模型8且兩EOS；三題組分開。", "逐token/prompt核對成功；calculator10/10、unavailable10/10、tool_return5/5各自獨立。", "沒有把5筆獨立讀回題加進10筆完整工具往返。", {"validation_total":84,"calculator":10,"unavailable":10,"independent_tool_return":5,"numeric_families":5})),
    claim("c7", "empirical", "joint及後續DPO分支在同一validation保留固定題型的工具流程；這份驗證未觀察到共同圖音訓練破壞工具流程。", "19.7details joint/DPO段落", "同一合成validation的觀察，不是因果保證、圖音能力驗收或DPO工具能力提升。", [ev("joint-records", "/records中三工具task；/by_task三分組", "joint固定工具分組全部通過。"), ev("dpo-records", "/records中三工具task；/by_task三分組", "DPO同分組同題全部通過。"), ev("trainer", "train_stage117-120及194-213", "SFT只text_rows；joint包含全部模態rows；DPO從即時前一階段並配合replay。"), ev("sft-provenance", "/inference_export/sha256", "SFT父階段export。"), ev("joint-provenance", "/parent_checkpoint_sha256,/inference_export/sha256", "joint承接SFT。"), ev("dpo-provenance", "/parent_checkpoint_sha256", "DPO承接joint。"), ev("cpu-run", "recomputed_tool_summaries/joint,dpo", "同題id、原始token/EOS/prompt和分母本人重算。"), ev("lineage-run", "lineage stdout", "上述parent hashes相等。")], ["joint-records-snapshot", "dpo-records-snapshot", "trainer-snapshot", "provenance-inspection", "cpu-execution", "lineage-execution"], executed("joint與DPO同validation完整工具10、關閉10、獨立回填5各不混算；DPO parent=joint。", "兩版本都calculator10/10、unavailable10/10、tool_return5/5；父checkpoint與code SHA吻合。", "有限觀察僅支持未見破壞；不推出任意圖音或自然語言泛化。", {"each_stage_validation_total":84,"each_stage_calculator":10,"each_stage_unavailable":10,"each_stage_tool_return":5,"seed":42})),
    claim("c8", "empirical", "最後joint在12個計算請求全部正確並真執行，最終同模型10/12答對；0+1→DIRECT:0及1+0→DIRECT:111都EOS，因此工具正確不保證模型答對。", "19.7details最後3段及失敗表", "指定joint checkpoint與指定test的原始測量；兩次錯誤是內容而非缺EOS，不能泛化成所有任務的不確定性校準。", [ev("test-records", "/by_task/calculator; /records/18,/records/20; /checkpoint_sha256", "12請求、runtime都ok，10最終正確與兩筆實際生成。"), ev("data", "/splits/test/18 id6452e1d5197f5dc7987c; /splits/test/20 id246ec17075366e4bea5f", "失敗表原問句和參數順序。"), ev("capstone", "evaluate_rows499-547; parse_action550-561", "第一生成精確有序request和EOS、第二生成parse/direct內容判定。"), ev("cpu-run", "historical_failures與recomputed_tool_summaries/test-joint", "本人重算12/12與10/12；token EOS、raw文字、回填結果1都吻合。"), ev("lineage-run", "deployment/checkpoint_sha256 comparison", "final test匹配joint export hash。")], ["test-records-snapshot", "data-snapshot", "cpu-execution", "raw-recalculation", "lineage-execution"], executed("12次請求與工具成功、10模型最後答對；兩次不正確答案都是EOS。", "精確重算12/12、10/12；records18/20分別0、111，runtime均1，兩EOS=True。", "generated_ids本人解碼，與raw和stop_reason一致；prompt IDs對上真返回值；其餘12題獨立關閉和6題獨立回填不加進分母。", {"test_total":90,"calculator_requests":12,"calculator_families":6,"actual_successful_runtime":12,"correct_model_final":10,"unavailable_separate":12,"tool_return_separate":6})),
    claim("c9", "software", "數字family與訓練分離，但問句只有固定加法/回填模板；結果不能代表任意自然語言計算或所有任務校準。", "19.7details限制敘述", "原始資料生成與split方法的可驗範圍；不重新證明業界知識。", [ev("capstone", "build_dataset139-160,211-282", "unordered numbers family的所有方向/availability/回填同組拆分；只固定模板。"), ev("data", "/manifest/families,/sha256,/counts,/split_policy; /splits/*的numeric rows", "實際各split numeric families和問句規則。"), ev("cpu-run", "data_split_verification及numeric-family交集assert", "本人核對44/5/6家族互斥、split digest及每calculator模板。")], ["data-snapshot", "capstone-snapshot", "cpu-execution", "raw-recalculation"], executed("numeric families互斥、固定模板且manifest SHA可重算。", "train44、validation5、test6 numbers families無交集，rows552/84/90 digest一致；每calculator問句符合兩模板之一。", "資料切分只對家族外數值組合，沒有任意語言題/其他不確定性校準測試。")),
    claim("c10", "concept", "DPO是用好壞回答配對進行偏好訓練的方法；本課分支只是synthetic配對實作。", "19.7details對DPO的一句定義及13.2連結", "核基本方法定義，不宣稱人類偏好、純DPO或此處能力提升。", [ev("dpo", "Section4 pp4-5 Eq7及DPO outline", "偏好三元組(x,yw,yl)最優化policy。"), ev("capstone", "preference_pairs341-351; preference_loss354-368", "本repo建立chosen/rejected配對並使用frozen reference。"), ev("trainer", "train_stage204-213", "分支為DPO與replay CE方法，沒有被正文說成純DPO結果。")], ["dpo-paper", "capstone-snapshot", "trainer-snapshot", "inspection"]),
    claim("c11", "numeric", "圖中1+2的真算術值3、各框/箭頭與正文工具往返一致，模型第一/第二段明示應請求/應回答，是期待示意。", "19.7rewrite-19-tool-roundtrip.svg與alt文字", "只核算例及圖的忠實表達；不把圖當成1+2的已觀測完整模型成功。", [ev("arithmetic-and-criteria", "1+2=3計算", "基本整數加法。"), ev("fence-run", "calculator available True result3", "原程式實算結果。"), ev("tool-loop", "Definitions Tool153", "箭頭順序為request→execution→return→assistant。")], ["svg-snapshot", "figure-intrinsic", "figure-mobile", "render-execution", "inspection", "fence-execution"], {"method":"executed","expected":"1+2=3；五框有序，第一和最後明示應，圖說標期待路徑。","observed":"CPU真回3；640/390px render本人view_image確認相同五框與箭頭、數字3、正常結束檢查和可能答錯提示，無截字。","details":"圖不是測量截圖；最後3是期待模型回答，與正文人工請求/無模型生成一致。","tolerance":"整數精確相等；視覺標籤/箭頭逐項一致。"}),
]

report = {
    "schema_version":1, "review_stage":"technical", "lesson_id":"19.7", "source":"course/chapters/19.md#19.7",
    "source_sha256":sha(ART / "inputs/section.md"), "reviewer_task":TASK, "reviewer_context":"fresh", "verdict":"pass",
    "figure_sha256":{"course/figures/rewrite-19-tool-roundtrip.svg":sha(ART / "inputs/rewrite-19-tool-roundtrip.svg")},
    "read_scope":{"section":"本人讀目前19.7全文/圖及必要原始方法；沒有讀其他小節或章導言。","frozen_complete_chapter":{"path":rel(ART / "inputs/19-frozen.md"),"sha256":sha(ART / "inputs/19-frozen.md"),"meaning":"initial frozen input only; not a current full-chapter assertion"},"inspection_artifact":"inspection"},
    "review_limits":"CPU only。原始固定模板測量按指定commit核對；未載入模型權重、未重訓/新評模型、未下載訓練資料。未讀舊review結論或作者結果摘要。圖為預期示意，完整成品協定仍是計畫。",
    "sources":sources, "artifacts":artifacts, "claims":claims, "issues":[],
    "checks":{
        "factual_accuracy":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"逐項核權威工具契約、原始方法及指定版本historical traces；人工demo、scripted控制流witness、既有模型生成與未来計畫分開。"},
        "numeric_verification":{"status":"pass","claim_ids":["c6","c7","c8","c11"],"details":"原fence1+2=3；SFT4+4=8；逐token/初始與回填prompt重建並重算10/10及12/12→10/12，各工具組分母不相加。"},
        "figure_consistency":{"status":"pass","claim_ids":["c1","c11"],"details":"實際Chromium render並view_image查看640和390寬圖；數字、五框、箭頭、期待措辭及正文無矛盾。"},
        "source_verification":{"status":"pass","claim_ids":[c["id"] for c in claims],"details":"原稿指定commit的JSON和必要code完整副本SHA吻合；官方Model Spec/SDK重取並核version，DPO首面版本與方法親讀；每claim locator及支持範圍各自列明。"},
        "limitations":{"status":"pass","claim_ids":["c2","c3","c4","c5","c6","c7","c8","c9","c10","c11"],"details":"只支持固定模板、family外split與指定历史checkpoint觀察；無因果保證/任意語言/全任務校準；回放診斷不是正常得分；未實作成品角色化協定只作設計。"},
    },
}
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_19_7"
assert report["source_sha256"] == "713f6d3ddd667feedd5c4053db4641926484e3e94c943df0f9482670e39ca35c"
(ROOT / "docs/technical-reviews/19.7.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print("Wrote independently authored canonical report", report["reviewer_task"], report["source_sha256"])
