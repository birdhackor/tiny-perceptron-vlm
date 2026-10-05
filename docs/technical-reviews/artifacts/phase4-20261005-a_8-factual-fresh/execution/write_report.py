"""Write this reviewer's complete new report without reading the previous report."""
import datetime
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent.parent
BASE = ART.relative_to(ROOT).as_posix()
TASK = "/root/phase4_factual_coordinator/factual_a_8"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
raw = (ROOT / "course/chapters/0A.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
index = next(i for i, h in enumerate(headers) if h[0].startswith(b"## A.8 "))
section = raw[headers[index].start():headers[index+1].start()]
assert section == (ART / "inputs/section.md").read_bytes(), "A.8 changed during review"
assert sha(ROOT / "course/figures/rewrite-A-clue-position.svg") == sha(ART / "inputs/course/figures/rewrite-A-clue-position.svg")
assert sha(ROOT / "tiny_perceptron/data.py") == sha(ART / "inputs/tiny_perceptron/data.py")
audit = json.loads((ART / "execution/audit-result.json").read_text())
render = json.loads((ART / "figures/render-result.json").read_text())
assert audit["assertions"] == "all passed"
assert audit["source_section_sha256"] == sha(ART / "inputs/section.md")
assert not (ART / "execution/audit-stderr.txt").read_bytes()
assert not (ART / "figures/render-stderr.txt").read_bytes()
date = datetime.datetime.now(datetime.UTC).date().isoformat()
cpu_cmd = f".venv/bin/python {BASE}/execution/audit_a8.py > {BASE}/execution/audit-stdout.txt 2> {BASE}/execution/audit-stderr.txt"
render_cmd = f".venv/bin/python {BASE}/execution/render_figure.py > {BASE}/figures/render-stdout.txt 2> {BASE}/figures/render-stderr.txt"
commands = [
    {"command": ".venv/bin/python docs/review-tools/section_facts.py 'course/chapters/0A.md#A.8' --output /tmp/a8-factual-20261005-extract", "exit_code": 0, "result": "0 Python fences; 0 other fences; 1 SVG; raw section extracted"},
    {"command": f"curl --fail --location --max-time 60 https://arxiv.org/pdf/2307.03172v3 -o {BASE}/sources/lost-in-the-middle-2307.03172v3.pdf", "exit_code": 0, "result": "Downloaded original arXiv v3 PDF; first page independently confirms 2307.03172v3, 20 Nov 2023"},
    {"command": f"pdftotext -layout {BASE}/sources/lost-in-the-middle-2307.03172v3.pdf {BASE}/sources/lost-in-the-middle-2307.03172v3.txt", "exit_code": 0, "result": "Original PDF layout text created and relevant sections read"},
    {"command": f"curl --fail --location --max-time 60 https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation -o {BASE}/sources/transformers-v4.57.1-text-generation.html", "exit_code": 0, "result": "Original versioned official generation documentation downloaded"},
    {"command": f"curl --fail --location --max-time 60 https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating -o {BASE}/sources/transformers-v4.57.1-chat-templating.html", "exit_code": 0, "result": "Original versioned official chat-template documentation downloaded"},
    {"command": f"curl --fail --location --max-time 60 https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/stopping_criteria.py -o {BASE}/sources/transformers-v4.57.1-stopping_criteria.py", "exit_code": 0, "result": "Official original tagged source downloaded, MaxLengthCriteria/EosTokenCriteria inspected"},
    {"command": cpu_cmd, "exit_code": 0, "result": "All bounded deterministic assertions passed; 8 records × 3 versions; T at 1/4/8; full-input byte-token counts 296/296/296; short input 114; no model run"},
    {"command": render_cmd, "exit_code": 0, "result": "System Chromium rendered original SVG at widths 850/640/390, preserving aspect ratio; all three final PNGs subsequently viewed"}
]
(ART / "execution/commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
inspection = {
    "reviewer_task": TASK,
    "read_scope": {"manuscript": "完整 A.8 原始 bytes，原稿 253–288 行；A.7 必要前置 222–251 行；最初 context 讀取還包含 218–221 行當前原稿，不是作者工作筆記或審閱摘要。", "instructions": "完整 factual-reviewer-instructions、section_facts.py、checker schema、clear-tutorial SKILL 與 review-protocol。", "repository_code": "先 AST 定位，再親讀 tiny_perceptron/data.py 11、14–28 行 ByteTokenizer；tokenization.py 1–55 行普通類別方法及 API 契約，未採用該模組為本節實測證據。", "external_sources": "arXiv v3 首頁版本、Abstract/Introduction、§§2.1–2.3、§§3.1–3.2、Figure 2/3/6/7 圖說、Appendix F 表2–4；官方 v4.57.1 GenerationConfig 長度/特殊 token 參數；chat templates 原理/apply_chat_template/special-token 避重段落；官方 stopping_criteria.py 59–85、452–473 行。"},
    "source_authenticity": "PDF 由 arxiv.org 指定 v3 取得且首頁自身識別版本；HF 文件 URL 明確 v4.57.1，原碼由 HF 官方 GitHub v4.57.1 tag 取得。官方文檔原 HTML 保留，另保存剔除 script/style 的可讀 plain-text 供定位。沒有使用摘要庫認證內容。",
    "figure_inspection": "本人以 view_image 看最終 850、640、390 PNG。三列各8框；綠框在1/4/8，灰框U1–U7不變，圖首部的攝影社→M17是語義對應，底部同題/同答案相符。此圖沒有數值軸、成績曲線、資料流箭頭或模型 tensor 形狀；格數是紀錄數，不是 token。完整標籤無裁切，390px縮放仍可辨識；只檢查 SVG 本體，不宣稱測過整個課程頁面。",
    "code_applicability": "當前 A.8 無任何 Python 或 shell fence，因此原 fence 執行不適用，非未執行卻宣稱成功。audit_a8.py 是 reviewer 的短 CPU 重建/原 ByteTokenizer 檢查，非原論文復現或模型生成。",
    "empirical_applicability": "本節沒有提供本課模型分數或指定本 repo 原始結果 JSON，因此沒有需要讀的既有本課實測/分母；引用原論文的定性結果直接核其原文與支持範圍。",
    "paper_support_scope": "原 §2.1 固定問題/素材後移動答案文件，§2.2 固定 greedy 解碼，§2.3 限當時指定模型與10/20/30文件；§3.1 改成UUID取回，§3.2 的75/140/300 KV、各500例報告有Claude接近全對，也有LongChat的不同趨勢。這些原始樣本數用來核範圍，不是本節八筆示例分母。原稿只列到§3.1；本人另實讀§3.2核『部分合成取回接近全對』，內容有直接支持，未因此改稿。",
    "independence": "未讀舊或他人技術/reader審閱正文或判定，未讀作者 TRAINING/DATA/worklog 或模型結果修正摘要。初始檔名定位過寬而列到歷史 artifact 路徑名，沒有開啟其內容；之後只讀指定原檔及原來源。未接觸舊判定或作者修正評語。"
}
(ART / "inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")

artifacts = []
def artifact(identifier, relative, kind, description, **extra):
    item = {"id": identifier, "path": f"{BASE}/{relative}", "sha256": sha(ART / relative), "kind": kind, "description": description}
    item.update(extra); artifacts.append(item)
artifact("section", "inputs/section.md", "source_snapshot", "A.8 初讀原始 UTF-8 bytes；不 strip、不正規化換行")
artifact("frozen_chapter", "inputs/course/chapters/0A.md", "source_snapshot", "當次 frozen input 全章原始 bytes；明標快照，不作目前或未來整章版本宣告")
artifact("frozen_manifest", "inputs/frozen-input-manifest.json", "source_snapshot", "輸入複本指紋、真實讀取範圍與 frozen whole-file 語義")
artifact("extraction", "inputs/extraction.json", "derivation", "原 bytes 擷取與0 fences、1 SVG事實")
artifact("svg_snapshot", "inputs/course/figures/rewrite-A-clue-position.svg", "source_snapshot", "本人讀取、渲染的原始 SVG bytes")
artifact("paper_pdf", "sources/lost-in-the-middle-2307.03172v3.pdf", "source_snapshot", "arxiv.org 原始2307.03172v3 PDF")
artifact("paper_text", "sources/lost-in-the-middle-2307.03172v3.txt", "source_snapshot", "同一原PDF的 pdftotext -layout，支援精確讀取定位")
for identifier, relative, description in [
    ("generation_html", "sources/transformers-v4.57.1-text-generation.html", "官方固定v4.57.1 GenerationConfig原HTML"),
    ("generation_text", "sources/transformers-v4.57.1-text-generation.txt", "官方GenerationConfig可讀原文定位"),
    ("chat_html", "sources/transformers-v4.57.1-chat-templating.html", "官方固定v4.57.1 Chat templates原HTML"),
    ("chat_text", "sources/transformers-v4.57.1-chat-templating.txt", "官方Chat templates可讀原文定位"),
    ("stopping_source", "sources/transformers-v4.57.1-stopping_criteria.py", "官方GitHub v4.57.1 tag原始停止條件實作"),
    ("byte_source", "inputs/tiny_perceptron/data.py", "原 ByteTokenizer 檔完整bytes複本，已本人比原件SHA相等")]:
    artifact(identifier, relative, "source_snapshot", description)
artifact("audit_code", "execution/audit_a8.py", "code", "本人的有界CPU算例：重建紀錄、AST執行原ByteTokenizer、驗圖座標/格數")
artifact("cpu_execution", "execution/audit-stdout.txt", "execution", "真CPU命令 stdout；全部assert通過，沒有模型推論", command=cpu_cmd, result="exit 0；8筆×3版；T=1/4/8；296/296/296 token；短版114；全部assert通過", environment=audit["environment"])
artifact("cpu_result", "execution/audit-result.json", "derivation", "本人重建的原ID、長度、位置、示例判準與精確相等記錄；不是模型輸出")
artifact("cpu_stderr", "execution/audit-stderr.txt", "execution", "真CPU執行stderr（空）", command=cpu_cmd, result="exit 0；stderr為空", environment=audit["environment"])
artifact("render_code", "execution/render_figure.py", "code", "以原SVG及系統Chromium渲染三種寬度的本人程式")
artifact("render_execution", "figures/render-result.json", "execution", "最終真渲染結果/瀏覽器版本/PNG指紋；本人已view三圖", command=render_cmd, result="exit 0；850/640/390 render成功，source SVG SHA與原檔相同", environment={"python": render["python"], "platform": render["platform"], "chromium": render["browser"], "playwright": "1.63.0", "device": "cpu"})
artifact("render_stdout", "figures/render-stdout.txt", "execution", "最終真Chromium渲染stdout", command=render_cmd, result="exit 0；輸出最終渲染metadata", environment={"chromium": render["browser"], "device": "cpu"})
artifact("render_stderr", "figures/render-stderr.txt", "execution", "最終真Chromium渲染stderr（空）", command=render_cmd, result="exit 0；stderr為空", environment={"chromium": render["browser"], "device": "cpu"})
for width in [850, 640, 390]:
    artifact(f"figure_{width}", f"figures/clue-position-{width}.png", "figure_render", f"本人實際view過的{width}px原SVG渲染")
artifact("commands", "execution/commands.json", "derivation", "實際命令、真exit code與作用域記錄")
artifact("inspection", "inspection.json", "derivation", "本人原來源版本/實讀locator/圖面核對/獨立性與NA範圍")
artifact("report_writer", "execution/write_report.py", "code", "本人完整新報告寫入與原 bytes/version/task assert程式")

sources = [
    {"id": "liu_v3", "kind": "paper", "title": "Liu et al., Lost in the Middle: How Language Models Use Long Contexts", "url": "https://arxiv.org/pdf/2307.03172v3", "version": "arXiv:2307.03172v3, 20 Nov 2023（PDF首頁本人核對）", "accessed_on": date, "authority_reason": "作者原論文的 arXiv 固定版本，直接報告設計與觀察，並非二手摘要。", "verified": True, "checked_original": True, "inspection_note": "親取原PDF並讀首頁、§§2.1–2.3、3.1–3.2、相關圖說及Appendix F；§3.2而非僅§3.1支持部分合成取回近全對，且有不同趨勢的模型。", "artifact_ids": ["paper_pdf", "paper_text", "inspection"]},
    {"id": "chat_v4571", "kind": "official_docs", "title": "Hugging Face Transformers Chat templates", "url": "https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating", "version": "Transformers v4.57.1 versioned documentation", "accessed_on": date, "authority_reason": "套件維護者的固定版本官方格式與tokenization文件。", "verified": True, "checked_original": True, "inspection_note": "親讀 The critical insight/Using apply_chat_template 及 special-token 避重段落；原HTML與可讀文原段落已保存。未下載文件示例的任何模型或tokenizer。", "artifact_ids": ["chat_html", "chat_text"]},
    {"id": "generation_v4571", "kind": "official_docs", "title": "Hugging Face Transformers GenerationConfig", "url": "https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation", "version": "Transformers v4.57.1 versioned documentation", "accessed_on": date, "authority_reason": "生成 API 維護者的官方長度及EOS參數契約。", "verified": True, "checked_original": True, "inspection_note": "親讀 max_length/max_new_tokens/min_new_tokens 與 eos_token_id 參數；此版本未安裝，本輪沒有呼叫其模型或把版本假稱成本地API。", "artifact_ids": ["generation_html", "generation_text"]},
    {"id": "stop_v4571", "kind": "official_source", "title": "Transformers v4.57.1 MaxLengthCriteria and EosTokenCriteria", "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/stopping_criteria.py", "version": "Official huggingface/transformers tag v4.57.1", "accessed_on": date, "authority_reason": "維護者官方 GitHub 固定 tag 的原始停止機制實作。", "verified": True, "checked_original": True, "inspection_note": "先AST定位，親讀59–85行與452–473行：完整序列長度達上限及最後token為EOS是不同停止條件。只檢查原碼，沒有模型推論。", "artifact_ids": ["stopping_source"]},
    {"id": "byte_tokenizer", "kind": "repository_code", "title": "Original repository ByteTokenizer", "path": f"{BASE}/inputs/tiny_perceptron/data.py", "sha256": sha(ART / "inputs/tiny_perceptron/data.py"), "version": "SHA-256 frozen original copy, identical to tiny_perceptron/data.py at inspection", "verified": True, "inspection_note": "AST定位後親讀11、14–28行；encode每個UTF-8 byte加8，角色ID0–7；CPU只執行原class與SPECIALS AST。不是訓練過的BPE或部署模型tokenizer。"},
    {"id": "bounded_cpu", "kind": "execution", "title": "A.8 independent bounded deterministic CPU audit", "artifact_id": "cpu_execution", "verified": True},
    {"id": "paper_derivation", "kind": "derivation", "title": "A.8 table-to-three-orders and gold-answer derivation", "verified": True, "details": "親讀表：只有T把攝影社連到M17；U3雖含M17但只講樓層，不能提供社團→代碼。將同一T插在U1…U7列表index0/3/7即第1/4/8；多重集合相同，剩餘U序不變；8筆×3版=24圖框。"}
]
claims = [
    {"id": "materials_and_figure", "kind": "numeric", "statement": "八筆同一紀錄分成三個排列；T在第1、第4、第8筆，U1–U7相對序不變，三版紙上答案都是M17；SVG每格是一筆紀錄。", "location": "course/chapters/0A.md:255–272；rewrite-A-clue-position.svg三列", "scope": "教材紙上材料與視覺示意，不是模型回答、token軸或能力分數。", "status": "verified", "evidence": [{"source_id": "paper_derivation", "locator": "表T/U1–U7（259–266行）與SVG文字座標y=182/288/394", "supports": "只有T含攝影社→M17，重排不改gold；逐列8個框，T座標與紀錄位置吻合。"}, {"source_id": "bounded_cpu", "locator": "audit-result.json /counts /measurements/{first,middle,last}/order /svg_rows", "supports": "逐版原始紀錄多重集合、U相對序、答案唯一值及24圖框的精確assert。"}], "artifact_ids": ["section", "svg_snapshot", "audit_code", "cpu_execution", "cpu_result", "figure_850", "figure_640", "figure_390", "inspection"], "verification": {"method": "executed", "expected": "每版8筆，T=1/4/8；U1–U7序相同，gold=M17；三列共24框。", "observed": "CPU全部assert通過；XML解析列序恰相等；本人查看三個最終render標籤與圖說一致。", "details": "未用alt文字代替核圖；从SVG文字座標與綠框位置獨立重建，並真render/view。", "tolerance": "整數、紀錄字串、順序與gold精確相等；無四捨五入。", "denominators": {"records_per_condition": 8, "conditions": 3, "illustrated_cells": 24}}},
    {"id": "position_controls", "kind": "concept", "statement": "位置比較應固定同一句問題、線索、完整輸入token數、生成上限、權重及解碼設定；首中尾指資料段，問題仍在其後，完整格式組合後才數token。", "location": "course/chapters/0A.md:272–274；依賴A.7的格式/預算說明", "scope": "方法規約；示例要求重新核對token數，並未宣稱任何tokenizer下重排自然等長。尾端靠近問題是此設計有意保留的位置變化。", "status": "verified", "evidence": [{"source_id": "liu_v3", "locator": "§2.1與Figures 2–3；§2.2；§3.1 Figure6；Appendix F", "supports": "原論文移動同一答案文件/鍵位置而不改期望輸出，標準prompt、greedy解碼，問題置於資料後；原token數按不同模型tokenizer計。"}, {"source_id": "chat_v4571", "locator": "Chat templates intro、Using apply_chat_template；saved text lines197–230、269–270", "supports": "聊天是包含角色control tokens的完整序列；先格式化再tokenize，不能省略/重複特殊標記。"}, {"source_id": "generation_v4571", "locator": "GenerationConfig max_length/max_new_tokens；saved text lines203–207", "supports": "新輸出token上限與輸入長度有不同定義，應另固定。"}, {"source_id": "byte_tokenizer", "locator": "original data.py lines11、14–28；audit-result /measurements /char_vs_token_example", "supports": "原byte tokenizer一個中文字可成三ID；本人小型序列化核三版等長與每筆完整性，僅示範核對流程。"}], "artifact_ids": ["paper_pdf", "paper_text", "chat_html", "generation_html", "byte_source", "audit_code", "cpu_execution", "cpu_result"], "reviewer_calculation": "原byte格式明含bos/user/assistant共3角色ID，3版296/296/296；T起始ID12/105/194（0-based），record位置1/4/8。此296非教材模型輸入測量、非BPE保證、非新模型score。"},
    {"id": "diagnostic_conditions", "kind": "concept", "statement": "內容正確與回答完整結束需分記；M17/B04/M耗盡上限是不同情形。只含T的短版是另一診斷條件，不與三個等長版本混算，長短差異仍需查長度/位置/干擾。", "location": "course/chapters/0A.md:276–278", "scope": "條件與評分記錄建議，不宣称短版能單獨證明模型懂格式、或長版失敗確定由位置造成；三個輸出是舉例而非實測。", "status": "verified", "evidence": [{"source_id": "generation_v4571", "locator": "GenerationConfig max_new_tokens/eos_token_id；saved text lines206–207、360–361", "supports": "生成長度有上限；EOS標記結束但不保證語義正確。"}, {"source_id": "stop_v4571", "locator": "MaxLengthCriteria lines59–85；EosTokenCriteria lines452–473", "supports": "達長度上限與產生EOS是不同判準，光有完整停止不能證明M17內容正確。"}, {"source_id": "liu_v3", "locator": "§2.1 accuracy criterion、§2.3 Table1 closed-book/oracle；§3.1 retrieval criterion", "supports": "原研究把只有答案文件的oracle與多文件另作條件，也明列內容正確判準；本節診斷比原論文substring accuracy另外記完成狀態。"}, {"source_id": "bounded_cpu", "locator": "audit-result.json /short_diagnostic_input_tokens /diagnostic_examples", "supports": "示例判準真檢查M17為true、B04/M為false；短版114與三版296分開，沒有產生模型回答或合成成功率。"}], "artifact_ids": ["generation_html", "stopping_source", "paper_text", "cpu_execution", "cpu_result", "inspection"]},
    {"id": "position_effect_and_scope", "kind": "concept", "statement": "不同模型/題目可能有中間較差、三處都成功或別的趨勢；可對指定模型與題組描述位置敏感，單線索取回成功不能替代多線索能力。原論文在多個當時模型/設定見首尾較好，也有合成取回接近全對；八筆紙上例未重現原評測或提供本課模型成績。", "location": "course/chapters/0A.md:280、285", "scope": "引用2023原研究的有界定性觀察與本節例子邊界；沒有新模型score、普遍U形保證或本課/現代所有模型能力宣稱。", "status": "verified", "evidence": [{"source_id": "liu_v3", "locator": "Abstract/Introduction；§2.3 Figure5；§3.1–3.2 Figure7；saved text lines294–329、350–383、407–416", "supports": "原研究用指定模型、資料與prompt；§3.2明确Claude近全對，LongChat一條件不同趨勢，其他模型常中間差。QA與合成單KV取回是分開任務，不能擴大單任務成功範圍。"}, {"source_id": "paper_derivation", "locator": "A.8八筆紀錄、唯一T與本节無fences/成績表的完整原文", "supports": "本節只有單一社團→代碼匹配的紙上例，沒有生成、訓練、部署驗收或本課位置分數。"}], "artifact_ids": ["section", "paper_pdf", "paper_text", "inspection"]}
]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "A.8",
    "source": "course/chapters/0A.md#A.8", "source_sha256": sha(ART / "inputs/section.md"),
    "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": [],
    "reviewed_on": date, "verdict": "pass",
    "figure_sha256": {"course/figures/rewrite-A-clue-position.svg": sha(ART / "inputs/course/figures/rewrite-A-clue-position.svg")},
    "frozen_input": {"artifact_id": "frozen_chapter", "sha256": sha(ART / "inputs/course/chapters/0A.md"), "meaning": "最初讀取的整章frozen input，原檔實bytes保存；source_sha256仍單獨對應A.8未正規化原bytes。"},
    "read_scope": inspection["read_scope"], "code_applicability": {"status": "not_applicable", "details": inspection["code_applicability"]},
    "empirical_applicability": {"status": "not_applicable", "details": inspection["empirical_applicability"]},
    "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "逐核原材料、控制、診斷與成熟研究定性敘述；每項證據有自己的實讀locator與支持邊界，無未定實質主張。"},
        "numeric_verification": {"status": "pass", "claim_ids": ["materials_and_figure", "position_controls", "diagnostic_conditions"], "details": "短CPU精確核8筆/3條件/T1,4,8/24格/gold一致；自己的byte序列化296×3、短版114與位置ID另標審閱檢查，未冒充本節實測成績。"},
        "figure_consistency": {"status": "pass", "claim_ids": ["materials_and_figure", "position_controls"], "details": inspection["figure_inspection"]},
        "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "親取arxiv v3并自行核首頁版本/§§2.1–2.3、3.1–3.2；親讀官方固定v4.57.1文件及原停止機制。§3.2直接支持合成近全對。URL、版本、存取日期、原資料SHA、locator與inspection皆保存。"},
        "limitations": {"status": "pass", "claim_ids": ["position_controls", "diagnostic_conditions", "position_effect_and_scope"], "details": "本節是單線索紙上材料，無位置score或原評測復現；token相等是將來實測前須驗的條件，296只代表本人byte例；短診斷不混分母，答案正確與EOS/上限分開，原研究不保證所有模型U形；未測長度外推、多資訊/部署、完整課程頁面，未做GPU/模型/資料下載。"}
    },
    "independence_record": inspection["independence"],
    "unresolved_substantive_claims": []
}
target = ROOT / "docs/technical-reviews/A.8.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
written = json.loads(target.read_text())
assert written["reviewer_task"] == TASK
assert written["source_sha256"] == sha(ART / "inputs/section.md")
assert written["verdict"] == "pass" and written["issues"] == []
for item in written["artifacts"]:
    assert sha(ROOT / item["path"]) == item["sha256"], item["id"]
print(json.dumps({"written": target.relative_to(ROOT).as_posix(), "reviewer_task": TASK,
                  "source_sha256": written["source_sha256"], "verdict": written["verdict"],
                  "artifacts": len(artifacts), "claims": len(claims), "assertions": "all passed"}, ensure_ascii=False))
