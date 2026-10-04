from pathlib import Path
import datetime
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

understanding = {
    "A.1": "提示裡的例子提供本次任務線索而不更新權重；三種提示印得出來與小模型真的按新規則答對是兩項不同證據。",
    "A.2": "RAG先取外部資料再放入當前提示，新地址能被使用不代表永久記住；正式新店名與改地址比較也只支持有限的逐題觀察。",
    "A.3": "本例檢索器按中文字及英文數字片段的重合數排序，能取到含查詢字的文件，卻不理解同義詞或知道文件是否提供答案。",
    "A.4": "來源編號存在、所引原文支持地址、地址回答了當前店名是三個判斷，簡單字串包含檢查只能展示其中的窄範圍。",
    "A.5": "留下hit與correct並介入提供正確公告，可定位檢索或用資料的瓶頸；按UNKNOWN協議成功與真正地址引用答對不能合算。",
    "A.6": "空檢索與命中但未公布地址都可能無法回答，衝突和不可信文件還要另核對，固定分支或UNKNOWN格式不能證明一般誠實能力。",
    "A.7": "完整序列化輸入與新增答案一起占上下文位置，縮短歷史或公告要另查必要證據，KV快取及容得下不保證遠距內容使用正確。",
    "B.1": "工具請求是名稱和參數的資料，寫出JSON或聲稱已用工具不會執行；工具題與COPY題的固定協議成績不能當中文自然選工具能力。",
    "B.2": "解析JSON與白名單、欄位、確切數字型別驗證是不同關卡，合法參數仍可能抄錯問題，正式嚴格入口另處理非有限數與重複欄位。",
    "B.3": "真正call_tool的返回值要依模型協議送回，示例tool角色紀錄不是render_chat直接支援的角色，工具算對之後仍可能抄錯或格式錯。",
    "B.4": "done、步數上限、非法請求及缺後續输出有不同意思，工具呼叫數與完成訊息數不同，EOS停止率也不等於原題完成成功率。",
    "B.5": "DIRECT、TOOL、ASK是先明定的教學策略，工具狀態與任務需求一起决定動作，人工標籤的印出不等於模型已判斷任務。",
    "B.6": "只對assistant的動作字母與EOS計答案代價可訓練選卡；選卡器和填JSON、真正運算、最後回答仍是分開的能力。",
    "B.7": "可靠選卡要分任務、狀態及問法檢查，96題全對仍被新問法中6道應用工具題全失敗推翻廣泛泛化解讀，並未量端到端助手。",
    "B.8": "可以比較直接回答及完整工具路線的失敗代價和額外成本，但動作token機率不是算術答對機率，人工損失算例需要獨立驗證資料才可成策略。",
    "C.1": "步驟提供後續可讀的中間結果並增加成本，本輪步驟模型改善是在更多訓練token下取得，GSM8K片段通路段落會打斷正在建立的比較問題。",
    "C.2": "等式正確、步驟接續、參數符合原題及最後答案正確應分開核對，窄驗證器的通過不能當任何論證或內部推理忠實的證據。",
    "C.3": "1-(1-p)^k算固定成功率及獨立試驗下集合含正解的機會，n=k的pass@k特例可用本組命中；共同錯誤偏好與統計不獨立需說得更精確。",
    "C.4": "同一份含正解的候選可能讓多數決選錯、精確驗證選對，全部題目的最後成功率與已命中子集中的選對率有不同分母。",
    "C.5": "生成与驗證都占預算，token數、運算及等待時間各有單位，batch可令更多候選生成更快，但整輪訓練保存時間應與一次推論請求分開。",
    "C.6": "reward把固定格式與答案真值變為回饋，弱規則可獎勵列舉全部數字而未交單一答案，有限動作策略的抽樣不是語言模型生成。",
    "C.7": "正advantage沿所選動作log機率的策略梯度更新能略增其機率，負值方向相反；真正有限策略訓練及高reward仍未改善新題泛化。",
}
issue_by_section = {"C.1": ["F2"], "C.3": ["F1"], "C.5": ["F3"]}
read_ranges = {"0A": [[1,85],[86,170],[171,239]], "0B": [[1,85],[86,170],[171,250],[251,336]], "0C": [[1,80],[81,160],[161,230],[231,292]]}
chapters = []
sections = []
figures = json.loads((OUT / "figure-source-identity.json").read_text())
code = json.loads((OUT / "local-example-execution.json").read_text())
read_order = []
ordinal = 0
for chapter in ("0A", "0B", "0C"):
    path = ROOT / f"course/chapters/{chapter}.md"
    snapshot = OUT / f"{chapter}.source.md"
    raw = snapshot.read_bytes()
    lines = raw.splitlines(keepends=True)
    headings = []
    for i, line in enumerate(lines):
        match = re.match(rb"## ([A-C]\.\d+) (.*)", line.rstrip(b"\r\n"))
        if match:
            headings.append((i, match.group(1).decode(), match.group(2).decode()))
    chapters.append({
        "source": str(path.relative_to(ROOT)),
        "snapshot": str(snapshot.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "line_count": len(lines),
        "intro_sha256": hashlib.sha256(b"".join(lines[:headings[0][0]])).hexdigest(),
        "complete_read_ranges_in_order": read_ranges[chapter],
        "read_mechanism": "Every numbered line displayed to reviewer in sequential chunks; hashes recorded separately, not used as reading substitutes.",
    })
    for j, (start, section_id, title) in enumerate(headings):
        end = headings[j+1][0] if j+1 < len(headings) else len(lines)
        body = b"".join(lines[start:end])
        section_snapshot = OUT / "sections" / f"{section_id}.source.md"
        section_snapshot.parent.mkdir(exist_ok=True)
        section_snapshot.write_bytes(body)
        section_figures = []
        for reference in re.findall(rb"!\[[^\]]*\]\(([^)]+)\)", body):
            figpath = (path.parent / reference.decode()).resolve()
            section_figures.extend([f for f in figures if f["source"] == str(figpath.relative_to(ROOT))])
        ordinal += 1
        read_order.append(section_id)
        sections.append({
            "read_order": ordinal, "section_id": section_id, "title": title,
            "source": str(path.relative_to(ROOT)), "line_start": start+1, "line_end_inclusive": end,
            "chapter_sha256": hashlib.sha256(raw).hexdigest(),
            "section_sha256": hashlib.sha256(body).hexdigest(), "section_bytes": len(body),
            "section_snapshot": str(section_snapshot.relative_to(ROOT)),
            "reader_understanding": understanding[section_id],
            "issues": issue_by_section.get(section_id, []),
            "figures": section_figures,
            "exact_python_fence_executed": next(r for r in code["records"] if r["section_id"] == section_id),
            "exercise_execution": "not run by this reviewer; parent owns full notebook execution",
        })

final_identity = []
for chapter in chapters:
    current = (ROOT / chapter["source"]).read_bytes()
    final_identity.append({
        "source": chapter["source"], "initial_sha256": chapter["sha256"],
        "rechecked_sha256": hashlib.sha256(current).hexdigest(),
        "same_bytes_as_read": current == (ROOT / chapter["snapshot"]).read_bytes(),
    })
for figure in figures:
    current = (ROOT / figure["source"]).read_bytes()
    final_identity.append({
        "source": figure["source"], "initial_sha256": figure["sha256"],
        "rechecked_sha256": hashlib.sha256(current).hexdigest(),
        "same_bytes_as_read": current == (OUT / "figures" / Path(figure["source"]).name).read_bytes(),
    })
inspection = {
    "schema": "independent-wholebook-module-precheck-v1",
    "task_identity": "/root/v4_wholebook_scan_foundations",
    "parent_task_identity": "/root",
    "recorded_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": ["course/chapters/0A.md", "course/chapters/0B.md", "course/chapters/0C.md"],
    "review_type": "Fresh independent full text module precheck; not wholebook final approval.",
    "reader_profile": "Chinese-reading high-school student with good mathematics or university beginner with basic mathematics, initially learning LLMs; terms and explicit prerequisite links considered.",
    "isolation": {
        "prior_review_verdicts_read": False,
        "author_history_read": False,
        "chapter_20_drafts_read": False,
        "canonical_chapters_or_checkers_or_reviews_or_reading_times_modified": False,
        "child_agents_spawned": False,
    },
    "chapter_intro_understanding": {
        "0A": "當前輸入可給例子或新文件，但必須追蹤是否真的讀對資料。",
        "0B": "先接通工具請求到結果與停止，再把選工具的判斷放到請求之前檢查。",
        "0C": "寫步驟、多試、選答案、用回饋更新是四種行為，需要分別計數與驗證。",
    },
    "read_order": read_order,
    "full_read_complete": True,
    "section_count": len(sections),
    "chapters": chapters,
    "sections": sections,
    "figure_review": {
        "svg_sources_read_in_full": True,
        "rendered_pngs_visually_inspected_with_view_image": True,
        "inspected_order": ["rag.svg", "tools.svg", "tool_choice.svg", "architecture_candidate_selection.svg", "architecture_policy_update.svg"],
        "render_receipts": "figure-render-receipts.json",
        "finding": "All five static diagrams match the corresponding textual process and numeric example.",
        "limitation": "Animation timing, browser layout at phone widths, and accessibility behavior were not tested.",
    },
    "local_code_reads": [
        {"source":"tiny_perceptron/retrieval.py","full_file_read":True,"sha256":hashlib.sha256((ROOT/'tiny_perceptron/retrieval.py').read_bytes()).hexdigest(),"reason":"Check lexical scoring, call_tool, and tool_loop semantics against explanations."},
        {"source":"tiny_perceptron/data.py","lines_read":[1,104],"full_file_read":False,"sha256":hashlib.sha256((ROOT/'tiny_perceptron/data.py').read_bytes()).hexdigest(),"reason":"Read ByteTokenizer and render_chat definitions used by snippets."},
    ],
    "authority_receipts": json.loads((OUT/'authority/fetch-receipts.json').read_text()),
    "authority_use": [
        {"source":"authority/human-eval-pass-at-k.pdf","locations":"§2.1 equation (1), Figure 3, Appendix A","understanding":"The pass@k estimator averages per-problem success over candidate sets; its binomial proof uses c~Binom(n,p) at fixed task, and n=k reduces to the actual set-hit indicator."},
        {"source":"authority/python-json.html","locations":"Standard Compliance and Interoperability; Infinite and NaN Number Values; Repeated Names Within an Object","understanding":"Python JSON defaults accept non-finite values and repeated names, so the strict-wrapper caveat in B.2 is justified."},
        {"source":"authority/pytorch-distributions-source.py.txt","locations":"Score function / REINFORCE module documentation","understanding":"Negative log_prob times reward implements gradient ascent on reward through a gradient-descent optimizer; C.7 single-step sign and arithmetic match."},
    ],
    "numeric_execution": {"exact_short_python_fences":22,"executed_without_exception":22,"log":"local-example-execution.log","structured_result":"local-example-execution.json","python_version":code['python_version'],"torch_version":code['torch_version'],"c3_probability_comparison":code['c3_probability_comparison']},
    "findings": [
        {"id":"F1","priority":"medium","section":"C.3","lines":[105,105],"kind":"probability wording","issue":"Sharing an error preference under randomized generation is not sufficient to establish statistical dependence; fixed-task iid sampling and heterogeneous task success rates need to be separated.","recommendation":"Clarify the formula applies to fixed problem/weights/prompt/decoding with independent randomness, and distinguish low per-task p or mixed p across problems from dependence caused by deterministic/reused randomness or sequentially conditioned samples."},
        {"id":"F2","priority":"medium","section":"C.1","lines":[44,56],"kind":"pedagogical sequence","issue":"The unrelated GSM8K fragmented-record loss pilot and lengthy fingerprint/LFS/checkpoint instructions interrupt the steps-versus-short-answer argument before step validation is learned.","recommendation":"Keep same-start and unequal-token-budget conclusions plus a compact results link in C.1; move full hashes, GSM8K pipeline pilot, LFS instructions and output inventory into an optional experiment/reproduction page linked after C.7."},
        {"id":"F3","priority":"medium","section":"C.5","lines":[204,206],"kind":"cost/latency boundary","issue":"The paragraph moves from per-request latency to whole training/evaluation/save/setup/upload wall time, then loosely calls these stages a complete latency comparison, risking unit/scope confusion.","recommendation":"State explicitly that a ready service's request latency includes input preparation, candidate generation, selection and output formatting; training, saving, environment setup and uploading are experiment/preparation cost, while cold-start latency is an optional separate scenario."},
    ],
    "major_numeric_or_implementation_error_found": False,
    "toy_capability_overclaim_found": False,
    "not_run_or_not_verified": [
        "No long RAG/tools/tool_choice/reasoning training commands rerun; historical model metrics/checkpoints were not independently reproduced.",
        "No notebooks or exercise variants executed by this reviewer; parent owns full notebook execution.",
        "No CUDA or L4 timing reproduction; only current CPU short snippets checked.",
        "Explicit prerequisite chapters and all linked reports were not read as full documents; this review records only the assigned appendices and the listed code/source excerpts.",
        "No browser animation/mobile/accessibility execution.",
        "No chapter 20 draft read; this is a module precheck and cannot establish wholebook final readiness.",
    ],
    "source_identity_recheck": final_identity,
    "all_reviewed_canonical_bytes_unchanged_at_report_time": all(r['same_bytes_as_read'] for r in final_identity),
}
(OUT/'inspection.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+'\n')
(OUT/'source-identity-recheck.json').write_text(json.dumps({'checked_at_utc':inspection['recorded_at_utc'],'sources':final_identity},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'sections':len(sections),'short_fences_executed':len(code['records']),'unchanged':inspection['all_reviewed_canonical_bytes_unchanged_at_report_time']},ensure_ascii=False))
