"""把本次逐項核對的原始判斷寫成正式技術審閱報告。"""

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TARGET = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_readme/technical-readme-original-01.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((HERE / "audit-output.json").read_text())
    supplement = json.loads((HERE / "supplementary-output.json").read_text())
    network = json.loads((HERE / "network-output.json").read_text())
    env = audit["environment"]
    artifacts = []
    for ident, name, kind, description in [
        ("audit_program", "audit_readme.py", "code", "教材枚舉、Notebook bootstrap、資料及權重清單與短命令核對程式。"),
        ("audit", "audit-output.json", "execution", "枚舉結果、實際命令及 stdout/stderr；另含 CPU dry-run 參數逐張量不變檢查、圖中文字編號推導與生成輸入的完整小份副本。"),
        ("network_program", "network_evidence.py", "code", "以 curl 讀官方文件、公開網站與固定 HF 版本，只保存短摘錄與收據。"),
        ("network", "network-output.json", "execution", "curl 命令、原文 SHA、短摘錄、網站腳本來源、build-info 與固定 HF 版本的目錄回應。"),
        ("supplement_program", "supplementary_checks.py", "code", "讀原始 train/test 收據、實際重建參數量、執行公開 CPU 下載與 Auto 控制流程。"),
        ("supplement", "supplementary-output.json", "execution", "原始結果定位、參數重算、LFS pointer、匿名下載與七個 CPU 子程序結果；MathJax 摘錄另經 curl 精確定位，補讀 JupyterLab 官方文件。"),
        ("report_program", "write_report.py", "code", "本審閱者原始判斷與報告組裝程式；先在記憶體檢查格式，再以 exclusive create 保存。"),
    ]:
        item = {"id":ident, "kind":kind, "path":(HERE / name).relative_to(ROOT).as_posix(), "sha256":sha(HERE / name), "description":description}
        if kind == "execution":
            item["environment"] = env
            item["command"] = {
                "audit":"PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-readme/audit_readme.py；另以 .venv/bin/python - 執行 audit-output.json 的 runtime_verification 所記 AdamW 包裝器檢查。",
                "network":"PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-readme/network_evidence.py",
                "supplement":"PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-readme/supplementary_checks.py；另以 curl --fail --silent --show-error --location --max-time 40 讀取 network 所列 MathJax 與 JupyterLab URL。",
            }[ident]
            item["result"] = {
                "audit":"312 份章節 Notebook、26 個額外編號小節；dry-run loss 5.770333290100098、grad_norm 0.9595053791999817、140 個有效位置，參數逐張量不變且沒有權重檔。build_course --check 通過；Notebook 單節檢查因本機未安裝 matplotlib 失敗；uv notebook 安裝 dry-run 可解析。",
                "network":"所有 curl returncode 為 0；官方文件已讀原文。公開站列出 312 節、executed_cpu_outputs=true；固定 v2 目錄有四個子目錄、16 個檔案。kernel 文件沒有 sys-prefix 字串，已另讀 ipykernel 的版本化原始碼，未將未命中當成支持。",
                "supplement":"重算 MoE 5447107、Dense 2288067；匿名 text_foundation 下載、六組權重 probe 與一個 infer 子命令均 passed。v2 實體資料包在本機只有 LFS pointer，沒有把它當成已解包資料。Auto 使用替身只核對控制流程，不計模型品質。",
            }[ident]
        artifacts.append(item)
    sources = []
    source_ids = {}

    def repo(path, note):
        if path in source_ids:
            return source_ids[path]
        ident = "repo_" + str(len(source_ids) + 1)
        source_ids[path] = ident
        sources.append({"id":ident, "kind":"repository_code", "title":path, "path":path, "sha256":sha(ROOT / path),
            "version":"2026-10-20 實際讀取的工作樹；內容以本項 SHA-256 固定", "verified":True, "inspection_note":note})
        return ident

    def official(ident, title, rec, version, note):
        sources.append({"id":ident,"kind":"official_source" if "raw.githubusercontent" in rec["url"] else "official_docs",
            "title":title,"url":rec["url"],"version":version,"verified":True,"checked_original":True,
            "accessed_on":"2026-10-20","authority_reason":"維護者直接發布的文件或原始碼，核對其自身 API／產品行為。",
            "inspection_note":note + " 本次 curl 原文指紋：" + rec["raw_sha256"]})

    n = {r["id"]:r for r in network["sources"]}
    sn = {r["id"]:r for r in supplement["network"]}
    official("uv_docs","uv Locking and syncing",n["uv"],"2026-10-20 讀取的官方文件", "已讀 Automatic lock and sync、Syncing optional dependencies、Syncing development dependencies；uv run 會同步，extra 須明確指定，--frozen 沿用 lock，--group 選群組。")
    official("lfs_docs","Git LFS 設定原始手冊",n["lfs"],"main，2026-10-20 原文 SHA 固定", "已讀 GIT_LFS_SKIP_SMUDGE 的完整條目；1 會跳過 smudge/filter-process 的物件實體化。")
    official("git_windows","Git for Windows FAQ",n["longpaths"],"2026-10-20 讀取的 FAQ", "已讀 long path 問題：預設 260 字元限制與 core.longpaths=true 的作用。--global 是目前使用者設定範圍，本文在同一讀者帳號的操作情境下成立。")
    official("torch_backward","torch.Tensor.backward",n["autograd"],"PyTorch 2.14 文件；本機 torch 2.14.1", "已讀 backward 描述：沿 chain rule 計算葉節點梯度並累積至 .grad；它與 optimizer.step 分開。梯度可用不等於已完成參數更新。")
    official("kernel_source","ipykernel kernelspec 安裝入口",sn["ipykernel"],"ipykernel v7.4.0", "已讀 InstallIPythonKernelSpecApp 的 --sys-prefix、--name、--display-name：sys-prefix 指向目前 Python 的 sys.prefix，名稱與顯示名功能分開。")
    official("jupyter_docs","Starting JupyterLab",sn["jupyterlab"],"2026-10-20 讀取的官方文件", "已讀啟動段：jupyter lab 啟動服務，可把工作目錄路徑當引數。與 README 的 jupyter lab notebooks 一致。未在本機開 GUI。")
    official("colab_docs","Google Colab 入口",sn["colab_entry"],"2026-10-20 匿名取得的產品入口", "匿名入口提供 Sign in；另讀 FAQ 的帳號與 Workspace 使用條件。只核對雲端使用需 Google 帳號，不把本機 Notebook 當成 Colab 實機驗證。")
    official("mathjax_docs","Hosting Your Own Copy of MathJax",sn["mathjax"],"MathJax v3.2 官方文件；站內資產固定 3.2.2", "已讀 Hosting Your Own Copy of MathJax 的開頭與 Acquiring the MathJax Code；官方明說可在自己伺服器或硬碟本機提供資產以離線使用。排版本身在瀏覽器執行，不需要遠端排版服務。")
    for ident, artifact in (("audit_exec","audit"),("network_exec","network"),("supplement_exec","supplement")):
        sources.append({"id":ident,"kind":"execution","title":"本次獨立核對的執行紀錄 " + artifact,"verified":True,"artifact_id":artifact})
    claims = []

    def claim(ident, kind, statement, location, expected, observed, details, evidence, artifact="audit", scope="只支持指定版本與此處列明的行為；不外推為所有硬體、版本或模型品質保證。", status="verified", denominators=None):
        refs = [{"source_id":s,"locator":loc,"supports":support} for s,loc,support in evidence]
        item = {"id":ident,"kind":kind,"statement":statement,"location":location,"status":status,"evidence":refs,"artifact_ids":[artifact],"scope":scope}
        if kind != "concept":
            item["verification"] = {"method":"executed","expected":expected,"observed":observed,"details":details}
            if kind == "numeric":item["verification"]["tolerance"]="整數與清單精確相等；不以浮點比率代替計數。"
            if kind == "empirical":item["verification"]["denominators"]=denominators
        claims.append(item)

    def ev(path, locator, supports):
        return (repo(path,"已實際讀取 " + locator + "；" + supports), locator, supports)

    def exec_ev(which, locator, support):
        return (which + "_exec", locator, support)

    A=lambda loc,support:exec_ev("audit",loc,support)
    S=lambda loc,support:exec_ev("supplement",loc,support)
    N=lambda loc,support:exec_ev("network",loc,support)
    loc="開頭教材預覽說明"
    claim("website","software","網站提供正文、SVG 與 CPU 執行輸出，閱讀不需要先安裝專案。",loc,"靜態網頁包含正文／圖與輸出","公開 1.1 頁包含輸出、圖、Colab 與 Notebook 入口；build-info executed_cpu_outputs=true。","curl 讀取現站；另讀 exporter 如何把已執行 Notebook 輸出寫入靜態頁。",[N("site/site_info","實際公開產物與版本狀態"),ev("scripts/export_course.py","checked_notebook、build","輸出與原始 cell 逐格核對後匯入頁面")],"network")
    claim("notebook_source","software","同一節 Notebook 由正文產生，頁尾提供下載與 Colab 入口。",loc,"同一小節對應同一 Notebook","build_course --check 回傳 0；公開頁有下載與 Colab 連結。","比對產生器與 exporter section_footer 的 notebook 路徑。",[ev("scripts/build_course.py","notebook、build","Markdown 是 Notebook 來源"),ev("scripts/export_course.py","section_footer","下載與 Colab 指向同節 Notebook"),A("commands/build_course --check","現行來源與 Notebook 同步")])
    claim("colab_bootstrap","software","有程式的 Notebook 在第一個程式格下載專案並安裝套件。",loc,"有程式的 Notebook 先執行準備格","audit notebook_bootstrap 的所有有 code 檔案第一格皆為 course_setup。","逐份解析 Notebook；讀 BOOTSTRAP 的 google.colab 偵測、GIT_LFS_SKIP_SMUDGE、git clone 與 pip install -e。",[ev("scripts/build_course.py","BOOTSTRAP、notebook","Colab 分支取得專案與套件，本機沿用 repo"),A("checks.notebook_bootstrap","逐份枚舉第一格")],scope="只支持準備格內容與產生結果；沒有在 Colab 雲端執行。")
    claim("colab_account","software","使用 Colab 雲端需要 Google 帳號。",loc,"雲端登入條件","Google 產品入口提供 Sign in；FAQ 列帳號與 Workspace 存取。","curl 讀產品入口與官方 FAQ；沒有登入或啟動雲端運算。",[("colab_docs","Sign in／FAQ account access","帳號使用情境")],"supplement")
    claim("cpu_small","software","局部短實驗可使用 CPU，不必先租 GPU。",loc,"不需外部 GPU 通路","CPU toy dry-run、公開文字模型下載與推論成功；已發布網站保存 CPU 輸出。","本次 CPU 執行只支持被實跑入口；全站 CPU 依發布產物標記與 kernel 發布流程核對。",[A("commands/train","短流程用 CPU"),S("checks.anonymous_cpu_probe","公開模型 CPU 操作"),ev(".github/workflows/pages.yml","Execute every lesson in an independent CPU kernel","發布流程以 CPU kernel 執行")],"supplement",scope="不表示正式全量訓練在筆電上很快，也不表示已做 Colab 雲端驗證。")
    for ident,statement,expected,observed,where in [
        ("chapter_count","教材有20個主章，另有A／B與C單元。","20個主章、3個延伸／推理單元","01–20、0A、0B、0C","checks.catalogue.chapters"),
        ("notebook_count","逐課 Notebook 共312份。","312","312","checks.catalogue.chapter_sections／notebooks"),
        ("aux_count","閱讀指南、暖身、訓練操作及名詞頁另有26個編號小節。","26","26","checks.catalogue.auxiliary_sections"),
        ("total_sections","合計338個編號小節。","338","312+26=338","checks.catalogue"),
    ]:
        claim(ident,"numeric",statement,"教材提供什麼／數量段",expected,observed,"直接枚舉當前 Markdown 編號標題與 .ipynb；非沿用 README 的數字。",[A(where,"原始檔案枚舉與加總")])
    claim("topic_progression","software","前段涵蓋接字表、MLP與手寫注意力，後段涵蓋MTP、現代架構、Dense／MoE、長上下文、cache／SDPA、量化與蒸餾。","教材提供什麼／主題段","課程索引有對應主題","章檔標題與 lesson-index 對應上述主題。","讀取章標題及指定前置的內容定位；此處是涵蓋範圍，未替各方法作效益保證。",[A("checks.catalogue.chapters","獨立枚舉各章小節"),ev("course/lesson-index.json","各項 id/title/source","正文主題與入口")])
    claim("feedback_navigation","software","7.17介紹預訓練／後訓練分工，第13章涵蓋偏好、獎勵模型、PPO與DPO。","教材提供什麼／預訓練段","指定位置有相應內容","7.17、13.1–13.17 的題名與內容符合導覽。","核對目標小節與各段命名；此處僅判斷導覽描述。",[A("checks.catalogue.chapters","導覽主題存在")])
    claim("instruction_context_navigation","software","課程把指令遵循及長上下文分成文中列出的不同檢查面向。","教材提供什麼／預訓練段","範圍、格式、停止與資訊使用等獨立討論","第8章、14章、16章與A延伸有對應小節。","核對索引標題與被連結教材的位置，不推導通用品質定律。",[A("checks.catalogue.chapters","分項教學位置可追蹤")])
    v2=ev("docs/selftrained/results/v2-final-public-results.json","architectures、data_scope","固定模型及測試範圍")
    claim("v2_from_zero","empirical","v2 的文字核心與感知入口由本專案從隨機初始化訓練，包含 MoE 與 Dense。","教材提供什麼／v2段","兩架構都有自行訓練來源","原始 train-receipt 的 origin 與模型建構器使用本專案隨機初始化模組，沒有載入預訓練編碼器。","讀原始來源與收據，並實際重建兩個模型；本次不重做訓練。",[S("checks.raw_training","原始 train 收據及來源鏈"),ev("tiny_perceptron/selftrained/model.py","LimitedAssistant.__init__、SelftrainedLanguageModel","文字與感知模組直接建構")],"supplement",denominators={"architectures":2,"training_rows":28876,"validation_rows":2435,"seed":20261006})
    for ident,name,number in (("moe_parameters","moe",5447107),("dense_parameters","dense",2288067)):
        claim(ident,"numeric",f"{name} 共有 {number:,} 個參數。","教材提供什麼／v2段",str(number),str(supplement["checks"]["recomputed_parameters"][name]),"按公開固定 config 建立 LimitedAssistant，sum(p.numel())，含文字與三個感知入口，不把 active 參數當總參數。",[S("checks.recomputed_parameters."+name,"實際由模型重算參數量"),v2],"supplement")
    claim("v2_tests","empirical","MoE與Dense各在3,734題固定最後測試上評估。","教材提供什麼／v2段","各3734題且完成","兩份原始 metrics 的 count/expected_count=3734，evaluation-receipt status=complete、completed_count=3734。","逐一讀原始結果與 journal，核對各任務 count 加總及選中權重 SHA；沒有重新跑模型測試。",[S("checks.raw_final_tests","原始測試收據"),v2],"supplement",denominators={"moe_test_rows":3734,"dense_test_rows":3734,"per_task_counts":"audit.checks.v2_results.architectures.*.per_task_counts"},scope="固定資料的描述性結果；題目不是獨立自然語言能力抽樣，兩架構也非等計算公平比較。")
    claim("v2_unmet","empirical","兩版仍未通過全部原定能力判準。","教材提供什麼／v2段","至少各有一項低於原門檻","tool_roundtrip 門檻0.95；MoE 0/276，Dense 7/276；MoE OCR exact 252/324 亦低於0.8。","把原始 metrics.thresholds 與相同任務的分子／分母比較，沒有挪用成功示範。",[S("checks.raw_final_tests","原始門檻與原始分數")],"supplement",denominators={"tool_rows_per_architecture":276,"moe_ocr_rows":324})
    claim("v2_scope","software","v2 任務限於三類服飾／兩格位置、12個已知字的指定連續1–4字區域、三種銀行客服主題與有限文字／計算器。","教材提供什麼／v2段","固定有限任務","data_scope 記三類服飾、12字、caller-provided ROI、地址／App／卡片；程式有有限感知與工具入口。","核對模型輸入契約、結果 data_scope 與 manifest.scope；不把接受模態輸入解讀為一般能力。",[v2,ev("tiny_perceptron/selftrained/model.py","MODALITY_TOKENS、LimitedAssistant._embeddings","兩個影像格與明示模態 payload 契約"),ev("docs/selftrained/v2-manifest.json","scope","有限資料範圍")],"supplement")
    claim("separate_synthetic","software","合成小世界的固定實驗是另一組模型，不能替代 v2 成績。","教材提供什麼／v2段","不同模型格式與固定來源","capstone-public 清單使用 capstone-v1／capstone-ptq-v1，v2 使用 selftrained_multimodal_v1 及不同固定 HF 版本。","比較模型清單、格式與結果來源；沒有混合兩組分數。",[ev("docs/course-experiments/capstone-public.json","models.format_version、checkpoint","舊合成模型輸出"),v2])
    claim("mature_pipeline","software","第20章沿用 Qwen3-VL 處理圖文，Whisper 先將語音轉成文字。","教材提供什麼／第20章段","語音先 ASR 再接圖文助理","natural_assistant 使用 Qwen3VLForConditionalGeneration 與 WhisperForConditionalGeneration，ASR 文字進入聊天歷史。","讀 from_pretrained 與語音轉文字入口；沒有下載大模型或執行正式成品。",[ev("tiny_perceptron/natural_assistant.py","MODEL_ID、load_model、transcribe","預訓練模型與語音前處理分工"),ev("docs/natural-assistant/v4/STUDENT.md","本版指定配置","Qwen3-VL-2B-Instruct 與 Whisper-large-v3-turbo")],"supplement")
    claim("required_units","software","Thinking effort、Looped Transformer與Jev式判斷已在必修路線；C.1–C.5接第7章後。","教材提供什麼／必修段","指定單元及順序存在","7.23、14.11、9.11 與 C.1–C.5 存在；zensical 導覽在第7章後接 C.1–C.5。","核對 lesson-index 與 zensical nav，而非只看 README 的連結字面。",[ev("zensical.toml","project.nav","實際站點閱讀順序"),A("checks.catalogue.chapters","指定編號可定位")])
    claim("three_products","software","三個成品分別是MoE助理、有限文字Looped解題器與機率判斷器。","教材提供什麼／三成品段","三個獨立用途","v2 與 lesson-products 的 B/C 有不同模型格式及入口；B 解兩盒／三數有限題，C 對候選計分。","讀固定產品操作與公開入口契約。",[v2,ev("tiny_perceptron/lesson_products/inference.py","Product.solve、Product.judge","B與C的輸入與輸出契約")],"supplement")
    claim("auto_control","software","Auto先用機率判斷器確認支援任務與資訊充分性，再按規則選Looped輪數。","教材提供什麼／Auto段","choice→info→規則→solve","auto_solve 先做兩次 judge；unsupported／clarification_needed 會停止；其餘用 rule_depth 再依硬上限執行 solve。替身執行成功。","實際跑控制函式的替身，並讀原始函式；替身沒有產生模型品質證據。",[ev("tiny_perceptron/lesson_products/inference.py","auto_solve、rule_depth","呼叫順序及規則選輪數"),S("checks.auto_order","實際控制通路")],"supplement",scope="只支持有限語法的規則控制；不是模型預測不同預算效益的學習式控制器。")
    claim("products_l4","empirical","Looped解題器與機率判斷器已在NVIDIA L4完成訓練。","教材提供什麼／三成品段","B/C均完成L4更新","initial-b/c-execution status=completed、gpu_name=NVIDIA L4、weights_changed=true；分別1000與800更新。","讀原始執行紀錄的 training 與 environment，沒有以CPU smoke 代替GPU訓練。",[ev("docs/lesson-products/results/initial-b-execution.json","status、gpu_name、training","B的實際執行"),ev("docs/lesson-products/results/initial-c-execution.json","status、gpu_name、training","C的實際執行"),A("checks.lesson_training","原始輸出擷取")],denominators={"b_updates":1000,"c_updates":800,"b_parameters":170003,"c_parameters":84545})
    claim("test_after_freeze","empirical","兩產品在設定凍結後才開始最後測試。","教材提供什麼／三成品段","先凍結，再啟動test","settings-freeze 記07:30:19 UTC，pretest記07:31:35，test-started記07:33:06；開始測試的 SHA 與設定檔一致。","讀當時設定、測試前與測試啟動標記；比對 SHA 與時間，沒有把完成後報告時間單獨當成事前凍結證明。",[ev("docs/lesson-products/results/settings-freeze-v1.json","recorded_at、test_outcomes_observed","事前固定設定"),ev("docs/lesson-products/results/pretest-freeze-record-v1.json","test_marker_absent、settings_freeze_sha256","啟動前紀錄"),ev("docs/lesson-products/results/settings-freeze-v1.test-started.json","recorded_at、settings_freeze_sha256","實際啟動標記")],denominators={"settings_freeze_files":1,"formal_test_start_markers":1},scope="依保存的同一批次原始紀錄核對時序；本次不再啟動測試或宣稱獨立量測訓練全史。")
    product_result=ev("docs/lesson-products/results/final-report-v1.json","b.5、c、shifts.c_wording","固定測試分子與分母")
    for ident,statement,expected,observed,where,denom in [
        ("b_boxes","Looped解題器兩盒追蹤答對2928/2956。","2928/2956","b.5.boxes correct=2928,count=2956","checks.lesson_products.b_effort5.boxes",{"test_rows":2956,"families":1478,"loops":5}),
        ("b_sum","Looped解題器三數加法答對2/42。","2/42","b.5.sum correct=2,count=42","checks.lesson_products.b_effort5.sum",{"test_rows":42,"families":10,"loops":5}),
        ("c_fixed","機率判斷器在固定測試題的候選判斷上全對。","三種問題皆全對","Choice11797/11797；Info10890/10890；Rubric10890/10890；校準前後argmax一致。","checks.lesson_products.c",{"choice_rows":11797,"info_rows":10890,"rubric_rows":10890}),
        ("c_wording","機率判斷器換一種說法會出錯。","變化測試有錯題","Choice11493/11797；Info與Rubric各10773/10890。","checks.lesson_products.c_wording",{"choice_rows":11797,"info_rows":10890,"rubric_rows":10890}),
    ]:
        claim(ident,"empirical",statement,"教材提供什麼／三成品成績段",expected,observed,"實際解析原始固定報告，對照分子、分母、輪數與問題型別；沒有重跑模型測試。",[product_result,A(where,"獨立擷取及對照")],denominators=denom,scope="有限固定語法與保存的同一測試配置；家族切分仍存在跨家族相同通用觀測，不能推成未見中文或開放領域能力。")
    claim("skip_smudge","software","GIT_LFS_SKIP_SMUDGE=1讓clone跳過訓練資料實體下載。","安裝與開啟Notebook／clone段","先取得LFS pointer","官方手冊說1跳過smudge；本機v2檔為133 bytes pointer，內含真實物件大小與SHA。","讀官方手冊條件與 .gitattributes 的LFS規則；沒有執行額外clone。",[("lfs_docs","GIT_LFS_SKIP_SMUDGE","跳過物件實體化"),S("checks.archive","pointer與實體物件明確區分")],"supplement")
    claim("no_lfs_short","software","只閱讀與跑局部短程式不需要先下載LFS訓練包；列清單也不需要Git LFS。","安裝及最小流程／LFS說明","列清單與短流程不呼叫git lfs","prepare_data、dry-run及fetch_training_assets --list成功；程式僅在選中包仍是pointer時呼叫git lfs pull。","實際執行不下載的入口，讀參數分支；未測試已損壞且全域filter必填但git-lfs遺失的特殊Git安裝。",[ev("scripts/fetch_training_assets.py","main 的 --list 與 pointer 分支","只有下載已選包才需LFS"),A("commands","短命令實跑")])
    claim("windows_paths","software","Windows預設Git長路徑受限；core.longpaths=true可允許Git處理較長路徑。","安裝與開啟Notebook／PowerShell段","設定長路徑處理","官方FAQ的260字元限制與core.longpaths設定與正文一致；CI在checkout前也作相同設定。","讀官方FAQ與ci.yml；沒有在Windows執行。",[("git_windows","long path FAQ","預設與選項作用"),ev(".github/workflows/ci.yml","Allow long review evidence paths on Windows","實際CI配置")],"network",scope="Git本身的支援，不保證所有Windows程式都能處理長路徑；--global依同一使用者帳號生效。")
    install=ev("pyproject.toml","project.optional-dependencies、dependency-groups、tool.uv.sources","CPU／CUDA來源與Notebook工具群組")
    claim("cpu_install","software","uv sync --frozen --extra cpu --group notebook準備CPU閱讀與Notebook工具。","安裝命令區塊","lock可解析，包含kernel、JupyterLab與matplotlib","uv sync --dry-run回傳0，計畫安裝ipykernel7.4.0、JupyterLab4.6.4、matplotlib3.11.2與nbclient。","依相同命令加--dry-run，核對pyproject與lock；沒有變更共用.venv。",[install,ev("uv.lock","tiny-perceptron-vlm、torch與Notebook群組解析","固定依賴"),("uv_docs","extras/groups/frozen","uv旗標契約"),A("commands/uv sync --dry-run","本機解析成功")],scope="安裝命令的固定配置與本機解析成立；不是全新Windows／Linux／CUDA安裝實跑。")
    claim("kernel_install","software","ipykernel install --sys-prefix --name tiny-perceptron --display-name設定指定kernel。","安裝命令區塊","內部名稱與顯示名稱配對","官方v7.4.0原始碼將sys-prefix指定至Python環境；Notebook metadata的kernelspec同名。","讀版本化安裝程式與Notebook產生器，避免寫入共用環境的kernel設定。",[("kernel_source","InstallIPythonKernelSpecApp","旗標含義"),ev("scripts/build_course.py","notebook metadata.kernelspec","相同name/display_name")],"supplement")
    claim("jupyter_start","software","jupyter lab notebooks從notebooks工作目錄列出檔案，先開01/1.1.ipynb。","安裝命令區塊後段","路徑參數有效且Notebook存在","官方啟動文件支援工作目錄參數；312份Notebook枚舉含notebooks/01/1.1.ipynb。","讀官方啟動文件與實際檔案；沒有啟動GUI伺服器。",[("jupyter_docs","Starting JupyterLab","工作目錄引數"),A("checks.catalogue／notebook_bootstrap","實體Notebook位置")],"supplement")
    claim("hardware_extras","software","CPU、Apple Silicon預設與CUDA12.6／13.0使用表列uv安裝選項。","安裝選擇表","三個互斥extra指向各自官方index，Mac有對應wheel","tool.uv.sources有cpu/cu126/cu130；uv.lock有darwin arm64的torch wheel，Mac預設與cpu extra都可解析。","解析pyproject與lock的source與wheel平台；實際本機為Apple Silicon、torch2.14.1，可完成CPU計算。",[install,A("checks.installation","來源與wheel枚舉")],scope="CUDA仍需讀者具相容硬體與驅動；本次未在CUDA12.6／13.0或MPS重做全課。")
    claim("uv_extra_repeat","software","使用extra後應直接用環境Python，或uv run重帶相同extra以免換PyTorch來源。","安裝選擇表後段","uv run同步選取依賴，extras非永久選項","官方文件說uv run先同步、extras需指定；專案torch由extra選index，與提示一致。","讀官方同步語義與本專案來源條件；沒有故意重裝共用torch。",[("uv_docs","Automatic lock and sync／Syncing optional dependencies","同步與extra契約"),install],"network")
    claim("l4_torch_environment","empirical","前19章的正式GPU實驗在Modal的NVIDIA L4、PyTorch2.14.1+cu126實測。","安裝／實測環境段","GPU報告有固定設備與版本","本次枚舉的GPU實驗原始報告device=cuda、gpu=NVIDIA L4、torch_version=2.14.1+cu126。CPU實驗獨立列明。","逐份解析results當前JSON的設備／版本／狀態；沒有重新開GPU。",[A("checks.experiments","逐份原始實驗環境枚舉"),ev("docs/course-experiments/results/text_foundation.json","device、gpu、torch_version、status","完整正式實驗範例")],denominators={"gpu_experiment_reports":sum(x.get("device")=="cuda" for x in audit["checks"]["experiments"]),"scope":"只計有device=cuda的原始報告"})
    claim("natural_environment","software","第20章使用Python3.12、PyTorch2.8.0的獨立環境，GPU配套為CUDA12.8。","安裝／實測環境段","獨立固定配套","requirements-natural.txt固定torch2.8.0／torchvision0.23.0；學生指引使用Python3.12 venv並提供cu128 index。","讀固定requirements與實際操作指引，不把前19章lock套入成品環境。",[ev("requirements-natural.txt","固定torch/torchvision配對","獨立套件版本"),ev("docs/natural-assistant/v4/STUDENT.md","建立專用環境","Python3.12與CPU/cu128命令")],"supplement")
    claim("mps_ci_scope","empirical","Apple MPS在macOS CI通過基本矩陣前向／反向與CPU比對，未據此宣稱全課或正式模型MPS訓練已驗證。","安裝／MPS範圍段","基本測試通過且範圍有限","保存的CI紀錄macOS裝置為MPS、compute=OK，330 passed；check_env做256x256前後向，test_accelerator_matches_cpu做32x32比較。","讀固定CI執行收據及當前測試程式；沒有重新連GitHub觸發或用目前CPU成功替代MPS證據。",[ev("docs/validation-artifacts/release-compatibility-ci.json","jobs macos-latest、environment_check","固定執行觀察"),ev("scripts/check_env.py","main","256x256前後向"),ev("tests/test_env.py","test_accelerator_matches_cpu","32x32 CPU比對")],"supplement",denominators={"macos_ci_runs":1,"forward_backward_matrix":"256x256","cpu_comparison_matrix":"32x32"},scope="保存的特定CI執行；未取得遠端原始log重新認證，也未量測整套MPS訓練。")
    claim("site_features","software","Zensical網站有搜尋、章節與頁內目錄、深淺色模式及程式複製。","安裝／網站功能段","建置器與功能配置一致","live build-info.builder=zensical、version0.0.67；zensical配置與頁面包含search、toc、palette及content.code.copy。","讀實際網站標記、固定builder版本與主題配置；未逐一點擊每個瀏覽器互動。",[ev("zensical.toml","project.theme、palette、features","功能明確啟用"),N("site_info／site.features","實際發布內容")],"network")
    claim("svg_interaction","software","SVG可放大並保留靜態標註，頁面支援prefers-reduced-motion。","安裝／網站功能段","圖解放大與減少動態設定","course.js切換is-enlarged；CSS明定min-width與reduced-motion時animation/transition=none；圖中文字是靜態SVG。","讀JS、CSS、SVG並看指定PNG；未據此宣稱完整無障礙驗證。",[ev("course/web/course.js","initializeDiagrams","放大控制"),ev("course/web/course.css","diagram-viewport／prefers-reduced-motion","顯示與減少動態"),A("checks.figure","實際圖文記錄")])
    claim("site_group","software","網站建置需另加--group site。","安裝／網站功能段","site群組提供builder","pyproject site群組固定zensical==0.0.67；發布workflow同步notebook與site群組。","解析群組與workflow設定。",[install,ev(".github/workflows/pages.yml","uv sync --group site","實際發布安裝配置")])
    claim("mathjax_network","software","公式由MathJax連網排版。","安裝與開啟Notebook／網站功能段末句「公式由 MathJax 連網排版。」","描述應符合當前公式排版與資產來源","當前站頁載入./javascripts/mathjax/tex-chtml.js；exporter用stage_assets把固定MathJax3.2.2資產及字型隨站提供。官方支援本機離線使用；排版在瀏覽器執行。","curl核對公開頁script src，並讀prepare_assets/stage_assets、zensical extra_javascript與官方hosting文件；區分建置時取得第三方資產、載入網站資產和實際公式排版。",[ev("scripts/stage_mathjax.py","prepare_assets、stage_assets","建置時下載／核對並隨站提供資產"),ev("scripts/export_course.py","build與stage_assets呼叫","MathJax被複製入站點"),ev("zensical.toml","project.extra_javascript","使用站內相對路徑"),("mathjax_docs","Hosting Your Own Copy of MathJax開頭","MathJax可在本機離線使用"),N("site.scripts","公開站實際使用自帶MathJax")],"network",scope="載入公開網站自然需要取得網站檔案，但MathJax不是連網排版服務，也不必為排版連線至外部CDN；未測試本機完整離線鏡像。",status="contradicted")
    claim("toy_prepare","software","prepare_data --kind toy-text準備小段文字資料。","最小流程／第一條命令","生成玩具文字的train/validation/test","3色×2形狀×2位置=12個家族，seed42分成9／1／2筆；stdout含各切分SHA。","按README只改輸出路徑到自己的artifact目錄；生成內容已嵌入audit-output。",[ev("scripts/prepare_data.py","generate_records、main","規則文字與家族切分"),A("commands/prepare_data與generated_inputs","實際輸入輸出")])
    claim("dry_no_update","software","預設train命令只計算forward/backward，不更新參數、不保存權重。","最小流程／第二條命令","mode=dry-run-no-weight-update且參數不變","CPU命令回傳該mode；另包裝AdamW.step並逐張量比較初末參數，parameters_unchanged=true、checkpoint_exists=false。","不加--train；保持預設seed42、batch4、width32、1層、128最大長度，只把資料／權重路徑放到自己的artifact目錄並明確指定CPU。",[ev("scripts/train.py","main的loss.backward、if args.train: optimizer.step／save_checkpoint","反向與更新／存檔分開"),A("commands/train與runtime_verification","實際no-update行為")])
    claim("loss_grad_numeric","numeric","這個通路檢查應得到有限loss及有限且大於0的grad_norm。","最小流程／報告判讀段","loss有限；grad_norm有限且>0","loss=5.770333290100098；grad_norm=0.9595053791999817；140有效token。","核對history[0]與實際執行輸出；grad_norm為clip_grad_norm_回傳的裁切前範數，不把梯度存在說成已學會任務。",[A("commands/train.stdout／runtime_verification","實際數值"),ev("scripts/train.py","entry／clip_grad_norm_","報告欄位含義")],scope="固定seed、batch和環境下的通路結果；README未要求跨版本精確同值，本次只檢查有限與正值。")
    claim("backward_update_distinction","concept","反向傳播先取得參數梯度，仍須執行更新器才會更新權重；通路成功不代表已學會回答。","最小流程／原理解說段","","","讀PyTorch官方backward契約，並對照本次參數不變實跑。",[("torch_backward","Tensor.backward／leaf gradients","梯度計算與更新分開"),A("runtime_verification.parameters_unchanged","本算例沒有更新權重")],scope="梯度是局部代價對參數的變化率；並非保證一次有限步長會改善所有題目的方法。")
    claim("explicit_train","software","正式更新模型需明確加--train。","最小流程／正式訓練段","更新與存檔受--train控制","argparse action=store_true；optimizer.step與save_checkpoint皆在args.train分支。","實際dry-run驗證預設行為，另讀更新分支；沒有跑長時間訓練。",[ev("scripts/train.py","parser --train、main更新／存檔條件","命令旗標與程式行為"),A("commands/train","未加旗標時不更新")])
    claim("asset_selection","software","--list列包名；--asset NAME才選取、核對與解包所選訓練包。","最小流程／資料包段","明確選包，不自動取全部","--list實際成功；程式拒絕未知名稱，pointer才git lfs pull，unpack_asset負責校驗與解包。","執行list並讀下載分支；沒有下載全部LFS包。",[ev("scripts/fetch_training_assets.py","main","明確選擇及下載路徑"),A("commands/fetch_training_assets --list","本次清單輸出")])
    claim("ignored_storage","software","原始快取與權重存於被Git忽略的data/與checkpoints/。","最小流程／資料包段","目錄有ignore規則",".gitignore含/data/、/checkpoints/、*.pt、*.safetensors等；腳本預設輸出位於相應目錄。","讀.gitignore與prepare_data/train/fetch模型預設路徑；沒有對使用者資料作追蹤狀態改動。",[ev(".gitignore","原始資料快取、權重與實驗輸出規則","目錄及權重檔被忽略"),ev("scripts/train.py","--output預設","checkpoints/text.pt")])
    claim("gpu_smoke","software","手動GPU training smoke test檢查GPU更新、HF checkpoint上傳下載與續訓。","最小流程／GitHub Actions段","需手動入口與帳號配置","gpu-smoke.yml只有workflow_dispatch，檢查Modal及HF配置，再呼叫modal_smoke。程式具更新、備份、下載與resume比對。","讀workflow與modal_smoke流程；沒有觸發或進行任何HF寫入。",[ev(".github/workflows/gpu-smoke.yml","workflow_dispatch／smoke job","手動工作與帳號依賴"),ev("scripts/modal_smoke.py","main／smoke訓練與HF檢查","訓練、上下載及續訓檢查")],scope="只支持該工作設計及入口，不表示本次觸發成功或已設定使用者帳號。")
    claim("v2_public_count","numeric","固定HF版本有四組v2推論輸出，共16個配對檔案。","最小流程／公開v2段","4組、每組4檔、共16檔","固定版本API列四個directory與16個file；各含inference-manifest/model-config/model.safetensors/tokenizer。","匿名curl列完整固定版本目錄並只計file，沒有把directory算檔案。",[N("hf_v2.response","固定版本原始目錄")],"network")
    claim("v2_license","software","v2公開推論輸出採MIT授權。","最小流程／公開v2段","v2授權欄為MIT","固定結果safe_weights.license=MIT、redistribution_approved=true；本專案LICENSE為MIT。","核對固定釋出描述與本專案授權；外部訓練資料仍有各自條款，未把資料許可擴張成相同MIT。",[v2,ev("LICENSE","MIT條款","本專案程式與文件的許可")],"supplement")
    for ident,statement,expected,observed in [
        ("moe_selected","MoE joint最後段完成4000步，選用第1000步。","selected1000/completed4000","原始moe-native train收據steps=4000；最後選版selected_step=1000。"),
        ("dense_selected","Dense joint完成10000步，選用第1000步。","selected1000/completed10000","原始dense-weighted train收據steps=10000；最後選版selected_step=1000。"),
    ]:
        claim(ident,"numeric",statement,"最小流程／公開v2段",expected,observed,"讀最後選定來源與對應原始train-receipt，區分最後訓練段與更早接續階段，也區分選中步數與完成步數。",[S("checks.raw_training／selected_steps","原始步數與選版紀錄")],"supplement")
    for ident,statement,expected,observed,locator in [
        ("package_bytes","固定v2資料包為67,862,431 bytes。","67862431","LFS pointer size=67862431，manifest.package.bytes一致。","checks.archive／audit.checks.v2_package"),
        ("package_files","固定v2資料包宣告含8,962個檔案。","8962","manifest有12筆records與8950筆assets，合計8962。","audit.checks.v2_package.files"),
        ("train_rows","v2訓練題數為28,876。","28876","原始train-receipt與data_scope均為28876。","checks.raw_training"),
        ("validation_rows","v2驗證題數為2,435。","2435","原始train-receipt與data_scope均為2435。","checks.raw_training"),
        ("test_rows","v2測試題數為3,734。","3734","原始test metrics/journal與data_scope均為3734。","checks.raw_final_tests"),
    ]:
        claim(ident,"numeric",statement,"最小流程／重做資料段",expected,observed,"本機實體包是LFS pointer；檔案總數由manifest枚舉、題數由原始訓練／測試收據交叉核對，沒有假稱本次解包逐列重數。",[S(locator,"固定來源與原始紀錄"),ev("docs/selftrained/v2-manifest.json","package、records、assets","包大小與逐檔清單"),v2],"supplement",scope="支持固定manifest／LFS指標及原始執行紀錄中所宣告的數量；本次未下載68MB實體包重解或重驗所有媒體。")
    claim("local_checkpoint","software","從隨機初始化重做時逐段接自己的完整checkpoint，不需要作者私人Modal Volume。","最小流程／重做資料段","本機入口可載入自己保存的checkpoint","TRAINING.md提供train_local_stage；本機程式從--output與--source讀自己檔案，不要求作者Volume。","讀本機指引及本機stage入口的參數、checkpoint路徑；未重做全量訓練。",[ev("docs/selftrained/TRAINING.md","本機逐階段指令與自己的checkpoint","讀者獨立重做方式"),ev("scripts/selftrained/train_local_stage.py","parser／stage source與output","本機檔案入口")],"supplement")
    claim("inference_resume","software","公開推論權重不含optimizer／RNG等狀態，不能精確接續原訓練。","最小流程／v2與公開學生權重段","推論輸出剔除訓練狀態","selftrained export_inference只保存model tensors、config與tokenizer；舊權重inference_payload用allowlist，實際輸出沒有optimizer/rng/training_state。","讀兩個匯出器並執行舊格式allowlist；缺少狀態不能保證還原同一次下一步更新。",[ev("scripts/selftrained/train.py","export_inference","safetensors只保存神經參數"),ev("scripts/course_release.py","inference_payload","allowlist排除更新器與RNG"),S("checks.public_export_keys","實際測試訓練欄位被排除")],"supplement")
    for ident,statement,expected,observed,locator in [
        ("public_experiments","公開學生權重涵蓋30組實驗。","30","30","experiments"),
        ("public_weights","30組實驗共120份權重檔。","120","120","weight_files"),
        ("capstone_weights","合成小世界整合清單另有11份階段、量化及Dense學生檔。","11","11","capstone_weights"),
    ]:
        claim(ident,"numeric",statement,"最小流程／各章局部比較段",expected,observed,"直接解析public-models與capstone-public，按.pt權重檔輸出項計數，metadata與dataset檔不計入。",[A("checks.public_counts."+locator,"逐檔獨立計數"),ev("docs/course-experiments/public-models.json","models/files","固定版本與檔案指紋"),ev("docs/course-experiments/capstone-public.json","models/files","另組固定整合輸出")])
    claim("public_cpu_command","software","check_course_models --model text_foundation會匿名下載、驗檔並在CPU推論。","最小流程／公開學生權重命令區塊","token=False、SHA/bytes核對與CPU推論","在自己的暫存artifact子目錄與隔離HF快取實際執行，anonymous_download=true、status=passed；六個probe加一個infer全部成功。","僅以--output與--evidence改變輸出位置；下載固定revision，不送出憑證、不寫HF、不持久保存權重副本。",[ev("scripts/fetch_course_models.py","fetch_model","token=False與固定版本指紋驗證"),ev("scripts/check_course_models.py","check_model、probe_checkpoint、inference_command","CPU子命令與完整操作證據"),S("checks.anonymous_cpu_probe","本次匿名下載與推論")],"supplement",scope="文件與執行通路成功；不表示生成答案正確或通過全部測試。")
    claim("figure_ids","numeric","圖中貓看狗，狗看貓。依序對應3、2、1、4、1、2、3、0，ID可逆查字且不表示字義大小。","文字與識別編號的對照／整張SVG","3,2,1,4,1,2,3,0；同字同ID","runtime_verification.figure_derivation排序唯一字為。狗看貓，；enumerate得到圖上相同映射。","實際逐字建立字典並逆向查驗；PNG先上排再下排與SVG座標一致，貓兩處皆3、狗兩處皆1。",[A("runtime_verification.figure_derivation／checks.figure","透明字典算例與目視檢查"),ev("course/figures/rewrite-01-character-ids.svg","各字與ID、雙向箭頭與底部說明","圖中標籤和示意結論")],scope="這份手工詞表示意；未宣稱所有tokenizer使用此編號，也未用ID數字推論語義關係。")
    claim("project_structure","software","正文、圖、Notebook、模型模組、腳本、測試與文件位於專案結構所列目錄。","專案結構與維護／text區塊","列示目錄與用途相符","章檔產生Notebook；tiny_perceptron有模型與模態零件；scripts、tests、docs與course/figures存在且對應用途。","核對實體檔案及build_course的輸入輸出。",[ev("scripts/build_course.py","build sources與Notebook輸出","正文單一來源"),A("checks.catalogue","實體章檔與Notebook")])
    claim("maintenance_commands","software","維護命令涵蓋圖解、範例格式化、Notebook生成同步、審閱、Notebook執行與pytest/ruff檢查。","專案結構與維護／bash區塊","所有列出腳本與旗標可定位","build_course --check實跑成功；其他入口的parser與預設功能符合命令。check_notebooks區分python與kernel模式。pytest與ruff屬dev群組預設依賴。","讀build_visuals、format_course_code、build_course、兩個review checker及check_notebooks；避免執行會改寫凍結正文／圖的命令。",[ev("scripts/build_visuals.py","build","圖產生入口"),ev("scripts/format_course_code.py","main sources","格式化指定Markdown的Python範例"),ev("scripts/check_notebooks.py","parser、run_python、run_kernel","mode與lesson選項"),install,A("commands/build_course --check","唯讀同步檢查")],scope="不表示本次已跑全套pytest/ruff、全書kernel或實際重建網站。")
    claim("independent_review_gate","software","讀者與技術審閱分開，來源／圖版本不符會阻止發布。","專案結構與維護／審閱流程段","當前版本與不同審閱身分受檢查","兩種checker核對SHA及reviewer身分，pages workflow先執行這些檢查才build/deploy。","讀現行checker函式及workflow，未讀其引用的審閱報告或歷史裁決。",[ev("scripts/check_technical_reviews.py","_validate／_sources／_hash_file","來源hash、身分與證據檢查"),ev("scripts/check_course_reviews.py","main","讀者版本檢查"),ev(".github/workflows/pages.yml","build前置steps與deploy needs","失敗阻斷後續發布")])
    claim("kernel_isolation","software","完整--mode kernel檢查為每份Notebook建立獨立kernel，輸出預設在outputs。","專案結構與維護／kernel段","每份建立NotebookClient並執行","run_kernel每次讀新Notebook、建立新NotebookClient；輸出寫--output或預設outputs/notebooks；exporter預設outputs/site。","讀函式作用域與main排程。這次python-mode檢查因未安裝matplotlib失敗；沒有把快速namespace模式說成完整kernel驗證。",[ev("scripts/check_notebooks.py","run_kernel、main --output","獨立client與輸出位置"),ev("scripts/export_course.py","parser --output","網站輸出預設位置")])
    claim("ci_platforms","software","GitHub Actions保留Linux、macOS、Windows核心測試，Notebook全套由發布流程另跑。","專案結構與維護／CI段","三平台pytest與獨立Notebook工作","ci.yml的test matrix有ubuntu/windows/macos；pages.yml單獨安裝Notebook群組並執行所有kernel。","直接讀目前workflow，不觸發CI。",[ev(".github/workflows/ci.yml","test.strategy.matrix與pytest step","三平台核心測試"),ev(".github/workflows/pages.yml","Execute every lesson","全套Notebook另跑")])
    claim("mit_license","software","專案程式、教材與自製圖解使用MIT；外部資料依各自授權。","參考與授權","根LICENSE是MIT，外部資料條款分開","LICENSE含MIT許可與Copyright；training manifest逐資產列license，未一律標為MIT。","讀根LICENSE與實際--list輸出中的各包license；參考專案僅作引用，README未宣稱其內容改採本專案授權。",[ev("LICENSE","Permission is hereby granted及條款","程式與associated documentation"),A("commands/fetch_training_assets --list","不同外部資料授權")])
    for path in ("scripts/selftrained/train_local_stage.py","scripts/modal_smoke.py"):
        assert (ROOT / path).exists(), path
    issue = {"claim_id":"mathjax_network","status":"contradicted","details":"README 的『公式由 MathJax 連網排版』將網路資產載入與公式排版混為一談。當前程式已把固定 MathJax 3.2.2 主程式、字型等隨網站提供，公開頁實際使用 ./javascripts/mathjax/tex-chtml.js；MathJax 官方 hosting 文件明確支援本機離線使用。讀者不應由此得到公式排版必須連線外站或由遠端服務處理的事實。","necessary":True,"suggestion":"改成『公式由網站隨附的 MathJax 在瀏覽器排版』；若要說首次載入需取得資產，另明確限定為網站檔案載入，避免寫成連網排版。"}
    report = {
        "schema_version":1,"review_stage":"technical","lesson_id":"readme","source":"README.md#readme",
        "reviewer_task":"/root/rv1019_technical_readme","reviewer_context":"fresh","author_tasks":["/root"],
        "source_sha256":"d35d7b95f1d8c102e8fdecbd51aceb0dcf7b43c1caacb1b364d56200612b4568",
        "source_freeze":{"path":"docs/course-review-20261019/freeze-04.json","sha256":"a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"},
        "figure_sha256":{"course/figures/rewrite-01-character-ids.svg":"61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad"},
        "verdict":"revise","claims":claims,"sources":sources,"artifacts":artifacts,"issues":[issue],
        "checks":{
            "factual_accuracy":{"status":"revise","details":"逐項核對整份凍結README；數量、參數與固定結果一致，MathJax『連網排版』與現行自帶資產及本機排版方式不符。","claim_ids":["mathjax_network"]},
            "numeric_verification":{"status":"pass","details":"實數312份Notebook與26個額外小節；重建兩模型參數量；核對LFS pointer、manifest、原始train/test收據與公開權重清單。loss有限、梯度為正；固定成績限於所列分母。資料包未在本機實體化，數量以固定manifest與原始收據交叉核對。","claim_ids":[c["id"] for c in claims if c["kind"] in ("numeric","empirical")]},
            "figure_consistency":{"status":"pass","details":"已用view_image看 outputs/course-review-20261019/figure-renders-04/61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad.png，另讀SVG。先上排後下排、貓↔3／狗↔1等五個對照、上下位置與雙向箭頭都一致，句子ID實算3,2,1,4,1,2,3,0；圖未用尺度或ID大小暗示語義量，文字沒有裁切。PNG僅在details記路徑，未列artifact。","claim_ids":["figure_ids"]},
            "source_verification":{"status":"revise","details":"以curl讀官方uv、Git LFS、Git for Windows、PyTorch2.14、ipykernel7.4.0、JupyterLab、Colab與MathJax3.2原文，保存短摘錄／原文SHA及定位；研究方法沒有在此頁宣稱效益。原始程式、訓練及測試輸出直接核對。MathJax原始文件支持離線／本機使用，不能支持原句連網排版。","claim_ids":["mathjax_network","backward_update_distinction","uv_extra_repeat","kernel_install"]},
            "limitations":{"status":"pass","details":"未將短CPU執行、匿名下載或固定成功示範當成模型品質。B/C、v2與舊合成模型分開，固定分母與有限語法明記；未做開放中文外推、未重訓、未觸發workflow或HF寫入。全新環境安裝、Colab、Windows/CUDA/MPS整課、全量Notebook及原始資料包實體逐檔重驗不在本次已實跑範圍。","claim_ids":["cpu_small","v2_tests","c_fixed","mps_ci_scope","public_cpu_command"]},
        },
        "review_scope":{"original_read":"整份freeze-02/sources/readme.md；實際SHA與README.md及派工一致。","independence":"原始判斷存檔前未讀被排除的歷史／本輪他人審閱報告、處置、seal或閱讀計畫；沒有使用候選摘要當作證據。git status只看到一個既有dispatch-log檔名，未讀內容。","execution":"CPU短通路、參數重算、清單／同步檢查、匿名固定文字模型下載與七個CPU子程序，以及curl唯讀原始來源。","not_verified":"未重跑正式訓練及全量模型測試；本機缺Notebook依賴，未完整安裝或執行全課kernel；未重取遠端CI原始log；未下載／解包LFS實體包或核對全部媒體；未完成Colab／Windows／CUDA／MPS整課或互動瀏覽器測試。"},
        "process_notes":["Notebook單節檢查未指定--output，現行程式寫入outputs/notebooks/validation-python.json；這是本審閱者的越界寫入，沒有查看其他outputs內容，也沒有擅自還原可能原先存在的檔案。","supplementary_checks的第一次替身Auto配方漏version，拋ValueError『需要凍結的Auto規則版本』；補齊替身契約後執行成功。此為審閱程式設定，不是教材回歸。","圖的PNG已按指定路徑觀看；除了該PNG沒有讀取outputs。未修改正文、程式、workflow或既有審閱紀錄；未commit、push、stash或啟動付費運算。"],
        "escalation_signals":{
            "A":{"answer":"無","check_method":"已讀technical-review-guide、指定checker與凍結原文／圖；以SHA與自我檢查核對派工中的版本與工具行為，未發現派工對既有系統的宣稱被反證。"},
            "B":{"answer":"無","check_method":"本派工未附已駁回／不要重提清單，故無可核對的編號區間；沒有自行讀歷史裁決。"},
            "C":{"answer":"有","details":"一次check_notebooks --mode python --lesson 1.1未指定--output，預設寫outputs/notebooks/validation-python.json，超出指定可寫目錄。未讀其他審閱內容；未重訓／重跑全量測試與跨平台安裝，其餘範圍限制列於review_scope。"},
            "D":{"answer":"無","check_method":"只新增本審閱者報告與artifact；原先此報告路徑不存在。沒有修改實作位置，也未在所讀README／SVG／程式引文看到需要援引的先前裁決編號或被改寫註解。"},
        },
    }
    spec=importlib.util.spec_from_file_location("validator",ROOT/"scripts/check_technical_reviews.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=(ROOT/"docs/course-review-20261019/freeze-02/sources/readme.md").read_text()
    errors=module._validate(ROOT,ROOT/"README.md","readme",raw,report,{report["reviewer_task"]:["readme"]},{"/root/rv1019_reader_entry"})
    expected=["審閱要求修訂／尚未核實","claim mathjax_network: 未核實／矛盾主張不能通過，應維持revise","尚有未解決問題，不能記為pass","factual_accuracy: 未通過或不適用的NA","source_verification: 未通過或不適用的NA"]
    if errors!=expected:
        raise RuntimeError(json.dumps(errors,ensure_ascii=False))
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    with TARGET.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"report":TARGET.relative_to(ROOT).as_posix(),"sha256":sha(TARGET),"verdict":report["verdict"],"kinds":dict(Counter(c["kind"] for c in claims)),"unresolved":[c["id"] for c in claims if c["status"]=="unresolved"],"contradicted":[c["id"] for c in claims if c["status"]=="contradicted"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
