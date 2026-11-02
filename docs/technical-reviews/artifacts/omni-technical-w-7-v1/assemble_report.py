"""Save this reviewer's first W.7 report and compact source receipts once."""

import datetime
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_7/technical-original-v1.json"
CACHE = Path("/workspace/work/reference/omni-technical-w-7-v1")
BODY = (ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4/sections/W.7.md").read_text()
ROUTES = json.loads((OUT / "route-inspection.json").read_text())
CPU = json.loads((OUT / "cpu-verification.json").read_text())
ENV = CPU["environment"]
COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-7-v1/audit_w7.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def location(quote):
    assert quote in BODY, quote
    line = BODY[:BODY.index(quote)].count("\n") + 1
    return f"W.7 凍結切片 L{line}；course/first-steps.md L{line + 266}"


def reference(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def main():
    assert not REPORT.exists()
    sources = []
    code_ids = {}
    primary = dict(ROUTES["primary_code"])
    for path in [".gitignore", "scripts/check_env.py", "scripts/prepare_data.py", "scripts/evaluate.py", "scripts/export_course.py", "pyproject.toml", "scripts/train_simple.py", "scripts/pretrain_encoders.py"]:
        primary[path] = {"sha256": sha(ROOT / path), "symbols": []}
    for index, (path, record) in enumerate(primary.items(), 1):
        identifier = f"code{index}"
        code_ids[path] = identifier
        assert sha(ROOT / path) == record["sha256"]
        symbols = ", ".join(f"{s['name']} L{s['line']}–{s['end_line']}" for s in record["symbols"])
        notes = {
            ".gitignore": "實讀 .venv/ 與 /data/、/checkpoints/、/outputs/ 排除規則，並實跑 git check-ignore --no-index。",
            "scripts/check_env.py": "實讀模組用法與 main：套件檢查及前向／反向計算入口；只證明 scripts 目錄的用途。",
            "scripts/prepare_data.py": "實讀檔頭與 generate_records：自建規則資料的製作入口；本次未製作新資料。",
            "scripts/evaluate.py": "實讀檔頭、evaluate 介面與 main L164–175：評估入口與 outputs/evaluation.json 預設輸出。",
            "scripts/export_course.py": "實讀 main L418–430：網站輸出目的地預設 ROOT / outputs/site；未重建網站。",
            "pyproject.toml": "實讀專案名、套件目錄與 dependency-groups；確認根目錄標記檔與 tiny_perceptron 套件。",
            "scripts/train_simple.py": "實讀全檔：--train、optimizer.step、存檔與 mode 皆由同一 args.train 控制。未執行訓練。",
            "scripts/pretrain_encoders.py": "實讀全檔：--train、optimizer.step、存檔與 mode 皆由同一 args.train 控制。未執行訓練。",
        }
        sources.append({"id": identifier, "kind": "repository_code", "title": path,
                        "path": path, "sha256": record["sha256"],
                        "version": "2026-10-29 實讀工作樹；HEAD " + ENV["head"] + "；以本筆 SHA 綁定實際 bytes",
                        "verified": True,
                        "inspection_note": notes.get(path, "實讀目前主要程式的相關定義，並對照目標小節的節錄／說明：" + symbols + "。位置與節錄相符；本報告不代審該節的研究或品質結論。")})

    original_receipts = []
    for name in ["download-receipts.json", "download-receipts-additional.json"]:
        original_receipts += json.loads((CACHE / name).read_text())
    original_receipts.append(json.loads((CACHE / "python-ospath.receipt.json").read_text()))
    receipt_by_name = {Path(r["path"]).name: r for r in original_receipts if "sha256" in r}
    originals = [
        ("py_pathlib", "python-pathlib.rst", [(754, 770), (913, 920), (1021, 1040)], "pathlib.Path、Path.cwd、Path.exists：建立路徑物件、讀目前工作目錄、檢查檔案或資料夾存在；不以這些方法進行寫檔。"),
        ("py_ospath", "python-ospath.rst", [(57, 64)], "os.path.abspath：一般平台以 normpath(join(os.getcwd(), path)) 解讀相對位置；配合本次兩個 cwd 的直接執行。"),
        ("py_os", "python-os.rst", [(2044, 2056), (2198, 2207)], "os.chdir 會更換目前工作目錄；getcwd 回傳目前工作目錄。"),
        ("py_exceptions", "python-exceptions.rst", [(255, 279), (504, 512), (551, 560), (766, 770)], "逐項讀 ModuleNotFoundError、IndexError、KeyError、SyntaxError、IndentationError、FileNotFoundError 定義；縮排錯誤屬 SyntaxError。"),
        ("py_errors", "python-errors.rst", [(15, 32), (63, 79)], "語法錯誤列檔名、行號及偵測位置；一般未處理例外的 traceback 最後一行列類型與原因。原文也提醒偵測位置不一定就是修正位置。"),
        ("py_venv", "python-venv.rst", [(34, 41), (69, 81), (188, 237)], "虛擬環境含 Python 與套件；POSIX bin、Windows Scripts；activate 把環境目錄放到 PATH 前端，也可直接指定 Python 路徑而不啟用。"),
        ("py_interpreter", "python-interpreter.rst", [(87, 104)], "互動模式主提示通常為 >>>；多行續行則為 ...。"),
        ("py_constants", "python-constants.rst", [(82, 98)], "site 預設自動載入，提供互動 shell 使用的 exit()/quit()，呼叫會引發 SystemExit；-S 是明載例外，本節使用一般啟動。"),
        ("py_stdtypes", "python-stdtypes.rst", [(975, 983), (1111, 1113)], "序列 s[i] 從 0 起算；超出序列範圍的索引引發 IndexError；不同於可裁切的 slice。"),
        ("uv_layout", "uv-structure-current.md", [(3, 7), (29, 45)], "實裝 uv 0.12.19 同版官方原始文件：專案由 pyproject.toml 辨識，預設 .venv 在其旁；uv sync 可明確建立 project environment，不建議把 .venv 納入版本控制。"),
    ]
    excerpts = []
    for identifier, name, spans, note in originals:
        receipt = receipt_by_name[name]
        assert sha(CACHE / name) == receipt["sha256"]
        lines = (CACHE / name).read_text().splitlines()
        selections = [{"start_line": a, "end_line": b, "text": "\n".join(lines[a - 1:b])} for a, b in spans]
        excerpts.append({"source_id": identifier, "url": receipt["url"], "full_original_sha256": receipt["sha256"],
                         "original_cache_path": receipt["path"], "accessed_on": receipt["accessed_on"],
                         "excerpts": selections, "inspection_note": note})
        sources.append({"id": identifier, "kind": "official_docs", "title": name,
                        "url": receipt["url"], "version": "uv 0.12.19 tag" if identifier == "uv_layout" else "CPython v3.13.5 tag；與本次 Python 3.13.5 對應",
                        "verified": True, "checked_original": True, "accessed_on": receipt["accessed_on"],
                        "authority_reason": "工具維護者官方 repository 的版本化文件原文，直接定義本節使用的介面與行為。",
                        "inspection_note": note + " 實讀行號：" + ", ".join(f"L{a}–{b}" for a, b in spans) + "；原文 SHA=" + receipt["sha256"] + "。全文留 Git 外，必要小摘錄見 a_originals。"})
    save(OUT / "original-source-excerpts.json", excerpts)

    # Exact currently needed prerequisite slices; whole chapter files are not
    # pinned merely to support one routing sentence.
    prerequisite_specs = [
        ("course/first-steps.md", [(23, 28), (31, 36), (39, 47), (59, 61)]),
        ("course/chapters/05.md", [(257, 259), (294, 307)]),
        ("course/training.md", [(41, 61)]),
    ]
    prerequisite = []
    for path, spans in prerequisite_specs:
        lines = (ROOT / path).read_text().splitlines()
        for first, last in spans:
            text = "\n".join(lines[first - 1:last]) + "\n"
            prerequisite.append({"path": path, "start_line": first, "end_line": last,
                                 "text": text, "slice_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "observed_at": datetime.datetime.now(datetime.UTC).isoformat()})
    save(OUT / "necessary-current-slices.json", prerequisite)

    artifacts = []
    for identifier, name, kind, description in [
        ("a_code", "audit_w7.py", "code", "本人實跑的小型 CPU 查核程式：由 W.7 凍結正文擷取原樣 pathlib 程式、讀目前 route destinations 與 primary code，及真 default dry-run observer。"),
        ("a_routes", "route-inspection.json", "execution", "16 列、65 目標小節的當前精確摘錄、section SHA、primary code SHA／符號位置、58 個原樣節錄 AST 相符結果。"),
        ("a_cpu", "cpu-verification.json", "execution", "原樣 pathlib 根目錄／notebooks、互動 shell、環境啟用、例外與無更新 dry-run 的完整短輸出。"),
        ("a_originals", "original-source-excerpts.json", "source_snapshot", "本人實際讀取的官方版本化原文小摘錄與全文 Git 外 SHA 收據；不複製第三方全文。"),
        ("a_prereq", "necessary-current-slices.json", "source_snapshot", "W.1、W.2、5.7 及 T.2 當前真正必要小範圍摘錄；不引用歷史課文當現行前提。"),
        ("a_assembly", "assemble_report.py", "code", "首判組裝程式；以 exclusive create 保存原件，逐 source_quote 核對真正文；不修改課文或 checker。"),
    ]:
        item = {"id": identifier, "kind": kind, "path": relative(OUT / name), "sha256": sha(OUT / name), "description": description}
        if kind == "execution":
            item.update(command=COMMAND, environment=ENV,
                        result="65 個目標存在且有指名實作／說明，58 個原樣節錄 AST 全同。" if identifier == "a_routes" else "根目錄 True/True、notebooks False/False；7 個例外情境符合；預設文字 dry-run 17 參數張量逐位未變、step/save 呼叫皆 0。")
        artifacts.append(item)
    sources += [{"id": "run_routes", "kind": "execution", "title": "本人執行的當前路線與程式版本查核", "verified": True, "artifact_id": "a_routes"},
                {"id": "run_cpu", "kind": "execution", "title": "本人執行的 W.7 原樣程式與無更新觀察", "verified": True, "artifact_id": "a_cpu"},
                {"id": "index_derivation", "kind": "derivation", "title": "兩項清單的合法索引手算", "verified": True,
                 "details": "手算而非程式執行：animals 有兩項，長度 n=2；非負整數索引須滿足 0≤i<n，所以合法值只有 0、1。i=2 是第 i+1=3 項，2<2 不成立，故超出範圍。另於 a_cpu 實際執行取 0、1 得貓狗，取 2 引發 IndexError；兩者分別記錄。"}]

    claims = []

    def add(identifier, kind, statement, quote, evidence, artifacts_used, scope, observed=None, expected=None, details=None):
        claim = {"id": identifier, "kind": kind, "statement": statement, "source_quote": quote,
                 "location": location(quote), "status": "verified", "evidence": evidence,
                 "artifact_ids": artifacts_used, "scope": scope}
        if kind in ("software", "numeric"):
            claim["verification"] = {"method": "hand_calculation" if kind == "numeric" else "executed",
                                     "expected": expected or statement, "observed": observed,
                                     "details": details or "本人實跑 audit_w7.py；依 artifact 的精確欄位與當前原始程式核對，不把 exit code 當作全部內容正確的證明。"}
            if kind == "numeric":
                claim["verification"]["tolerance"] = "整數索引與位置精確相等，無浮點容忍差。"
        claims.append(claim)

    # One claim per directory role.
    directory_evidence = {
        "course/chapters/": ("course/chapters/01.md 存在，當前 65 個連結目的小節都由該目錄章檔擷取。", "run_routes", "rows[*].targets[*].path / heading"),
        "notebooks/": ("notebooks/01/1.1.ipynb 存在且 notebooks 是資料夾。", "run_cpu", "example_files / directory_inventory.notebooks"),
        "tiny_perceptron/": ("目前 22 份主要程式具有所指函式或類別；16 列共 65 個實作／說明目的地成立。", "run_routes", "primary_code / rows"),
        "scripts/": ("check_env.py、prepare_data.py、train.py、evaluate.py 均存在；實讀入口目的並實跑 train.py 預設 dry-run。", "run_cpu", "example_files / dry_run"),
        "data/": ("git check-ignore --no-index 對 data/__w7_probe__ 確認被排除。", code_ids[".gitignore"], "/data/ 規則"),
        "checkpoints/": ("train.py 預設 --output checkpoints/text.pt；training.py 保存 checkpoint；git check-ignore 確認 checkpoints 被排除。", code_ids["scripts/train.py"], "parser --output L70 與 main 的 args.train 保存分支"),
        "outputs/": ("evaluate.py 預設 outputs/evaluation.json；export_course.py 預設 outputs/site；check-ignore 確認 outputs 被排除。", code_ids["scripts/export_course.py"], "main L418–430"),
        ".venv/": ("當前 .venv 可直接執行；官方 uv 0.12.19 同版文件寫明 sync 建立預設專案環境；check-ignore 確認被排除。", "uv_layout", "Project environment L29–45"),
    }
    for index, (path, (observed, source, locator)) in enumerate(directory_evidence.items(), 1):
        quote = next(line for line in BODY.splitlines() if line.startswith(f"| `{path}`"))
        evidence = [reference(source, locator, observed)]
        if path == "scripts/":
            evidence += [reference(code_ids[p], "檔頭及 main", "該檔是本列列出的實際入口。") for p in ("scripts/check_env.py", "scripts/prepare_data.py", "scripts/evaluate.py")]
        if path == "outputs/":
            evidence.append(reference(code_ids["scripts/evaluate.py"], "main L174", "JSON 報告預設位置。"))
        add(f"dir{index}", "software", f"專案路徑 {path} 的用途與正文表格相符。", quote, evidence,
            ["a_routes", "a_cpu", "a_prereq"], "路徑是本專案慣例；資料、成果與環境目錄可在操作後才建立。忽略規則表示通常不納入 Git，並非聲稱 Git 絕不容許例外。", observed)

    # A separate claim for every file-to-section edge, not one generic table pass.
    prose_primary = {
        ("tiny_perceptron/simple.py", "2.5"): ["tiny_perceptron/simple.py"],
        ("tiny_perceptron/capstone.py", "6.6"): ["tiny_perceptron/capstone.py"],
        ("tiny_perceptron/tokenization.py", "6.6"): ["tiny_perceptron/tokenization.py"],
        ("tiny_perceptron/natural_concepts.py", "11.14"): ["tiny_perceptron/natural_concepts.py"],
        ("tiny_perceptron/natural_concepts.py", "11.15"): ["tiny_perceptron/natural_concepts.py"],
        ("tiny_perceptron/natural_concepts.py", "12.14"): ["tiny_perceptron/natural_concepts.py"],
        ("tiny_perceptron/selftrained/", "12.15"): ["tiny_perceptron/selftrained/model.py"],
        ("tiny_perceptron/selftrained/", "19.9"): ["tiny_perceptron/selftrained/model.py"],
        ("tiny_perceptron/lesson_products/", "14.15"): ["tiny_perceptron/lesson_products/data.py"],
    }
    count = 0
    for row_number, row in enumerate(ROUTES["rows"]):
        for target_number, target in enumerate(row["targets"]):
            count += 1
            target_id = target["lesson_id"]
            files = sorted({s["path"] for s in target["primary_symbols"]} | set(prose_primary.get((row["path"], target_id), [])))
            if not files and row["path"] in code_ids:
                files = [row["path"]]
            observed = "；".join(f"{e['start_line']}: {e['text']}" for e in target["excerpts"])
            evidence = [reference("run_routes", f"rows/{row_number}/targets/{target_number} ({target['path']}#{target_id}, section SHA {target['section_sha256']})", "實際開啟目前目的小節，保留指名實作／零件說明的精確行；不是僅檢查連結存在。")]
            evidence += [reference(code_ids[path], "; ".join(f"{s['symbol']} L{s['line']}–{s['end_line']} ({s['excerpt_kind']})" for s in target["primary_symbols"] if s["path"] == path) or "route-inspection.primary_code 對應相關定義及其行號", "目前主要程式具備目標小節指名的零件；原樣節錄已逐 AST 比較，精簡版依其省略聲明核對。") for path in files]
            add(f"route{count:02d}", "software", f"{row['path']} 的實作節錄或具名零件說明可在 {target_id} 找到。", row["source_quote"], evidence,
                ["a_routes", "a_code"], "只核本表這一條導覽關係及當前程式版本。目標節的數學、實測數字與圖解不因此取得本節的技術通過；只匯入而無解釋不足作證。", observed)
    assert count == 65
    quote = "下表依各檔第一次出現的閱讀順序排列，列出節錄該檔程式、或點名該檔並說明其中零件的小節；只匯入來用的小節不列。"
    add("route_policy", "software", "本表所列 16 個檔案／目錄按其第一個列出目的地的閱讀順序排列，65 個目的地均有實作或具名解釋。", quote,
        [reference("run_routes", "rows 依序為 1.2、2.5、3.6、4.4、4.5、5.6、6.6、6.6、8.8、10.1、11.14、12.15、13.10、14.15、17.5、A.3", "順序與表中規則一致；named prose 的 capstone.state、ByteLevelBPE、natural_concepts helpers、write_dataset 也實讀目前內容。")],
        ["a_routes"], "這是本表所列內容的閱讀索引；未把每個套件檔案或只匯入的所有小節都當成實作解說。", "16 列、65 個目標；58 個原樣定義 AST 相同，精簡來源標記與具名解釋另核。")

    quote = "相對路徑是從「目前所在資料夾」出發找東西，而不是從電腦固定的位置找。"
    add("relative_path", "concept", "相對路徑的查找起點取決於目前工作目錄。", quote,
        [reference("py_ospath", "abspath L57–64", "一般平台以 getcwd 與相對路徑組合成絕對位置。"), reference("py_os", "chdir L2044–2056", "切換工作目錄會改變此起點。")],
        ["a_cpu"], "指本節普通檔案系統路徑及終端機 cwd；不是說 import、套件資源或 URL 都依相同查找規則。")
    quote = "例如 `scripts/check_env.py` 的意思是先進 `scripts` 再找那個檔案；若目前在 `notebooks/`，就會走錯地方。"
    add("script_relative", "software", "scripts/check_env.py 相對專案根目錄存在；從 notebooks 當起點會指向不同位置。", quote,
        [reference("run_cpu", "example_files / project_root / notebooks_directory", "根目錄的 scripts/check_env.py 存在，cwd 變為 notebooks 時相對起點改變。"), reference("py_ospath", "abspath L57–64", "路徑由當前目錄起算。")],
        ["a_cpu"], "按本 repo 的 notebooks 目錄布局；不是聲稱所有 notebooks 目錄都沒有同名 scripts。", "根目錄查核檔案存在；notebooks cwd 位於根目錄下一層。")
    quote = "在終端機先 `cd tiny-perceptron-vlm`，是把起點移到有 `pyproject.toml` 的專案根目錄。"
    add("project_root", "software", "依 W.1 的預設 clone 資料夾名稱進入 repo 後，pyproject.toml 在該根目錄。", quote,
        [reference("uv_layout", "pyproject.toml L3–7", "uv 用該檔辨識根目錄。"), reference("run_cpu", "project_root / activated_shell", "在本 repo 根目錄用原樣程式得到 True。")],
        ["a_cpu", "a_prereq"], "cd 指令承接 W.1 的 clone 起點與預設資料夾名；未聲稱在任何任意 cwd 都能直接 cd 成功。", "本次根目錄 /workspace/tiny-perceptron-vlm；pyproject.toml=True。")
    quote = "Notebook 的工作位置也可能不同，故短實驗盡量直接寫小資料，不先要求讀檔。"
    add("notebook_cwd", "software", "工作頁可從 notebooks 內開啟，故不應假定每個 Python 執行程序的 cwd 都是 repo 根目錄。", quote,
        [reference("run_cpu", "project_root / notebooks_directory", "同一段程式在兩個真 cwd 得到不同查找結果。"), reference("py_pathlib", "Path.cwd L913–920", "讀取的是程序目前目錄。")],
        ["a_cpu", "a_prereq"], "這裡使用兩個 Python subprocess 證明 cwd 差異的機制，沒有冒稱實際執行 Jupyter、Colab 或所有 kernel 設定。", "同一環境可由根目錄與 notebooks 兩個 cwd 執行。")
    quote = "`Path` 是表示檔案路徑的工具，`cwd()` 回傳目前資料夾，`exists()` 回答某個路徑是否存在。這段不會建立、修改或刪除任何檔案。"
    add("pathlib_api", "software", "W.7 原樣程式只建立 Path 物件、讀 cwd 及查詢是否存在，沒有檔案寫入操作。", quote,
        [reference("py_pathlib", "Path、cwd、exists", "三個介面的定義與讀取用途。"), reference("run_cpu", "inline_code / inline_ast_calls / project_root / notebooks_directory", "執行的是本節原樣 fenced code；AST 呼叫只包含 print、Path、cwd、exists。")],
        ["a_cpu", "a_code"], "不建立 Path 指向的資料夾或檔案；說明的是該短程式的顯式操作，不宣稱 OS 或 Python 啟動永無其他環境副作用。", "原樣程式成功完成；AST 沒有 mkdir、write、unlink、rename 等變更操作。")
    quote = "第一行輸出依你電腦而不同。在專案根目錄執行時，後兩行應都是 `True`；若都是 `False`，先看第一行，確認是否站在別的資料夾。"
    add("pathlib_output", "software", "原樣程式在本專案根目錄輸出 cwd 與兩個 True；不同起點會改變兩個 exists 結果。", quote,
        [reference("run_cpu", "project_root.stdout / notebooks_directory.stdout", "同一份原樣程式得到預期布林值。"), reference("py_pathlib", "Path.cwd / Path.exists", "輸出分別是目錄與存在性。")],
        ["a_cpu"], "cwd 的字串依電腦而變；True 表示檔案或目錄存在，不保證環境安裝完整。兩個 False 是檢查起點的線索，不是完整診斷。", "根目錄：/workspace/tiny-perceptron-vlm, True, True；notebooks：/workspace/tiny-perceptron-vlm/notebooks, False, False。")
    quote = "macOS 與 Linux 是 `.venv/bin/python`，Windows PowerShell 是 `.\\.venv\\Scripts\\python.exe`"
    add("venv_paths", "software", "POSIX 與 Windows 的虛擬環境 Python 路徑分別位於 bin 與 Scripts。", quote,
        [reference("py_venv", "建立環境 L69–81；How venvs work L188–237", "官方平台目錄及直接指定環境 Python 的約定。"), reference("run_cpu", "project_root.command", "Linux 直接指定 .venv/bin/python 的執行成功。")],
        ["a_cpu", "a_prereq", "a_originals"], "Linux 本人實跑；Windows PowerShell 與 macOS 的寫法由同版官方契約核對，未宣稱在本容器實跑這兩個平台。", "Linux 的 .venv/bin/python 可用；Windows 的 Scripts/python.exe 由官方目錄定義支持。")
    quote = "已照 [W.1](#W.1) 啟用工具箱時，只輸入 `python` 也可以。"
    add("activation", "software", "啟用環境把其 Python 放到 PATH 前端，故可用 python 啟動同一環境。", quote,
        [reference("py_venv", "L200–205", "activate 的 PATH 效果。"), reference("run_cpu", "activated_shell", "本人在 bash source .venv/bin/activate 後以 python 在根目錄及 notebooks 執行。")],
        ["a_cpu", "a_prereq"], "承接 W.1 的正常啟用成功情況；不把失敗的啟用或其他 shell 的額外設定算作已成功。", "啟用後 python 在根目錄 True/True，在 notebooks False/False。")
    quote = "看到 `>>>` 就進入 Python 互動模式；逐行貼上包含 `from pathlib import Path` 在內的程式，輸出會緊接在後。輸入 `exit()` 並按 Enter 可回到終端機指令模式，才能再用 `cd` 切換資料夾。"
    add("interactive", "software", "一般 Python 互動模式提示 >>>，執行該單行序列可即時得到輸出，exit() 結束該程序。", quote,
        [reference("py_interpreter", "Interactive Mode L87–104", "主提示通常為 >>>。"), reference("py_constants", "site constants L82–98", "exit() 由預設 site 提供並引發 SystemExit。"), reference("run_cpu", "interactive", "實際 -i 執行取得提示與預期輸出，exit() 後 returncode=0。")],
        ["a_cpu"], "正常啟動包含 site；-S、不載入 site 或自行覆寫 exit 的特例不在這段新手流程內。容器用 -i 令非 TTY 輸入仍進互動模式；未假稱 GUI 終端機操作。", "stderr 包含 >>>；stdout 含 cwd、True、True；exit() 正常結束。")
    quote = "練習先退出 Python，在專案根目錄輸入 `cd notebooks`，再進互動模式重跑這段：工具箱這時在上一層，路徑要改成 `../.venv/bin/python`（Windows 是 `..\\.venv\\Scripts\\python.exe`），啟用時仍只要輸入 `python`。"
    add("cwd_exercise", "software", "進入 notebooks 後，環境位於上一層，可用 ../.venv/bin/python；回到根目錄可用 cd ..，檔案本身未被移動。", quote,
        [reference("run_cpu", "notebooks_directory / activated_shell", "實際從 notebooks 用上一層 Python 重跑原樣程式，且 shell cd notebooks 後仍可用已啟用 python。"), reference("py_venv", "bin / Scripts 與直接路徑", "支持相應平台的環境路徑形式。")],
        ["a_cpu"], "本 repo 的 notebooks 是根目錄直接子目錄；Windows 路徑做官方文件核對，未執行 Windows。", "notebooks 的輸出是 False/False；同一 repo 根目錄檔案仍存在。")

    error_quotes = [
        ("error_trace", "紅色錯誤通常會列出哪個檔案、哪行程式，最後一行說明原因。", "一般 traceback 或語法錯誤可用檔名、行號與最後一行定位原因。", "py_errors", "L15–32、L63–79", "exceptions[*].stderr", "未處理例外與語法錯誤的實際 stderr 提供類型／行號／訊息；顏色由介面決定，不是 Python 語義。"),
        ("error_module", "`ModuleNotFoundError: No module named 'torch'` 表示現在的 Python 找不到工具", "ModuleNotFoundError 的指定訊息表示目前 import 找不到 torch 模組。", "py_exceptions", "ModuleNotFoundError L255–260", "exceptions.ModuleNotFoundError", "以 -I -S 隔離套件查找，確實得到 ModuleNotFoundError: No module named 'torch'；沒有刪除已安裝套件。"),
        ("error_file", "`FileNotFoundError` 是找不到指定檔案，先查目前起點和檔名。", "不存在的檔案或目錄請求可引發 FileNotFoundError。", "py_exceptions", "FileNotFoundError L766–770", "exceptions.FileNotFoundError", "讀取不存在的 __w7_missing_input_20261029__ 得到 FileNotFoundError。"),
        ("error_key", "`KeyError` 是字典中沒有那個鍵，可回看 [W.2](#W.2)。", "字典查找不存在的鍵引發 KeyError。", "py_exceptions", "KeyError L273–277", "exceptions.KeyError", "只有 animal 鍵的字典取 caption，得到 KeyError: 'caption'。"),
        ("error_index", "`IndexError` 是索引超出範圍。", "序列以單一位置取值超出範圍會引發 IndexError。", "py_exceptions", "IndexError L264–270", "exceptions.IndexError", "兩項 animals 的第 0、1 項為貓狗；第 2 項得到 IndexError: list index out of range。"),
        ("error_syntax", "`SyntaxError` 則先檢查括號、冒號與縮排。", "未關閉括號、遺漏冒號與錯誤縮排是應優先檢查的語法錯誤例子。", "py_exceptions", "SyntaxError L504–512、IndentationError L551–560", "exceptions.SyntaxError_parenthesis / SyntaxError_colon / SyntaxError_indentation", "實際得到 '(' was never closed、expected ':' 及 IndentationError；後者是 SyntaxError 子類。"),
    ]
    for identifier, quote, statement, source, locator, run_locator, observed in error_quotes:
        add(identifier, "software", statement, quote,
            [reference(source, locator, statement), reference("run_cpu", run_locator, observed)],
            ["a_cpu", "a_originals"], "普通 Python 內建例外的新手診斷提示；不是所有紅字的完整分類，也不保證提示的檢查一步就能排除所有原因。", observed)
    quote = "先確認 Notebook 選 **Tiny Perceptron**，或終端機已啟用 `.venv`；也可以在專案根目錄直接用 `.venv/bin/python` 執行，Windows 是 `.\\.venv\\Scripts\\python.exe`（見 [W.1](#W.1)）。"
    add("module_remedy", "software", "選 W.1 登記的 Tiny Perceptron kernel 或使用 .venv Python，是先確認 import 使用哪個環境的相符排查步驟。", quote,
        [reference("py_venv", "How venvs work L188–237", "啟用與直接指定 Python 的環境選擇作用。"), reference("run_cpu", "project_root / activated_shell / exceptions.ModuleNotFoundError", "本人正常 .venv 可執行，隔離套件查找則找不到 torch，展示 interpreter 搜尋環境的差異。")],
        ["a_cpu", "a_prereq"], "確認 kernel 的名稱承接 W.1 的 ipykernel install --name / --display-name 指令；本次未開 Jupyter。這是首先核對環境，不是聲稱只選名稱必然修好 torch。", "Linux .venv Python 與啟用 python 均可用；W.1 當前確實登記 Tiny Perceptron 名稱。")
    quote = "例如 `animals = [\"貓\", \"狗\"]` 從前面用 0、1 編號；取 `animals[2]` 是取第三項，就會超出範圍。"
    add("index_example", "numeric", "animals 長度為 2，合法非負索引為 0、1；索引 2 表示第三項並超出範圍。", quote,
        [reference("py_stdtypes", "Sequence operations s[i] L975–983、L1111–1113", "從 0 起算與超範圍例外的原始契約。"), reference("index_derivation", "n=2；0≤i<n；i+1=3", "透明整數手算。")],
        ["a_cpu"], "本例普通 Python list 的非負整數單一索引；未討論負索引或 slice。", "手算合法索引集合 {0,1}；2 對應第三項且不合法。另有實際執行 IndexError 收據。", "合法索引 0、1；animals[2] 引發 IndexError。", "手算 n=2、合法條件 0≤i<2。執行證據是另行實跑，不把推導假標成 executed。")

    quote = "模型訓練的指令碼有些預設只檢查資料與流程。"
    add("default_dryrun", "software", "本 repo 的部分訓練入口預設只做流程查核，沒有開啟權重更新。", quote,
        [reference(code_ids["scripts/train.py"], "parser --train；main steps、loss.backward、if args.train: optimizer.step", "預設一次 forward/backward，而 update 被 args.train 控制。"), reference("run_cpu", "dry_run", "真預設文字命令只做一次紀錄，17 個參數張量未變。")],
        ["a_cpu", "a_prereq"], "核對 training.md T.2 對應的 train.py 預設路線；不是說 repo 所有訓練入口都預設 dry-run。", "mode=dry-run-no-weight-update；history 1 筆，optimizer_step_calls=0、checkpoint_save_calls=0、all_parameters_bitwise_unchanged=true。")
    quote = "這類命令最後印出的那行 JSON 是報告；其中的 `mode` 若是 `dry-run-no-weight-update`，表示這次沒有更新模型"
    add("dryrun_report", "software", "此類入口最後一行 JSON 的 dry-run-no-weight-update 標記對應未更新權重的分支。", quote,
        [reference(code_ids["scripts/train.py"], "main 最後 report、args.train 的 mode 與 optimizer.step 分支", "同一布林條件控制 mode 與更新。"), reference(code_ids["scripts/train_simple.py"], "main L58–83", "相同的 --train / step / mode 分支。"), reference(code_ids["scripts/pretrain_encoders.py"], "main L45–74", "相同標記與不更新分支。"), reference("run_cpu", "dry_run.raw_stdout / report / all_parameters_bitwise_unchanged", "最後一行能解析為完整 report，全部觀察到的參數逐位未變。")],
        ["a_cpu"], "本節明指的報告語義由目前程式控制流程與一次真 CPU 執行支持；不把 JSON 出現、有限 loss 或非零梯度當品質證據。", "真原始 stdout 先印 step JSON，再印完整 report；最後 mode=dry-run-no-weight-update，無 optimizer step。")
    quote = "依[操作配方](training.md)加上 `--train` 才正式訓練。不要只因看到報告就以為模型學過了。"
    add("train_gate", "software", "上述 dry-run 入口須明確加 --train 才走 optimizer.step 與保存分支；報告存在不代表模型已學會任務。", quote,
        [reference(code_ids["scripts/train.py"], "parser L40、main 的 if args.train 更新／保存分支", "更新需明確的 --train；反向傳播本身不呼叫 optimizer.step。"), reference("run_cpu", "dry_run.report / optimizer_step_calls / checkpoint_save_calls", "沒有 --train 也產生報告，但本次沒有更新。")],
        ["a_cpu", "a_prereq"], "本次沒有執行 --train，未做新訓練或品質測試。這裡只確認是否更新的控制流程；實際學習成效另需獨立量測。", "未加 --train 的真執行已有 JSON 報告但 step/save 呼叫皆為 0；--train 的啟用效果由目前原始碼核對。")

    ids = [c["id"] for c in claims]
    numeric = [c["id"] for c in claims if c["kind"] == "numeric"]
    report = {"schema_version": 1, "review_stage": "technical", "lesson_id": "W.7",
              "source": "course/first-steps.md#W.7", "reviewer_task": "/root/omni_technical_coordinator/technical_w_7",
              "reviewer_context": "fresh", "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
              "source_sha256": hashlib.sha256(BODY.encode()).hexdigest(), "figure_sha256": {},
              "verdict": "pass", "claims": claims, "sources": sources, "artifacts": artifacts,
              "issues": [], "original_issue_resolutions": [],
              "checks": {
                  "factual_accuracy": {"status": "pass", "details": "逐項核對 8 個目錄用途、65 條獨立實作路線、路徑／環境／互動模式、6 類錯誤與預設 dry-run。每項保留真原句及可追蹤證據；未發現需修訂的矛盾。", "claim_ids": ids},
                  "numeric_verification": {"status": "pass", "details": "本節唯一具體數值算例是兩項 list 的 0、1、2 索引；明記手算，再以真執行補核 IndexError。布林輸出另以軟體查核保存；沒有測得品質／速度主張。", "claim_ids": numeric},
                  "figure_consistency": {"status": "not_applicable", "details": "W.7 v4 凍結切片沒有 SVG 或其他圖引用，freeze figures_sha256 恰為 {}。所連目標小節圖並非 W.7 的圖，本報告只核其文字實作路由，沒有聲稱實看或重審那些圖。", "claim_ids": []},
                  "source_verification": {"status": "pass", "details": "本人讀 CPython v3.13.5 與 uv 0.12.19 官方版本化原文，實讀目前目的小節與主要程式。每個節錄目的地記精確切片 SHA；58 個原樣函式／類別 AST 相同。不以來源庫摘要或他人審閱作原證據。", "claim_ids": ids},
                  "limitations": {"status": "pass", "details": "Linux CPU 的一次短執行只支持本配置行為；Windows/macOS 路徑由官方契約核對，未宣稱實跑。route 通過不代審連結頁理論、圖或實測。dry-run 反向傳播不等於權重更新，未開始正式訓練，未碰 GPU、付費或模型／正文／checker／active pointer。", "claim_ids": ids},
              },
              "review_scope": {"source_freeze": "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4.json",
                               "snapshot": "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4/sections/W.7.md",
                               "current_source_equal_to_frozen_slice": True,
                               "reader_task_exclusions_only": ["/root/rv1019_reader_entry", "/root/omni_reader_first_steps"],
                               "reader_completion_claim": False,
                               "claims_count": len(claims), "route_edges": count,
                               "independence": "未讀作者 PLAN／OUTLINE、其他 technical／reader／sidecar 答案；本人完整 task 名稱按實際派工填寫。"}}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    save(REPORT, report)
    print(json.dumps({"report": relative(REPORT), "sha256": sha(REPORT), "verdict": report["verdict"],
                      "claims": len(claims), "sources": len(sources), "artifacts": len(artifacts),
                      "created_at": report["created_at"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
