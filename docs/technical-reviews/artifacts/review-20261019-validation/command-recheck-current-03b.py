"""核對新增命令說明，保存原始文件小段摘錄及隔離執行結果。"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path.cwd()
ART = ROOT / "docs/technical-reviews/artifacts/review-20261019-validation"
PREVIOUS = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-current-02.json"
DESTINATION = ART / "command-recheck-results-current-03.json"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(argv, cwd=ROOT, env=None):
    p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    return {"argv": [str(x) for x in argv], "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, text):
        self.parts.append(text)

report = json.loads(PREVIOUS.read_text())
policy = json.loads((ART / "policy-recheck-results-current-02.json").read_text())
prior_evidence = json.loads((ART / "evidence-results-current-01.json").read_text())
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-validation/command-recheck-current-03b.py",
    "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform()},
    "accessed_on": datetime.now().astimezone().date().isoformat(),
    "previous_report": {"path": str(PREVIOUS.relative_to(ROOT)), "sha256": digest(PREVIOUS)},
    "repository_head": run(["git", "rev-parse", "HEAD"]),
}
result["sources"] = [{"id": s["id"], "path": s["path"], "previous_sha256": s["sha256"], "current_sha256": digest(ROOT / s["path"]), "unchanged": s["sha256"] == digest(ROOT / s["path"])} for s in report["sources"] if s["kind"] == "repository_code"]
result["artifacts"] = [{"id": a["id"], "path": a["path"], "previous_sha256": a["sha256"], "current_sha256": digest(ROOT / a["path"]), "unchanged": a["sha256"] == digest(ROOT / a["path"])} for a in report["artifacts"]]
result["previous_input_files"] = [{"path": x["path"], "previous_sha256": x["current_sha256"], "current_sha256": digest(ROOT / x["path"]), "unchanged": x["current_sha256"] == digest(ROOT / x["path"])} for x in policy["previous_input_files"]]
result["input_drift"] = [x for x in result["previous_input_files"] if not x["unchanged"]]
old_source = "docs/course-review-20261019/freeze-05/sources/validation.md"
new_source = "docs/course-review-20261019/freeze-07/sources/validation.md"
result["page_diff"] = run(["git", "diff", "--no-index", old_source, new_source])
result["page_and_freeze"] = [{"path": p, "sha256": digest(ROOT / p)} for p in ["docs/validation.md", new_source, "docs/course-review-20261019/freeze-07.json", "docs/technical-review-guide.md"]]
result["resolved_issue_basis"] = {
    "retained_artifact": {"path": "docs/technical-reviews/artifacts/review-20261019-validation/evidence-results-current-01.json", "sha256": digest(ART / "evidence-results-current-01.json")},
    "checks": prior_evidence["checks"],
    "authorized_full_book_summary": prior_evidence["authorized_full_book_summary"],
    "kernel_probe": prior_evidence["kernel_probe"],
    "dataset_counts": prior_evidence["dataset_counts"],
    "scoring_publication_identity": prior_evidence["scoring_publication_identity"],
    "recalculation_identity_checks": prior_evidence["recalculation_identity_checks"],
    "scope": "本次逐檔重算原始證據指紋，沿用先前已保存的欄位與執行；沒有重新訓練、逐題生成、啟動kernel或重算他人審閱。",
}
result["shells"] = {}
for shell, flags in [("/bin/bash", ["--noprofile", "--norc"]), ("/bin/zsh", ["-f"])]:
    entry = {"version": run([shell, "--version"]), "cases": []}
    python = shlex.quote(sys.executable)
    for status in (0, 7):
        body = python + " -c " + shlex.quote("import sys; sys.exit(" + str(status) + ")") + "\necho $?\n"
        observed = run([shell, *flags, "-c", body])
        entry["cases"].append({"expected_exit_code": status, "execution": observed, "matches": observed["stdout"].strip() == str(status)})
    body = python + " -c " + shlex.quote("import sys; sys.exit(7)") + "\n:\necho $?\n"
    overwritten = run([shell, *flags, "-c", body])
    entry["intervening_success"] = {"execution": overwritten, "matches": overwritten["stdout"].strip() == "0"}
    result["shells"][shell] = entry
result["powershell"] = {"pwsh": shutil.which("pwsh"), "powershell": shutil.which("powershell"), "execution": "沒有安裝PowerShell；Windows PowerShell的原生命令狀態契約由下列官方5.1文件核對，未聲稱Windows實跑。"}
git = shutil.which("git")
result["environment"]["git"] = run([git, "--version"])["stdout"].strip()
with tempfile.TemporaryDirectory(prefix="command-probe-", dir=ART) as temporary:
    fixture = Path(temporary)
    repository = fixture / "repository"
    repository.mkdir()
    global_config = fixture / "gitconfig"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "EMAIL"}
    env.update({"GIT_CONFIG_GLOBAL": str(global_config), "GIT_CONFIG_NOSYSTEM": "1", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"})
    calls = []
    def g(*arguments):
        observed = run([git, *arguments], cwd=repository, env=env)
        calls.append(observed)
        return observed
    init = g("init", "--quiet", "--initial-branch=probe")
    with (repository / "selected.txt").open("x") as f:
        f.write("初始內容\n")
    with (repository / "unselected.txt").open("x") as f:
        f.write("保留未選定\n")
    status_before = g("status", "--short")
    staged = g("add", "selected.txt")
    status_staged = g("status", "--short")
    blob_before = g("rev-parse", ":selected.txt")
    # 只改合成工作檔，用來區分目前檔案與已選定的內容。
    with (repository / "selected.txt").open("a") as f:
        f.write("新增內容\n")
    status_modified = g("status", "--short")
    working_blob = g("hash-object", "selected.txt")
    blob_preserved = g("rev-parse", ":selected.txt")
    staged_again = g("add", "selected.txt")
    blob_after = g("rev-parse", ":selected.txt")
    identity_missing = g("-c", "user.useConfigOnly=true", "var", "GIT_AUTHOR_IDENT")
    name_set = g("config", "--global", "user.name", "命令測試")
    email_set = g("config", "--global", "user.email", "command-probe@example.invalid")
    identity = g("var", "GIT_AUTHOR_IDENT")
    head_before = g("rev-parse", "--verify", "HEAD")
    dry_run = g("commit", "--dry-run", "-m", "核對提交說明")
    head_after = g("rev-parse", "--verify", "HEAD")
    global_content = global_config.read_text()
    result["git_probe"] = {
        "scope": "測試repo與替代global設定均在自己的artifact暫存子目錄內；離開後移除。只執行commit --dry-run，不建立提交、不讀寫使用者global設定、不接網路。",
        "executions": calls,
        "isolated_global_config": global_content,
        "checks": {
            "initialized": init["exit_code"] == 0,
            "status_lists_untracked": "?? selected.txt" in status_before["stdout"] and "?? unselected.txt" in status_before["stdout"],
            "add_selects_only_named_file": "A  selected.txt" in status_staged["stdout"] and "?? unselected.txt" in status_staged["stdout"],
            "index_keeps_snapshot_until_added_again": blob_before["stdout"] == blob_preserved["stdout"] and blob_before["stdout"] != working_blob["stdout"] and "AM selected.txt" in status_modified["stdout"],
            "adding_again_updates_snapshot": blob_after["stdout"] == working_blob["stdout"] and staged_again["exit_code"] == 0,
            "missing_identity_message": identity_missing["exit_code"] != 0 and "Author identity unknown" in identity_missing["stderr"],
            "global_settings_resolve_identity": name_set["exit_code"] == 0 and email_set["exit_code"] == 0 and "command-probe@example.invalid" in identity["stdout"],
            "commit_dry_run_accepts_message": dry_run["exit_code"] == 0 and "selected.txt" in dry_run["stdout"],
            "no_commit_created": head_before["exit_code"] != 0 and head_after["exit_code"] != 0,
        },
    }

def external(identifier, url, needles, excerpts, version, locator, summary):
    fetched = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", url], capture_output=True)
    if fetched.returncode:
        raise RuntimeError("原始文件讀取失敗：" + identifier)
    body = fetched.stdout
    parser = Text()
    parser.feed(body.decode("utf-8"))
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    if not all(n in text for n in needles):
        raise RuntimeError("原始文件找不到指定內容：" + identifier)
    if not all(e in text for e in excerpts):
        raise RuntimeError("原始文件摘錄不相符：" + identifier)
    matched = re.search(r"last updated in ([0-9.]+)", text)
    if matched:
        version = "線上最新版；此命令頁最後更新 " + matched.group(1) + "；版本選單列2.56.0"
    return {"id": identifier, "url": url, "accessed_on": result["accessed_on"], "version": version, "response_sha256": hashlib.sha256(body).hexdigest(), "response_bytes": len(body), "curl_command": ["curl", "--fail", "--location", "--silent", "--show-error", url], "curl_exit_code": fetched.returncode, "locator": locator, "excerpts": excerpts, "inspection_note": summary, "checked_original": True}

specifications = [
    ("bash_exit", "https://www.gnu.org/s/bash/manual/html_node/Exit-Status.html", ["an exit status of zero indicates success", "special parameter $?"], ["a non-zero exit status indicates failure.", "The exit status of the last command is available in the special parameter $?"], "GNU Bash Reference Manual 5.3", "Exit Status", "0為成功、非0為失敗，上一個命令的結束碼可由特殊參數取得。"),
    ("bash_version", "https://www.gnu.org/software/bash/manual/html_node/index.html", ["This is Edition 5.3"], [], "GNU Bash Reference Manual 5.3，18 May 2025", "首頁版本", "確認實際讀取的線上手冊版本。"),
    ("zsh_status", "https://zsh.sourceforge.io/Doc/Release/Parameters.html", ["The exit status returned by the last command."], ["The exit status returned by the last command."], "zsh manual 5.9.2，July 12, 2026", "Parameters Used By The Shell，? <S>", "問號特殊參數保存上一個命令的結束碼。"),
    ("zsh_version", "https://zsh.sourceforge.io/Doc/Release/index.html", ["Version 5.9.2"], [], "zsh manual 5.9.2，July 12, 2026", "首頁版本", "確認實際讀取的線上手冊版本。"),
    ("powershell_exit", "https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_automatic_variables?view=powershell-5.1", ["$LASTEXITCODE", "Contains the exit code of the last native program or PowerShell script that ran."], ["Contains the exit code of the last native program or PowerShell script that ran."], "Windows PowerShell 5.1 view；本次HTTP內容以SHA固定", "$LASTEXITCODE", "本頁Python與pytest是原生外部命令，適用LASTEXITCODE；不能把此說法擴成所有PowerShell cmdlet都會更新它。"),
    ("git_status", "https://git-scm.com/docs/git-status", ["Displays paths that have differences between the index file", "not tracked by Git"], ["Displays paths that have differences between the index file and the current HEAD commit"], "線上最新版", "DESCRIPTION", "列出HEAD/index/工作樹差異及未忽略的未追蹤檔案。"),
    ("git_add", "https://git-scm.com/docs/git-add", ["at the time the add command is run"], ["It only adds the content of the specified file(s) at the time the add command is run"], "線上最新版", "DESCRIPTION", "選取指定檔案當下內容至index；之後編輯須再add才能包含。"),
    ("git_commit", "https://git-scm.com/docs/git-commit", ["Create a new commit containing the current contents of the index", "Use <msg> as the commit message.", "--dry-run"], ["Create a new commit containing the current contents of the index and the given log message"], "線上最新版", "DESCRIPTION；OPTIONS，-m、--dry-run", "建立包含暫存內容及訊息的提交，-m給定訊息；dry-run只列出將包含的內容。"),
    ("git_identity", "https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup", ["git config --global user.name", "git config --global user.email", "every Git commit uses this information"], ["every Git commit uses this information"], "Pro Git 2nd Edition，線上英文版", "Your Identity", "作者name/email用於每份提交，global設定對該系統使用者生效；repo local可覆蓋。"),
]
result["official_sources"] = [external(*spec) for spec in specifications]
result["publication_sources"] = []
for path, needles in [
    ("docs/publishing.md", ["先把這次要發布的教材與核對資料commit", 'python scripts/export_course.py --executed outputs/notebooks --require-reading-times --revision "$(git rev-parse HEAD)"']),
    (".github/workflows/pages.yml", ["branches: [main]", 'scripts/export_course.py --executed outputs/notebooks --require-reading-times --revision "$GITHUB_SHA"']),
    ("scripts/export_course.py", ['colab = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{revision}/', '"revision": revision,']),
]:
    body = (ROOT / path).read_text()
    result["publication_sources"].append({"path": path, "sha256": digest(ROOT / path), "excerpts": [line.strip() for line in body.splitlines() if any(n in line for n in needles)], "all_locators_found": all(n in body for n in needles)})
result["checks"] = {
    "previous_report_identity": result["previous_report"]["sha256"] == "db70d2d4f147721a55bfa1f4c39ab7da50e692e9ecb5913aa23f23b69a10c94c",
    "all_previous_sources_unchanged": all(x["unchanged"] for x in result["sources"]),
    "all_previous_artifacts_unchanged": all(x["unchanged"] for x in result["artifacts"]),
    "input_changes_only_expected_page": all(x["path"] == "docs/validation.md" for x in result["input_drift"]),
    "page_matches_frozen": digest(ROOT / "docs/validation.md") == digest(ROOT / new_source) == "fd11cbc2a41852b8a3d1a517e81b101dd4bef5bdc01c6a711f90b0a755812e84",
    "freeze_identity": digest(ROOT / "docs/course-review-20261019/freeze-07.json") == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753",
    "shell_exit_codes": all(c["matches"] for shell in result["shells"].values() for c in shell["cases"]),
    "immediate_status_required": all(shell["intervening_success"]["matches"] for shell in result["shells"].values()),
    "git_contract": all(result["git_probe"]["checks"].values()),
    "publication_locators_found": all(x["all_locators_found"] for x in result["publication_sources"]),
    "previous_issue_checks_retained": all(prior_evidence["checks"].values()),
}
result["limitations"] = [
    "沒有在Windows PowerShell 5.1實跑；該變數契約讀原廠5.1文件，實際退出碼探針跑於macOS的Bash與zsh。",
    "沒有建立Git提交；-m的建立語意由官方文件核對，隔離執行只用commit --dry-run驗證選定內容與訊息選項。",
    "沒有重新執行全書Notebook、模型訓練、最後測試或發布workflow；原始紀錄沿用已保存證據並核對未變的指紋。",
]
with DESTINATION.open("x", encoding="utf-8") as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
    output.write("\n")
print(json.dumps({"artifact": str(DESTINATION.relative_to(ROOT)), "sha256": digest(DESTINATION), "checks": result["checks"], "git_checks": result["git_probe"]["checks"], "input_drift": result["input_drift"]}, ensure_ascii=False))
raise SystemExit(0 if all(result["checks"].values()) else 1)
