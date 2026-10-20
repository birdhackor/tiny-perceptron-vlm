"""複查頁面差異、證據完整性與 venv 的預設 pip 安裝行為。"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_20-2/technical-20-2-original-01.json"
ORIGINAL_SHA = "f6566c8c4d02704dfa5f7311ac6a2d32d126872d4fc1037c887b0beb53aa800a"
OLD_SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/20.2.md"
NEW_SOURCE = ROOT / "docs/course-review-20261019/freeze-05/sources/20.2.md"
FREEZE = ROOT / "docs/course-review-20261019/freeze-06.json"
NOTEBOOK = ROOT / "notebooks/20/20.2.ipynb"
os.chdir(ROOT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
result = {"environment": {"python": sys.version, "platform": platform.platform()}, "commands": []}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, expected=0):
    process = subprocess.run(args, capture_output=True, text=True)
    result["commands"].append({"command": args, "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
    assert process.returncode == expected, "指令結束狀態與預期不符"
    return process.stdout


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def original(identifier, url, locator, excerpts):
    args = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "35", url]
    process = subprocess.run(args, capture_output=True)
    assert process.returncode == 0, "官方原文讀取失敗"
    raw = process.stdout
    parser = Text()
    parser.feed(raw.decode())
    plain = " ".join(" ".join(parser.parts).split())
    assert all(excerpt in plain for excerpt in excerpts), "官方原文未找到指定摘錄"
    title_match = re.search(r"<title>(.*?)</title>", raw.decode(), re.S)
    result.setdefault("official_sources", []).append({
        "id": identifier, "url": url, "version": "Python 3.12.15 documentation",
        "accessed_on": "2026-10-20", "locator": locator, "excerpts": excerpts,
        "title": title_match.group(1) if title_match else "",
        "checked_original": True, "full_response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw), "command": args, "returncode": process.returncode,
    })


assert sha(ORIGINAL) == ORIGINAL_SHA, "原始報告內容已變動"
report = json.loads(ORIGINAL.read_text())
previous = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-20-2/verification.json").read_text())
result["head"] = run(["git", "rev-parse", "HEAD"]).strip()
for package in ["torch", "pytest", "Pillow", "soundfile"]:
    result["environment"][package] = importlib.metadata.version(package)
result["original_report"] = {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": sha(ORIGINAL)}
result["prior_artifacts"] = []
for item in report["artifacts"]:
    actual = sha(ROOT / item["path"])
    assert actual == item["sha256"], "原有 artifact 內容已變動"
    result["prior_artifacts"].append({"path": item["path"], "sha256": actual, "unchanged": True})
self_check = ARTIFACT / "self-check.txt"
assert sha(self_check) == "42c9dfc00035cfa0c9f781fe5ea8bf5a98097fb83b070720722d7522c4d0d196", "原有自我檢查內容已變動"
result["prior_self_check"] = {"path": str(self_check.relative_to(ROOT)), "sha256": sha(self_check)}
result["prior_input_integrity"] = []
for path, expected in previous["source_hashes"].items():
    actual = sha(ROOT / path)
    if path == "notebooks/20/20.2.ipynb":
        assert actual == "e8d389b0d6233fd42bde2dc2c727eb756811c6c3933180186cfae9962c5c381e"
    else:
        assert actual == expected, "先前執行依據出現額外變動"
    result["prior_input_integrity"].append({"path": path, "previous_sha256": expected, "current_sha256": actual, "unchanged": expected == actual})
result["historical_input_integrity"] = []
for item in previous["historical_inputs"].values():
    actual = sha(ROOT / item["path"])
    assert actual == item["sha256"], "原始量測證據內容已變動"
    result["historical_input_integrity"].append({"path": item["path"], "sha256": actual, "unchanged": True})
result["repository_sources"] = []
for item in report["sources"]:
    if item["kind"] == "repository_code":
        actual = sha(ROOT / item["path"])
        result["repository_sources"].append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": actual == item["sha256"]})
assert sha(NEW_SOURCE) == "3b75be66712cdb377b6e1c51d4c03355e7858839b1ec8f1f8250abc6de6f9e27"
assert sha(FREEZE) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
result["source_freeze"] = {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)}
result["source_sha256"] = sha(NEW_SOURCE)
freeze = json.loads(FREEZE.read_text())
page = next(page for page in freeze["pages"] if page["page_id"] == "20.2")
assert page["source_sha256"] == sha(NEW_SOURCE) and page["figures_sha256"] == {}
chapter = (ROOT / "course/chapters/20.md").read_text()
headings = list(re.finditer(r"(?m)^## ", chapter))
start = next(index for index, match in enumerate(headings) if chapter[match.start():].startswith("## 20.2 "))
current_section = chapter[headings[start].start():headings[start + 1].start()]
assert current_section == NEW_SOURCE.read_text(), "現行課文與凍結原文不符"
result["current_chapter_matches_freeze"] = True
result["source_diff"] = run(["git", "diff", "--no-index", str(OLD_SOURCE.relative_to(ROOT)), str(NEW_SOURCE.relative_to(ROOT))], expected=1)
result["notebook_diff"] = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "notebooks/20/20.2.ipynb"])
notebook = json.loads(NOTEBOOK.read_text())
markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
changed_text = "`venv` 是 Python 標準函式庫裡建立環境的工具，`pip` 是安裝套件的工具；`pip` 通常隨 Python 安裝程式提供，`venv` 建新環境時預設也會裝好它"
assert changed_text in NEW_SOURCE.read_text() and changed_text in markdown
source_commands = re.findall(r"```bash\n(.*?)```", NEW_SOURCE.read_text(), re.S)
notebook_commands = re.findall(r"```bash\n(.*?)```", markdown, re.S)
assert source_commands == notebook_commands, "Notebook 與凍結原文的 Bash 命令不符"
assert all(cell["cell_type"] == "markdown" for cell in notebook["cells"])
result["notebook"] = {"path": str(NOTEBOOK.relative_to(ROOT)), "sha256": sha(NOTEBOOK), "cell_types": [cell["cell_type"] for cell in notebook["cells"]], "bash_blocks_match_freeze": True, "bash_blocks": len(source_commands), "changed_text_present": True}
result["source_structure"] = {"python_fences": len(re.findall(r"(?m)^```python", NEW_SOURCE.read_text())), "svg_references": re.findall(r"[^\s()]+\.svg", NEW_SOURCE.read_text())}
original("venv", "https://docs.python.org/3.12/library/venv.html", "Creating virtual environments / How venvs work", ["Installs pip by default", "ensurepip"])
original("installing", "https://docs.python.org/3.12/installing/index.html", "Key terms", ["pip is the preferred installer program.", "it is included by default with the Python binary installers."])
original("ensurepip", "https://docs.python.org/3.12/library/ensurepip.html", "Introduction", ["pip is an independent project with its own release cycle"])
python = shutil.which("python3.12") or sys.executable
result["venv_python"] = {"executable": python, "version": run([python, "--version"]).strip(), "is_python312": bool(shutil.which("python3.12"))}
with tempfile.TemporaryDirectory(prefix="venv-check-", dir=ARTIFACT) as temporary:
    temporary = Path(temporary)
    default = temporary / "default"
    without = temporary / "without-pip"
    run([python, "-m", "venv", str(default)])
    venv_python = str(default / "bin/python")
    run([venv_python, "-m", "pip", "--version"])
    inspection = json.loads(run([venv_python, "-c", "import json,sys,pip; print(json.dumps({'python':sys.version,'isolated':sys.prefix!=sys.base_prefix,'pip_inside_environment':pip.__file__.startswith(sys.prefix),'pip_version':pip.__version__}))"]))
    assert inspection["isolated"] and inspection["pip_inside_environment"]
    run([python, "-m", "venv", "--without-pip", str(without)])
    no_pip = json.loads(run([str(without / "bin/python"), "-c", "import importlib.util,json; print(json.dumps({'pip_absent':importlib.util.find_spec('pip') is None}))"]))
    assert no_pip["pip_absent"]
    result["venv_observation"] = {"default": inspection, "without_pip": no_pip, "scope": "本機命令列預設行為的短驗證；Python 3.12 的契約另依版本化官方文件核對，不將本機結果外推為成品的 Linux 安裝驗證。"}
result["inherited_execution"] = {"path": "docs/technical-reviews/artifacts/review-20261019-20-2/verification.json", "sha256": sha(ARTIFACT / "verification.json"), "tests": "先前 91 項測試輸出沿用；所測程式與測試的 SHA-256 未變，本次未重跑。", "historical_observation": "原始 Linux 單機量測未變，依已核對的執行紀錄沿用；本次沒有重新量測模型。"}
for item in report["artifacts"]:
    assert sha(ROOT / item["path"]) == item["sha256"], "查核後原有 artifact 內容變動"
assert sha(ORIGINAL) == ORIGINAL_SHA
result["status"] = "passed"
output = ARTIFACT / "verification-current-01.json"
with output.open("x") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": result["status"], "source_sha256": result["source_sha256"], "notebook": result["notebook"], "venv_observation": result["venv_observation"], "output": str(output.relative_to(ROOT)), "sha256": sha(output)}, ensure_ascii=False))
