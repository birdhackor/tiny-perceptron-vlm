"""核對凍結原文差異、沿用證據及 round 的行為，另存實跑收據。"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fences(raw):
    return re.findall(rb"```python\n(.*?)```", raw, re.S)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--original-report", type=Path, required=True)
parser.add_argument("--old-source", type=Path, required=True)
parser.add_argument("--new-source", type=Path, required=True)
parser.add_argument("--freeze", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
require(not args.output.exists(), "輸出檔已存在；請使用新檔名")
original_raw = args.original_report.read_bytes()
original = json.loads(original_raw)
old_raw = args.old_source.read_bytes()
new_raw = args.new_source.read_bytes()
freeze_raw = args.freeze.read_bytes()
freeze = json.loads(freeze_raw)
page = next(p for p in freeze["pages"] if p["page_id"] == original["lesson_id"])
require(digest(old_raw) == original["source_sha256"], "舊原文與原報告不符")
require(digest(new_raw) == page["source_sha256"], "新原文與凍結清單不符")
require(page["snapshot"] == args.new_source.as_posix(), "凍結快照路徑不符")
require(old_raw.count("四捨五入到小數四位".encode()) == 1, "預期措辭的出現次數不符")
require(old_raw.replace("四捨五入到小數四位".encode(), "取到小數四位".encode()) == new_raw, "存在其他原文差異")
require(fences(old_raw) == fences(new_raw), "Python 程式區塊有改動")

reused_artifacts = []
for item in original["artifacts"]:
    actual = digest((ROOT / item["path"]).read_bytes())
    require(actual == item["sha256"], "舊 artifact 已變更")
    reused_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
repo_sources = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = digest((ROOT / item["path"]).read_bytes())
        require(actual == item["sha256"], "引用的 repo 來源已變更")
        repo_sources.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})

diff_command = ["git", "diff", "--no-index", str(args.old_source), str(args.new_source)]
diff_result = subprocess.run(diff_command, cwd=ROOT, capture_output=True, check=False)
require(diff_result.returncode == 1, "原文差異指令未得到預期差異")

section_command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", original["source"], "--execute"]
section_env = os.environ.copy()
section_env["PYTHONDONTWRITEBYTECODE"] = "1"
section_result = subprocess.run(section_command, cwd=ROOT, env=section_env, capture_output=True, check=False)
require(section_result.returncode == 0, "原始小節程式執行失敗")
section_summary = json.loads(section_result.stdout)
require(section_summary["source_sha256"] == digest(new_raw), "現行章檔與新版凍結原文不符")
run_directory = Path(section_summary["artifact_directory"]).resolve()
require(run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs"), "執行輸出不在工具原定目錄")
section_copy = {"command_argv": section_command, "environment_overrides": {"PYTHONDONTWRITEBYTECODE": "1"}, "returncode": section_result.returncode, "summary": section_summary}
for name in ("extraction.json", "environment.json", "execution.json"):
    section_copy[name] = json.loads((run_directory / name).read_bytes())
for name in ("stdout.txt", "stderr.txt"):
    section_copy[name] = (run_directory / name).read_text()
section_copy["python_fences"] = {p.name: p.read_text() for p in sorted(run_directory.glob("fence-*.py"))}
require(section_copy["stderr.txt"] == "", "原始小節有非空 stderr")
require(section_summary["python_fences"] == len(fences(new_raw)), "執行區塊數不符")

python_sources = [
    ("python-round", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/builtins/functions.rst", "round(number, ndigits=None)", "rounding is done toward the even choice", "round 依 ndigits 指定小數精度，取最接近的十進位間距；等距時取偶數。float 不保證保留字串尾端的零。"),
    ("python-list-comprehension", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/tutorial/datastructures.rst", "List Comprehensions", "List comprehensions provide a concise way to create lists.", "方括號內的運算式對 for 指定的元素逐一求值並組成新清單；本頁運算式為 round(value, 4)。"),
]
receipts = []
for identifier, url, locator, excerpt, note in python_sources:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "40", url]
    response = subprocess.run(command, capture_output=True, check=False)
    require(response.returncode == 0, "原始文件讀取失敗")
    require(excerpt.encode() in response.stdout, "摘錄未在原始文件中找到")
    receipts.append({"id": identifier, "url": url, "version": "CPython v3.13.16", "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(), "locator": locator, "checked_original": True, "inspection_note": note, "excerpt": excerpt, "retrieved_sha256": digest(response.stdout), "retrieved_bytes": len(response.stdout), "command_argv": command, "returncode": response.returncode, "full_text_saved": False})

import torch

namespace = {"__name__": "__main__"}
code = fences(new_raw)[0].decode()
code = code[:code.index('print("IGNORE的值"')]
exec(compile(code, str(args.new_source), "exec"), namespace)
logits, labels = namespace["logits"], namespace["labels"]
valid = labels != namespace["IGNORE"]
safe = labels.masked_fill(~valid, 0)
selected = logits.log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
values = selected[0].tolist()
original_values = values.copy()
rounded = [round(value, 4) for value in values]
require(rounded == [-2.2395, -0.2395, -0.2395], "教材清單輸出不符")
require(rounded is not values and values == original_values, "運算未建立獨立的新清單")
tie_inputs = [1 / 32, 3 / 32, -1 / 32, -3 / 32]
tie_actual = [round(value, 4) for value in tie_inputs]
require(tie_actual == [0.0312, 0.0938, -0.0312, -0.0938], "等距取偶數的行為不符")
require(round(1.2, 4) == 1.2, "數值精度測例不符")

result = {
    "command_argv": ["PYTHONDONTWRITEBYTECODE=1", ".venv/bin/python", str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "result": "全部核對成立；原始三段程式成功執行，唯一文字變動符合 Python 官方契約；舊 artifact 與 repo 來源全數未變。",
    "original_report": {"path": str(args.original_report), "sha256": digest(original_raw), "issues": original["issues"]},
    "source_versions": {"old": {"path": str(args.old_source), "sha256": digest(old_raw)}, "new": {"path": str(args.new_source), "sha256": digest(new_raw)}, "freeze": {"path": str(args.freeze), "sha256": digest(freeze_raw)}},
    "diff": {"command_argv": diff_command, "returncode": diff_result.returncode, "stdout": diff_result.stdout.decode(), "stderr": diff_result.stderr.decode(), "only_wording_changed": True, "python_fences_identical": True},
    "changed_sentence": {"before": "[round(value, 4) for value in ...]是把清單裡每個數四捨五入到小數四位、再收成新清單的簡寫。", "after": "[round(value, 4) for value in ...]是把清單裡每個數取到小數四位、再收成新清單的簡寫。", "status": "verified", "supports": "官方 round 契約和 list comprehension 契約，加上實際教材值、可精確表示的等距數值與新清單測例。"},
    "reused_artifacts": reused_artifacts,
    "repository_sources": repo_sources,
    "section_execution": section_copy,
    "python_source_receipts": receipts,
    "round_verification": {"input": values, "output": rounded, "new_list": rounded is not values, "input_unchanged": values == original_values, "tie_inputs": tie_inputs, "tie_actual": tie_actual, "trailing_zero_probe": {"input": 1.2, "ndigits": 4, "output": round(1.2, 4)}, "scope": "取到小數四位指數值捨入，不是固定四個字元的小數格式；沒有承諾每個十進位中點採傳統進位。"},
    "unchanged_claims": [c["id"] for c in original["claims"] if c["id"] != "round-comprehension"],
    "unverified_scope": original["unverified_scope"],
}
with args.output.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({"path": str(args.output), "sha256": digest(args.output.read_bytes()), "result": result["result"], "section_stdout": section_copy["stdout.txt"], "round_verification": result["round_verification"]}, ensure_ascii=False, indent=2))
