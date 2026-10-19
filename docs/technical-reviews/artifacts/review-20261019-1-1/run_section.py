"""在指定證據目錄執行現有小節工具，保存完整輸出收據。"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ARTIFACTS = Path(__file__).resolve().parent
ROOT = ARTIFACTS.parents[3]
HELPER = ROOT / "docs/review-tools/section_facts.py"
RUN = ARTIFACTS / "section-run"
raw = HELPER.read_bytes()
original = 'if not (output.is_relative_to(ROOT / "outputs") or output.is_relative_to(Path("/tmp"))):'
replacement = "if output != REVIEW_OUTPUT:"
text = raw.decode("utf-8")
assert text.count(original) == 1
namespace = {
    "__file__": str(HELPER),
    "__name__": "review_section_runner",
    "REVIEW_OUTPUT": RUN,
}
exec(compile(text.replace(original, replacement), str(HELPER), "exec"), namespace)
metadata = namespace["extract"]("course/chapters/01.md#1.1", RUN)
exit_code = namespace["execute"](RUN, metadata, 120)
files = {}
for name in (
    "section.md", "extraction.json", "fence-1.py", "bootstrap.py",
    "execution.json", "environment.json", "stdout.txt", "stderr.txt",
):
    data = (RUN / name).read_bytes()
    files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "text": data.decode("utf-8")}
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "requested_command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/01.md#1.1" --execute',
    "adaptation": "原工具只允許 outputs 或 /tmp；僅在記憶體將 extract 的輸出位置限制改為本程式的 section-run，其他程式原文不變，子程序仍執行未改動的 section_facts.py --worker。完整收據合併後移除本次暫存工作目錄。",
    "helper_path": str(HELPER.relative_to(ROOT)),
    "helper_sha256": hashlib.sha256(raw).hexdigest(),
    "original_guard": original,
    "replacement_guard": replacement,
    "parent_environment": {"python": sys.version, "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")},
    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
    "exit_code": exit_code,
    "files": files,
}
with (ARTIFACTS / "section-execution.json").open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
shutil.rmtree(RUN)
print(json.dumps({"exit_code": exit_code, "stdout": files["stdout.txt"]["text"], "stderr": files["stderr.txt"]["text"], "environment": json.loads(files["environment.json"]["text"])}, ensure_ascii=False, indent=2))
raise SystemExit(exit_code)
