"""核對改句、證據延續性與當前小節的實際執行。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_C-3/technical-C-3-original-01.json"
OLD = ROOT / "docs/course-review-20261019/freeze-01/sources/C.3.md"
NEW = ROOT / "docs/course-review-20261019/freeze-05/sources/C.3.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = json.loads(REPORT.read_bytes())
assert sha(REPORT) == "038d7d773be668746a2b002ed0ed2c8c5acb95872682815b7ef32e90e3db5ff2"
assert sha(NEW) == "0d13815586b270c897199826f0911090e34efb75ecb57f3c08db4bf8dd69bb3b"
old_lines, new_lines = OLD.read_bytes().splitlines(keepends=True), NEW.read_bytes().splitlines(keepends=True)
assert len(old_lines) == len(new_lines)
changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
assert len(changed) == 1
assert b"greedy" in changed[0][1]
old_fences = re.findall(rb"```python\n(.*?)```", OLD.read_bytes(), re.S)
new_fences = re.findall(rb"```python\n(.*?)```", NEW.read_bytes(), re.S)
assert old_fences == new_fences
references = []
for item in original["artifacts"]:
    actual = sha(ROOT / item["path"])
    assert actual == item["sha256"]
    references.append({"kind": "artifact", "path": item["path"], "sha256": actual, "unchanged": True})
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = sha(ROOT / item["path"])
        assert actual == item["sha256"]
        references.append({"kind": "repository_code", "path": item["path"], "sha256": actual, "unchanged": True})
previous_verification = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-C-3/verification.json").read_bytes())
for path, expected in previous_verification["inputs"].items():
    actual = sha(ROOT / path)
    assert actual == expected
    references.append({"kind": "prior_execution_input", "path": path, "sha256": actual, "unchanged": True})

command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/0C.md#C.3", "--execute"]
completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
assert completed.returncode == 0, completed.stderr
run = json.loads(completed.stdout)
assert run["source_sha256"] == sha(NEW)
assert run["python_fences"] == 1 and run["svg_references"] == 0
run_directory = Path(run["artifact_directory"])
assert run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
files = {}
for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_directory / name).read_bytes()
    files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "content": raw.decode()}
assert (run_directory / "section.md").read_bytes() == NEW.read_bytes()
old_execution = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-C-3/section-execution.json").read_bytes())
assert files["stdout.txt"]["content"] == old_execution["files"]["stdout.txt"]["content"]
assert files["fence-1.py"]["content"] == old_execution["files"]["fence-1.py"]["content"]
endpoint_calculation = [{"p": p, "k": k, "coverage": 1 - (1 - p) ** k} for p in (0, 1) for k in (1, 2, 4, 8)]
for item in endpoint_calculation:
    assert item["coverage"] == item["p"]
diff_command = ["git", "diff", "--no-index", str(OLD.relative_to(ROOT)), str(NEW.relative_to(ROOT))]
diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 1
execution_environment = json.loads(files["environment.json"]["content"])
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-3/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": execution_environment["torch"], "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": str(REPORT.relative_to(ROOT)), "sha256": sha(REPORT)},
    "source": {"path": str(NEW.relative_to(ROOT)), "sha256": sha(NEW)},
    "source_freeze": {"path": "docs/course-review-20261019/freeze-05.json", "sha256": sha(ROOT / "docs/course-review-20261019/freeze-05.json")},
    "diff": {"command_argv": diff_command, "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr},
    "changed_line_count": len(changed),
    "other_source_lines_unchanged": True,
    "all_python_fences_unchanged": True,
    "retained_evidence": references,
    "greedy_endpoint_calculation": endpoint_calculation,
    "mathematical_check": "固定題目與確定性解碼的成功指標恆為 0 或 1；對 k>=1，1-(1-p)^k=p。常數隨機變數的任意聯合事件與其邊際事件乘積相同，因此沒有重試增益不代表破壞獨立性。",
    "adaptive_prompt_check": "後續分布依先前答案改變的改寫流程，不能直接當成固定分布的獨立重抽；沿用機率乘法的原始定義及先前的具體相依反例。這不表示僅因文字出現於 prompt 就能證明任意忽略該文字的模型相依。",
    "section_execution": {"command_argv": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "files": files},
    "result": "舊證據與相關實作及資料的 SHA-256 全數相同；只有一段中的三處句子變更；新小節程式實跑成功且輸出完全相同；greedy 端點與無增益說明成立。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
