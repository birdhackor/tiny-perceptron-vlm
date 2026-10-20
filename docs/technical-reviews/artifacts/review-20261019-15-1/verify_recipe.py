"""核對教學命令的解析、派送與比較表的存在，不啟動訓練。"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.run import experiment_spec

environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]
result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
assert result.returncode == 0 and "--experiment ID" in result.stdout
spec = experiment_spec("moe")
assert spec["module"] == "architecture" and spec["function"] == "run_moe"
body = (ROOT / "course/chapters/15.md").read_text()
comparison = re.search(r"(?ms)^## 15\.13 .*?(?=^## |\Z)", body).group()
full_table = comparison.split("| 模型 |", 1)[1]
rows = [line for line in full_table.splitlines() if line.startswith("| Dense，匹配") or line.startswith("| MoE top-")]
assert len(rows) == 7
record = json.loads((ROOT / "docs/course-experiments/results/moe.json").read_text())
historical = []
for path in ("scripts/course_experiments/architecture.py", "scripts/course_experiments/run.py"):
    argv = ["git", "show", record["revision"] + ":" + path]
    process = subprocess.run(argv, cwd=ROOT, capture_output=True)
    historical.append({"command_argv": argv, "exit_code": process.returncode, "stdout_bytes": len(process.stdout), "stderr": process.stderr.decode(), "sha256": hashlib.sha256(process.stdout).hexdigest() if process.returncode == 0 else None})
out = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-1/verify_recipe.py",
    "help": {"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
    "experiment_spec": spec,
    "comparison_rows": rows,
    "comparison_scope": "只核對被本頁指向的七組比較表與分母文字，不審閱其他小節。",
    "denominators_present": all(value in comparison for value in ("39,256", "41,914", "337,761")),
    "historical_source_lookup": historical,
    "limitations": "結果紀錄列出的 revision 在本機 Git 裡不含這兩個實驗腳本，無法由該 revision 還原當時腳本；目前模型、FFN、attention 與評估 common.py 的 hash 與實驗紀錄相同，目前 architecture.py 與 run.py 不同。未聲稱完全重現歷史訓練。",
    "environment": {"python": sys.version, "device": "未啟動模型運算"},
    "result": "命令解析與 moe 派送符合正文；比較表確實包含七組與有效目標數。",
}
(HERE / "recipe-verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": out["result"], "historical_source_lookup_exit_codes": [item["exit_code"] for item in historical]}, ensure_ascii=False))
