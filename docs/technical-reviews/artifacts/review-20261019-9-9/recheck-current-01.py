import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition):
    if not condition:
        raise AssertionError("核對結果與預期不符")


def command(argv, expected_codes=(0,)):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    require(result.returncode in expected_codes)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


old_report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-9/technical-9-9-original-01.json"
require(digest(old_report_path) == "d85e0443981c59cdba4ff8f853700d988b2df31316e05083121094ce78ebe8ee")
old = json.loads(old_report_path.read_text())
source_audit = []
artifact_audit = []
for source in old["sources"]:
    if "path" in source:
        actual = digest(ROOT / source["path"])
        source_audit.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})
require([s["id"] for s in source_audit if not s["unchanged"]] == ["s_training"])
for artifact in old["artifacts"]:
    actual = digest(ROOT / artifact["path"])
    require(actual == artifact["sha256"])
    artifact_audit.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

snapshot = ROOT / "docs/course-review-20261019/freeze-05/sources/9.9.md"
require(digest(snapshot) == "25112c3910c3199a4935f0d9df8124e7274e8dff38b08241d0c86fd614a21c92")
freeze = ROOT / "docs/course-review-20261019/freeze-06.json"
require(digest(freeze) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9")
run = (ROOT / sys.argv[1]).resolve()
require(run.is_relative_to(ROOT / "outputs/reviewer-tools/runs"))
section = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.9" --execute'}
for name in ["environment.json", "extraction.json", "execution.json"]:
    section[name] = json.loads((run / name).read_text())
for name in ["stdout.txt", "stderr.txt", "fence-1.py"]:
    section[name] = (run / name).read_text()
require(section["extraction.json"]["source_sha256"] == digest(snapshot))
require(section["extraction.json"]["figure_sha256"] == {})
require(section["execution.json"]["exit_code"] == 0)
require(section["stdout.txt"] == "倍率 1.0 信心 0.6652 選擇 0 正確 False\n倍率 10.0 信心 1.0 選擇 0 正確 False\n")
original_section = json.loads((BASE / "section-execution.json").read_text())
require(section["fence-1.py"] == original_section["code"])

page_diff = command(["git", "diff", "--no-index", "docs/course-review-20261019/freeze-01/sources/9.9.md", "docs/course-review-20261019/freeze-05/sources/9.9.md"], (1,))
training_diff = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
training = (ROOT / "course/training.md").read_text()
linked_command = ".venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda"
require(linked_command in training)
start = training.index("## T.6 ")
stop = training.index("\n## ", start + 1)
training_section = training[start:stop]
require(linked_command in training_section)
require(digest(ROOT / "course/training.md") == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94")
cli = command([".venv/bin/python", "-m", "scripts.course_experiments.run", "--help"])
from scripts.course_experiments.run import experiment_spec
spec = experiment_spec("encoders")
require(spec["module"] == "modalities" and spec["function"] == "run_encoders")

numeric_rerun = command([".venv/bin/python", str((BASE / "verify.py").relative_to(ROOT))])
numeric_results = json.loads(numeric_rerun.pop("stdout"))
require(numeric_results["chosen_temperature"] == 0.5)
require(numeric_results["sample_confidence_original"] == 0.7490114569664001)
require(numeric_results["sample_confidence_calibrated"] == 0.8990483283996582)
round_cases = [{"input": number, "ndigits": digits, "result": round(number, digits)} for number, digits in [(0.6652409434318542, 4), (0.9999545812606812, 4), (1.25, 1), (1.75, 1)]]
require([r["result"] for r in round_cases] == [0.6652, 1.0, 1.2, 1.8])
require(isinstance(round_cases[0]["result"], float))

url = "https://docs.python.org/3.13/builtins/functions.html#round"
curl_argv = ["curl", "--fail", "--location", "--silent", "--show-error", url]
curl = subprocess.run(curl_argv, capture_output=True, check=True)
parser = Plain()
parser.feed(curl.stdout.decode())
plain = " ".join(" ".join(parser.parts).split())
quote = "Return number rounded to ndigits precision after the decimal point."
require(quote in plain and "rounding is done toward the even choice" in plain)
require("Python 3.13.16 documentation" in plain)
source_receipt = {
    "url": url,
    "version": "Python 3.13.16 documentation",
    "accessed_on": datetime.now(timezone.utc).date().isoformat(),
    "command_argv": curl_argv,
    "exit_code": curl.returncode,
    "retrieved_bytes": len(curl.stdout),
    "original_sha256": hashlib.sha256(curl.stdout).hexdigest(),
    "locator": "Built-in Functions / round(number, ndigits=None)",
    "short_excerpts": [quote, "rounding is done toward the even choice"],
    "inspection_note": "取最接近的十進位指定精度值，恰好中點選偶數；float 二進位表示可能影響十進位中點。此函式傳回數值，不保證字串補零成固定長度。本節兩個數都不是中點，原碼捨入結果與正文相符。第三方全文僅在記憶體中讀取，未寫入 repo。",
}
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + sys.argv[1],
    "environment": numeric_results["environment"],
    "result": "全部核對通過；原碼實跑、既有證據完整性、數值重算、round 官方契約與重做入口均相符。",
    "original_report": {"path": str(old_report_path.relative_to(ROOT)), "sha256": digest(old_report_path), "issues": old["issues"]},
    "source_sha256": digest(snapshot),
    "freeze_sha256": digest(freeze),
    "section_execution": section,
    "page_diff": page_diff,
    "training_diff": training_diff,
    "repository_sources": source_audit,
    "previous_artifacts": artifact_audit,
    "round_source": source_receipt,
    "round_cases": round_cases,
    "training_navigation": {"command": linked_command, "section_excerpt": training_section[training_section.index("### 重跑"):training_section.index("這套 `vqa`")], "cli": cli, "experiment_spec": spec},
    "unchanged_numeric_rerun": {**numeric_rerun, "result": numeric_results, "scope": "唯讀執行先前 verify.py；其 source_sha256 仍是原先固定輸入的 hash，只用來重算未變的程式、數值與歷史實驗資料。新版正文的一致性由 section_execution 與本次 source_sha256 核對。"},
    "evidence_access": "只讀原報告及其明確指名 artifact 與實驗證據；outputs 僅讀這次 section_facts 回傳執行目錄。未閱讀其他角色報告或先前審閱目錄。",
}
output = BASE / "recheck-current-01.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": digest(output), "result": result["result"], "round_cases": round_cases}, ensure_ascii=False))
