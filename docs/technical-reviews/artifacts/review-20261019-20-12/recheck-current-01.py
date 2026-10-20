"""核對凍結差異、沿用證據指紋與本次小節執行輸出。"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.natural_concepts import text_error_report


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


original_path = "docs/course-review-20261019/reviews/rv1019_technical_20-12/technical-20-12-original-01.json"
original_raw = (ROOT / original_path).read_bytes()
assert digest(original_raw) == "ae61d09277d0a8ab72472e37e44ffe15173ce66e7c2e0a234c9d51f47db9bc6c"
original = json.loads(original_raw)
old_path = "docs/course-review-20261019/freeze-01/sources/20.12.md"
current_path = "docs/course-review-20261019/freeze-06/sources/20.12.md"
old, current = [(ROOT / name).read_bytes() for name in (old_path, current_path)]
assert digest(old) == original["source_sha256"]
assert digest(current) == "45e1e1d5ec66a7a1d792c7ab76bdbfbbe5749a47cb31ac927d1759b384e52dcc"
previous_sentence = "AISHELL 提供朗讀錄音與正確轉寫文字，助理回答的判準由本課另訂。"
current_sentence = "AISHELL 提供朗讀錄音與正確轉寫文字，助理回答的判準由本課另訂；判準由 AI 助理撰寫，沒有逐題經人審核。"
assert old.decode().replace(previous_sentence, current_sentence).encode() == current
changed = subprocess.run(["git", "diff", "--no-index", old_path, current_path], cwd=ROOT, capture_output=True, check=False)
assert changed.returncode == 1

reuse = []
for group in ("sources", "artifacts"):
    for item in original[group]:
        if item.get("path"):
            raw = (ROOT / item["path"]).read_bytes()
            actual = digest(raw)
            assert actual == item["sha256"], item["path"]
            reuse.append({"group": group, "id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": actual, "matches": True})

run_path = ROOT / "outputs/reviewer-tools/runs/20261020T101619701235Z"
files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_path / name).read_bytes()
    files[name] = {"sha256": digest(raw), "text": raw.decode()}
extraction = json.loads(files["extraction.json"]["text"])
execution = json.loads(files["execution.json"]["text"])
assert extraction["source_sha256"] == digest(current) == execution["source_sha256"]
assert execution["exit_code"] == 0
assert files["stdout.txt"]["text"] == "最少編輯次數 1\n正確答案字數 9\nCER百分比 11.1\n兩份問題相同 False\n"
assert files["stderr.txt"]["text"] == ""
previous_execution = json.loads((ARTIFACT / "section-execution.json").read_bytes())
assert files["fence-1.py"]["sha256"] == previous_execution["files"]["fence-1.py"]["sha256"]

numeric = []
for prediction in ("請推薦辣的晚餐。", "請推薦不辣的晚飯。"):
    result = text_error_report("請推薦不辣的晚餐。", prediction)
    assert result["edits"] == 1 and result["reference_characters"] == 9
    assert result["cer"] == 1 / 9 and round(result["cer"] * 100, 1) == 11.1
    numeric.append(result)

figure = ROOT / "course/figures/natural-v4-asr-two-routes.svg"
figure_sha = digest(figure.read_bytes())
assert original["figure_sha256"][figure.relative_to(ROOT).as_posix()] == figure_sha
png = ROOT / "outputs/course-review-20261019/figure-renders-04/aaed40e6fc4eebfa6ba59a98b2c08ea42624a481553570313b146d8e1f0085ac.png"
assert digest(png.read_bytes()) == "f39bac41451c8c50bdc0c0b4192f779e8d1eb190b5d25272d34657a7a577cb67"
freeze = ROOT / "docs/course-review-20261019/freeze-06.json"
assert digest(freeze.read_bytes()) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
value = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-12/recheck-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": original_path, "sha256": digest(original_raw)},
    "previous_issues": original["issues"],
    "source_change": {"old_path": old_path, "current_path": current_path, "old_sha256": digest(old), "current_sha256": digest(current), "only_replacement": {"old": previous_sentence, "current": current_sentence}},
    "diff": {"command": "git diff --no-index " + old_path + " " + current_path, "exit_code": changed.returncode, "stdout": changed.stdout.decode(), "stderr": changed.stderr.decode()},
    "reused_evidence": reuse,
    "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.12" --execute', "files": files},
    "numeric": numeric,
    "figure": {"path": figure.relative_to(ROOT).as_posix(), "sha256": figure_sha, "png": png.relative_to(ROOT).as_posix(), "png_sha256": digest(png.read_bytes())},
    "result": "凍結版本、差異範圍、沿用來源與證據、相同程式區塊、新版執行與兩個算例均符合；新增來源身分與人工審核揭露另行判斷。"
}
with (ARTIFACT / "recheck-current-01.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": value["result"], "previous_issue_count": len(original["issues"]), "reused_evidence_count": len(reuse)}, ensure_ascii=False))
