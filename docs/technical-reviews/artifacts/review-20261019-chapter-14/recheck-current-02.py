"""重跑短程式並核對新版章首及依據；舊報告與證據只讀。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
OLD_REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-14/technical-chapter-14-original-01.json"
OLD_PROBE = ARTIFACTS / "review_probe.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if digest(OLD_REPORT) != "0077405eb388571d55f55a93489212816ca89d32e7f15ae2b9008dd51765f1f2":
    raise ValueError("原始報告的內容已改變")
original = json.loads(OLD_REPORT.read_bytes())
integrity = []
for artifact in original["artifacts"]:
    actual = digest(ROOT / artifact["path"])
    integrity.append({"path": artifact["path"], "recorded_sha256": artifact["sha256"], "current_sha256": actual, "unchanged": actual == artifact["sha256"]})
    if actual != artifact["sha256"]:
        raise ValueError("沿用證據的內容已改變：" + artifact["path"])

probe_source = OLD_PROBE.read_text()
probe_source = probe_source.replace("freeze-01/sources/chapter-14.md", "freeze-05/sources/chapter-14.md")
probe_source = probe_source.replace("freeze-04.json", "freeze-05.json")
probe_source = probe_source.replace("chapter-14/review_probe.py", "chapter-14/recheck-current-02.py")
probe_source = probe_source.replace('"library_scope_counterexample"', '"qk_local_example_boundary"')
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(probe_source, str(OLD_PROBE), "exec"), {"__file__": str(OLD_PROBE), "__name__": "__main__"})
probe = json.loads(output.getvalue())

chapter_path = ROOT / "course/chapters/14.md"
chapter = chapter_path.read_text()
headings = list(re.finditer(r"^## (14\.\d+) (.+)$", chapter, re.M))
needles = [
    "假設每個位置轉45度", "[2,2,2]", "Q=[1,1]", "輸入1與2時輸出6與12",
    "內容路徑輸出 `[2,2,2]`", "E三列為", "每段文字最多八張卡", "第一張的座標是0",
    "第一組每往後一個位置就轉60度", "下面把α設成1、β設成2",
    "假設字的embedding表與輸出層共20萬參數", "把要接近的答案向量簡化成數字4",
    "w 初值0.2", "loss_by_depth = {1: 1.2", "state = -1",
    "假設同一份驗證資料上的代價在1到5輪依序是1.2", "predictions = [torch.tensor([0.95, 0.05])",
    "other, block = 200_000, 1_000_000",
]
if len(headings) != len(needles):
    raise ValueError("固定算例的核對清單與目前小節數不符")
inventory = []
for index, (heading, needle) in enumerate(zip(headings, needles, strict=True)):
    section = chapter[heading.end():headings[index + 1].start() if index + 1 < len(headings) else len(chapter)]
    position = section.find(needle)
    if position < 0:
        raise ValueError("找不到固定數字算例：" + heading[2])
    inventory.append({"section_id": heading[1], "section_title": heading[2], "fixed_example": needle, "context": section[max(0, position - 80):position + len(needle) + 180]})

commands = [
    ["git", "diff", "--no-index", "docs/course-review-20261019/freeze-01/sources/chapter-14.md", "docs/course-review-20261019/freeze-05/sources/chapter-14.md"],
    ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/14.md"],
    ["git", "diff", "90ebf972aecddd555f18443624974a3f8ddc60a6", "HEAD", "--", "course/glossary.md"],
]
diffs = []
for command in commands:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode not in (0, 1) or completed.stderr:
        raise ValueError("差異核對未成功：" + completed.stderr)
    diffs.append({"command": command, "exit_code": completed.returncode, "stdout": completed.stdout})

import torch

signed_zero_left = torch.tensor([0.0], dtype=torch.float32)
signed_zero_right = torch.tensor([-0.0], dtype=torch.float32)
current_equal_paragraph = next(p for p in chapter.split("\n\n") if "要求每個元素完全相等的" in p)
equal_check = {
    "changed_dependency_paragraph": current_equal_paragraph,
    "installed_official_docstring": torch.equal.__doc__.split("Example::")[0].strip(),
    "equal_positive_negative_zero": torch.equal(signed_zero_left, signed_zero_right),
    "same_storage_bits_for_signed_zero": torch.equal(signed_zero_left.view(torch.int32), signed_zero_right.view(torch.int32)),
    "scope": "只核對章檔作為導覽依據的額外改句；該句不在凍結章首頁面，不新增為章首主張。",
}

old_output = json.loads((ARTIFACTS / "review-probe-output.json").read_bytes())
unchanged_dependencies = []
for path, previous_digest in old_output["file_sha256"].items():
    if path.startswith("tiny_perceptron/") or path in ("course/chapters/04.md", "course/glossary.md"):
        unchanged_dependencies.append({"path": path, "sha256": digest(ROOT / path), "unchanged": digest(ROOT / path) == previous_digest})
if not all(item["unchanged"] for item in unchanged_dependencies if item["path"] != "course/glossary.md"):
    raise ValueError("實作或前置文件有額外變動，需要擴大複查")

guard_command = ["uv", "run", "--no-sync", "python", str(ARTIFACTS / "recheck-current-01.py")]
guard_run = subprocess.run(guard_command, cwd=ROOT, capture_output=True, text=True)

print(json.dumps({
    "initial_guard_run": {"command": guard_command, "exit_code": guard_run.returncode, "stdout": guard_run.stdout, "stderr": guard_run.stderr, "scope_note": "先前的守門條件把查詞連結目標也要求逐字不變，因此中止；已檢視名詞頁差異。章首僅主張該頁可供查詞，連結仍存在，沒有沿用舊版名詞頁來證明新內容。"},
    "initial_guard_stdout_file": {"path": (ARTIFACTS / "recheck-output-current-01.json").relative_to(ROOT).as_posix(), "sha256": digest(ARTIFACTS / "recheck-output-current-01.json"), "bytes": (ARTIFACTS / "recheck-output-current-01.json").stat().st_size},
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-14/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "original_report_sha256": digest(OLD_REPORT),
    "original_artifact_integrity": integrity,
    "unchanged_dependencies": unchanged_dependencies,
    "replay_transformations": ["只在記憶體把 probe 的凍結原文與清單改為本次凍結版本", "命令標示改為實際複查入口", "QK 局部例子的輸出欄名改為範圍界線，沒有修改計算"],
    "diff_receipts": diffs,
    "fixed_number_example_inventory": inventory,
    "dependency_wording_check": equal_check,
    "current_probe": probe,
}, ensure_ascii=False, indent=2))
