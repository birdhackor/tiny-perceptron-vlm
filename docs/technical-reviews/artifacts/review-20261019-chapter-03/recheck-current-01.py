"""複查現行依據與兩處措辭變動，沿用原始核對方法。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old_output_path = ARTIFACTS / "verification-output.json"
old_code_path = ARTIFACTS / "verify_intro.py"
old_report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-03/technical-chapter-03-original-01.json"
expected_old = {
    old_output_path: "9de8d4a8be4d3126bcd080b637a54086d40b2a39f9d613a4c6ff25d5affdeb85",
    old_code_path: "3e5e12683442d18fbd69276bb673b97c19867ac0bcf7b6963f2641b8756bb104",
    old_report_path: "7a322e071cb3a292abbbb110a99aa2414ca83760d2169129d2c2c20044f1b5f6",
}
for path, expected in expected_old.items():
    assert sha(path) == expected, "原始報告與證據應保持不變"
old_output = json.loads(old_output_path.read_text())
old_report = json.loads(old_report_path.read_text())
assert old_report["issues"] == [], "原始報告應無待解決問題"

captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    execution = runpy.run_path(str(old_code_path), run_name="__review_recheck__")
current = json.loads(captured.getvalue())
assert current["section_code_outputs"] == old_output["section_code_outputs"], "現行課文程式輸出應與先前核對相同"

base = "5ef56bc756f3"
target = "3af7f13ef959"
chapter_paths = ["course/chapters/02.md", "course/chapters/03.md"]
changes = []
for relative in chapter_paths:
    previous = subprocess.run(["git", "show", f"{base}:{relative}"], capture_output=True, check=True).stdout
    revised = subprocess.run(["git", "show", f"{target}:{relative}"], capture_output=True, check=True).stdout
    old_source = next(source for source in old_report["sources"] if source.get("path") == relative)
    assert hashlib.sha256(previous).hexdigest() == old_source["sha256"], "差異起點應等於原報告引用版本"
    assert revised == (ROOT / relative).read_bytes(), "差異終點應等於現行檔案"
    command = ["git", "diff", base, target, "--", relative]
    diff = subprocess.run(command, capture_output=True, text=True, check=True)
    changes.append({"path": relative, "command": command, "base_sha256": hashlib.sha256(previous).hexdigest(), "current_sha256": sha(ROOT / relative), "diff": diff.stdout})
    if relative.endswith("02.md"):
        before_block = next(block for block in execution["blocks"](previous.decode()) if block.startswith('examples = [("紅色物體是"'))
        after_block = next(block for block in execution["blocks"](revised.decode()) if block.startswith('examples = [("紅色物體是"'))
        assert before_block == after_block, "紅藍資料及切片程式應未變動"
    else:
        assert previous.decode().split("## ", 1)[0] == revised.decode().split("## ", 1)[0], "導言應未變動"
        for title in ["## 3.6 整段一起算，怎樣禁止讀後面的答案？", "## 3.7 幾種讀法怎樣並行，再交回同一位置？"]:
            assert execution["lesson"](previous.decode(), title) == execution["lesson"](revised.decode(), title), "注意力程式與逐項說明應未變動"

sampling = execution["lesson"]((ROOT / "course/chapters/03.md").read_text(), "## 3.5 匹配特徵越多，為什麼要除以平方根？")
sampling_stdout = io.StringIO()
namespace = {"__name__": "__review_sampling__"}
with contextlib.redirect_stdout(sampling_stdout):
    for block in execution["blocks"](sampling):
        exec(compile(block, "匹配特徵縮放算例", "exec"), namespace)
rows = []
for line in sampling_stdout.getvalue().splitlines():
    d, raw, scaled = line.split()
    dimension = int(d)
    rows.append({"dimension": dimension, "samples": 1000, "theoretical_std": math.sqrt(dimension), "observed_std": float(raw), "observed_scaled_std": float(scaled), "scaled_relative_error": abs(float(scaled) - 1.0)})
assert all(row["scaled_relative_error"] < 0.1 for row in rows), "本次抽樣縮放後標準差應在理論尺度附近"
finite_std = execution["torch"].tensor([-2.0, 0.0, 2.0]).std().item()
assert finite_std == 2.0, "有限樣本也可能恰好等於理論標準差"

url = "https://www.cs.cornell.edu/courses/cs4780/2023fa/lectures/lecturenote03.html"
curl = ["curl", "-fLsS", "--max-time", "45", url]
response = subprocess.run(curl, capture_output=True, check=True)
parser = execution["PlainText"]()
parser.feed(response.stdout.decode())
plain = " ".join(parser.parts)
formula = r"h(x_i) = \textrm{sign}(\mathbf{w}^\top \mathbf{x}_i + b)"
assert formula in plain, "感知器講義應明列加權和經符號判定的公式"
assert "Fall 2023" in plain, "應讀取指定學期版本"

for path, expected in expected_old.items():
    assert sha(path) == expected, "執行後原始報告與證據應仍保持不變"
fresh_attention = current.pop("original_source")
current.pop("command")
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-chapter-03/recheck-current-01.py",
    "executed_at": datetime.now().astimezone().isoformat(),
    "environment": current["environment"],
    "original_evidence_unchanged": {str(path.relative_to(ROOT)): sha(path) for path in expected_old},
    "original_issue_ids": [],
    "diff_receipts": changes,
    "target_intro_unchanged": True,
    "relied_fixture_code_unchanged": True,
    "relied_attention_sections_unchanged": True,
    "original_verification_rerun": current,
    "sampling_execution": {"stdout": sampling_stdout.getvalue(), "rows": rows, "tolerance": "本次尺度示範採縮放後標準差相對理論值一的誤差小於 10%；不作統計品質或訓練改善判定。", "finite_sample_equality_example": {"possible_scores_at_dimension_four": [-2, 0, 2], "sample_mean": 0, "sample_variance_formula": "(4 + 0 + 4) / (3 - 1) = 4", "observed_corrected_sample_std": finite_std, "population_std": 2}},
    "perceptron_source": {"url": url, "version": "Cornell CS 4/5780, Fall 2023", "locator": "Classifier；bias absorbed into a constant feature", "command": curl, "curl_exit": response.returncode, "response_sha256": hashlib.sha256(response.stdout).hexdigest(), "formula_excerpt": formula, "full_text_retained": False},
    "attention_source_retrieval": {key: value for key, value in fresh_attention.items() if key != "short_excerpts"},
    "result": "導言及被依賴的資料與程式未變；現行小節實跑通過且輸出與先前相同。兩處來源措辭變更均已獨立核對。"
}, ensure_ascii=False, indent=2))
