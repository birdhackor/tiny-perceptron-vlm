"""只重算 5.6 的固定排程與 SVG；不載入模型、不更新權重。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
PAGE = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == "5.6")
BODY = (ROOT / PAGE["snapshot"]).read_bytes()
assert hashlib.sha256(BODY).hexdigest() == PAGE["source_sha256"]
current_chapter = (ROOT / "course/chapters/05.md").read_text()
current_slice = re.search(r"^## 5\.6 .+?(?=^## |\Z)", current_chapter, re.M | re.S)[0]
assert current_slice.encode() == BODY
blocks = re.findall(r"```python\n(.*?)```", BODY.decode(), re.S)
assert len(blocks) == 2
inline_function = ast.parse(blocks[0]).body[0]
library_path = ROOT / "tiny_perceptron/training.py"
library_tree = ast.parse(library_path.read_text())
library_function = next(n for n in library_tree.body if isinstance(n, ast.FunctionDef) and n.name == "learning_rate")
assert ast.dump(inline_function, include_attributes=False) == ast.dump(library_function, include_attributes=False)

namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for index, code in enumerate(blocks):
        exec(compile(code, f"frozen-5.6-inline-{index + 1}", "exec"), namespace)
lr = namespace["learning_rate"]
rates = namespace["rates"]
expected_first_ten = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.001, 0.000998, 0.000993, 0.000984, 0.000971]
expected_last_three = [0.000116, 0.000107, 0.000102]
assert [round(v, 6) for v in rates[:10]] == expected_first_ten
assert [round(v, 6) for v in rates[-3:]] == expected_last_three
assert len(rates) == 40
assert all(0.0001 <= value <= 0.001 + 1e-15 for value in rates)
assert all(a <= b for a, b in zip(rates[:4], rates[1:5], strict=True))
assert rates[4] == rates[5] == 0.001
assert all(a > b for a, b in zip(rates[5:-1], rates[6:], strict=True))

rates_ten = [lr(i, 40, peak=0.001, warmup=10) for i in range(40)]
rates_thirty = [lr(i, 40, peak=0.001, warmup=30) for i in range(40)]
assert rates_ten == rates_thirty
assert [round(rates_ten[i], 6) for i in (0, 4, 9)] == [0.0001, 0.0005, 0.001]
assert lr(40, 40, warmup=5) == lr(50, 40, warmup=5) == 0.0001
assert rates[-1] > 0.0001
exceptions = []
for kwargs in ({"total": 0}, {"total": 40, "peak": 0}, {"total": 40, "peak": -0.001}):
    try:
        lr(step=0, **kwargs)
    except ValueError as error:
        exceptions.append({"arguments": kwargs, "error": str(error)})
    else:
        raise AssertionError("missing ValueError")

same_update_different_plan = {str(total): lr(19, total, peak=0.001, warmup=5) for total in (40, 100)}
assert same_update_different_plan["40"] != same_update_different_plan["100"]
assert [lr(i, 40, warmup=5) for i in range(20, 40)] == rates[20:]

figure_path = ROOT / next(iter(PAGE["figures_sha256"]))
figure_bytes = figure_path.read_bytes()
assert hashlib.sha256(figure_bytes).hexdigest() == next(iter(PAGE["figures_sha256"].values()))
svg = ET.fromstring(figure_bytes)
polyline = svg.find("{http://www.w3.org/2000/svg}polyline")
points = [tuple(map(float, pair.split(","))) for pair in polyline.attrib["points"].split()]
assert len(points) == 40
expected_points = [(184 + i / 39 * 416, 424 - rate / 0.001 * 300) for i, rate in enumerate(rates)]
max_coordinate_error = max(abs(a - b) for actual, expected in zip(points, expected_points, strict=True) for a, b in zip(actual, expected, strict=True))
assert max_coordinate_error <= 0.0051
for i, x in ((0, 184.0), (19, 386.66666666666663), (39, 600.0)):
    assert math.isclose(expected_points[i][0], x, abs_tol=1e-12)

quadratic_steps = [{"lr": value, "w_after": 1 - value * -4, "loss_after": (1 - value * -4 - 3) ** 2} for value in (0.1, 0.6, 2)]
assert math.isclose(quadratic_steps[0]["loss_after"], 2.56, abs_tol=1e-14)
assert math.isclose(quadratic_steps[1]["loss_after"], 0.16, abs_tol=1e-14)
assert quadratic_steps[2]["loss_after"] == 36

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_5_6",
    "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform(), "device": "CPU; Python standard library only; no neural model loaded"},
    "scope": "完整執行 5.6 兩段凍結內嵌程式，另重算正文練習、正值輸入邊界、必要前置純算例及 SVG 座標；沒有訓練、forward 或品質評測。",
    "source": {"freeze": str(FREEZE.relative_to(ROOT)), "freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(), "snapshot": PAGE["snapshot"], "source_sha256": PAGE["source_sha256"], "current_slice_byte_equal": True},
    "library": {"path": str(library_path.relative_to(ROOT)), "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(), "function_ast_equal_to_inline": True},
    "inline_stdout": captured.getvalue(),
    "warmup5": {"rates": rates, "first_ten_rounded6": [round(v, 6) for v in rates[:10]], "last_three_rounded6": [round(v, 6) for v in rates[-3:]], "maximum": max(rates), "last_progress": (39 - 5) / (40 - 5), "floor_at_step40": lr(40, 40, warmup=5)},
    "warmup10": {"first_ten": rates_ten[:10], "last_three": rates_ten[-3:], "warmup30_identical": rates_ten == rates_thirty},
    "same_twentieth_update_different_total": same_update_different_plan,
    "same_step_and_plan_resume_sequence_identical": True,
    "invalid_total_peak_checks": exceptions,
    "figure": {"path": str(figure_path.relative_to(ROOT)), "sha256": hashlib.sha256(figure_bytes).hexdigest(), "point_count": len(points), "x_mapping": "184 + zero_based_step / 39 * 416; labels 1,20,40 are zero_based_step + 1", "y_mapping": "424 - learning_rate / 0.001 * 300", "max_coordinate_rounding_error_px": max_coordinate_error, "tolerance_px": 0.0051},
    "prerequisite_1_12_quadratic_recalculation": quadratic_steps,
    "all_assertions_passed": True,
    "limits": ["僅測正文的正整數步數與非負訓練索引範圍，未替任意型別、負 step 或負 warmup 建立 API 契約。", "續訓只核對排程相同輸入保留相同值這項必要條件，未核對完整 optimizer/RNG/checkpoint 還原。", "未做模型 forward、梯度更新、訓練、GPU/MPS/Colab 或全頁桌面/手機排版驗證。"],
}
json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
print()
