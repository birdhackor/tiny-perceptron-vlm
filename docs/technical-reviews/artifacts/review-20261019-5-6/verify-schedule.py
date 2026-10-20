"""獨立核算凍結頁面的學習率、練習變體與 SVG 座標。"""

import argparse
import ast
import hashlib
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.section_run.resolve()
    assert run.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    copied = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "fence-2.py"):
        path = run / name
        copied[name] = {"sha256": sha(path), "text": path.read_text()}
    execution = json.loads(copied["execution.json"]["text"])
    environment = json.loads(copied["environment.json"]["text"])
    assert execution["exit_code"] == 0
    assert environment["guard_events"] == []
    assert environment["attempted_fences"] == [1, 2]
    save("section-execution.json", {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.6" --execute',
        "original_run_directory": run.relative_to(ROOT).as_posix(),
        "files": copied,
        "result": "兩個原始 Python 區塊執行成功；stdout 與其他執行紀錄原字串複製保存。",
    })

    source = ROOT / "docs/course-review-20261019/freeze-01/sources/5.6.md"
    text = source.read_text()
    expected_source_sha = "703f4c882048ecd6f162edc6c9d5c1c3d4268b35d5d21a13f2e7a4f5f95e2ba7"
    assert sha(source) == expected_source_sha
    chapter = (ROOT / "course/chapters/05.md").read_bytes()
    start = chapter.index(b"## 5.6 ")
    end = chapter.index(b"\n## ", start) + 1
    assert chapter[start:end] == source.read_bytes()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) == 2
    lesson_tree = ast.parse(blocks[0])
    lesson_fn = next(n for n in lesson_tree.body if isinstance(n, ast.FunctionDef))
    implementation = ROOT / "tiny_perceptron/training.py"
    implementation_tree = ast.parse(implementation.read_text())
    implementation_fn = next(n for n in implementation_tree.body if isinstance(n, ast.FunctionDef) and n.name == "learning_rate")
    ast_equal = ast.dump(lesson_fn, include_attributes=False) == ast.dump(implementation_fn, include_attributes=False)
    assert ast_equal
    namespace = {}
    exec(compile(blocks[0], str(source), "exec"), namespace)
    learning_rate = namespace["learning_rate"]

    rates = [learning_rate(i, 40, peak=0.001, warmup=5) for i in range(40)]
    reference = []
    for i in range(40):
        if i < 5:
            reference.append(0.001 * (i + 1) / 5)
        else:
            reference.append(0.001 * (1 - 0.9 * math.sin(math.pi * (i - 5) / 70) ** 2))
    error = max(abs(a - b) for a, b in zip(rates, reference, strict=True))
    assert error < 1e-18
    assert [round(x, 6) for x in rates[:10]] == [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.001, 0.000998, 0.000993, 0.000984, 0.000971]
    assert [round(x, 6) for x in rates[-3:]] == [0.000116, 0.000107, 0.000102]
    assert max(rates) <= 0.001 + 1e-9
    assert all(a >= b for a, b in zip(rates[5:-1], rates[6:], strict=True))
    assert min(rates) > 0.0001
    exercises = {}
    for warmup in (10, 30):
        exercises[str(warmup)] = [learning_rate(i, 40, peak=0.001, warmup=warmup) for i in range(40)]
    assert exercises["10"] == exercises["30"]
    assert abs(exercises["10"][0] - 0.0001) < 1e-18
    assert abs(exercises["10"][4] - 0.0005) < 1e-18
    assert abs(exercises["10"][9] - 0.001) < 1e-18
    total_comparison = {str(total): learning_rate(19, total, peak=0.001, warmup=5) for total in (40, 100)}
    assert total_comparison["40"] != total_comparison["100"]
    assert learning_rate(40, 40, peak=0.001, warmup=5) == 0.0001
    assert learning_rate(80, 40, peak=0.001, warmup=5) == 0.0001
    boundaries = {"total_1_warmup_0": learning_rate(0, 1, warmup=0), "total_1_default": learning_rate(0, 1)}
    validation = []
    for total, peak in ((0, 0.001), (-1, 0.001), (40, 0), (40, -0.001)):
        try:
            learning_rate(0, total, peak=peak)
        except ValueError as exc:
            validation.append({"total": total, "peak": peak, "error": str(exc)})
        else:
            raise AssertionError("無效的 total 或 peak 沒有報錯")

    figure = ROOT / "course/figures/rewrite-05-06-learning-rate.svg"
    assert sha(figure) == "48f3e87e7d411f757e04c0c678dfced438fdf6671f776948fc3c61c68b45cbd9"
    svg = ET.fromstring(figure.read_bytes())
    polyline = svg.find("{http://www.w3.org/2000/svg}polyline")
    points = [[float(z) for z in pair.split(",")] for pair in polyline.attrib["points"].split()]
    assert len(points) == 40
    # 由刻度反推映射：x 由 184 到 600，y=424-300000*rate。
    expected_points = [[184 + 416 * i / 39, 424 - 300000 * rate] for i, rate in enumerate(rates)]
    coordinate_error = max(abs(a - b) for actual, expected in zip(points, expected_points, strict=True) for a, b in zip(actual, expected, strict=True))
    assert coordinate_error <= 0.0051
    png = ROOT / "outputs/course-review-20261019/figure-renders-04/48f3e87e7d411f757e04c0c678dfced438fdf6671f776948fc3c61c68b45cbd9.png"
    assert sha(png) == "71020479a0ec2a99876afcb2ea1f84521d9cc0e5322a16302499701a6f478d05"

    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-6/verify-schedule.py --section-run " + run.relative_to(ROOT).as_posix(),
        "environment": {"python": sys.version, "torch": environment["torch"], "platform": platform.platform(), "device": "cpu (pure Python scalar arithmetic)"},
        "source_sha256": sha(source),
        "implementation_sha256": sha(implementation),
        "lesson_and_library_function_ast_equal": ast_equal,
        "rates_warmup_5": rates,
        "first_ten_rounded": [round(x, 6) for x in rates[:10]],
        "last_three_rounded": [round(x, 6) for x in rates[-3:]],
        "first_cosine_progress": 0.0,
        "last_planned_progress": 34 / 35,
        "maximum": max(rates),
        "minimum": min(rates),
        "reference_max_abs_error": error,
        "rates_warmup_10": exercises["10"],
        "rates_warmup_30_equal_10": exercises["10"] == exercises["30"],
        "warmup_10_last_three_rounded": [round(x, 6) for x in exercises["10"][-3:]],
        "update_20_total_comparison": total_comparison,
        "invalid_argument_results": validation,
        "denominator_boundaries": boundaries,
        "python_semantics": {"range_40": list(range(40)), "slice_first_ten_length": len(rates[:10]), "slice_last_three_length": len(rates[-3:]), "integer_division": 40 // 4, "warmup_upper_bound": max(1, 40 // 4), "warmup_requested_30_effective": min(30, max(1, 40 // 4)), "one_e_minus_nine": 1e-9},
        "oversized_step_example": {"old_weight": 1, "gradient": -4, "old_cost": 4, "rate": 2, "new_weight": 1 - 2 * (-4), "new_cost": (1 - 2 * (-4) - 3) ** 2},
        "figure": {"svg_sha256": sha(figure), "png_path": png.relative_to(ROOT).as_posix(), "png_sha256": sha(png), "point_count": len(points), "mapping": "x=184+416*i/39; y=424-300000*learning_rate(i,40,0.001,5)", "max_coordinate_error": coordinate_error, "tolerance": 0.0051, "note": "SVG 座標保留兩位小數，容許半個百分之一 SVG 單位的捨入差。PNG 由審閱者用 view_image 看過，未複製或列為 artifact。"},
        "derivation": "設 p=(i-5)/35。cos(pi*p)=1-2*sin(pi*p/2)^2，因此 peak*(0.1+0.45*(1+cos(pi*p)))=peak*(1-0.9*sin(pi*p/2)^2)。p∈[0,1] 時導數為 -0.45*pi*peak*sin(pi*p)≤0，下限是 0.1*peak。i=39 時 p=34/35<1，故最後一次仍高於下限；i=40 才取到下限。SVG 的 0.001、0.0005、0.0001 對應 y=124、274、394，尺度一致。固定梯度時，較小學習率按比例減少更新量；w=1、梯度=-4、學習率2得到w=9、代價36，證明正確梯度方向配大步仍可增加代價。",
        "result": "所有斷言通過；核對的是純排程，未更新模型，也未測訓練品質。",
    }
    save("schedule-verification.json", output)
    print(json.dumps({"result": output["result"], "first_ten": output["first_ten_rounded"], "last_three": output["last_three_rounded"], "reference_max_abs_error": error, "svg_max_coordinate_error": coordinate_error, "update_20_total_comparison": total_comparison}, ensure_ascii=False))


if __name__ == "__main__":
    main()
