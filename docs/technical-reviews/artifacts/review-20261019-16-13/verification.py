"""獨立枚舉可見配對，核對遮罩、練習與 SVG 格子。"""

import hashlib
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.attention import attention_mask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enumerate_rule(name, length, width=3, group_size=3):
    rows = []
    for query in range(length):
        rows.append(
            [
                key
                for key in range(length)
                if key <= query
                and (name != "sliding" or query - key < width)
                and (name != "block_local" or query // group_size == key // group_size)
            ]
        )
    return rows


def check_equal(observed, expected, description):
    if observed != expected:
        raise AssertionError(f"{description}：實際 {observed}；預期 {expected}")


position = torch.arange(6)
causal_full = attention_mask(position, position)
distance = position[:, None] - position[None, :]
group = position // 3
rules = {
    "causal": causal_full[0, 0],
    "sliding": causal_full[0, 0] & (distance < 3),
    "block_local": attention_mask(position, position, segments=group[None])[0, 0],
}
check_equal(list(causal_full.shape), [1, 1, 6, 6], "因果遮罩形狀")
check_equal(list(distance.shape), [6, 6], "距離表形狀")
check_equal(group.tolist(), [0, 0, 0, 1, 1, 1], "分組")
check_equal(list(group[None].shape), [1, 6], "分組筆數軸")
check_equal(str(causal_full.dtype), "torch.bool", "遮罩型別")

observed = {}
for name, mask in rules.items():
    rows = [position[row].tolist() for row in mask]
    expected = enumerate_rule(name, 6)
    check_equal(rows, expected, f"{name} 的全部可讀位置")
    observed[name] = {
        "rows": rows,
        "row_counts": [len(row) for row in rows],
        "allowed_pairs": mask.sum().item(),
        "query_3": rows[3],
        "query_4": rows[4],
        "shape": list(mask.shape),
        "stored_elements": mask.numel(),
        "storage_bytes": mask.untyped_storage().nbytes(),
    }

check_equal([v["allowed_pairs"] for v in observed.values()], [21, 15, 12], "允許配對總數")
check_equal([v["query_3"] for v in observed.values()], [[0, 1, 2, 3], [1, 2, 3], [3]], "查詢位置的可讀範圍")
check_equal([v["query_4"] for v in observed.values()], [[0, 1, 2, 3, 4], [2, 3, 4], [3, 4]], "切換查詢位置")

distance_only = distance < 3
without_causal = {
    "allowed_pairs": distance_only.sum().item(),
    "query_3": position[distance_only[3]].tolist(),
    "future_query_3": [key for key in position[distance_only[3]].tolist() if key > 3],
}
check_equal(without_causal["allowed_pairs"], 30, "僅使用距離條件的配對數")
check_equal(without_causal["query_3"], [1, 2, 3, 4, 5], "僅使用距離條件的可讀位置")
two_group = attention_mask(position, position, segments=(position // 2)[None])[0, 0]
two_rows = [position[row].tolist() for row in two_group]
check_equal(two_rows, enumerate_rule("block_local", 6, group_size=2), "改變分組大小")
check_equal(two_group.sum().item(), 9, "較小分組的配對數")
check_equal(two_rows[3], [2, 3], "較小分組的可讀位置")

svg = ROOT / "course/figures/rewrite-16-13-visible-pairs.svg"
tree = ET.parse(svg)
rectangles = tree.findall(".//{http://www.w3.org/2000/svg}rect")
figure = {}
for name, y_start in (("causal", 148), ("sliding", 508), ("block_local", 868)):
    cells = {}
    for rect in rectangles:
        if rect.get("width") != "34" or rect.get("height") != "34":
            continue
        x, y = int(rect.get("x")), int(rect.get("y"))
        if not (y_start <= y <= y_start + 5 * 38):
            continue
        row, column = (y - y_start) // 38, (x - 69) // 38
        check_equal((y - y_start) % 38, 0, "格子列座標")
        check_equal((x - 69) % 38, 0, "格子欄座標")
        if (row, column) in cells:
            raise AssertionError("SVG 有重複格子")
        if rect.get("fill") not in ("#107b6e", "#e2e8f0"):
            raise AssertionError("SVG 格子顏色不符圖例")
        cells[row, column] = rect.get("fill") == "#107b6e"
    check_equal(len(cells), 36, "每張配對表的格子數")
    svg_rows = [[key for key in range(6) if cells[row, key]] for row in range(6)]
    check_equal(svg_rows, enumerate_rule(name, 6), "圖中配對與獨立枚舉")
    figure[name] = {"cell_count": len(cells), "rows": svg_rows, "green_count": sum(map(len, svg_rows))}

receipt_path = ROOT / "docs/course-review-20261019/figure-renders-04.json"
receipt = json.loads(receipt_path.read_text())
record = next(record for record in receipt["records"] if record["figure"] == str(svg.relative_to(ROOT)))
png = ROOT / record["png"]["path"]
check_equal(digest(png), record["png"]["sha256"], "PNG 收據雜湊")
check_equal(digest(svg), record["frozen_svg"]["sha256"], "SVG 收據雜湊")

scores = [0.0, math.log(2), math.log(4)]
values = [10.0, 20.0, 30.0]
weighted_mean = sum(math.exp(s) * v for s, v in zip(scores, values)) / sum(map(math.exp, scores))
online = {}
for block_size in [1, 2, 3]:
    maximum, denominator, numerator = -math.inf, 0.0, 0.0
    states = []
    for start in range(0, len(scores), block_size):
        block_scores = scores[start : start + block_size]
        block_values = values[start : start + block_size]
        new_maximum = max(maximum, max(block_scores))
        factor = math.exp(maximum - new_maximum)
        weights = [math.exp(s - new_maximum) for s in block_scores]
        denominator = denominator * factor + sum(weights)
        numerator = numerator * factor + sum(w * v for w, v in zip(weights, block_values))
        maximum = new_maximum
        states.append({"denominator": denominator, "numerator": numerator})
    result = numerator / denominator
    if not math.isclose(result, weighted_mean, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("分塊加權平均與完整結果不符")
    covered = [(q, k) for q in range(6) for start in range(0, 6, block_size)
               for k in range(start, min(start + block_size, 6)) if k <= q]
    check_equal(sorted(covered), [(q, k) for q in range(6) for k in range(q + 1)], "分塊遍歷的完整因果配對")
    online[str(block_size)] = {"states": states, "result": result, "causal_pairs_covered": len(covered)}

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-13/verification.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "status": "pass",
    "attention_module_sha256": digest(ROOT / "tiny_perceptron/attention.py"),
    "causal_shape": list(causal_full.shape),
    "distance": distance.tolist(),
    "group": group.tolist(),
    "group_shape": list(group[None].shape),
    "rules": observed,
    "without_causal": without_causal,
    "group_size_two": {"rows": two_rows, "allowed_pairs": two_group.sum().item(), "query_3": two_rows[3]},
    "svg": {"path": str(svg.relative_to(ROOT)), "sha256": digest(svg), "tables": figure},
    "render_receipt": {"path": str(receipt_path.relative_to(ROOT)), "sha256": digest(receipt_path), "record": record, "png_sha256_matches": True},
    "online_softmax": {"full_result": weighted_mean, "block_sizes": online, "absolute_tolerance": 1e-12},
    "limitations": "只驗證確定性 CPU 算例與圖表；沒有量測 GPU kernel、速度、訓練品質或記憶體收益。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
