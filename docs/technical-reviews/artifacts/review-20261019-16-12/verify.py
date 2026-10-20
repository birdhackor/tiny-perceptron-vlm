"""獨立核對視窗端點、跨層可達範圍、配對數及圖文版本。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/16.12.md"
FIGURE = ROOT / "course/figures/rewrite-16-12-sliding-path.svg"
FREEZE = ROOT / "docs/course-review-20261019/freeze-04.json"
RENDER_RECEIPT = ROOT / "docs/course-review-20261019/figure-renders-04.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def direct(position, width, length):
    return {key for key in range(length) if 0 <= position - key < width}


def reachable(position, width, length, layers):
    current = {position}
    for _ in range(layers):
        current = {key for query in current for key in direct(query, width, length)}
    return current


def run_code(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, str(SOURCE), "exec"), namespace)
    return namespace, stdout.getvalue()


raw = SOURCE.read_text()
code_blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
require(len(code_blocks) == 1, "原文應有唯一的 Python 程式區塊")
code = code_blocks[0]
original, original_stdout = run_code(code)
require(code.count("window = 3") == 1, "視窗設定必須可唯一辨識")
changed, changed_stdout = run_code(code.replace("window = 3", "window = 2", 1))

cases = []
for width, namespace in ((3, original), (2, changed)):
    expected_direct = direct(7, width, 8)
    expected_two = reachable(7, width, 8, 2)
    count = sum(len(direct(query, width, 8)) for query in range(8))
    require(namespace["visible"](7) == expected_direct, "直接可見集合與獨立判定式不符")
    require(namespace["two_layers"] == expected_two, "兩層可達集合與獨立圖走訪不符")
    history = []
    slots = {}
    for position in range(8):
        slots[position % width] = position
        history.append(len(slots))
        require(set(slots.values()) == direct(position, width, 8), "環狀快取內容不符視窗")
    cases.append(
        {
            "width": width,
            "visible_at_last": sorted(expected_direct),
            "two_layers_at_last": sorted(expected_two),
            "visible_at_five": sorted(direct(5, width, 8)),
            "row_counts": [len(direct(query, width, 8)) for query in range(8)],
            "pair_count": count,
            "ring_size_history": history,
            "ring_final_positions": sorted(slots.values()),
        }
    )

configurations = 0
reachable_queries = 0
for length in range(1, 33):
    for width in range(1, length + 3):
        count = sum(len(direct(query, width, length)) for query in range(length))
        clipped_width = min(width, length)
        formula = length * clipped_width - clipped_width * (clipped_width - 1) // 2
        require(count == formula, "配對數與封閉公式不符")
        configurations += 1
        for layers in range(1, 5):
            for query in range(length):
                expected = set(range(max(0, query - layers * (width - 1)), query + 1))
                require(reachable(query, width, length, layers) == expected, "跨層端點與封閉公式不符")
                reachable_queries += 1

weights = [Fraction(0) for _ in range(8)]
for intermediate in direct(7, 3, 8):
    for key in direct(intermediate, 3, 8):
        weights[key] += Fraction(1, len(direct(7, 3, 8))) * Fraction(1, len(direct(intermediate, 3, 8)))
require({index for index, weight in enumerate(weights) if weight} == reachable(7, 3, 8, 2), "均勻平均的非零依賴不符圖路徑")
require(weights[3] == Fraction(1, 9), "最早位置的兩層係數不符")

dense_cases = []
vectors = [[position, 1, -position] for position in range(8)]
for width in (3, 2):
    dots = 0
    scores = []
    for query in range(8):
        row = []
        for key in range(8):
            row.append(sum(a * b for a, b in zip(vectors[query], vectors[key])))
            dots += 1
        scores.append(row)
    masked = [[scores[query][key] if key in direct(query, width, 8) else None for key in range(8)] for query in range(8)]
    dense_cases.append({"width": width, "dot_products_before_mask": dots, "entries_kept_after_mask": sum(value is not None for row in masked for value in row)})

freeze = json.loads(FREEZE.read_bytes())
page = next(page for page in freeze["pages"] if page["page_id"] == "16.12")
require(sha(SOURCE) == page["source_sha256"], "原文與凍結清單不符")
require(sha(FIGURE) == page["figures_sha256"][FIGURE.relative_to(ROOT).as_posix()], "SVG 與凍結清單不符")
receipt = json.loads(RENDER_RECEIPT.read_bytes())
render = next(record for record in receipt["records"] if record["figure"] == FIGURE.relative_to(ROOT).as_posix())
png = ROOT / render["png"]["path"]
require(sha(png) == render["png"]["sha256"], "已觀看的 PNG 與算繪收據不符")

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-12/verify.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU", "torch": "未使用；獨立核對僅需標準函式庫"},
    "source_sha256": sha(SOURCE),
    "figure_sha256": sha(FIGURE),
    "source_freeze_sha256": sha(FREEZE),
    "original_stdout": original_stdout,
    "changed_window_stdout": changed_stdout,
    "full_causal_pair_count": sum(range(1, 9)),
    "cases": cases,
    "exhaustive_check": {"lengths": [1, 32], "widths": "每個長度 n 對應 1 至 n+2", "layers": [1, 4], "configurations": configurations, "reachable_queries": reachable_queries, "result": "全部精確相等"},
    "two_layer_uniform_average_coefficients": [str(weight) for weight in weights],
    "information_retention_limit": "有向路徑只提供潛在依賴；若 value 映射恆為零，所有注意力輸出也恆為零，不能從路徑推論保留任意原始字詞。",
    "dense_then_mask": dense_cases,
    "derivation": {
        "direct": "V(p,w)={j: 0<=j<=p 且 p-j<w}；自己計入 w 個位置。",
        "layers": "L 層來源為 [max(0,p-L(w-1)),p]；以相鄰層來源集合的聯集歸納。",
        "pair_count": "令 m=min(n,w)，總數=sum(min(p+1,w),p=0..n-1)=nm-m(m-1)/2；完整因果為 n(n+1)/2。",
        "asymptotic": "n>=w>=1 時，滑動視窗配對為 nw-w(w-1)/2，固定 w 時隨 n 線性增加；完整因果為 Θ(n²)。只計注意力配對，不計線性投影、MLP 或牆鐘時間。",
        "cache": "固定 batch B、KV heads H、head dimension D、元素 bytes b；只存視窗內 K 和 V 時最多 2BHwDb bytes。完整快取將 w 換成 n。暫存區、權重與其他層不由此式界定。",
    },
    "figure_render_receipt": {"path": RENDER_RECEIPT.relative_to(ROOT).as_posix(), "sha256": sha(RENDER_RECEIPT), "record": render},
    "result": "全部核對通過；未量測模型品質、速度或 GPU。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
