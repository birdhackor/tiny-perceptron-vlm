"""核對字串範例、正規化的資訊損失、裁切座標及能力卡用途分類。"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
reference = "牛奶\n麵包"
reversed_lines = "麵包\n牛奶"
no_newline = "牛奶麵包"

def facts(text):
    return {
        "text": text,
        "length": len(text),
        "counter": dict(Counter(text)),
        "line_count": len(text.splitlines()),
        "lines": text.splitlines(),
        "counter_equal_reference": Counter(text) == Counter(reference),
        "exact_equal_reference": text == reference,
        "without_whitespace": "".join(ch for ch in text if not ch.isspace()),
    }

assert len(reference) == len(reversed_lines) == 5
assert dict(Counter(reference)) == {"牛": 1, "奶": 1, "\n": 1, "麵": 1, "包": 1}
assert Counter(reference) == Counter(reversed_lines)
assert reference != reversed_lines
assert reference.splitlines() == ["牛奶", "麵包"]
assert reversed_lines.splitlines() == ["麵包", "牛奶"]
assert len(no_newline) == 4
assert Counter(reference) != Counter(no_newline)
assert reference != no_newline
assert no_newline.splitlines() == ["牛奶麵包"]
assert "\n".isspace()
assert facts(reference)["without_whitespace"] == facts(no_newline)["without_whitespace"]
assert facts(reference)["without_whitespace"] != facts(reversed_lines)["without_whitespace"]

# 相同局部內容可來自不同全頁位置；裁切原點不能由局部內容唯一推回。
page = [[0] * 100 for _ in range(200)]
for origin in (20, 150):
    for y in range(origin, origin + 10):
        for x in range(30, 50):
            page[y][x] = (y - origin) * 20 + (x - 30)
def crop(x, y, width, height):
    return [row[x:x+width] for row in page[y:y+height]]
top = crop(30, 20, 20, 10)
bottom = crop(30, 150, 20, 10)
assert top == bottom
zoom = [[pixel for pixel in row for _ in range(2)] for row in top for _ in range(2)]
assert (len(zoom[0]), len(zoom)) == (40, 20)

chapter = ROOT / "course/chapters/20.md"
raw = chapter.read_bytes()
text = raw.decode()
section = re.search(r"^## 20\.11 .*?(?=^## |\Z)", text, re.M | re.S).group()
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/20.11.md").read_bytes()
assert section.encode() == frozen
card = re.search(r"^## 20\.13 .*?(?=^## |\Z)", text, re.M | re.S).group()
labels = ["有沒有清楚可見的中文字", "真實照片裡指定區域的逐字抄寫", "真實文字的指定讀序與分行"]
rows = [next(line for line in card.splitlines() if line.startswith("| " + label + " |")) for label in labels]
assert len(rows) == 3

paths = [
    "docs/course-review-20261019/freeze-01/sources/20.11.md",
    "docs/course-review-20261019/freeze-04.json",
    "course/figures/natural_reading_order.svg",
    "course/chapters/20.md",
    "docs/review-tools/section_facts.py",
]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-11/verify.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; Python 標準函式庫"},
    "result": "全部斷言通過",
    "strings": {"reference": facts(reference), "reversed_lines": facts(reversed_lines), "no_newline": facts(no_newline)},
    "crop": {"page_size": [100, 200], "origins": [[30, 20], [30, 150]], "crop_size": [20, 10], "identical_crop_pixels": top == bottom, "zoom_size": [len(zoom[0]), len(zoom)], "scope": "透明的離散幾何範例；放大顯示不表示增加原始資訊或 OCR 準確率"},
    "capability_card": {"headings_found": labels, "rows": rows, "scope": "只核對用途分類；不複審相鄰小節的模型得分"},
    "current_equals_frozen": section.encode() == frozen,
    "input_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
