"""核對字串範例、正規化的資訊損失、裁切座標及能力卡用途分類。"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import platform
import re
import sys
import subprocess

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
    "docs/course-review-20261019/freeze-06.json",
    "course/figures/natural_reading_order.svg",
    "course/chapters/20.md",
    "docs/review-tools/section_facts.py",
]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-11/verify-current-01.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; Python 標準函式庫"},
    "result": "全部斷言通過",
    "strings": {"reference": facts(reference), "reversed_lines": facts(reversed_lines), "no_newline": facts(no_newline)},
    "crop": {"page_size": [100, 200], "origins": [[30, 20], [30, 150]], "crop_size": [20, 10], "identical_crop_pixels": top == bottom, "zoom_size": [len(zoom[0]), len(zoom)], "scope": "透明的離散幾何範例；放大顯示不表示增加原始資訊或 OCR 準確率"},
    "capability_card": {"headings_found": labels, "rows": rows, "scope": "只核對用途分類；不複審相鄰小節的模型得分"},
    "current_equals_frozen": section.encode() == frozen,
    "input_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
}

original_report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_20-11/technical-20-11-original-01.json"
original_report = json.loads(original_report_path.read_bytes())
assert sha256(original_report_path.read_bytes()).hexdigest() == "43d95ce95d47783e7b795d836bf162a8e671e5daf2d4dc75ac9629bb3511c3c1"
assert original_report["issues"] == []
old_artifact_checks = []
for item in original_report["artifacts"]:
    current_hash = sha256((ROOT / item["path"]).read_bytes()).hexdigest()
    assert current_hash == item["sha256"]
    old_artifact_checks.append({"path": item["path"], "sha256": current_hash, "unchanged": True})

def git_bytes(*args):
    command = ["git", *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return completed.stdout

old = git_bytes("show", "5ef56bc756f3:course/chapters/20.md")
new = git_bytes("show", "6a7aa1de5227:course/chapters/20.md")
assert sha256(old).hexdigest() == "c27ddfc292f9d23cb044fc9370e01ed86831f6212270d2c4a150ed35737bb9bf"
assert sha256(raw).hexdigest() == "57b74075aff027a450f5a8cdf453955028b8cef0f124f70b5cbfba86ec21d425"
assert new == raw

def sections(blob):
    headings = list(re.finditer(rb"(?m)^## ([0-9]+\.[0-9]+) [^\r\n]+", blob))
    return {m[1].decode(): blob[m.start():headings[i+1].start() if i+1 < len(headings) else len(blob)] for i,m in enumerate(headings)}

old_sections, new_sections = sections(old), sections(new)
assert old_sections["20.11"] == new_sections["20.11"] == frozen
changed_sections = [key for key in old_sections if old_sections[key] != new_sections.get(key)]
old_card = old_sections["20.13"].decode()
old_rows = [next(line for line in old_card.splitlines() if line.startswith("| " + label + " |")) for label in labels]
assert rows == old_rows
prior_probe_path = ROOT / "docs/technical-reviews/artifacts/review-20261019-20-11/verification.json"
prior_probe = json.loads(prior_probe_path.read_bytes())
for field in ("strings", "crop", "capability_card"):
    assert result[field] == prior_probe[field]
png_path = ROOT / "outputs/course-review-20261019/figure-renders-04/a2e552443a5458386a8aabc26af43ae110d1a2256d3f3d2db8184f87f14cea98.png"
png_hash = sha256(png_path.read_bytes()).hexdigest()
assert png_hash == "df8a6f41ebd1df3cb301c817f288542f45592faf25067b8336bbf0b954a5f43a"
assert result["input_sha256"]["docs/course-review-20261019/freeze-06.json"] == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
result["change_review"] = {
    "diff_command": "git diff 5ef56bc756f3 6a7aa1de5227 -- course/chapters/20.md",
    "diff_exit_code": 0,
    "diff_stdout": git_bytes("diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/chapters/20.md").decode(),
    "old_chapter_sha256": sha256(old).hexdigest(),
    "current_chapter_sha256": sha256(raw).hexdigest(),
    "current_matches_new_commit": new == raw,
    "changed_sections": changed_sections,
    "target_changed": False,
    "target_old_current_frozen_equal": old_sections["20.11"] == new_sections["20.11"] == frozen,
    "referenced_text_task_rows_unchanged": rows == old_rows,
    "original_issue_count": len(original_report["issues"]),
    "old_artifacts": old_artifact_checks,
    "reexecuted_results_match_prior": True,
    "figure_viewed": {"path": str(png_path.relative_to(ROOT)), "sha256": png_hash, "method": "本次以 view_image 實看；原位、箭頭及兩組轉寫與原文一致"},
    "scope": "比較整章差異以判斷本頁及所引用證據是否受影響；不將其他頁的新句子納入本頁的技術通過結論。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))

