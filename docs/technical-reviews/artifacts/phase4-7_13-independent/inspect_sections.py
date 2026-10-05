"""Capture exact current original sections and inspect the disputed cross-reference."""
import hashlib
import json
import re
from pathlib import Path

A = Path(__file__).resolve().parent
ROOT = A.parents[3]


def section(path, lesson):
    raw = path.read_bytes()
    raw.decode("utf-8")
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headings) if h[0].startswith(f"## {lesson} ".encode()))
    start = headings[index].start()
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    body = raw[start:end]
    return body, raw[:start].count(b"\n") + 1


captured = {}
for filename, lesson in [("course/chapters/07.md", "7.12"), ("course/chapters/07.md", "7.13"), ("course/training.md", "T.4")]:
    raw, line = section(ROOT / filename, lesson)
    (A / f"current-{lesson}.md").write_bytes(raw)
    captured[lesson] = {"source": filename + "#" + lesson, "section_first_line": line,
                        "raw_utf8_sha256": hashlib.sha256(raw).hexdigest(),
                        "svg_references": re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", raw.decode())}
assert (A / "current-7.13.md").read_bytes() == (A / "section.md").read_bytes()
current_7_12 = (A / "current-7.12.md").read_text()
current_7_13 = (A / "current-7.13.md").read_text()
captured["cross_reference"] = {
    "quote": "也就是[7.12](#7.12)那條從零練900次的模型",
    "quote_is_current": "也就是[7.12](#7.12)那條從零練900次的模型" in current_7_13,
    "7.12_mentions_900": "900" in current_7_12,
    "7.12_explicit_scope_quotes": [q for q in ["沒有訓練模型或產生答題成績", "本節只示範材料與切分，沒有新訓練結果"] if q in current_7_12],
    "interpretation": "The actual 900-step direct-SFT experiment is true, but the current linked lesson no longer describes that model branch.",
}
(A / "current-section-inspection.json").write_text(json.dumps(captured, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(captured, ensure_ascii=False, indent=2))
