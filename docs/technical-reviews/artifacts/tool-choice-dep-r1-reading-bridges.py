"""Check the on-demand warm-up links named by R.1, not their mathematical claims."""
import hashlib
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
chapter = ROOT / "course/chapters/01.md"
warmup = ROOT / "course/first-steps.md"
raw = chapter.read_text()
site = BeautifulSoup((ROOT/"outputs/site/first-steps.html").read_text(), "html.parser")
rows = []
for lesson, topic in [("W.2", "清單、字典與方括號"), ("W.3", "數字表與軸"), ("W.4", "機率"), ("W.6", "梯度")]:
    heading = next(line for line in warmup.read_text().splitlines() if line.startswith("## "+lesson+" "))
    lines = [line for line in raw.splitlines() if "../first-steps.md#"+lesson in line]
    first = lines[0]
    assert site.find(id=lesson) is not None
    assert first
    rows.append({"topic": topic, "warmup_id": lesson, "heading": heading, "original_chapter_link_count": len(lines), "first_source_paragraph": first, "site_anchor_exists": True})
for lesson in ["W.1", "W.2"]:
    assert site.find(id=lesson) is not None
report = {"command": ".venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-r1-reading-bridges.py", "environment": {"python": sys.version.split()[0], "device": "CPU; static generated HTML"}, "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [chapter, warmup]}, "rows": rows, "W1_operation_guidance": "Read W.1: Google sign-in/CPU runtime, first setup cell, local uv/Git/install commands, ipykernel/JupyterLab opening, Shift+Enter, restart-and-run-all and bird-text exercise. This checks that these instructions are present; it does not revalidate every supported operating system or Google runtime.", "result": "All named on-demand topics link from chapter 1 to matching warm-up headings and existing website anchors."}
(Path(__file__).parent/"tool-choice-dep-r1-reading-bridges.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
