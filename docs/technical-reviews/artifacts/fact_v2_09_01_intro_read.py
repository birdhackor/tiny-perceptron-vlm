"""Persist the exact chapter introduction already read by this reviewer."""

import hashlib
import json
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/technical-reviews/artifacts"
INTRO_SHA = "519303bb81a71983bf537e42f4f0848cdec58b35a5c30bc6e42bbbd7fdce3f5c"
LESSON_SHA = "7aa3582fdbdd0be9646a049fa8ccb4b9dd4601aae4937fe917cfaa616ed9c42d"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    source = ROOT / "course/chapters/09.md"
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    headings = list(re.finditer(r"^## .+$", text, re.M))
    assert headings[0][0] == "## 9.1 想讓模型學到哪些行為？"
    intro = text[: headings[0].start()].encode("utf-8")
    lesson = text[headings[0].start() : headings[1].start()].encode("utf-8")
    assert sha256(intro) == INTRO_SHA
    assert sha256(lesson) == LESSON_SHA
    assert b"\r" not in raw
    report = ROOT / "docs/technical-reviews/9.1.json"
    previous = BASE / "fact_v2_09_01_pre_intro_report.json"
    if not previous.exists():
        current = json.loads(report.read_text())
        assert "intro_sha256" not in current
        assert current["reviewer_task"] == "/root/integration_technical_coordinator/fact_v2_09_01"
        previous.write_bytes(report.read_bytes())
    snapshot = BASE / "fact_v2_09_01_intro_source.md"
    snapshot.write_bytes(intro)
    receipt = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_09_01_intro_read.py",
        "result": "Complete raw prefix before the first ## persisted; matches the introduction personally read; lesson 9.1 matches the previously reviewed body hash",
        "environment": {"python": platform.python_version(), "device": "CPU filesystem inspection"},
        "reviewer_task": "/root/integration_technical_coordinator/fact_v2_09_01",
        "reviewed_on": "2026-10-04",
        "source": "course/chapters/09.md",
        "source_file_sha256": sha256(raw),
        "first_heading": headings[0][0],
        "intro_snapshot": str(snapshot.relative_to(ROOT)),
        "intro_sha256": sha256(intro),
        "intro_utf8_bytes": len(intro),
        "previously_reviewed_lesson_sha256": sha256(lesson),
        "pre_intro_report_snapshot": str(previous.relative_to(ROOT)),
        "pre_intro_report_sha256": sha256(previous.read_bytes()),
        "program_sha256": sha256(Path(__file__).read_bytes()),
    }
    (BASE / "fact_v2_09_01_intro_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(intro.decode("utf-8"), end="")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
