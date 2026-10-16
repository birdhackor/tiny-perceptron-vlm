"""Informed 19.12 callback: source delta, retained pins and navigation only."""

import difflib
import hashlib
import json
import platform
import re
import urllib.request
from pathlib import Path

from scripts.reading_time import lesson_slices

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old_path = ROOT / "docs/course-revision-20261016/reviews/technical-19.12-original.json"
old = json.loads(old_path.read_text())
freeze_path = ROOT / "docs/course-revision-20261016/freeze-18.json"
freeze = json.loads(freeze_path.read_text())
pages = {p["page_id"]: p for p in freeze["pages"]}
source = ROOT / "course/chapters/19.md"
sections = lesson_slices(source.read_bytes().decode())
body = sections["19.12"]
before = (ROOT / "docs/course-revision-20261016/freeze-03/sources/19.12.md").read_bytes().decode()
old_route = "接著從[19.14](19.md#19.14)讀共同切分，再完成Looped解題器、機率判斷器與Auto的路線。"
new_route = "接著在[19.13](19.md#19.13)對照三個成品的用途，再從[19.14](19.md#19.14)讀共同切分，完成Looped解題器、機率判斷器與Auto的路線。"
assert before.count(old_route) == 1
assert before.replace(old_route, new_route) == body
assert hashlib.sha256(before.encode()).hexdigest() == old["source_sha256"]
assert hashlib.sha256(body.encode()).hexdigest() == pages["19.12"]["source_sha256"]
assert (ROOT / pages["19.12"]["snapshot"]).read_bytes() == body.encode()
pins = []
for entry in old["sources"]:
    if entry["kind"] != "repository_code":
        continue
    actual = digest(ROOT / entry["path"])
    assert actual == entry["sha256"]
    pins.append({"id": entry["id"], "path": entry["path"], "sha256": actual, "same_as_original": True})
artifacts = []
for entry in old["artifacts"]:
    actual = digest(ROOT / entry["path"])
    assert actual == entry["sha256"]
    artifacts.append({"id": entry["id"], "path": entry["path"], "sha256": actual, "same_as_original": True})
dependencies = []
for lesson in ["19.13", "19.14"]:
    value = sections[lesson]
    sha = hashlib.sha256(value.encode()).hexdigest()
    assert sha == pages[lesson]["source_sha256"]
    assert (ROOT / pages[lesson]["snapshot"]).read_bytes() == value.encode()
    dependencies.append({"lesson_id": lesson, "source_sha256": sha, "heading": value.splitlines()[0]})
assert "三個成品，分別拿來做什麼" in sections["19.13"]
assert all(term in sections["19.13"] for term in ["MoE v2", "Looped Dense", "Jev", "Auto"])
assert "家族" in sections["19.14"] and "全部繼承相同split" in sections["19.14"]
preview = []
for lesson in ["19.12", "19.13", "19.14"]:
    url = "http://127.0.0.1:8898/" + lesson + ".html"
    with urllib.request.urlopen(url, timeout=15) as response:
        rendered = response.read().decode()
        assert response.status == 200
        preview.append({"url": url, "status": response.status, "html_sha256": hashlib.sha256(rendered.encode()).hexdigest()})
    if lesson == "19.12":
        match = re.search(r"接著在.*?完成Looped解題器、機率判斷器與Auto的路線。", rendered, re.S)
        assert match
        paragraph = match.group(0)
        assert re.search(r'href="[^"]*19\.13\.html[^"]*"', paragraph)
        assert re.search(r'href="[^"]*19\.14\.html[^"]*"', paragraph)
        assert paragraph.index("19.13.html") < paragraph.index("19.14.html")
        preview[-1]["changed_transition_html"] = paragraph

result = {
    "reviewer_task": "/root/tech_20261016_19_12",
    "current_context": "informed",
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/19.12/check_callback.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; byte, metadata and local HTTP checks only"},
    "original_report": {"path": str(old_path.relative_to(ROOT)), "sha256": digest(old_path)},
    "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path)},
    "source_sha256": pages["19.12"]["source_sha256"],
    "source_file_sha256": digest(source),
    "diff": "".join(difflib.unified_diff(before.splitlines(True), body.splitlines(True))),
    "all_other_lesson_bytes_unchanged": True,
    "repository_source_pins": pins,
    "inherited_artifacts": artifacts,
    "dependency_sections": dependencies,
    "strict_preview": preview,
    "limits": "This callback reread current 19.12 and linked 19.13/19.14, verified the sole transition delta and every retained repository source/artifact pin, and checked canonical HTTP navigation. The initial exploratory /19.12/ URL returned 404 because this preview uses .html routes; canonical .html pages all passed. Did not rerun prior arithmetic/helper checks, Notebook kernel, training, final tests, GPU, HF downloads, or visual layout inspection.",
    "result": "PASS: sole transition delta, current freeze-18 bytes, all 19 repository pins, 3 inherited artifacts and 19.13-before-19.14 canonical navigation verified.",
}
(HERE / "callback-execution-01.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(result["result"])
