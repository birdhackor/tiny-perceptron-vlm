"""Narrow chapter-19 ordering callback; no training, test generation or network."""

import hashlib
import json
import platform
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from scripts.check_lesson_order import audit
from scripts.export_course import DOCUMENTS, reading_markdown
from scripts.reading_time import chapter_introduction, lesson_slices


def raw(path):
    return (ROOT / path).read_bytes()


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def data(path):
    return json.loads(raw(path))


prior_path = "docs/course-revision-20261016/reviews/technical-chapter-19-backfill-original-01.json"
prior = data(prior_path)
f16_path = "docs/course-revision-20261016/freeze-16.json"
f18_path = "docs/course-revision-20261016/freeze-18.json"
old = {p["page_id"]: p for p in data(f16_path)["pages"]}
new = {p["page_id"]: p for p in data(f18_path)["pages"]}
text = raw("course/chapters/19.md").decode("utf-8")
sections = lesson_slices(text)
expected = [f"19.{i}" for i in range(1, 21)]
assert list(sections) == expected
intro = chapter_introduction(text)
assert intro.encode() == raw(new["chapter-19"]["snapshot"])
assert hashlib.sha256(intro.encode()).hexdigest() == new["chapter-19"]["source_sha256"]
old_intro = raw(old["chapter-19"]["snapshot"]).decode()
old_sentence = "先讀19.13的總覽，再沿19.1–19.12看既有助理，接著由19.14讀共同資料、兩個新增產品與Auto整合。"
new_sentence = "本章依節號順讀：19.1–19.12完成既有助理的資料、訓練與驗收；19.13對照三個成品的用途，再由19.14–19.20讀共同資料、兩個新增產品與Auto整合。"
assert old_intro.replace(old_sentence, new_sentence) == intro
rows, figure_pins = [], {}
for lesson_id in expected:
    a, b = old[lesson_id], new[lesson_id]
    current = sections[lesson_id].encode()
    same16 = current == raw(a["snapshot"])
    assert current == raw(b["snapshot"])
    assert hashlib.sha256(current).hexdigest() == b["source_sha256"]
    assert same16 == (lesson_id != "19.12")
    assert a["figures_sha256"] == b["figures_sha256"]
    for path, digest in b["figures_sha256"].items():
        assert sha(path) == digest
        figure_pins[path] = digest
    assert sha(b["notebook"]) == b["notebook_sha256"]
    if lesson_id != "19.12":
        assert a["notebook_sha256"] == b["notebook_sha256"]
    rows.append({"lesson_id": lesson_id, "source_sha256": b["source_sha256"], "same_freeze16": same16, "same_freeze18_snapshot": True, "notebook_current_matches_freeze18": True, "notebook_same_freeze16": a["notebook_sha256"] == b["notebook_sha256"]})
before12 = raw(old["19.12"]["snapshot"]).decode()
after12 = sections["19.12"]
old_transition = "接著從[19.14](19.md#19.14)讀共同切分，再完成Looped解題器、機率判斷器與Auto的路線。"
new_transition = "接著在[19.13](19.md#19.13)對照三個成品的用途，再從[19.14](19.md#19.14)讀共同切分，完成Looped解題器、機率判斷器與Auto的路線。"
assert before12.replace(old_transition, new_transition) == after12
assert re.findall(r"```.*?```", before12, re.S) == re.findall(r"```.*?```", after12, re.S)
notebook = data("notebooks/19/19.12.ipynb")
markdown = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
notebook_transition = new_transition.replace("(19.md#19.13)", "(https://birdhackor.github.io/tiny-perceptron-vlm/19.13.html)").replace("(19.md#19.14)", "(https://birdhackor.github.io/tiny-perceptron-vlm/19.14.html)")
assert notebook_transition in markdown and "接著從[19.14]" not in markdown

current_prior_pins = []
for source in prior["sources"]:
    if source["kind"] != "repository_code":
        continue
    observed = sha(source["path"])
    assert observed == source["sha256"]
    current_prior_pins.append({"source_id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": observed, "same_bytes": True})
prior_artifacts = []
original_primary_pins = {}
for artifact in prior["artifacts"]:
    observed = sha(artifact["path"])
    assert observed == artifact["sha256"]
    prior_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": observed})
    if artifact["kind"] == "execution":
        for path, digest in data(artifact["path"]).get("source_sha256", {}).items():
            assert sha(path) == digest
            original_primary_pins[path] = digest

index = data("course/lesson-index.json")
indexed = [x for x in index if x["source"] == "course/chapters/19.md"]
assert [x["id"] for x in indexed] == expected
for row in indexed:
    assert Path(row["notebook"]).name == row["id"] + ".ipynb"
    assert sections[row["id"]].splitlines()[0] == "## " + row["id"] + " " + row["title"]


def navpaths(value):
    if isinstance(value, str):
        return [value]
    values = value.values() if isinstance(value, dict) else value
    return [p for child in values for p in navpaths(child)]


navigation = tomllib.loads(raw("zensical.toml").decode())
nav19 = [x.removesuffix(".md") for x in navpaths(navigation["project"]["nav"]) if re.fullmatch(r"(?:chapter-19|19\.\d+)\.md", x)]
assert nav19 == ["chapter-19", *expected]
route = next(x for x in data("course/reading-time-routes.json")["routes"] if x["id"] == "capstone")
assert route["page_ids"] == ["chapter-19", *expected]
lesson_list = re.findall(r"^\| (19\.\d+) .*", raw("course/lessons.md").decode(), re.M)
assert lesson_list == expected
assert [p["page_id"] for p in data(f18_path)["pages"] if p["page_id"] in expected] == expected
order_result = audit(ROOT)
assert order_result["failures"] == []

targets = {(ROOT / path).resolve(): name + ".md" for name, path in DOCUMENTS.items()}
targets.update({(ROOT / item["source"]).resolve(): "chapter-" + Path(item["source"]).stem + ".md" for item in index})
lesson_targets = {}
for item in index:
    lesson_targets.setdefault((ROOT / item["source"]).resolve(), {})[item["id"]] = item["id"] + ".md"
source = ROOT / "course/chapters/19.md"
converted_intro = reading_markdown(intro, source, targets, lesson_targets, "main")
converted12 = reading_markdown(after12, source, targets, lesson_targets, "main")
assert "[19.12](19.12.md)" in converted_intro and "[19.13](19.13.md)" in converted_intro
assert "[19.13](19.13.md)" in converted12 and "[19.14](19.14.md)" in converted12
first11 = "".join(sections[f"19.{i}"] for i in range(1, 12))
assert "19.13" not in first11 and "Looped" not in first11 and "Jev" not in first11

extra_sources = ["course/chapters/19.md", "course/lesson-index.json", "course/lessons.md", "course/reading-time-routes.json", "zensical.toml", "scripts/check_lesson_order.py", "scripts/reading_time.py", "scripts/export_course.py", "course/README.md", "docs/curriculum.md", f16_path, f18_path, new["chapter-19"]["snapshot"], new["19.12"]["snapshot"], new["19.13"]["snapshot"], new["19.14"]["snapshot"], "notebooks/19/19.12.ipynb"]
output = {
    "reviewer_task": "/root/backfill_tech_chapter_19",
    "current_context": "informed",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/chapter-19/order-callback-01.py",
    "environment": {"python": platform.python_version(), "device": "cpu", "libraries": "Python standard library; existing exporter/reading_time functions"},
    "result": "all assertions passed",
    "prior_report": {"path": prior_path, "sha256": sha(prior_path)},
    "current_intro": new["chapter-19"],
    "intro_only_change": {"before": old_sentence, "after": new_sentence},
    "chapter_sections": rows,
    "unchanged_chapter_figures": figure_pins,
    "transition19_12_only_change": {"before": old_transition, "after": new_transition, "code_blocks_same": True, "current_notebook_markdown_has_new_transition": True},
    "current_prior_repository_pins": current_prior_pins,
    "prior_artifacts_unchanged": prior_artifacts,
    "original_primary_source_pins_unchanged": original_primary_pins,
    "order": {"authored_source": list(sections), "index": [x["id"] for x in indexed], "nav": nav19, "capstone": route["page_ids"], "lesson_list": lesson_list},
    "check_lesson_order_actual_result": order_result,
    "link_conversion": {"intro_targets": re.findall(r"\[[^]]+\]\(([^)]+)\)", converted_intro), "transition_targets": re.findall(r"\[[^]]+\]\(([^)]+)\)", converted12)[-3:]},
    "dependency_inspection": {"19.1_to_19.12": "Personally inspected current intro, actual diff, early A presentation and section-level dependencies: 19.1 introduces the finite MoE assistant and input roles; 19.2 introduces experts in this product; first eleven sections do not cite or require 19.13/B/C product names. 19.12 closes the A evaluation before introducing the next products.", "19.12_to_19.14": "Personally read full current 19.12/19.13/19.14. New final transition leads to the unchanged A/B/C purpose comparison, then shared B/C/Auto family split. A/B/C labels are defined in 19.13 before 19.14 uses them.", "19.15_to_19.20": "Inspected each current opening and explicit link dependencies: B task in 19.15, C contract in 19.16, Auto composition in 19.17, comparison in 19.18, download identity in 19.19 and final acceptance in 19.20 retain those preceding definitions. No model/test fact or identity is inferred from file position."},
    "current_extra_source_pins": {path: sha(path) for path in extra_sources},
    "scope": "Narrow order/source-dependency callback. Original seven substantive claims use my preserved original evidence, after verifying all its referenced repository bytes and original artifacts are unchanged. No model execution, optimizer, training, formal-test generation, benchmark, new HF download, network action, broad score recalculation or reading-time estimate performed. No new visual/phone-layout approval claimed.",
}
out = Path(__file__).with_suffix(".json")
assert not out.exists()
out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": output["result"], "source_sections": len(rows), "unchanged_prior_repository_sources": len(current_prior_pins), "order_failures": order_result["failures"], "output": str(out.relative_to(ROOT))}, ensure_ascii=False))
