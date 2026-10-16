"""Late curriculum-link dependency callback; read-only and offline."""

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from scripts.export_course import DOCUMENTS, reading_markdown
from scripts.reading_time import chapter_introduction


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


previous_path = "docs/course-revision-20261016/reviews/technical-order-chapter-19-callback-01.json"
previous = read(previous_path)
f18_path, f19_path = [f"docs/course-revision-20261016/freeze-{n}.json" for n in (18, 19)]
f18, f19 = [{p["page_id"]: p for p in read(path)["pages"]} for path in (f18_path, f19_path)]
oldpath, newpath = [f["curriculum"]["snapshot"] for f in (f18, f19)]
old, current = [(ROOT / p).read_bytes() for p in (oldpath, "docs/curriculum.md")]
assert current == (ROOT / newpath).read_bytes()
assert sha("docs/curriculum.md") == f19["curriculum"]["source_sha256"]
old_phrase = "課程索引按下表區分「主題初讀」與「按需回讀」"
new_phrase = "課程索引依[閱讀指南](../course/README.md#R.2)區分「主題初讀」與「按需回讀」"
assert old.count(old_phrase.encode()) == 1
assert old.replace(old_phrase.encode(), new_phrase.encode()) == current
oldroute = next(p for p in old.decode().split("\n\n") if "第19章依19.1–19.20順讀" in p)
assert oldroute in current.decode()
intro = chapter_introduction((ROOT / "course/chapters/19.md").read_bytes().decode())
assert intro.encode() == (ROOT / f19["chapter-19"]["snapshot"]).read_bytes()
assert hashlib.sha256(intro.encode()).hexdigest() == f18["chapter-19"]["source_sha256"] == f19["chapter-19"]["source_sha256"] == previous["source_sha256"]
repo_pins = []
for source in previous["sources"]:
    if source["kind"] != "repository_code":
        continue
    observed = sha(source["path"])
    changed = observed != source["sha256"]
    assert changed == (source["id"] == "s-curriculum-routing-current")
    if changed:
        assert source["sha256"] == f18["curriculum"]["source_sha256"]
    repo_pins.append({"source_id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": observed, "changed": changed})
old_artifacts = []
for artifact in previous["artifacts"]:
    assert sha(artifact["path"]) == artifact["sha256"]
    old_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": artifact["sha256"]})
guide = (ROOT / "course/README.md").read_bytes().decode()
assert guide.encode() == (ROOT / f19["course"]["snapshot"]).read_bytes()
section = re.search(r"^## R\.2 .*?(?=^## |\Z)", guide, re.M | re.S)[0]
assert "第一次接觸語言模型，先沿第 1–4 章閱讀" in section
assert "| 想理解的問題 | 建議起點 |" in section
assert "跳讀遇到陌生概念時" in section and "回讀連結" in section
targets = {(ROOT / path).resolve(): name + ".md" for name, path in DOCUMENTS.items()}
converted = reading_markdown(new_phrase, ROOT / "docs/curriculum.md", targets, {}, "main")
converted_guide = reading_markdown(guide, ROOT / "course/README.md", targets, {}, "main")
assert "[閱讀指南](course.md#R.2)" in converted
assert "## 依你想理解的問題選路 {#R.2}" in converted_guide
result = {
    "reviewer_task": "/root/backfill_tech_chapter_19",
    "current_context": "informed",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/chapter-19/dependency-callback-02.py",
    "environment": {"python": platform.python_version(), "device": "cpu", "library": "Python standard library and existing reading/export functions"},
    "result": "all assertions passed",
    "previous_callback": {"path": previous_path, "sha256": sha(previous_path)},
    "unique_dependency_change": {"source": "docs/curriculum.md", "old_snapshot": oldpath, "old_sha256": sha(oldpath), "new_snapshot": newpath, "new_sha256": sha(newpath), "before": old_phrase, "after": new_phrase, "rest_of_file_identical": True},
    "link_target_inspection": {"source_target": "../course/README.md#R.2", "physical_file": "course/README.md", "heading": section.splitlines()[0], "role": "R.2 gives the initial 1–4 chapter route, a topic-to-entry table, and return-to-text guidance when a jump reader needs a prerequisite. It is an appropriate navigation target for distinguishing initial topic reading from on-demand rereading; it does not contain a literal two-category classification table, and the corrected sentence no longer says one exists below.", "converted_link": converted.strip(), "preserved_export_anchor": "R.2"},
    "chapter19_route_paragraph_unchanged": oldroute,
    "chapter19_current_source_sha256": f19["chapter-19"]["source_sha256"],
    "current_previous_repository_pins": repo_pins,
    "previous_artifacts_unchanged": old_artifacts,
    "current_pins": {path: sha(path) for path in [f18_path, f19_path, oldpath, newpath, "docs/curriculum.md", "course/README.md", "scripts/export_course.py", "course/chapters/19.md"]},
    "scope": "Only inspect the unique curriculum navigation phrase, its actual R.2 source target and canonical conversion, unchanged chapter-19 route/source and continuation of this owner's original/callback claims and evidence. No model run, formal test, benchmark, training, HF/network download, reading-time reevaluation, whole-site build or shared technical-seal modification.",
}
out = Path(__file__).with_suffix(".json")
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "changed_repository_pins": [x["source_id"] for x in repo_pins if x["changed"]], "current_curriculum_sha256": sha("docs/curriculum.md"), "output": str(out.relative_to(ROOT))}, ensure_ascii=False))
