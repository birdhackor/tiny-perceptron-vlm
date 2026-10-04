"""Check the optional TRAINING link and unchanged CPU code after re-reading 20.8."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import build_course, export_course

BASE = ROOT / "docs/technical-reviews/artifacts"
source = ROOT / "course/chapters/20.md"
prior = (BASE / "natural-r2-v3-chapter20-source.md").read_text(encoding="utf-8")
current = source.read_text(encoding="utf-8")
def digest(data):
    return hashlib.sha256(data).hexdigest()
def section(raw, number):
    start = re.search(r"^## " + re.escape(number) + r" .+$", raw, re.M).start()
    nxt = re.search(r"^## ", raw[start + 1:], re.M)
    return raw[start:start + 1 + nxt.start() if nxt else len(raw)]
old_section, new_section = section(prior, "20.8"), section(current, "20.8")
assert prior[:prior.index("## 20.8 ")] == current[:current.index("## 20.8 ")]
assert re.findall(r"```bash\n(.*?)```", prior, re.S) == re.findall(r"```bash\n(.*?)```", current, re.S)
assert re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", prior) == re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", current)
def code_cells(raw):
    title = raw.splitlines()[0].removeprefix("## 20.8 ")
    notebook = build_course.notebook("20.8", title, raw.split("\n", 1)[1], source)
    return ["".join(cell["source"]).encode("utf-8") for cell in notebook["cells"] if cell["cell_type"] == "code"]
old_cells, current_cells = code_cells(old_section), code_cells(new_section)
assert old_cells == current_cells
v3 = json.loads((BASE / "natural-r2-v3-execution.json").read_text())
assert [digest(cell) for cell in current_cells] == [row["sha256"] for row in v3["code_cells"]]
assert v3["current_20.8_execution"]["status"] == "passed"

index = json.loads((ROOT / "course/lesson-index.json").read_text())
targets = {(ROOT / p).resolve(): n + ".md" for n, p in export_course.DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    path = (ROOT / item["source"]).resolve()
    targets[path] = "chapter-" + path.stem + ".md"
    lesson_targets.setdefault(path, {})[item["id"]] = item["id"] + ".md"
guide = (source.parent / "../../docs/natural-assistant/TRAINING.md").resolve()
assert guide.is_file() and guide not in targets and guide not in lesson_targets
guide_text = guide.read_text(encoding="utf-8")
assert "從固定的圖文底座建立一份新的LoRA" in guide_text
assert "先驗證自己的模型，再固定版本做最後測試" in guide_text
converted = export_course.reading_markdown(new_section, source, targets, lesson_targets, "main")
url = "https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/natural-assistant/TRAINING.md"
assert "[LoRA訓練實作](" + url + ")" in converted
reading_source = ROOT / "course/README.md"
body = section(reading_source.read_text(encoding="utf-8"), "R.2")
converted_reading = export_course.reading_markdown(body, reading_source, targets, lesson_targets, "main")
links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", converted_reading)
assert links == [row["published"] for row in json.loads((BASE / "natural-r2-v2-links.json").read_text())["links"]]
inventory = export_course.build_inventory(ROOT, index, export_course.DOCUMENTS, export_course.home_introduction(index))
assert not any(row.get("source") == "docs/natural-assistant/TRAINING.md" for row in inventory["pages"])

result = {
    "whole_chapter20_sha256": digest(source.read_bytes()),
    "current_20.8_section_sha256": digest(new_section.encode()),
    "previous_20.8_section_sha256": digest(old_section.encode()),
    "current_code_cells": [{"index": i, "sha256": digest(cell), "equal_to_last_executed_v3_bytes": True} for i, cell in enumerate(current_cells)],
    "history_binding": {"v3_execution_path": "docs/technical-reviews/artifacts/natural-r2-v3-execution.json", "sha256": digest((BASE / "natural-r2-v3-execution.json").read_bytes()),
        "note": "No new CPU model run needed: actual generated current program bytes are exactly the v3 executed bytes. Historical v3 section-prose hash remains unchanged in its original JSON."},
    "training_document": {"path": guide.relative_to(ROOT).as_posix(), "sha256": digest(guide.read_bytes()), "exists": True,
        "literal_route_checked": "Intro states new LoRA from fixed base; section 6 states validation before selecting/fixing own version and final test. No GPU/API/quality validation claimed."},
    "exported_training_url": url,
    "fallback": "repository GitHub blob; not a canonical lesson/page",
    "lesson_index_count": len(index), "canonical_pages": len(inventory["pages"]),
    "R.2_sha256": digest(body.encode()), "R.2_links_count": len(links), "R.2_links_unchanged": True,
    "unchanged_20.1_to_20.7_bash_commands_and_svg_references": True,
    "scope": "Same owner read complete current 20.8 and actual diff, plus training guide only for file/topic/ordering. Tests source bytes and real exporter fallback. No GPU, installation, Git operation, model download, training, quality assurance or extra review stage."
}
(BASE / "natural-r2-v4-execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"code_cells_match_last_execution": len(current_cells), "new_link": url, "R.2_links": len(links), "lessons": len(index), "canonical_pages": len(inventory["pages"])}))
