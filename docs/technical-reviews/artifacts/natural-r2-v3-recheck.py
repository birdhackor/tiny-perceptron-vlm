"""Recheck R.2 after the two 20.8 file-unit prose changes; preserve v2 outputs."""
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from scripts import build_course, export_course

BASE = ROOT / "docs/technical-reviews/artifacts"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def sections(raw):
    heads = list(re.finditer(r"^## ([A-Z\d]+\.\d+) (.+)$", raw, re.M))
    return {h[1]: {"title": h[2], "body": raw[h.start():heads[i + 1].start() if i + 1 < len(heads) else len(raw)]}
            for i, h in enumerate(heads)}

source = ROOT / "course/chapters/20.md"
current = source.read_text(encoding="utf-8")
prior = (BASE / "natural-r2-v2-chapter20-source.md").read_text(encoding="utf-8")
assert prior[:prior.index("## 20.8 ")] == current[:current.index("## 20.8 ")]
assert re.findall(r"```bash\n(.*?)```", prior, re.S) == re.findall(r"```bash\n(.*?)```", current, re.S)
assert re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", prior) == re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", current)

old_section = sections(prior)["20.8"]
new_section = sections(current)["20.8"]
def cells(section):
    notebook = build_course.notebook("20.8", section["title"], section["body"].split("\n", 1)[1], source)
    return ["".join(cell["source"]).encode("utf-8") for cell in notebook["cells"] if cell["cell_type"] == "code"]
old_cells, new_cells = cells(old_section), cells(new_section)
assert old_cells == new_cells

out = io.StringIO()
namespace = {"__name__": "__main__"}
with contextlib.redirect_stdout(out):
    for number, cell in enumerate(new_cells):
        exec(compile(cell.decode("utf-8"), str(source) + ":20.8:cell" + str(number), "exec"), namespace)

v2 = json.loads((BASE / "natural-r2-v2-execution.json").read_text())
current_hashes = {}
for path in (ROOT / "course/chapters").glob("*.md"):
    for number, section in sections(path.read_text(encoding="utf-8")).items():
        current_hashes[number] = sha(section["body"].encode("utf-8"))
unchanged = [row["lesson"] for row in v2["lessons"]
             if row["section_sha256"] == current_hashes[row["lesson"]]]
changed = [row["lesson"] for row in v2["lessons"]
           if row["section_sha256"] != current_hashes[row["lesson"]]]
assert len(unchanged) == 260 and changed == ["20.8"]
assert all(row["status"] == "passed" for row in v2["lessons"])

guide = ROOT / "course/README.md"
body = sections(guide.read_text(encoding="utf-8"))["R.2"]["body"]
index = json.loads((ROOT / "course/lesson-index.json").read_text())
targets = {(ROOT / p).resolve(): n + ".md" for n, p in export_course.DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    path = (ROOT / item["source"]).resolve()
    targets[path] = "chapter-" + path.stem + ".md"
    lesson_targets.setdefault(path, {})[item["id"]] = item["id"] + ".md"
published = export_course.reading_markdown(body, guide, targets, lesson_targets, "main")
actual_links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", published)
expected_links = [row["published"] for row in json.loads((BASE / "natural-r2-v2-links.json").read_text())["links"]]
assert actual_links == expected_links and len(actual_links) == 44
assert sha(body.encode()) == json.loads((BASE / "natural-r2-v2-site.json").read_text())["source_sha256"]

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU"},
    "current_whole_chapter20_sha256": sha(source.read_bytes()),
    "current_20.8_section_sha256": sha(new_section["body"].encode()),
    "previous_20.8_section_sha256": sha(old_section["body"].encode()),
    "code_cells": [{"index": i, "sha256": sha(cell), "identical_to_v2": True} for i, cell in enumerate(new_cells)],
    "current_20.8_execution": {"status": "passed", "stdout": out.getvalue(), "namespace": "fresh dict for current 20.8"},
    "v2_execution_binding": {"historical_file": "docs/technical-reviews/artifacts/natural-r2-v2-execution.json", "historical_file_sha256": sha((BASE / "natural-r2-v2-execution.json").read_bytes()),
        "current_unchanged_source_sections": len(unchanged), "changed_source_sections": changed,
        "note": "Keep v2 raw bytes and source hashes. Its 20.8 prose hash is historical; latest 20.8 was re-executed above. The other 260 sections remain exact-current."},
    "unchanged_bash_commands_and_svg_references": True,
    "R.2_source_sha256": sha(body.encode()),
    "current_exported_links": actual_links,
    "links_match_previous_real_Zensical_fixture": True,
    "scope": "Only R.2 navigation and CPU short-program source binding. No installation, Git operation, natural-model download, GPU, or full UI reproduction. Two 20.8 upload-unit paragraphs do not change the R.2 routes or short Python cells."
}
output = BASE / "natural-r2-v3-execution.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"current_20.8": "passed", "code_cells_unchanged": len(new_cells), "v2_exact_current_sections": len(unchanged), "links": len(actual_links), "source_sha256": result["current_whole_chapter20_sha256"]}))
