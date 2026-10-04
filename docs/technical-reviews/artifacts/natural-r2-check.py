"""CPU evidence for R.2: independent short examples and publishing links."""
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
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import matplotlib
import matplotlib.pyplot as plt
from scripts import build_course, export_course

PREFIX = ROOT / "docs/technical-reviews/artifacts/natural-r2-"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(name, value):
    Path(str(PREFIX) + name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

def sections(path):
    raw = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## ([A-Z\d]+\.\d+) (.+)$", raw, re.M))
    return {
        h[1]: {
            "title": h[2],
            "body": raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)],
            "line": raw[:h.start()].count("\n") + 1,
        }
        for i, h in enumerate(headings)
    }

source = ROOT / "course/README.md"
body = sections(source)["R.2"]["body"]
index = json.loads((ROOT / "course/lesson-index.json").read_text())
targets = {(ROOT / p).resolve(): n + ".md" for n, p in export_course.DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    path = (ROOT / item["source"]).resolve()
    targets[path] = "chapter-" + path.stem + ".md"
    lesson_targets.setdefault(path, {})[item["id"]] = item["id"] + ".md"
converted = export_course.reading_markdown(body, source, targets, lesson_targets, "main")
original_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body)
converted_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", converted)
assert len(original_links) == len(converted_links)
link_rows, excerpts = [], {}
for (label, target), (_, actual) in zip(original_links, converted_links, strict=True):
    relative, _, lesson = target.partition("#")
    path = (source.parent / relative).resolve()
    section = sections(path)[lesson]
    expected = lesson_targets[path][lesson] if path in lesson_targets else targets[path] + "#" + lesson
    assert actual == expected, (target, actual, expected)
    if path not in lesson_targets:
        page = export_course.reading_markdown(section["body"], path, targets, lesson_targets, "main")
        assert "{#" + lesson + "}" in page
    link_rows.append({"label": label, "original": target, "published": actual,
                      "destination_title": section["title"], "line": section["line"]})
    excerpts[target] = {"path": path.relative_to(ROOT).as_posix(),
                       "file_sha256": digest(path.read_bytes()), **section}
for relative, numbers in {
    "chapters/01.md": ["1.3", "1.4", "1.5", "1.6", "1.10", "1.12"],
    "chapters/13.md": ["13.11", "13.12", "13.13", "13.14", "13.15", "13.16"],
    "chapters/20.md": ["20.5", "20.6", "20.7"],
}.items():
    path = source.parent / relative
    inventory = sections(path)
    for number in numbers:
        excerpts[relative + "#" + number] = {
            "path": path.relative_to(ROOT).as_posix(), "file_sha256": digest(path.read_bytes()), **inventory[number]
        }
save("route-source.json", excerpts)
save("links.json", {"source_sha256": digest(body.encode()), "links": link_rows,
                     "converted_heading": converted.splitlines()[0], "all_links_verified": True})

torch.set_num_threads(1)
results = []
for path in sorted((ROOT / "course/chapters").glob("*.md")):
    inventory = sections(path)
    for lesson, section in inventory.items():
        # Build the actual current author's short code without writing notebooks.
        section_body = section["body"].split("\n", 1)[1]
        notebook = build_course.notebook(lesson, section["title"], section_body, path)
        namespace = {"__name__": "__main__"}
        out = io.StringIO()
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        error = None
        try:
            with contextlib.redirect_stdout(out):
                for number, cell in enumerate(cells):
                    code = "".join(cell["source"])
                    exec(compile(code, str(path) + ":" + lesson + ":" + str(number), "exec"), namespace)
        except Exception as caught:
            error = repr(caught)
        plt.close("all")
        results.append({"lesson": lesson, "source": path.relative_to(ROOT).as_posix(),
                        "section_sha256": digest(section["body"].encode()), "code_cells": len(cells),
                        "status": "passed" if error is None else "failed", "error": error,
                        "output": out.getvalue()})
        if len(results) % 40 == 0:
            print("independent CPU short examples:", len(results), flush=True)
environment = {"python": platform.python_version(), "torch": str(torch.__version__),
               "matplotlib": matplotlib.__version__, "device": "CPU; CUDA_VISIBLE_DEVICES empty",
               "mode": "new namespace per current source section; no full-model training or download"}
save("execution.json", {"environment": environment, "total": len(results),
                        "passed": sum(row["status"] == "passed" for row in results),
                        "failed": sum(row["status"] == "failed" for row in results), "lessons": results})
print(json.dumps({"links": len(link_rows), "total": len(results),
                  "failed": [row for row in results if row["status"] == "failed"],
                  "environment": environment}, ensure_ascii=False))
assert all(row["status"] == "passed" for row in results)
