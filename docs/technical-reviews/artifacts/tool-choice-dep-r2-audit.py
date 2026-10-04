"""R.2: current local reading routes and Notebook publication contract.

This checks saved CPU execution records; it does not execute lesson code,
start Jupyter kernels, train a model, use a GPU, or contact Colab.
"""
import hashlib
import importlib.util
import json
import platform
import re
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("tool-choice-dep-r2-audit.json")
SNAPSHOT = Path(__file__).with_name("tool-choice-dep-r2-read-sections.md")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path, identifier):
    raw = path.read_text(encoding="utf-8")
    match = re.search(r"^## " + re.escape(identifier) + r" .+$", raw, re.M)
    assert match, (path, identifier)
    next_heading = re.search(r"^## ", raw[match.end():], re.M)
    end = match.end() + next_heading.start() if next_heading else len(raw)
    return raw[match.start():end]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hrefs.append(attrs["href"])
        if "id" in attrs:
            self.ids.append(attrs["id"])


def html_section(page, identifier):
    raw = page.read_text(encoding="utf-8")
    match = re.search(r'<h[12] id="' + re.escape(identifier) + r'">', raw)
    assert match, (page, identifier)
    rest = raw[match.start():]
    end = re.search(r"<h2 |</article>", rest[match.end()-match.start():])
    return rest[:match.end()-match.start()+end.start()] if end else rest


build = load_module("r2_build_course", ROOT / "scripts/build_course.py")
export = load_module("r2_export_course", ROOT / "scripts/export_course.py")
index = json.loads((ROOT / "course/lesson-index.json").read_text())
by_id = {item["id"]: item for item in index}
assert len(index) == len(by_id)
readme = ROOT / "course/README.md"
r2 = section(readme, "R.2")
assert not re.search(r"^```python", r2, re.M)
assert not re.search(r"!\[.*?\]\(.*?\.svg\)", r2)
route_html = Links()
route_html.feed(html_section(ROOT / "outputs/site/course.html", "R.2"))
route_hrefs = set(route_html.hrefs)
snapshots = [("course/README.md#R.2", r2)]
route_records = []
for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", r2):
    path_text, _, identifier = target.partition("#")
    path = (readme.parent / path_text).resolve()
    body = section(path, identifier)
    relative = path.relative_to(ROOT).as_posix()
    snapshots.append((relative + "#" + identifier, body))
    if identifier in by_id:
        item = by_id[identifier]
        assert item["source"] == relative
        href = identifier + ".html"
    else:
        assert relative == "course/first-steps.md"
        href = "first-steps.html#" + identifier
    assert href in route_hrefs, (target, href)
    parsed = urlsplit(href)
    target_html = Links()
    target_html.feed((ROOT / "outputs/site" / parsed.path).read_text())
    assert identifier in target_html.ids, (identifier, href)
    route_records.append({"label": label, "source": relative, "id": identifier,
                          "section_sha256": hashlib.sha256(body.encode()).hexdigest(),
                          "site_href": href, "target_heading": body.splitlines()[0]})

# The example is 17.1 -> W.5 (weights/matrix/bias) -> W.6 (adjustable parameter).
for identifier in ["W.5", "W.6"]:
    body = section(ROOT / "course/first-steps.md", identifier)
    snapshots.append(("course/first-steps.md#" + identifier, body))
assert "../first-steps.md#W.5" in section(ROOT / "course/chapters/17.md", "17.1")
assert "[W.6](#W.6)" in section(ROOT / "course/first-steps.md", "W.5")
assert "可調旋鈕叫參數" in section(ROOT / "course/first-steps.md", "W.6")

manifest = []
code_cells = 0
for item in index:
    original_path = ROOT / item["notebook"]
    original = json.loads(original_path.read_text())
    source_body = section(ROOT / item["source"], item["id"])
    body_without_heading = source_body.split("\n", 1)[1]
    regenerated = build.notebook(item["id"], item["title"], body_without_heading, ROOT / item["source"])
    assert regenerated == original, ("source Notebook mismatch", item["id"])
    executed_path = ROOT / "outputs/tool-choice-site-kernels" / original_path.relative_to(ROOT / "notebooks")
    executed = export.checked_notebook(original, executed_path)
    counts = [cell["execution_count"] for cell in executed["cells"] if cell["cell_type"] == "code"]
    setups = [cell for cell in original["cells"] if cell.get("metadata", {}).get("course_setup")]
    assert len(setups) == 1 and export.joined(setups[0]["source"]) == build.BOOTSTRAP.rstrip("\n")
    assert counts == list(range(1, len(counts) + 1)), (item["id"], counts)
    code_cells += len(counts)
    page_path = ROOT / "outputs/site" / (item["id"] + ".html")
    html = Links()
    html.feed(page_path.read_text())
    assert item["id"] in html.ids
    assert item["notebook"] in html.hrefs
    colab = "https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/main/" + item["notebook"]
    assert colab in html.hrefs
    download = ROOT / "outputs/site" / item["notebook"]
    assert download.read_bytes() == original_path.read_bytes()
    manifest.append({"id": item["id"], "notebook": item["notebook"],
                     "notebook_sha256": sha(original_path),
                     "executed_sha256": sha(executed_path), "code_cells": len(counts)})

assert {"B.5", "B.6", "B.7", "B.8"}.issubset(by_id)
front_count = sum(len(re.findall(r"^## ", (ROOT / "course" / name).read_text(), re.M))
                  for name in ["README.md", "first-steps.md", "training.md", "glossary.md"])
SNAPSHOT.write_text("\n\n".join("<!-- " + target + " -->\n" + body for target, body in snapshots), encoding="utf-8")
result = {
    "reviewer_task": "/root/technical_dep_r2",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-r2-audit.py",
    "environment": {"python": platform.python_version(), "torch_distribution": version("torch"),
                    "device": "CPU text/JSON audit; no lesson code or GPU execution"},
    "result": "All assertions passed",
    "source_sha256": hashlib.sha256(r2.encode()).hexdigest(),
    "index_sha256": sha(ROOT / "course/lesson-index.json"),
    "notebooks": len(index), "numbered_sections": len(index) + front_count,
    "route_link_count": len(route_records), "route_links": route_records,
    "executed_code_records_inspected": code_cells,
    "new_lesson_ids_present": ["B.5", "B.6", "B.7", "B.8"],
    "quantization_bridge": ["17.1", "W.5", "W.6"],
    "lesson_python_executed_here": False,
    "fresh_kernels_run_here": 0,
    "limits": "Local current Markdown, Notebook and HTML inspection. Saved source-identical CPU execution records are inspected, not rerun. No live Colab, remote main-branch publication, empirical learner efficacy, training, GPU, or linked lesson scientific results are independently validated in this R.2 review.",
    "notebook_manifest": manifest,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ["result", "source_sha256", "notebooks", "numbered_sections", "route_link_count", "executed_code_records_inspected", "fresh_kernels_run_here"]}, ensure_ascii=False))
