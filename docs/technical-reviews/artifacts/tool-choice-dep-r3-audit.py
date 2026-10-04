"""Fresh R.3 navigation audit; reads current originals and local generated pages only."""

import hashlib
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path, lesson):
    raw = path.read_bytes()
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    chosen = [i for i, h in enumerate(headers) if h[0].startswith(f"## {lesson} ".encode())]
    assert len(chosen) == 1, (path, lesson)
    i = chosen[0]
    end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
    return raw[headers[i].start():end]


def write_json(name, value):
    (ARTIFACTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def html_section(path, fragment):
    soup = BeautifulSoup(path.read_text(), "html.parser")
    heading = soup.find("h2", id=fragment)
    assert heading is not None, (path, fragment)
    nodes = []
    for node in heading.next_siblings:
        if getattr(node, "name", None) == "h2":
            break
        nodes.append(str(node))
    return BeautifulSoup("".join(nodes), "html.parser")


source = ROOT / "course/README.md"
body = section(source, "R.3")
text = body.decode()
assert "```" not in text
assert not re.search(r"!\[[^\]]*\]\(", text)
table_rows = []
for line in text.splitlines():
    match = re.fullmatch(r"\| \[([^\]]+)\]\((chapters/[^)]+)\) \| (.+) \|", line)
    if match:
        table_rows.append({"label": match[1], "target": match[2], "question": match[3]})
assert len(table_rows) == 21

links = []
for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
    parts = urlsplit(target)
    assert not parts.scheme, target
    path = (source.parent / unquote(parts.path)).resolve()
    assert path.is_file() and path.is_relative_to(ROOT), target
    if parts.fragment:
        section(path, parts.fragment)
    links.append({"label": label, "target": target, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
assert len(links) == 23

outlines = []
for row in table_rows:
    path = (source.parent / row["target"]).resolve()
    original = path.read_text()
    intro = original.split("\n## ", 1)[0]
    headings = re.findall(r"^## ([A-Z\d]+\.\d+) (.+)$", original, re.M)
    assert headings
    outlines.append({
        **row,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest(path),
        "intro": intro,
        "headings": [{"id": number, "title": title} for number, title in headings],
    })
assert {item["path"] for item in outlines} == {path.relative_to(ROOT).as_posix() for path in (ROOT / "course/chapters").glob("*.md")}

index_path = ROOT / "course/lesson-index.json"
index = json.loads(index_path.read_text())
lessons_path = ROOT / "course/lessons.md"
lesson_text = lessons_path.read_text()
lesson_rows = re.findall(r"^\| ([A-Z\d]+\.\d+) (.*?) \| \[閱讀\]\(([^)]+)\) \| \[開啟\]\(([^)]+)\) \|$", lesson_text, re.M)
assert len(lesson_rows) == len(index) == 226
original_titles = {h["id"]: h["title"] for o in outlines for h in o["headings"]}
assert {row[0] for row in lesson_rows} == set(original_titles) == {entry["id"] for entry in index}
for number, title, reading, notebook in lesson_rows:
    assert title == original_titles[number], (number, title)
    markdown_path, fragment = reading.split("#")
    assert fragment == number
    chapter = (lessons_path.parent / markdown_path).resolve()
    assert section(chapter, number)
    assert (lessons_path.parent / notebook).resolve().is_file()

selected = [row for row in lesson_rows if row[0] in {"7.4", "B.5", "B.6", "B.7", "B.8"}]
assert len(selected) == 5
assert selected[0][1] == "回答第一個 token 在哪個位置被預測？"
assert "token、tokenizer" in section(ROOT / "course/glossary.md", "G.1").decode()
assert "token是模型一次處理的文字片段" in section(ROOT / "course/chapters/06.md", "6.1").decode()
assert "FlashAttention的關鍵做法是分塊與減少中間資料搬移" in section(ROOT / "course/chapters/16.md", "16.9").decode()

site_source = ROOT / "outputs/site/course.html"
rendered = html_section(site_source, "R.3")
rendered_rows = []
for row in rendered.select("tbody tr"):
    cells = row.find_all("td")
    link = cells[0].find("a")
    rendered_rows.append({"label": cells[0].get_text(), "href": link["href"], "question": cells[1].get_text()})
assert len(rendered_rows) == len(table_rows)
for original, generated in zip(table_rows, rendered_rows, strict=True):
    assert generated["label"] == original["label"]
    assert generated["question"] == original["question"]
    assert generated["href"] == "chapter-" + Path(original["target"]).stem + ".html"
    assert (site_source.parent / generated["href"]).is_file()
paragraphs = [line for line in text.splitlines() if line and not line.startswith(("#", "|"))]
paragraphs = [re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line) for line in paragraphs]
assert paragraphs == [node.get_text() for node in rendered.find_all("p")]
rendered_links = []
for anchor in rendered.find_all("a"):
    href = anchor["href"]
    target = (site_source.parent / href).resolve()
    assert target.is_file() and target.is_relative_to(ROOT / "outputs/site")
    rendered_links.append({"label": anchor.get_text(), "href": href})
assert len(rendered_links) == 23

catalogue = BeautifulSoup((ROOT / "outputs/site/index.html").read_text(), "html.parser")
article = catalogue.select_one("article")
assert article is not None
sidebar = catalogue.select_one(".md-sidebar--primary")
assert sidebar is not None
sidebar_lesson_ids = {
    link["href"].removeprefix("./").removesuffix(".html")
    for link in sidebar.find_all("a", href=True)
    if link["href"].removeprefix("./").removesuffix(".html") in original_titles
}
assert sidebar_lesson_ids == set(original_titles)
site_selected = []
for number, title, _, _ in selected:
    page = ROOT / "outputs/site" / f"{number}.html"
    assert page.is_file()
    link = sidebar.find("a", href=lambda value: value and value.removeprefix("./") == f"{number}.html")
    assert link is not None
    detail = BeautifulSoup(page.read_text(), "html.parser")
    assert detail.find("h1").get_text().strip() == f"{number} {title}¶"
    site_selected.append({"id": number, "page": page.relative_to(ROOT).as_posix(), "heading": title, "catalogue_sidebar_href": link["href"]})

kernel_root = ROOT / "outputs/tool-choice-site-kernels"
kernel_notebooks = list(kernel_root.glob("*/*.ipynb"))
assert len(kernel_notebooks) == 226
kernel_selected = []
for number, _, _, notebook in selected:
    original_path = (lessons_path.parent / notebook).resolve()
    executed_path = kernel_root / original_path.relative_to(ROOT / "notebooks")
    original = json.loads(original_path.read_text())
    executed = json.loads(executed_path.read_text())
    assert len(original["cells"]) == len(executed["cells"])
    for src, dest in zip(original["cells"], executed["cells"], strict=True):
        assert src["cell_type"] == dest["cell_type"] and src["source"] == dest["source"]
        if dest["cell_type"] == "code":
            assert dest.get("execution_count") is not None
            assert not any(output["output_type"] == "error" for output in dest.get("outputs", []))
    kernel_selected.append({"id": number, "path": executed_path.relative_to(ROOT).as_posix(), "sha256": digest(executed_path), "current_source_cells_match": True})

(ARTIFACTS / "tool-choice-dep-r3-original-section.md").write_bytes(body)
write_json("tool-choice-dep-r3-current-outlines.json", {"scope": "Personally read chapter introductions and headings; not a review of every section's technical or empirical claims.", "chapters": outlines})
write_json("tool-choice-dep-r3-audit-output.json", {
    "environment": {"python": platform.python_version(), "beautifulsoup4": version("beautifulsoup4"), "device": "CPU; static local filesystem checks only"},
    "source": "course/README.md#R.3",
    "section_sha256": hashlib.sha256(body).hexdigest(),
    "original_links": links,
    "chapter_table_rows": len(table_rows),
    "all_current_chapters_covered": True,
    "lesson_index": {"path": "course/lesson-index.json", "sha256": digest(index_path), "count": len(index)},
    "lessons": {"path": "course/lessons.md", "sha256": digest(lessons_path), "all_226_titles_and_reading_targets_match_originals": True, "selected_rows": selected},
    "rendered_R3": {"path": site_source.relative_to(ROOT).as_posix(), "sha256": digest(site_source), "paragraphs_and_table_match_original": True, "links": rendered_links},
    "selected_site_targets": site_selected,
    "rendered_catalogue": {"path": "outputs/site/index.html", "sha256": digest(ROOT / "outputs/site/index.html"), "sidebar_lesson_count": len(sidebar_lesson_ids), "scope": "The lessons.md link is converted to index.html, whose article lists chapters. The complete 226-lesson catalogue is in the site's primary navigation sidebar, not an HTML rendering of the lessons.md table."},
    "kernels": {"root": kernel_root.relative_to(ROOT).as_posix(), "notebook_count": len(kernel_notebooks), "selected_current_cell_matches": kernel_selected, "scope": "Only 7.4 and B.5-B.8 were checked for cell identity and recorded non-error execution metadata. No notebooks, training, or GPU runs were executed in this audit."},
    "result": "All navigation, current-title, local-page, and selected notebook-source checks passed.",
})
print(json.dumps({"result": "pass", "chapter_rows": len(table_rows), "original_links": len(links), "lesson_rows": len(lesson_rows), "selected_site_pages": len(site_selected), "kernel_notebook_count": len(kernel_notebooks), "GPU_or_training_executed": False}, ensure_ascii=False))
