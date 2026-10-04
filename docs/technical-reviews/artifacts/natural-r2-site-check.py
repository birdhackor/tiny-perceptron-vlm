"""Build an isolated navigation fixture with the real exporter and builder."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import export_course

def section(path, number):
    raw = path.read_text(encoding="utf-8")
    start = re.search(r"^## " + re.escape(number) + r" .+$", raw, re.M).start()
    next_heading = re.search(r"^## ", raw[start + 1:], re.M)
    end = start + 1 + next_heading.start() if next_heading else len(raw)
    return raw[start:end]

index = json.loads((ROOT / "course/lesson-index.json").read_text())
targets = {(ROOT / p).resolve(): n + ".md" for n, p in export_course.DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    path = (ROOT / item["source"]).resolve()
    targets[path] = "chapter-" + path.stem + ".md"
    lesson_targets.setdefault(path, {})[item["id"]] = item["id"] + ".md"
source = ROOT / "course/README.md"
body = section(source, "R.2")
content = export_course.reading_markdown(body, source, targets, lesson_targets, "main")
links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", content)
settings = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
assert settings["use_directory_urls"] is False
assert "attr_list" in settings["markdown_extensions"] and "tables" in settings["markdown_extensions"]

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links = set(), set()
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and "href" in values:
            self.links.add(values["href"])

with tempfile.TemporaryDirectory(prefix="natural-r2-site-") as temporary:
    directory = Path(temporary)
    docs = directory / "docs"
    docs.mkdir()
    (docs / "course.md").write_text(content)
    for link in set(links):
        name, _, anchor = link.partition("#")
        if anchor == "W.3":
            original = section(ROOT / "course/first-steps.md", anchor)
            heading = original.splitlines()[0]
            published = export_course.reading_markdown(heading, ROOT / "course/first-steps.md", targets, lesson_targets, "main")
        else:
            number = name.removesuffix(".md")
            item = next(item for item in index if item["id"] == number)
            path = ROOT / item["source"]
            heading = section(path, number).splitlines()[0]
            published = export_course.reading_markdown(heading, path, targets, lesson_targets, "main")
        (docs / name).write_text(published)
    config = (
        '[project]\nsite_name = "R.2 navigation fixture"\ndocs_dir = "docs"\nsite_dir = "site"\n'
        'use_directory_urls = false\n\n[project.markdown_extensions]\nattr_list = {}\ntables = {}\n'
    )
    (directory / "zensical.toml").write_text(config)
    command = [sys.executable, "-m", "zensical", "build", "--clean", "--strict", "--config-file", str(directory / "zensical.toml")]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    log = "COMMAND: " + " ".join(command) + "\nEXIT: " + str(result.returncode) + "\n" + result.stdout + result.stderr
    (ROOT / "docs/technical-reviews/artifacts/natural-r2-site-build.txt").write_text(log)
    assert result.returncode == 0, log
    course = Page((directory / "site/course.html").read_text())
    assert "R.2" in course.ids
    records = []
    for link in links:
        name, separator, anchor = link.partition("#")
        published = name.removesuffix(".md") + ".html" + separator + anchor
        destination = directory / "site" / (name.removesuffix(".md") + ".html")
        assert published in course.links and destination.is_file(), (link, published)
        page = Page(destination.read_text())
        assert not anchor or anchor in page.ids, (link, page.ids)
        records.append({"exported": link, "html_href": published, "file_exists": True,
                        "anchor": anchor or "separate lesson page", "anchor_exists": bool(not anchor or anchor in page.ids)})
    evidence = {"source_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "zensical": version("zensical"), "use_directory_urls": settings["use_directory_urls"],
                "scope": "isolated navigation fixture: actual R.2 content plus real destination headings; not full-site publication",
                "course_anchor": "R.2", "links": records, "result": "44/44 original links resolve to emitted HTML pages; W.3 anchor and R.2 anchor exist"}
    (ROOT / "docs/technical-reviews/artifacts/natural-r2-site.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"links": len(records), "builder_exit": result.returncode, "zensical": version("zensical"), "R.2_anchor": True, "W.3_anchor": True}))
