"""Supplemental review: capture the sources actually read and render their figures."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

chapters = []
figures = []
for n in range(1, 5):
    source = ROOT / f"course/chapters/{n:02d}.md"
    data = source.read_bytes()
    snapshot = OUT / "sources" / source.relative_to(ROOT)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(data)
    headings = list(re.finditer(rb"^## ([0-9]+\.[0-9]+) ([^\n]+)\n", data, re.M))
    sections = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(data)
        section = data[heading.start():end]
        sections.append({"section_id": heading[1].decode(), "title": heading[2].decode(),
                         "sha256": sha(section), "start_line": data[:heading.start()].count(b"\n")+1,
                         "end_line": data[:end].count(b"\n"),
                         "hash_boundary": "Exact UTF-8 bytes from this ## heading to just before the next ## heading, or EOF."})
    chapters.append({"source": str(source.relative_to(ROOT)), "snapshot": str(snapshot.relative_to(ROOT)),
                     "sha256": sha(data), "byte_count": len(data), "sections": sections})
    for match in re.finditer(rb"!\[([^]]*)\]\(([^)]+)\)", data):
        figure = (source.parent / match[2].decode()).resolve()
        raw = figure.read_bytes()
        copied = OUT / "sources" / figure.relative_to(ROOT)
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(raw)
        section_id = next(s["section_id"] for i, s in enumerate(sections)
                          if headings[i].start() <= match.start() < (headings[i+1].start() if i+1<len(headings) else len(data)))
        element = ET.fromstring(raw)
        viewbox = [float(v) for v in element.attrib["viewBox"].split()]
        rendered = OUT / "rendered" / (figure.stem + ".png")
        rendered.parent.mkdir(exist_ok=True)
        figures.append({"source": str(figure.relative_to(ROOT)), "snapshot": str(copied.relative_to(ROOT)),
                        "sha256": sha(raw), "section_id": section_id, "alt": match[1].decode(),
                        "viewbox": viewbox, "rendered": str(rendered.relative_to(ROOT))})

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    for figure in figures:
        _, _, width, height = figure["viewbox"]
        page = browser.new_page(viewport={"width": round(width), "height": round(height)}, device_scale_factor=1)
        svg = (ROOT / figure["source"]).read_text()
        svg = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[\s\S]*?>", "", svg)
        page.set_content(f'<html><head><style>body{{margin:0}}svg{{display:block;width:{width}px;height:{height}px}}</style></head><body>{svg}</body></html>')
        page.evaluate("document.fonts.ready")
        path = ROOT / figure["rendered"]
        page.screenshot(path=str(path))
        figure["rendered_sha256"] = sha(path.read_bytes())
        print(figure["section_id"], figure["source"], "rendered", path.name)
        page.close()
    browser.close()

(OUT / "source-manifest.json").write_text(json.dumps({"chapters": chapters, "figures": figures}, ensure_ascii=False, indent=2)+"\n")
print("Captured", len(chapters), "chapters,", sum(len(c["sections"]) for c in chapters), "sections,",len(figures), "figures")
