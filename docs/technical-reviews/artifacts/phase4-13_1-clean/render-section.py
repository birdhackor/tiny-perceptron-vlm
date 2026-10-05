"""Bounded source-based article render, using repository Markdown transformation.

This verifies visible section content, not the production Zensical navigation/theme.
"""
import ast
import hashlib
import html
import json
import re
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
source_path = ROOT / "course/chapters/13.md"
export_raw = (ROOT / "scripts/export_course.py").read_bytes()
tree = ast.parse(export_raw)
names = {"reading_markdown", "diagram_html"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = {"html": html, "re": re, "ROOT": ROOT,
      "COURSE_URL": "https://birdhackor.github.io/tiny-perceptron-vlm/",
      "REPOSITORY": "birdhackor/tiny-perceptron-vlm"}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "scripts/export_course.py:selected-functions", "exec"), ns)
body = (ART / "inputs/section13.1.md").read_text()
targets = {(ROOT / "course/chapters/08.md").resolve(): "chapter-08.md"}
lesson_targets = {source_path.resolve(): {"13.7": "13.7.md"},
                  (ROOT / "course/chapters/08.md").resolve(): {"8.3": "8.3.md"}}
transformed = ns["reading_markdown"](body, source_path, targets, lesson_targets, "frozen-input")
rendered = markdown.markdown(transformed, extensions=["attr_list", "tables", "md_in_html", "pymdownx.superfences", "pymdownx.details"])
css = (ROOT / "course/web/course.css").read_text()
(ART / "inputs/course.css").write_text(css)
page = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0;color:#202124;background:white;font-family:Arial,"Noto Sans CJK TC",sans-serif}'
        '.md-typeset{max-width:900px;margin:auto;padding:24px;font-size:18px;line-height:1.75}'
        'pre{overflow-x:auto;background:#f4f6f8;padding:16px;font-size:15px;line-height:1.6}'
        'details{border:1px solid #aaa;padding:12px}h2{font-size:28px}a{color:#007a73}'
        '@media(max-width:600px){.md-typeset{padding:16px;font-size:17px}h2{font-size:24px}}'
        + css + '</style></head><body><article class="md-typeset">' + rendered + '</article></body></html>')
(ART / "section-render.html").write_text(page)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    evidence = {"browser": browser.version, "renderer": "Repository reading_markdown + configured Markdown extensions; standalone article CSS shell",
                "production_theme_navigation": "not verified", "screenshots": []}
    for label, width, height in [("desktop",1280,800),("mobile",390,844)]:
        tab = browser.new_page(viewport={"width":width,"height":height}, device_scale_factor=1)
        tab.set_content(page, wait_until="load", timeout=20000)
        tab.locator("details").evaluate("e=>e.open=true")
        tab.screenshot(path=str(ART / f"{label}-head.png"))
        tab.locator("details").scroll_into_view_if_needed()
        tab.screenshot(path=str(ART / f"{label}-supplement.png"))
        evidence["screenshots"].extend([f"{label}-head.png", f"{label}-supplement.png"])
        evidence[label] = {"viewport":[width,height], "images":tab.locator("img").count(),
                           "horizontal_document_overflow":tab.evaluate("document.documentElement.scrollWidth>innerWidth"),
                           "details_open":tab.locator("details").evaluate("e=>e.open")}
        tab.close()
    browser.close()
(ART / "render-result.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(evidence,ensure_ascii=False,indent=2))
