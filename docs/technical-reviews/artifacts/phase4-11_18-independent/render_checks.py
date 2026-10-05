"""Render and verify the currently served section against current authored inputs."""

import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SECTION = OUT / "extraction/section.md"
FIGURE = ROOT / "course/figures/new-11.18-two-regions.svg"
url = "http://127.0.0.1:8765/11.18.html"
response = urlopen(url, timeout=15)
html = response.read()
(OUT / "served-section.html").write_bytes(html)
served_svg = urlopen("http://127.0.0.1:8765/figures/new-11.18-two-regions.svg", timeout=15).read()
assert served_svg == FIGURE.read_bytes()
inkscape = subprocess.run(["inkscape", str(FIGURE), "--export-type=png", f"--export-filename={OUT / 'figure-render.png'}"], capture_output=True, text=True, check=True)
(OUT / "inkscape-stderr.txt").write_text(inkscape.stderr)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
    page.goto(url)
    page.wait_for_load_state("networkidle")
    article = page.locator("article")
    assert article.count() == 1
    page.locator("details").evaluate_all("els => els.forEach(e => e.open=true)")
    visible_text = article.inner_text()
    (OUT / "rendered-article-text.txt").write_text(visible_text, encoding="utf-8")
    paragraphs = []
    for block in SECTION.read_text().split("\n\n"):
        if block.startswith(("##", "![", "<", "[")) or not block.strip():
            continue
        plain = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", block).strip()
        assert re.sub(r"\s+", "", plain) in re.sub(r"\s+", "", visible_text), plain
        paragraphs.append(plain)
    geometry = {}
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        page.screenshot(path=str(OUT / f"{name}-page.png"), full_page=True)
        image = page.locator("img[src$='new-11.18-two-regions.svg']")
        geometry[name] = image.bounding_box()
        assert image.evaluate("e => e.complete && e.naturalWidth === 640")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        image.screenshot(path=str(OUT / f"{name}-inline-figure.png"))
    version = browser.version
    browser.close()

result = {
    "url": url,
    "source_section_sha256": hashlib.sha256(SECTION.read_bytes()).hexdigest(),
    "current_figure_sha256": hashlib.sha256(FIGURE.read_bytes()).hexdigest(),
    "served_html_sha256": hashlib.sha256(html).hexdigest(),
    "served_figure_bytes_equal_current": True,
    "checked_prose_paragraphs": len(paragraphs),
    "figure_geometry": geometry,
    "environment": {"Chromium": version, "Playwright": importlib.metadata.version("playwright"), "Inkscape": subprocess.check_output(["inkscape", "--version"], text=True).strip()},
    "result": "current prose present; current figure loaded; no horizontal page overflow at both viewports",
    "scope": "Rendering and source correspondence only; screenshots require human/agent visual inspection.",
}
(OUT / "render-checks-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
