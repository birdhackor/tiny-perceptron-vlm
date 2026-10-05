"""Render only the independently verified current lesson page."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

base = Path(__file__).parent
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
    page = browser.new_page(viewport={"width": 1280, "height": 1600}, device_scale_factor=1)
    page.goto("http://127.0.0.1:8765/11.5.html", wait_until="domcontentloaded", timeout=15000)
    page.evaluate("document.fonts.ready")
    article = page.locator("article.md-content__inner")
    code = article.locator("pre code").inner_text()
    assert code == (base / "fence-1.py").read_text()
    assert article.locator("h1").get_attribute("id") == "11.5"
    page.screenshot(path=str(base / "page-desktop.png"), full_page=True, timeout=15000)
    table = article.locator("table")
    table.screenshot(path=str(base / "table-desktop.png"), timeout=10000)
    desktop_table = table.inner_text()
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path=str(base / "page-mobile.png"), full_page=True, timeout=15000)
    table.evaluate("t => t.scrollIntoView({block: 'center'})")
    page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    page.screenshot(path=str(base / "table-mobile-left.png"), timeout=10000)
    mobile_scroll = table.evaluate("""t => {
        const parent = t.closest('.md-typeset__scrollwrap');
        if (parent) parent.scrollLeft = parent.scrollWidth;
        return parent ? {scrollWidth: parent.scrollWidth, clientWidth: parent.clientWidth,
            scrollLeft: parent.scrollLeft} : null;
    }""")
    page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    page.screenshot(path=str(base / "table-mobile-right.png"), timeout=10000)
    result = {"url": page.url, "browser_version": browser.version, "source_code_exact": True,
        "desktop": "1280x1600", "mobile": "390x844", "desktop_table_text": desktop_table,
        "mobile_scroll": mobile_scroll,
        "scope": "Static current section only; no notebook execution or training."}
    browser.close()
(base / "render-playwright-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
