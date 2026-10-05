"""Render frozen section HTML through set_content; preserve actual viewport and results."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
html = (OUT / "section.html").read_text()
results = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                                timeout=20000)
    for name, width, height in (("desktop", 1280, 800), ("mobile", 390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.route("**/*", lambda route: route.abort())
        page.set_content(html, wait_until="domcontentloaded", timeout=15000)
        page.screenshot(path=str(OUT / (name + "-set-content.png")), full_page=True, timeout=15000)
        results.append({"viewport": {"width": width, "height": height},
                        "screenshot": name + "-set-content.png", "method": "set_content frozen section HTML",
                        "browser_version": browser.version,
                        "dimensions": page.evaluate("({scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight,innerWidth:innerWidth,innerHeight:innerHeight})")})
        page.close()
    browser.close()
(OUT / "playwright-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
