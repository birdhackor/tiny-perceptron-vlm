"""Render the current served section with bounded local assets only."""
import json
import platform
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                              args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
    for name, size in [("desktop", {"width": 1280, "height": 1500}),
                       ("mobile", {"width": 390, "height": 1300})]:
        page = browser.new_page(viewport=size, device_scale_factor=1)
        blocked = []
        def route(request):
            url = request.request.url
            if urlsplit(url).hostname in {"127.0.0.1", "localhost"}:
                request.continue_()
            else:
                blocked.append(url)
                request.abort()
        page.route("**/*", route)
        response = page.goto("http://127.0.0.1:8765/12.11.html", wait_until="domcontentloaded", timeout=25000)
        article = page.locator("article.md-content__inner")
        article.screenshot(path=str(BASE / f"article-{name}.png"), timeout=15000)
        page.screenshot(path=str(BASE / f"page-{name}.png"), full_page=True, timeout=15000)
        text = article.inner_text()
        assert "只收到同一字串的確定性回答規則" in text
        (BASE / f"article-{name}.txt").write_text(text)
        records.append({"viewport": size, "response_status": response.status,
                        "browser_version": browser.version, "external_asset_requests_aborted": blocked,
                        "article_screenshot": f"article-{name}.png", "page_screenshot": f"page-{name}.png"})
        page.close()
    browser.close()
(BASE / "render-environment.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False))
