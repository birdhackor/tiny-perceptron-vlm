"""Read-only rendering of current 7.14; open its existing disclosure for inspection."""
import importlib.metadata
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/usr/bin/chromium",
        headless=True,
        timeout=12000,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    response = page.goto("http://127.0.0.1:8765/7.14.html", wait_until="domcontentloaded", timeout=12000)
    assert response.status == 200
    article = page.locator("article.md-content__inner")
    assert article.count() == 1
    page.locator("article details").evaluate_all("nodes => nodes.forEach(n => n.open = true)")
    text = article.inner_text(timeout=5000)
    assert "7.13的直接SFT屬性基模" in text and "這份起點從零練900次" in text
    article.screenshot(path=str(BASE / "desktop-expanded.png"), timeout=8000, animations="disabled")
    info = {
        "url": page.url,
        "status": response.status,
        "playwright": importlib.metadata.version("playwright"),
        "chromium": browser.version,
        "viewport": {"width": 1280, "height": 800},
        "state": "Existing section details opened through DOM; source and teaching content unchanged",
        "article_image_count": article.locator("img").count(),
        "article_svg_count": article.locator("svg").count(),
        "baseline_link": article.locator("a", has_text="7.13的直接SFT屬性基模").get_attribute("href"),
        "table": article.locator("table").inner_text(),
        "screenshot": "desktop-expanded.png",
    }
    (BASE / "render-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    (BASE / "article-text.txt").write_text(text)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    browser.close()
