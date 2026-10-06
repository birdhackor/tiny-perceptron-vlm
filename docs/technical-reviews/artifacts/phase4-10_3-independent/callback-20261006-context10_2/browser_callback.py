"""Use the system Chromium binary with bounded navigation and screenshot calls."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
results = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/lib/chromium/chromium", headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--disable-background-networking"], timeout=15000)
    try:
        for lesson, height in [("10.2", 2000), ("10.3", 1800)]:
            page = browser.new_page(viewport={"width": 1280, "height": height})
            page.route("**/*", lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:8765/") or route.request.url.startswith("data:") else route.abort())
            url = "http://127.0.0.1:8765/" + lesson + ".html"
            response = page.goto(url, wait_until="domcontentloaded", timeout=15000)
            assert response.status == 200
            assert page.locator("h1").inner_text(timeout=5000).startswith(lesson)
            screenshot = HERE / (lesson + "-page.png")
            page.screenshot(path=str(screenshot), animations="disabled", timeout=15000)
            results.append({"url": url, "status": response.status, "title": page.title(), "viewport": {"width": 1280, "height": height},
                            "screenshot_path": str(screenshot), "visible_images": page.locator("img").count(),
                            "image_loaded": page.locator("img").evaluate_all("nodes => nodes.map(n => ({src:n.getAttribute('src'), complete:n.complete, naturalWidth:n.naturalWidth}))")})
            page.close()
        receipt = {"chromium_version": browser.version, "executable_path": "/usr/lib/chromium/chromium", "pages": results,
                   "scope": "Actual loopback pages rendered by installed system Chromium; external network requests blocked, no page content changed; personal image inspection follows."}
        (HERE / "browser-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    finally:
        browser.close()
