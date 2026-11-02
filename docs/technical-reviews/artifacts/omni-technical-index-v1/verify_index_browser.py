"""用本機 Chromium 操作 index 的搜尋與手機章節選單。"""

import hashlib
import json
import platform
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/omni-technical-index-v1")
SITE = ROOT / "outputs/omni-review-site-technical-fix13"


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(SITE)))
    Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium", args=["--disable-gpu"])
        browser_version = browser.version
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="networkidle")
        page.locator(".md-search__button").click()
        search = page.get_by_placeholder("Search")
        for query, reason in [("可還原", "小節標題的一部分"), ("索書號", "1.1正文詞語，非其標題")]:
            search.fill(query)
            # 搜尋結果位於Zensical產生的open shadow root；Playwright locator可穿透。
            match = page.locator(f'a[href*="1.1.html?h="]').filter(has_text=query)
            match.first.wait_for(state="visible", timeout=20000)
            href = match.first.get_attribute("href")
            text = match.first.inner_text()
            assert "1.1.html?h=" in href and query in text
            results.append({"query": query, "query_scope": reason, "target": "1.1.html",
                            "visible_result": True, "excerpt": " ".join(text.split())[:320]})
        desktop_path = CACHE / "index-search-body-v2.png"
        page.screenshot(path=str(desktop_path))
        page.keyboard.press("Escape")
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(url, wait_until="networkidle")
        button = page.locator('.md-header label[for="__drawer"]')
        button.wait_for(state="visible")
        box = button.bounding_box()
        assert box["x"] < 100 and box["y"] < 80
        button.click()
        assert page.locator("#__drawer").is_checked()
        page.wait_for_timeout(500)  # 等待CSS抽屜開啟動畫結束後才保存畫面。
        directory = page.locator('[id="__nav_3_label"]')
        directory.wait_for(state="visible")
        mobile_path = CACHE / "index-mobile-menu-v2.png"
        page.screenshot(path=str(mobile_path))
        mobile = {"viewport": {"width": 390, "height": 844}, "top_left_menu_box": box,
                  "drawer_checked_after_click": True, "course_directory_visible": directory.is_visible(),
                  "course_directory_label": directory.inner_text()}
        browser.close()
    server.shutdown()
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "python docs/technical-reviews/artifacts/omni-technical-index-v1/verify_index_browser.py",
        "environment": {"python": platform.python_version(), "playwright": "system tooling installation",
                        "chromium": browser_version, "device": "CPU headless Chromium --disable-gpu"},
        "site": str(SITE.relative_to(ROOT)), "site_index_sha256": sha(SITE / "index.html"),
        "search": results, "mobile": mobile,
        "screenshots_git_external": [{"path": str(path), "sha256": sha(path)}
                                     for path in [desktop_path, mobile_path]],
        "scope": "本機fix13預覽的搜尋與選單操作，沒有登入或執行Colab，沒有檢查其他頁的技術結論。",
    }
    destination = Path(__file__).with_name("browser-check-r2.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
