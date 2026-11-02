"""source15當前首頁的搜尋及手機選單callback。"""

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
BASE = Path(__file__).parent
SITE = ROOT / "outputs/omni-review-site-technical-fix15"
CACHE = Path("/workspace/work/omni-technical-index-v1/source15-current")


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(SITE)))
    Thread(target=server.serve_forever, daemon=True).start()
    queries, screenshots = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium", args=["--disable-gpu"])
        version = browser.version
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        page.goto(url, wait_until="networkidle")
        page.locator(".md-search__button").click()
        search = page.get_by_placeholder("Search")
        for query in ["可還原", "索書號"]:
            search.fill(query)
            result = page.locator('a[href*="1.1.html?h="]').filter(has_text=query).first
            result.wait_for(state="visible", timeout=20000)
            queries.append({"query": query, "target": "1.1.html", "visible": True,
                            "excerpt": " ".join(result.inner_text().split())[:190]})
        desktop = CACHE / "index-search-body.png"
        page.screenshot(path=str(desktop)); screenshots.append(desktop)
        page.keyboard.press("Escape")
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(url, wait_until="networkidle")
        menu = page.locator('.md-header label[for="__drawer"]')
        menu.wait_for(state="visible")
        box = menu.bounding_box()
        assert box["x"] < 100 and box["y"] < 80
        menu.click()
        assert page.locator("#__drawer").is_checked()
        page.wait_for_timeout(500)
        course = page.locator('[id="__nav_3_label"]')
        course.wait_for(state="visible")
        mobile = CACHE / "index-mobile-menu.png"
        page.screenshot(path=str(mobile)); screenshots.append(mobile)
        browser.close()
    server.shutdown()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "python docs/technical-reviews/artifacts/omni-technical-index-v1/verify_source15_browser.py",
        "environment": {"python": platform.python_version(), "chromium": version,
                        "device": "CPU headless Chromium --disable-gpu"},
        "site": str(SITE.relative_to(ROOT)), "site_index_sha256": sha(SITE / "index.html"),
        "search_queries": queries, "mobile_viewport": [390, 844], "mobile_menu_box": box,
        "mobile_drawer_checked": True, "mobile_course_navigation_visible": True,
        "screenshots_git_external": [{"path": str(path), "sha256": sha(path)} for path in screenshots],
        "scope": "source15當前本機建置UI實際操作；不代表公開部署、Google登入或Colab執行。",
    }
    with (BASE / "source15-browser-check.json").open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2); file.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
