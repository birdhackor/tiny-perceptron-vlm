"""Check the frozen home page's search/menu promises on the current local site."""

import argparse
import functools
import hashlib
import http.server
import json
import platform
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.site / "build-info.json").read_text())
    assert info["lessons"] == 313 and info["executed_cpu_outputs"]
    args.screenshots.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(args.site.resolve())))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    screenshots = []
    result = {"build_info": info, "checks": {}}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
            result["environment"] = {"python": platform.python_version(), "chromium": browser.version, "device": "headless Linux Chromium; local HTTP"}
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url + "/index.html", wait_until="networkidle")
            assert page.locator("main h1").filter(has_text="電腦怎麼學會接話？").count() == 1
            for title, target in [("從第一節開始", "1.1.html"), ("先看基礎暖身", "first-steps.html"), ("閱讀指南", "course.html")]:
                matches = page.locator(f'main a[href$="/{target}"]').filter(has_text=title)
                assert matches.count() >= 1
            assert "313 個小節" in page.locator("main").inner_text()
            path = args.screenshots / "index-desktop.png"
            page.screenshot(path=str(path))
            screenshots.append(path)
            result["checks"]["home_buttons_and_count"] = "First lesson, warmup, and reading-guide links point to real generated pages; 313 displayed."
            page.locator(".md-search__button").click()
            search = page.get_by_placeholder("Search", exact=True)
            searches = []
            for query, target in [("分塊邊界", "6.10.html"), ("Unicode", "1.1.html")]:
                search.fill(query)
                matched = page.locator(f'a.i[href*="/{target}?"]').first
                matched.wait_for(state="visible", timeout=20000)
                assert matched.is_visible()
                searches.append({"query": query, "expected_page": target, "visible_result": matched.inner_text()[:400]})
            result["checks"]["search_title_and_body"] = searches
            page.keyboard.press("Escape")
            page.goto(url + "/1.1.html", wait_until="networkidle")
            assert page.locator('main a[download][href$="/notebooks/01/1.1.ipynb"]').count() == 1
            assert page.locator('main a[href*="colab.research.google.com"]').count() == 1
            assert page.locator("main .output").count() >= 1
            assert "貓看狗，狗看貓。" in page.locator("main .output").inner_text()
            result["checks"]["sample_lesson"] = "1.1 shows its actual CPU result and both footer buttons."
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(url + "/index.html", wait_until="networkidle")
            drawer = page.locator('.md-header__button[for="__drawer"]')
            assert drawer.is_visible()
            box = drawer.bounding_box()
            assert box["x"] < 80 and box["y"] < 90
            drawer.click()
            assert page.locator("#__drawer").is_checked()
            page.wait_for_function("() => document.querySelector('.md-sidebar--primary').getBoundingClientRect().x >= -1")
            assert page.locator(".md-sidebar--primary").is_visible()
            assert page.locator('.md-sidebar--primary a[href$="/6.10.html"]').count() >= 1
            path = args.screenshots / "index-mobile-menu.png"
            page.screenshot(path=str(path))
            screenshots.append(path)
            result["checks"]["mobile_menu"] = {"viewport": "390x844", "button_box": box, "drawer_checked": True, "sidebar_visible": True, "contains_6_10_link": True}
            result["screenshots"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in screenshots]
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    result["scope"] = "Sample desktop search queries, home links, 1.1 footer/output, and mobile drawer; no Colab runtime, public deployment, all-device layout, or learning-quality validation."
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
