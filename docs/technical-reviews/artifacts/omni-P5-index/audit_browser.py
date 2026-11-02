"""Read-only local browser proof for the frozen index's search and mobile navigation."""

import functools
import hashlib
import http.server
import json
import platform
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
PREVIEW = ROOT / "outputs/omni-P5-preview-v8"
SCREENSHOTS = Path("/workspace/work/reference/omni-P5-index-review/screenshots")
BROWSER = "/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def screenshot(page, name):
    path = SCREENSHOTS / name
    page.screenshot(path=str(path), full_page=False)
    raw = path.read_bytes()
    return {"path_outside_git": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(PREVIEW)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    result = {"reviewer_task": "/root/omni_p5_technical_index", "started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": sys.version, "platform": platform.platform(), "browser_path": BROWSER,
                              "device": "CPU; local HTTP preview"}, "preview_path": str(PREVIEW.relative_to(ROOT)),
              "screenshots": [], "console_errors": [], "request_failures": []}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=BROWSER, headless=True, args=["--no-sandbox"])
            result["environment"]["browser_version"] = browser.version
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            page.on("pageerror", lambda e: result["console_errors"].append(str(e)))
            page.on("requestfailed", lambda r: result["request_failures"].append({"url": r.url, "failure": r.failure}))
            page.goto(origin + "/index.html", wait_until="networkidle")
            result["desktop_title"] = page.locator("article h1").inner_text()
            result["desktop_body_width"] = page.locator("article").bounding_box()["width"]
            result["screenshots"].append(screenshot(page, "desktop-index-v2.png"))
            page.locator(".md-search__button").click()
            search = page.get_by_role("combobox")
            result["desktop_search_visible"] = search.is_visible()
            probes = []
            for query, expected in (("文字怎麼變成可還原的編號", "1.1.html"), ("vocabulary", "1.1.html")):
                search.fill(query)
                page.wait_for_timeout(1800)
                result_links = search.evaluate(
                    "el => Array.from(el.getRootNode().querySelectorAll('a[href]')).map(x => ({href:x.getAttribute('href'),text:x.innerText}))"
                )
                probes.append({"query": query, "expected_target": expected, "links": result_links[:15],
                               "expected_target_found": any(expected in x["href"] for x in result_links),
                               "result_text_excerpt": "\n".join(x["text"] for x in result_links)[:1200]})
            result["search_probes"] = probes
            result["screenshots"].append(screenshot(page, "desktop-body-search-v2.png"))
            search.fill("")
            page.keyboard.press("Escape")
            mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
            mobile.goto(origin + "/index.html", wait_until="networkidle")
            result["mobile_no_horizontal_page_overflow"] = mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
            result["screenshots"].append(screenshot(mobile, "mobile-index-v2.png"))
            toggle = mobile.locator('header label[for="__drawer"]')
            result["mobile_menu_visible"] = toggle.is_visible()
            result["mobile_menu_position"] = toggle.bounding_box()
            result["mobile_drawer_before"] = mobile.locator("#__drawer").is_checked()
            toggle.click()
            mobile.wait_for_timeout(350)
            result["mobile_drawer_after"] = mobile.locator("#__drawer").is_checked()
            result["mobile_menu_text_excerpt"] = mobile.locator("nav.md-nav--primary").inner_text()[:1000]
            result["screenshots"].append(screenshot(mobile, "mobile-open-menu-v2.png"))
            mobile.locator("#__nav_3_label").click()
            mobile.wait_for_timeout(200)
            mobile.locator("#__nav_3_1_label").click()
            mobile.wait_for_timeout(200)
            first_lesson = mobile.locator('nav.md-nav--primary a[href="./1.1.html"]')
            result["mobile_first_lesson_link_visible_after_expand"] = first_lesson.is_visible()
            result["screenshots"].append(screenshot(mobile, "mobile-expanded-chapter-v2.png"))
            mobile.close()
            browser.close()
    finally:
        server.shutdown()
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    destination = Path(__file__).with_name("browser-audit-v2.json")
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"result": str(destination.relative_to(ROOT)), "desktop_search_visible": result["desktop_search_visible"],
                      "search_probes": [{"query": x["query"], "expected_target_found": x["expected_target_found"]} for x in result["search_probes"]],
                      "mobile_menu_visible": result["mobile_menu_visible"], "mobile_drawer_after": result["mobile_drawer_after"],
                      "console_errors": result["console_errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
