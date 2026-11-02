"""實際瀏覽器的有界 MathJax/數量/原圖觀察；截圖與第三方 bytes 在 Git 外。"""

import hashlib
import json
import sys
import threading
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

WORK = Path("/workspace/work/omni-technical-readme-v1/historical-callback")
OUT = Path(__file__).with_name("historical-browser-evidence-v1.json")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    assert not OUT.exists()
    site = WORK / "site"
    assert (site / "index.html").is_file()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors, external_blocked, requests, screenshots, observations = [], [], [], [], {}
    try:
        with sync_playwright() as play:
            browser = play.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
            context = browser.new_context(viewport={"width": 1024, "height": 900})

            def route_request(route):
                url = route.request.url
                if urlparse(url).hostname != "127.0.0.1":
                    external_blocked.append(url)
                    route.abort()
                else:
                    requests.append(urlparse(url).path)
                    route.continue_()

            context.route("**/*", route_request)
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(origin + "/index.html", wait_until="networkidle")
            page.wait_for_function("window.MathJax?.version === '3.2.2' && !!window.MathJax.startup?.promise")
            page.evaluate("async () => await window.MathJax.startup.promise")
            assert page.locator("mjx-container").count() == 1
            observations["mathjax_version"] = page.evaluate("window.MathJax.version")
            observations["typeset_container_count"] = page.locator("mjx-container").count()
            observations["mathjax_scripts"] = page.locator("script[src]").evaluate_all("es => es.map(e => e.getAttribute('src'))")
            observations["mathjax_errors"] = page.locator("mjx-merror").count()
            observations["formula_mathml"] = page.locator("mjx-assistive-mml").inner_text()
            observations["counts_desktop_text"] = page.locator("#counts").inner_text()
            math = WORK / "mathjax-desktop.png"
            page.locator("#math").screenshot(path=str(math))
            screenshots.append({"path": str(math), "sha256": hashlib.sha256(math.read_bytes()).hexdigest(), "scope": "Actual current MathJax3.2.2 typeset of literal current9.17 formula in a local harness; math semantics not re-reviewed here."})
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(100)
            observations["counts_mobile_text"] = page.locator("#counts").inner_text()
            observations["mobile_scroll_width"] = page.evaluate("document.documentElement.scrollWidth")
            count = WORK / "counts-mobile.png"
            page.locator("#counts").screenshot(path=str(count))
            screenshots.append({"path": str(count), "sha256": hashlib.sha256(count.read_bytes()).hexdigest(), "scope": "Actual literal current README count paragraph at390px; bounded harness, not full site layout."})
            page.set_viewport_size({"width": 900, "height": 1000})
            page.goto(origin + "/character-ids.svg", wait_until="networkidle")
            figure = WORK / "character-ids-current.png"
            page.locator("svg").screenshot(path=str(figure))
            screenshots.append({"path": str(figure), "sha256": hashlib.sha256(figure.read_bytes()).hexdigest(), "scope": "Current README exact SHA-bound SVG freshly rendered for appendix figure-map confirmation."})
            observations["browser_version"] = browser.version
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    assert not errors and not external_blocked and observations["mathjax_errors"] == 0
    assert observations["counts_desktop_text"] == observations["counts_mobile_text"]
    result = {"created_at": datetime.now(UTC).isoformat(), "command": "python docs/technical-reviews/artifacts/omni-technical-readme-v1/historical_browser_probe_v1.py", "environment": {"python": sys.version.split()[0], "browser": observations["browser_version"], "device": "CPU headless desktop/mobile emulation"}, "observations": observations, "requested_local_paths": sorted(set(requests)), "external_requests_blocked": external_blocked, "page_errors": errors, "screenshots_outside_git": screenshots, "scope": "Fresh browser run with all external hosts blocked. Initial same-origin asset loading is permitted. Proves local bundled-asset/browser typesetting in this bounded harness; not complete offline-site, full Zensical publication, Colab or hardware validation."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "observations": observations, "screenshots": screenshots}, ensure_ascii=False))


if __name__ == "__main__":
    main()
