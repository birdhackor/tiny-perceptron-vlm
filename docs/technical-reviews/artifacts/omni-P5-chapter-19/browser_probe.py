"""Read-only chapter-introduction navigation probe; screenshots stay outside Git."""
import functools
import hashlib
import json
import platform
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
PREVIEW = ROOT / "outputs/omni-P5-preview-v9"
SCREENSHOTS = Path("/workspace/work/omni-P5-chapter-19-screenshots-v1")
SCREENSHOTS.mkdir(exist_ok=True)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(PREVIEW)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}/"
result = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
          "device": "CPU; Chromium headless", "preview": "outputs/omni-P5-preview-v9", "views": []}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell",
            headless=True, args=["--no-sandbox"])
        result["chromium"] = browser.version
        for label, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            response = page.goto(base + "chapter-19.html", wait_until="networkidle")
            page.locator("article details summary").click()
            data = page.locator("article").evaluate("""article => ({
              text: article.innerText,
              links: [...article.querySelectorAll('a[href]')].filter(x => !x.classList.contains('headerlink')).map(x => ({label:x.innerText, href:x.getAttribute('href')})),
              width: article.getBoundingClientRect().width,
              documentWidth: document.documentElement.scrollWidth,
              viewportWidth: innerWidth
            })""")
            screenshot = SCREENSHOTS / f"chapter-19-{label}.png"
            if screenshot.exists():
                raise FileExistsError(screenshot)
            page.screenshot(path=str(screenshot), full_page=True)
            data.update({"view": label, "viewport": [width, height], "http_status": response.status,
                         "screenshot": str(screenshot), "screenshot_bytes": screenshot.stat().st_size,
                         "screenshot_sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest()})
            result["views"].append(data)
            page.close()
        page = browser.new_page()
        targets = []
        for link in result["views"][0]["links"]:
            href = link["href"]
            if not href.startswith(("http:", "https:")) and ".html" in href:
                response = page.goto(base + href.removeprefix("./"), wait_until="domcontentloaded")
                anchor = page.evaluate("decodeURIComponent(location.hash.slice(1))")
                found = page.evaluate("name => !name || !!document.getElementById(name)", anchor)
                targets.append({**link, "http_status": response.status, "anchor_exists": found,
                                "heading": page.locator("article h1,article h2").first.inner_text()})
        result["navigation_targets"] = targets
        page.close()
        browser.close()
finally:
    server.shutdown()

with (PROOF / "browser-receipt-v1.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"views": [{k: v[k] for k in ["view", "http_status", "documentWidth", "viewportWidth", "screenshot", "screenshot_sha256"]} for v in result["views"]],
                  "targets": result["navigation_targets"]}, ensure_ascii=False, indent=2))
