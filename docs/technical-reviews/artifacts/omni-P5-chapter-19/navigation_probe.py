"""Check rendered internal links, including browser-rewritten absolute local URLs."""
import functools
import hashlib
import json
import platform
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
PREVIEW = ROOT / "outputs/omni-P5-preview-v9"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(PREVIEW)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}/"
freeze = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json").read_bytes())
expected = {p["page_path"].replace(".md", ".html"): p for p in freeze["pages"]}
receipt = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
           "preview": "outputs/omni-P5-preview-v9", "targets": [],
           "correction": "First view-acquisition probe recorded screenshots but skipped JavaScript-rewritten absolute local links. This separate run verifies those links; original receipt remains unchanged."}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell",
            headless=True, args=["--no-sandbox"])
        receipt["chromium"] = browser.version
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base + "chapter-19.html", wait_until="networkidle")
        page.locator("article details summary").click()
        links = page.locator("article a[href]").evaluate_all("nodes => nodes.filter(x => !x.classList.contains('headerlink')).map(x => ({label:x.innerText,href:x.href}))")
        for link in links:
            target = urljoin(base, link["href"])
            parsed = urlsplit(target)
            if parsed.netloc != urlsplit(base).netloc:
                continue
            response = page.goto(target, wait_until="domcontentloaded")
            name = unquote(parsed.path.lstrip("/"))
            fragment = unquote(parsed.fragment)
            heading = page.locator("article h1,article h2").first.inner_text().rstrip("¶").strip()
            canonical = expected.get(name)
            ok = response.status == 200 and canonical is not None and heading.endswith(canonical["title"])
            anchor_ok = page.evaluate("name => !name || !!document.getElementById(name)", fragment)
            receipt["targets"].append({"label": link["label"], "path": name, "fragment": fragment,
                                       "http_status": response.status, "heading": heading,
                                       "canonical_page_id": canonical["page_id"] if canonical else None,
                                       "heading_matches_freeze": ok, "anchor_exists": anchor_ok})
        browser.close()
finally:
    server.shutdown()
receipt["all_match"] = bool(receipt["targets"]) and all(t["heading_matches_freeze"] and t["anchor_exists"] for t in receipt["targets"])
receipt["preview_html_sha256"] = hashlib.sha256((PREVIEW / "chapter-19.html").read_bytes()).hexdigest()
with (PROOF / "navigation-receipt-v1.json").open("x", encoding="utf-8") as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
assert receipt["all_match"]
