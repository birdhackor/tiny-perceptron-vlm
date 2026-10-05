"""Render the frozen section in Chromium; no external page or resource requests."""
from pathlib import Path
import json
import markdown
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
source = (OUT / "original-fence/section.md").read_text()
body = markdown.markdown(source, extensions=["fenced_code", "tables"])
html = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>
body{font:18px/1.7 sans-serif;margin:0;color:#202124;background:white}
main{max-width:900px;margin:auto;padding:24px}h2{font-size:26px;line-height:1.4}
pre{background:#f3f4f5;padding:16px;overflow-x:auto;font:15px/1.5 monospace}
code{font-family:monospace}details{border:1px solid #ddd;padding:12px}
@media(max-width:600px){main{padding:14px}body{font-size:17px}h2{font-size:24px}}
</style><main>""" + body + "</main></html>"
(OUT / "section-render.html").write_text(html)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                               args=["--no-sandbox", "--disable-gpu"])
    for name, w, h in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": w, "height": h})
        page.route("**/*", lambda route: route.abort())
        page.set_content(html, wait_until="load", timeout=15000)
        page.locator("details").evaluate_all("nodes=>nodes.forEach(n=>n.open=true)")
        page.screenshot(path=str(OUT / ("section-"+name+".png")), full_page=True, timeout=15000)
        print(json.dumps({"viewport": [w,h], "rendered_section": "13.3", "details_open": True,
                          "screenshot": "section-"+name+".png", "browser_version": browser.version}))
        page.close()
    browser.close()
