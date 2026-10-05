"""Render the frozen B.2 Markdown for visual factual inspection, not site acceptance."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import markdown
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
raw = (OUT / "section.md").read_bytes()
body = markdown.markdown(raw.decode("utf-8"), extensions=["fenced_code"])
css = """
body{margin:0;font-family:'Noto Sans CJK TC','Noto Sans CJK SC',sans-serif;color:#222;background:white}
main{max-width:900px;margin:32px auto;padding:0 20px;font-size:17px;line-height:1.65}
h2{font-size:26px}pre{background:#f3f4f6;padding:16px;overflow:auto;font-size:14px;line-height:1.5}
code{font-family:monospace}details{border:1px solid #bbb;padding:16px}summary{font-weight:bold}
a{color:#1254a1}
@media(max-width:500px){main{margin:16px auto;padding:0 14px;font-size:16px}pre{font-size:12px;padding:10px}}
"""
html = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>B.2 frozen review input</title><style>' + css + '</style><main>' + body + '</main></html>'
(OUT / "rendered-b2.html").write_text(html, encoding="utf-8")
views = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    for label, width, height in [("desktop",1280,900),("mobile",390,844)]:
        page = browser.new_page(viewport={"width":width,"height":height}, device_scale_factor=1)
        page.set_content(html, wait_until="load")
        page.locator("details").evaluate_all("elements => elements.forEach(element => element.open = true)")
        page.screenshot(path=str(OUT / (label + ".png")), full_page=True)
        views.append({"viewport":[width,height],"image":label+".png","details":"expanded",
            "body_has_current_user_marker_example": "專用 user 角色標記" in page.locator("main").inner_text(),
            "svg_count":page.locator("svg,img").count(),"heading":page.locator("h2").inner_text()})
        page.close()
    version = browser.version
    browser.close()
record = {"source_sha256":hashlib.sha256(raw).hexdigest(),"chromium":version,
    "input_transport":"page.set_content from saved HTML; initial file navigation was blocked by browser administrative policy, preserved as attempt1 stderr",
    "markdown":importlib.metadata.version("markdown"),"playwright":importlib.metadata.version("playwright"),
    "scope":"Actual browser render of own frozen Markdown; no diagrams referenced. This is factual visual inspection, not reader or production site acceptance.","views":views}
(OUT / "render-provenance.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
