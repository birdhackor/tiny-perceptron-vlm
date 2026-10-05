"""Isolated frozen section render and prior-context SVG inspection."""
from pathlib import Path
import hashlib
import json
import markdown
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
fig = ROOT / "course/figures/rewrite-02-embedding-purpose.svg"
(OUT / "inputs/prior-2.1-embedding-purpose.svg").write_bytes(fig.read_bytes())
body = markdown.markdown((OUT / "inputs/section14.6.md").read_text(), extensions=["fenced_code", "tables"])
html = """<!doctype html><html lang='zh-Hant'><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>
body { font-family: 'Noto Sans CJK TC',sans-serif; font-size:18px;line-height:1.7;margin:0 auto;padding:24px;max-width:920px; }
pre { font-size:14px; overflow:auto; background:#f0f2f5;padding:14px; } code { font-family:monospace; } summary { font-weight:bold; }
@media(max-width:600px){body{padding:16px;font-size:16px} pre{font-size:13px}}
</style><body>""" + body + "</body></html>"
(OUT / "section-render.html").write_text(html)
records = {"command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-14_6-independent/render.py",
    "scope": "isolated frozen Markdown rendering with reviewer CSS; not a published course-site validation; no own referenced figures",
    "prior_figure": {"path": str(fig.relative_to(ROOT)), "sha256": hashlib.sha256(fig.read_bytes()).hexdigest()}, "views": []}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"], headless=True)
    records["chromium_version"] = browser.version
    for name,width,height in [("desktop",1280,800),("mobile",390,844)]:
        page = browser.new_page(viewport={"width":width,"height":height},device_scale_factor=1)
        page.set_content(html, wait_until="load")
        page.locator("details").evaluate_all("nodes => nodes.forEach(node => node.open = true)")
        page.screenshot(path=str(OUT / (name+"-section.png")), full_page=True)
        records["views"].append({"name":name,"width":width,"height":height,"path":name+"-section.png","details_open":True})
        page.close()
    page = browser.new_page(viewport={"width":1100,"height":850},device_scale_factor=1)
    page.set_content((OUT / "inputs/prior-2.1-embedding-purpose.svg").read_text(), wait_until="load")
    page.screenshot(path=str(OUT / "prior-2.1-figure.png"),full_page=True)
    browser.close()
(OUT / "render-execution.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(records,ensure_ascii=False))
