"""Render the current reviewed Markdown section, not a generated course site.

This checks the presence of the actual section, code, and examples. There is no
referenced figure in 13.8, so diagram consistency remains not applicable.
"""
from pathlib import Path
from datetime import datetime, UTC
import importlib.metadata
import json
import markdown
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
source = (OUT / "inputs" / "section.md").read_text(encoding="utf-8")
html = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;background:#fff;color:#17212b;font:18px/1.8 sans-serif;}
main{max-width:900px;margin:auto;padding:24px;}
h2{font-size:1.4em;line-height:1.5;}
pre{background:#f2f4f7;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.5;}
code{font-family:monospace;} p{overflow-wrap:anywhere;}
@media(max-width:600px){body{font-size:16px;}main{padding:16px;}pre{font-size:13px;}}
</style><main>""" + markdown.markdown(source, extensions=["fenced_code"]) + "</main></html>"
(OUT / "section-render.html").write_text(html, encoding="utf-8")
record = {"started_at": datetime.now(UTC).isoformat(), "browser_executable": "/usr/bin/chromium",
          "markdown_version": importlib.metadata.version("Markdown"),
          "playwright_version": importlib.metadata.version("playwright"),
          "render_scope": "13.8 original source Markdown using local minimal review CSS; not a full-site visual acceptance test",
          "figures_referenced": 0, "viewports": []}
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"], timeout=15000)
        record["browser_version"] = browser.version
        for name, width, height in [("desktop",1280,800),("mobile",390,844)]:
            page = browser.new_page(viewport={"width":width,"height":height}, device_scale_factor=1)
            page.set_content(html, wait_until="load", timeout=15000)
            page.screenshot(path=str(OUT / f"section-{name}.png"), full_page=True, timeout=15000)
            record["viewports"].append({"name":name,"width":width,"height":height,
                "screenshot":f"section-{name}.png","rendered_text":page.locator("main").inner_text(),
                "horizontal_overflow":page.evaluate("document.documentElement.scrollWidth > window.innerWidth")})
            page.close()
        browser.close()
    record["exit_status"] = "success"
except Exception as error:
    record.update(exit_status="failed", error_type=type(error).__name__, error=str(error))
finally:
    (OUT / "render-results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
if record["exit_status"] != "success":
    raise SystemExit(1)
