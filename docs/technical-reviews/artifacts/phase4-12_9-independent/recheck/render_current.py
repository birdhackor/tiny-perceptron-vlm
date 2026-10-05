"""Render the personally checked revised source artifact, without production parity claims."""
import hashlib
import json
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
raw = (OUT / "section.md").read_bytes()
digest = hashlib.sha256(raw).hexdigest()
body = markdown.markdown(raw.decode(), extensions=["fenced_code"])
html = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>body{margin:0;font:18px/1.8 sans-serif;color:#172b45;background:#fff}article{max-width:880px;margin:32px auto;padding:0 20px}aside{overflow-wrap:anywhere;font-size:14px;background:#edf4fb;padding:10px}pre{background:#f3f4f6;padding:14px;overflow:auto;font:14px/1.5 monospace}code{font-family:monospace}h2{font-size:28px}</style><article><aside>獨立複查 current source artifact；不是 production 頁面。SHA-256: ' + digest + '</aside>' + body + '</article></html>'
(OUT / "current-artifact.html").write_text(html)
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"], headless=True)
    for name, w, h in [("desktop",1280,800),("mobile",390,844)]:
        page = browser.new_page(viewport={"width":w,"height":h},device_scale_factor=1)
        page.set_content(html,wait_until="load")
        assert page.locator("pre code").inner_text().rstrip() == (OUT / "fence-1.py").read_text().rstrip()
        assert "三個錯例的頻率都是 290 或 300 Hz" in page.locator("article").inner_text()
        assert "資料把振幅與時長一起改動，無法分辨兩者各自的影響" in page.locator("article").inner_text()
        page.screenshot(path=str(OUT/f"current-artifact-{name}.png"),full_page=True)
        records.append({"viewport":[w,h],"source_sha256":digest,"screenshot":f"current-artifact-{name}.png","original_fence_matches":True,"section_image_count":page.locator("article img").count(),"scope":"Revised source artifact only; production/live page parity unasserted."})
        page.close()
    browser.close()
(OUT / "render-receipt.json").write_text(json.dumps({"source_sha256":digest,"records":records},ensure_ascii=False,indent=2)+'\n')
print("Revised current source artifact rendered at 1280x800 and390x844; revised text and unchanged fence verified; no production parity claim.")
