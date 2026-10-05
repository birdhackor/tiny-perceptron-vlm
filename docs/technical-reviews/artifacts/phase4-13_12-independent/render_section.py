"""Render exactly this frozen section at two viewport sizes for visual inspection."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import json
import markdown

DEST=Path(__file__).resolve().parent
body=markdown.markdown((DEST/'frozen-input/section.md').read_text(),extensions=['fenced_code'])
html='<!doctype html><meta charset="utf-8"><style>body{font:18px/1.65 sans-serif;max-width:880px;margin:24px auto;padding:0 20px;color:#20252b}pre{background:#f4f6f8;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;font:14px/1.5 monospace}code{overflow-wrap:anywhere}h2{font-size:1.4em}details{margin-top:24px}a{color:#155ba5}</style>'+body
(DEST/'section-review.html').write_text(html)
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=25000)
    for width,height in [(1280,800),(390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.set_content(html,wait_until='load',timeout=25000)
        path=DEST/f'section-{width}x{height}.png'
        page.screenshot(path=str(path),full_page=True,timeout=25000)
        records.append({'viewport':[width,height],'screenshot':path.name,'scrollWidth':page.evaluate('document.documentElement.scrollWidth'),'clientWidth':page.evaluate('document.documentElement.clientWidth')})
        page.close()
    browser.close()
(DEST/'render-result.json').write_text(json.dumps({'renderer':'Chromium / Playwright, isolated frozen section HTML; not built course theme','screenshots':records},indent=2)+'\n')
print(json.dumps(records,indent=2))
