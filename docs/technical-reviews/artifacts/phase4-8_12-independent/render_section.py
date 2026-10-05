"""Render the exact section text locally; this section has no source figure."""
import hashlib,json,sys
from pathlib import Path
from importlib.metadata import version
import markdown
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
raw=(out/'original/section.md').read_bytes()
body=markdown.markdown(raw.decode('utf-8'),extensions=['fenced_code'])
html='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>8.12 原文檢視</title><style>body{margin:0;font:17px/1.65 sans-serif;color:#161616;background:#fff}main{max-width:860px;margin:32px auto;padding:0 20px}h2{font-size:24px}pre{font:14px/1.6 monospace;overflow-x:auto;background:#f4f4f4;padding:14px}code{background:#f4f4f4} @media(max-width:500px){main{margin:20px auto;padding:0 14px}h2{font-size:22px}pre{font-size:12px}}</style><main>'+body+'</main></html>'
(out/'section.html').write_text(html,encoding='utf-8')
receipts=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking'])
 for name,w,h in [('desktop',1280,800),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
  page.route('**/*',lambda route: route.continue_() if route.request.url.startswith('file:') else route.abort())
  page.set_content(html,wait_until='load',timeout=10000)
  page.screenshot(path=str(out/(name+'.png')),full_page=True,timeout=10000)
  receipts.append({'viewport':{'width':w,'height':h},'file':name+'.png','sha256':hashlib.sha256((out/(name+'.png')).read_bytes()).hexdigest(),'document_scroll_width':page.evaluate('document.documentElement.scrollWidth'),'rendered_image_count':page.locator('img,svg').count(),'visible_heading':page.locator('h2').inner_text(),'source_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Exact Markdown section body rendered in a standalone local typography harness. There is no original figure to render; this is not a production-course layout check.'})
  page.close()
 browser.close()
result={'python':sys.version,'markdown':version('markdown'),'playwright':version('playwright'),'chromium':'151.0.7922.173','source_sha256':hashlib.sha256(raw).hexdigest(),'html_sha256':hashlib.sha256((out/'section.html').read_bytes()).hexdigest(),'screenshots':receipts,'tool_limitations':'Initial file:// navigation was blocked by Chromium policy (net::ERR_BLOCKED_BY_ADMINISTRATOR). Final render uses page.set_content with the exact saved local HTML, no server, no HTTP navigation, and all external requests aborted.'}
(out/'render-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
