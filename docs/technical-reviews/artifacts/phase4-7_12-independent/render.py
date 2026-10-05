"""Local Markdown snapshot rendering, with no remote assets and a 30-second launch limit."""
import json
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

base=Path(__file__).resolve().parent
body=markdown.markdown((base/'inputs/section.md').read_text().replace('<details>','<details markdown="1">'),extensions=['extra'])
html='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>body{font-family:"Noto Sans CJK TC",sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.7;color:#172033}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:8px;border:1px solid #ccc}pre{overflow:auto;background:#f4f6f8;padding:12px;font-size:14px}code{font-family:monospace}details{border:1px solid #ddd;padding:10px}a{color:#1658a8}</style>'+body+'</html>'
(base/'rendered-section.html').write_text(html)
receipt=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=30000)
 for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  page.route('http**/*',lambda route:route.abort())
  page.set_content(html,wait_until='load',timeout=15000)
  page.locator('details').evaluate_all('(els)=>els.forEach(el=>el.open=true)')
  page.screenshot(path=str(base/(name+'.png')),full_page=True,timeout=15000)
  receipt.append({'name':name,'width':width,'height':height,'document_height':page.evaluate('document.documentElement.scrollHeight'),'svg_count':page.locator('svg,img').count(),'browser':browser.version,'scope':'Local Markdown snapshot, tables/fence rendered; current section references no figure. Not a deployed course-page test.'})
  page.close()
 browser.close()
(base/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
