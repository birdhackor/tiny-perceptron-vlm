import json,time,subprocess
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parents[1]; raw=(A/'sources/section.md').read_text()
html='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>body{font-family: sans-serif;line-height:1.65;margin:24px;background:#fff;color:#111}article{max-width:960px;margin:auto}pre{background:#f1f3f6;overflow:auto;padding:14px;font-size:14px;line-height:1.4}code{font-family:monospace}details{border:1px solid #aaa;padding:10px;margin:14px 0}summary{cursor:pointer}img{max-width:100%}</style><article>'+markdown.markdown(raw,extensions=['fenced_code','tables'])+'</article></html>'
p=A/'render/section.html'; p.write_text(html); result={'command':'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/phase4-13_15-independent/code/render-section.py','renderer':'installed Chromium via Playwright','targets':[],'scope':'standalone exact section Markdown render, not exported course site'}
started=time.monotonic()
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=20000)
  result['chromium_version']=browser.version
  for label,w,h in [('desktop',1280,800),('mobile',390,844)]:
   page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1); page.set_content(html,timeout=20000); page.screenshot(path=str(A/'render'/f'{label}-top.png'))
   page.locator('details').evaluate_all('(els)=>els.forEach(e=>e.open=true)'); page.locator('pre').first.scroll_into_view_if_needed(); page.screenshot(path=str(A/'render'/f'{label}-code.png'))
   result['targets'].append({'label':label,'width':w,'height':h,'top':'render/'+label+'-top.png','expanded_code':'render/'+label+'-code.png','body_scroll_width':page.evaluate('document.body.scrollWidth')}); page.close()
  browser.close(); result['status']='completed'
except Exception as e: result['status']='failed'; result['error']=type(e).__name__+': '+str(e)
result['elapsed_seconds']=time.monotonic()-started; (A/'render/render-result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if result['status']!='completed': raise SystemExit(1)
