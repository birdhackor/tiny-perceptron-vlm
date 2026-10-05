from pathlib import Path
import hashlib,json,sys
import markdown
from playwright.sync_api import sync_playwright
base=Path(__file__).resolve().parent
source=base.parent/'original-cpu/section.md'
raw=source.read_bytes()
html='<!doctype html><html lang="zh-Hant"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font-family:"Noto Sans CJK TC","Noto Sans CJK SC",sans-serif;margin:24px auto;padding:0 22px;max-width:920px;line-height:1.7;color:#111;background:#fff}h2{font-size:24px}table{width:100%;border-collapse:collapse;font-size:15px}th,td{border:1px solid #777;padding:7px;text-align:left}pre{padding:12px;background:#f4f4f4;overflow:auto;font-size:13px;line-height:1.5}code{font-family:monospace}p{overflow-wrap:anywhere}</style><body>'+markdown.markdown(raw.decode('utf-8'),extensions=['tables','fenced_code'])+'</body></html>'
(base/'section.html').write_text(html)
results=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--disable-gpu','--no-sandbox'])
 for name,width,height in [('desktop',1280,900),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  page.route('http://**/*',lambda route: route.abort())
  page.route('https://**/*',lambda route: route.abort())
  page.set_content(html,wait_until='load',timeout=10000)
  page.screenshot(path=str(base/(name+'.png')),full_page=True,timeout=10000)
  results.append({'name':name,'viewport':[width,height],'title':page.locator('h2').inner_text(),'table_rows':page.locator('tbody tr').count(),'images':page.locator('img').count(),'document_width':page.evaluate('document.documentElement.scrollWidth'),'document_height':page.evaluate('document.documentElement.scrollHeight'),'screenshot':name+'.png'})
  page.close()
 version=browser.version;browser.close()
(base/'render-receipt.json').write_text(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'markdown_version':markdown.__version__,'python':sys.version,'chromium':version,'command':'timeout 30 .venv/bin/python docs/technical-reviews/artifacts/phase4-8_10-independent/visual/render_section.py','scope':'Local offline rendering of original section Markdown with simple review CSS; no referenced figures exist; not a canonical site styling or full site validation.','results':results},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False))
