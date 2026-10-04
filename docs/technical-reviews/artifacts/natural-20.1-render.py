from pathlib import Path
import json,hashlib,importlib.metadata
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[3]
art=root/'docs/technical-reviews/artifacts'
svg=(root/'course/figures/natural_shared_chat.svg').read_text()
meta={'source_sha256':hashlib.sha256(svg.encode()).hexdigest(),'playwright':importlib.metadata.version('playwright'),'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-20.1-render.py','renders':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 meta['browser']=browser.version
 for name,width in [('desktop',720),('mobile',358)]:
  page=browser.new_page(viewport={'width':width,'height':1100},device_scale_factor=1)
  page.set_content('<html><head><style>html,body{margin:0;padding:0}svg{display:block;width:100%;height:auto}</style></head><body>'+svg+'</body></html>')
  page.evaluate('document.fonts.ready')
  element=page.locator('svg');path=art/f'natural-20.1-{name}.png';element.screenshot(path=str(path))
  box=element.bounding_box()
  texts=page.locator('svg text').evaluate_all('(es)=>es.map(e=>({text:e.textContent,rect: e.getBoundingClientRect().toJSON()}))')
  meta['renders'].append({'name':name,'viewport_width':width,'bbox':box,'png':str(path.relative_to(root)),'texts':texts})
  page.close()
 browser.close()
(art/'natural-20.1-render.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k!='renders'},ensure_ascii=False));print([(r['name'],r['bbox']) for r in meta['renders']])
