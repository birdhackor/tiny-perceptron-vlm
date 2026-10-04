from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/tiny-perceptron-vlm');OUT=ROOT/'docs/reader-reviews/artifacts/natural-v4-supplemental/training-course'
refs=json.loads((OUT/'prerequisite-read-receipt.json').read_text())
receipts=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1800,'height':1400},device_scale_factor=1)
 for section in refs:
  for ref in section['svg_refs']:
   source=(ROOT/section['source']).parent/ref;source=source.resolve();raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest()
   snapshot=OUT/(source.stem+'.'+h+'.svg');snapshot.write_bytes(raw)
   page.goto(snapshot.as_uri(),wait_until='load')
   page.locator('svg').screenshot(path=str(OUT/(source.stem+'.png')))
   png=(OUT/(source.stem+'.png')).read_bytes()
   receipts.append({'section':section['section'],'source':str(source.relative_to(ROOT)),'original_sha256':h,'source_snapshot':snapshot.name,'render':'Playwright Chromium /usr/bin/chromium; file:// snapshot URI; SVG element screenshot','render_url':snapshot.as_uri(),'png':source.stem+'.png','png_sha256':hashlib.sha256(png).hexdigest(),'svg_bounds':page.locator('svg').bounding_box()})
 b.close()
(OUT/'figure-render-receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipts,ensure_ascii=False,indent=2))
