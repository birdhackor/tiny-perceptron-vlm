from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
root=Path('docs/technical-reviews/artifacts/natural-v4-factual/20.1'); source=Path('course/figures/natural_shared_chat.svg')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':720,'height':1010},device_scale_factor=1)
 page.set_content('<html><head><style>html,body{margin:0;padding:0}svg{display:block;width:720px;height:1010px}</style></head><body>'+source.read_text()+'</body></html>')
 page.evaluate('document.fonts.ready')
 page.locator('svg').screenshot(path=str(root/'natural_shared_chat.png'))
 receipt={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'render':'natural_shared_chat.png','browser':b.version,'executable':'/usr/bin/chromium','viewport':{'width':720,'height':1010},'font_loaded':page.evaluate('document.fonts.check(\'34px "Noto Sans CJK TC"\')'),'method':'Direct current source SVG embedded at its intrinsic viewBox; no old screenshot reused'}
 (root/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));b.close()
