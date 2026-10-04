import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/technical-reviews/artifacts'
svgpath = ROOT / 'course/figures/natural_base_adapter.svg'
svg = svgpath.read_text()
html = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>body{margin:0;background:#fff}svg{display:block;width:100%;height:auto}</style>'+svg+'</html>'
(OUT / 'natural-20.4-render.html').write_text(html)
observations=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for width in [720,358]:
        page=browser.new_page(viewport={'width':width,'height':1300},device_scale_factor=1)
        page.set_content(html)
        page.evaluate('document.fonts.ready')
        path=OUT / f'natural-20.4-svg-{width}.png'
        page.locator('svg').screenshot(path=str(path))
        bounds=page.evaluate('''() => [...document.querySelectorAll('svg text')].map(t=>{
        const b=t.getBBox();return {text:t.textContent,x:b.x,y:b.y,width:b.width,height:b.height,
        inside:b.x>=0&&b.y>=0&&b.x+b.width<=720&&b.y+b.height<=1200};})''')
        assert all(item['inside'] for item in bounds)
        observations.append({'width':width,'screenshot':str(path.relative_to(ROOT)),
          'screenshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'svg_sha256':hashlib.sha256(svgpath.read_bytes()).hexdigest(),
          'all_text_inside_viewbox':True,'text_bounds':bounds,'browser_version':browser.version})
        page.close()
    browser.close()
(OUT/'natural-20.4-render.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='text_bounds'} for r in observations],ensure_ascii=False,indent=2))
