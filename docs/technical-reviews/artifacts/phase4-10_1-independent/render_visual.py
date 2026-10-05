"""Render the exact current section and figure locally using system Chromium."""
from pathlib import Path
import sys,base64,re,json,hashlib,subprocess,time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];sys.path.insert(0,str(ROOT))
import markdown
from playwright.sync_api import sync_playwright
from scripts.export_course import diagram_html
svg=ROOT/'course/figures/rewrite-10-pixels.svg'
raw=(OUT/'section.md').read_bytes();css=(ROOT/'course/web/course.css').read_bytes()
uri='data:image/svg+xml;base64,'+base64.b64encode(svg.read_bytes()).decode()
body=re.sub(r'!\[([^\]]*)\]\([^)]*\.svg\)',lambda m:diagram_html(uri,m[1],'factual-10-1'),raw.decode())
body=markdown.markdown(body,extensions=['fenced_code','tables','md_in_html'])
html='<html lang="zh-TW"><head><meta charset="utf-8"><style>html{font-size:20px}body{margin:0;font-family:"Noto Sans CJK TC",sans-serif;color:#17293c;background:white}main{max-width:780px;margin:24px auto;padding:0 16px;box-sizing:border-box}pre{overflow:auto;font-size:14px;background:#f6f7f8;padding:12px}code{font-family:monospace}'+css.decode()+'</style></head><body><main class="md-typeset">'+body+'</main></body></html>'
(OUT/'section-render.html').write_text(html)
started=time.perf_counter();renders=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=20000)
 for width,height in [(1280,800),(390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  page.set_default_timeout(20000);page.set_content(html,wait_until='load',timeout=20000)
  page.screenshot(path=str(OUT/f'section-{width}.png'),full_page=True)
  page.locator('figure').screenshot(path=str(OUT/f'figure-{width}.png'))
  measurement=page.locator('img.diagram').evaluate('(e)=>({complete:e.complete,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})')
  assert measurement['complete'] and measurement['naturalWidth']==640
  renders.append({'viewport':[width,height],'figure':measurement,'body_scroll_width':page.evaluate('document.body.scrollWidth')})
  page.close()
 version=browser.version;browser.close()
receipt={'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-10_1-independent/render_visual.py','exit_code':0,'elapsed_seconds':time.perf_counter()-started,'browser':version,'executable':'/usr/bin/chromium','screenshots':renders,'renderer':'Local set_content of exact current 10.1 raw source, original SVG via data URI, repository diagram_html and course.css; supplementary plain article framing. No server, no port 8765, no full-site build.','input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [svg,ROOT/'course/web/course.css',ROOT/'scripts/export_course.py']}}
(OUT/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
