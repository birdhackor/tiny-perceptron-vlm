"""Render actual existing README preview and SVG; screenshots remain outside Git."""
import datetime, functools, hashlib, http.server, json, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/work/omni-book-integrated-v1');RAW=Path('/workspace/work/omni-P5-readme-independent-raw');OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme'
PREVIEW=ROOT/'outputs/omni-P5-preview-v9'
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(PREVIEW)))
threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_address[1]
proof={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_readme','command':'python docs/technical-reviews/artifacts/omni-P5-readme/render_readme.py','preview_root':str(PREVIEW),'preview_html_sha256':hashlib.sha256((PREVIEW/'readme.html').read_bytes()).hexdigest(),'screenshots':[],'viewports':[]}
try:
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox'])
  proof['browser_version']=browser.version
  for name,width,height in [('desktop',1440,1000),('mobile',390,844)]:
   context=browser.new_context(viewport={'width':width,'height':height},reduced_motion='reduce');page=context.new_page();response=page.goto(f'http://127.0.0.1:{port}/readme.html',wait_until='networkidle');page.wait_for_timeout(300)
   picture=page.locator('img[alt="文字與識別編號的對照"]');picinfo=picture.evaluate('(e)=>({src:e.getAttribute("src"),complete:e.complete,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,rect:{x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height}})')
   for label,target in [('full',RAW/f'readme-{name}-full.png'),('figure',RAW/f'readme-{name}-figure.png')]:
    page.screenshot(path=str(target),full_page=True) if label=='full' else picture.screenshot(path=str(target))
    proof['screenshots'].append({'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'viewport':name,'kind':label})
   proof['viewports'].append({'name':name,'width':width,'height':height,'http_status':response.status,'document_width':page.evaluate('document.documentElement.scrollWidth'),'inner_width':page.evaluate('innerWidth'),'figure':picinfo,'title':page.title(),'text_contains_331':page.locator('article').inner_text().find('331')>=0})
   context.close()
  browser.close()
finally:server.shutdown()
with (OUT/'render-receipt-v1.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(proof,ensure_ascii=False,indent=2))
