import functools,http.server,json,threading,datetime
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('/workspace/work/omni-book-integrated-v1');D=R/'docs/technical-reviews/artifacts/omni-P5-readme'
class H(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
s=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(H,directory=str(R/'outputs/omni-P5-preview-v9')));threading.Thread(target=s.serve_forever,daemon=True).start()
try:
 with sync_playwright() as pw:
  b=pw.chromium.launch(executable_path='/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1000});page.goto(f'http://127.0.0.1:{s.server_address[1]}/readme.html',wait_until='networkidle');page.wait_for_timeout(500)
  d={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_readme','command':'python docs/technical-reviews/artifacts/omni-P5-readme/inspect_ui_dom.py'}
  for tag in ['button','input','nav']:
   d[tag]=page.locator(tag).evaluate_all('(xs)=>xs.slice(0,35).map(x=>({tag:x.tagName,cls:x.className,type:x.type,role:x.getAttribute("role"),component:x.getAttribute("data-md-component"),label:x.getAttribute("aria-label"),title:x.title,id:x.id,text:x.innerText?.slice(0,100),visible:!!(x.offsetWidth||x.offsetHeight||x.getClientRects().length)}))')
  d['code_ancestors']=page.locator('article pre').first.evaluate('(x)=>x.parentElement.outerHTML.slice(0,1500)');d['clipboard_available']=page.evaluate('!!navigator.clipboard');b.close()
finally:s.shutdown()
with (D/'ui-dom-inspection-v1.json').open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(d,ensure_ascii=False,indent=2))
