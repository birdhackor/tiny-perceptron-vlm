from pathlib import Path
import argparse,json,hashlib,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from functools import partial
from playwright.sync_api import sync_playwright
B=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text())
extra=B/'checks/supplemental-visual-assets.json'
if extra.exists():
 for name,record in json.loads(extra.read_text())['files'].items():
  M['figures_sha256'][name]=record['sha256']
p=argparse.ArgumentParser();p.add_argument('--figure');a=p.parse_args();names=[a.figure] if a.figure else sorted(M['figures_sha256'])
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args): pass
server=ThreadingHTTPServer(("127.0.0.1",0),partial(Quiet,directory=str(B/"freeze/original")))
threading.Thread(target=server.serve_forever,daemon=True).start()
base="http://127.0.0.1:"+str(server.server_port)+"/"
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=browser.new_page(reduced_motion='reduce')
 for name in names:
  assert name in M['figures_sha256']
  path=B/'freeze/original'/name
  for width in [640,360]:
   out=B/'renders'/f'{Path(name).stem}-{width}.png'
   if out.exists():continue
   page.set_viewport_size({'width':width,'height':1200})
   page.goto(base+name);page.evaluate('async()=>{await document.fonts.ready}')
   page.evaluate('(w)=>{const s=document.documentElement;const v=s.viewBox.baseVal;s.style.width=w+"px";s.style.height=(v.width?v.height/v.width*w:600)+"px";s.style.background="white"}',width)
   page.locator('svg').screenshot(path=str(out),animations='disabled')
  print(name,flush=True)
 browser.close()

server.shutdown();server.server_close()
