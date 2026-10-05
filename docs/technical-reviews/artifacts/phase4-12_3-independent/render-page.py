from pathlib import Path
import json, hashlib, subprocess
from playwright.sync_api import sync_playwright
a=Path(__file__).resolve().parent
records=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox","--disable-background-networking"],timeout=15000)
 for name,size in [("desktop",{"width":1280,"height":1000}),("mobile",{"width":390,"height":844})]:
  context=browser.new_context(viewport=size,device_scale_factor=1)
  # Abort only external destinations to keep this local content inspection bounded.
  blocked=[]
  def route_handler(route):
   if route.request.url.startswith(("http://127.0.0.1:8765/","data:","blob:")):
    route.continue_()
   else:
    blocked.append(route.request.url);route.abort()
  context.route("**/*",route_handler)
  page=context.new_page()
  page.goto("http://127.0.0.1:8765/12.3.html",wait_until="domcontentloaded",timeout=15000)
  page.locator("article img.diagram").wait_for(state="visible",timeout=10000)
  page.screenshot(path=str(a/f"page-{name}.png"),full_page=True,timeout=10000)
  records.append({"viewport":size,"name":name,"page_title":page.title(),"h1":page.locator("h1").inner_text(),"blocked_external_urls":blocked,"figure_loaded":page.locator("article img.diagram").evaluate("e => ({complete:e.complete,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight})"),"screenshot_sha256":hashlib.sha256((a/f"page-{name}.png").read_bytes()).hexdigest()})
  context.close()
 browser.close()
(a/"playwright-render.json").write_text(json.dumps({"chromium":subprocess.run(["/usr/bin/chromium","--version"],capture_output=True,text=True).stdout.strip(),"records":records},ensure_ascii=False,indent=2)+"\n")
print(json.dumps(records,ensure_ascii=False,indent=2))
