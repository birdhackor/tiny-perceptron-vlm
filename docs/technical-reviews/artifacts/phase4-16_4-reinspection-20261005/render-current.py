from pathlib import Path
import json,sys,hashlib
from playwright.sync_api import sync_playwright
p=Path(__file__).resolve().parent
results={"url":"http://127.0.0.1:8765/16.4.html","viewports":[],"scope":"This owner navigated current section page; expanded optional evidence details. No other page reviewed."}
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path="/usr/bin/chromium",headless=True,timeout=15000,args=["--no-sandbox","--disable-dev-shm-usage"])
  results["browser_version"]=browser.version
  for label,size in [("desktop",{"width":1280,"height":800}),("mobile",{"width":390,"height":844})]:
   page=browser.new_page(viewport=size,device_scale_factor=1)
   response=page.goto(results["url"],wait_until="domcontentloaded",timeout=15000)
   page.locator("h1").first.wait_for(timeout=10000)
   page.evaluate("document.querySelectorAll('details').forEach(x => x.open=true)")
   page.locator("img[src*='rewrite-16-query-kv-sharing']").first.wait_for(timeout=10000)
   page.screenshot(path=str(p/(label+"-page.png")),full_page=True,timeout=10000)
   figure=page.locator("img[src*='rewrite-16-query-kv-sharing']").first
   figure.screenshot(path=str(p/(label+"-figure.png")),timeout=10000)
   html=page.content();(p/(label+"-page.html")).write_text(html)
   item={"label":label,"viewport":size,"status":response.status,"title":page.title(),"figure_box":figure.bounding_box(),"html_sha256":hashlib.sha256(html.encode()).hexdigest(),"body_text":page.locator("main").inner_text() if page.locator("main").count() else page.locator("body").inner_text(),"document_width":page.evaluate("document.documentElement.scrollWidth")}
   results["viewports"].append(item);page.close()
  browser.close()
 results["success"]=True
except Exception as e:
 results["success"]=False;results["error"]=repr(e)
(p/"page-render-receipt.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in results.items() if k!="viewports"},ensure_ascii=False))
print(json.dumps([{k:v for k,v in x.items() if k!="body_text"} for x in results["viewports"]],ensure_ascii=False))
raise SystemExit(0 if results.get("success") else 1)
