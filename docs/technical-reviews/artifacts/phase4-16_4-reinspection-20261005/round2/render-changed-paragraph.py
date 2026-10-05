from pathlib import Path
import json,hashlib,sys
from playwright.sync_api import sync_playwright
p=Path(__file__).resolve().parent;record={"url":"http://127.0.0.1:8765/16.4.html","scope":"Only current corrected paragraph examined on page; unchanged diagram/page/CPU evidence reused after exact fingerprints confirmed.","viewports":[]}
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path="/usr/bin/chromium",headless=True,timeout=15000,args=["--no-sandbox","--disable-dev-shm-usage"]);record["browser_version"]=browser.version
 for label,size in [("desktop",{"width":1280,"height":800}),("mobile",{"width":390,"height":844})]:
  page=browser.new_page(viewport=size,device_scale_factor=1);response=page.goto(record["url"],wait_until="domcontentloaded",timeout=15000)
  page.evaluate("document.querySelectorAll('details').forEach(x=>x.open=true)")
  paragraph=page.locator("main p").filter(has_text="這份品質表使用").first;paragraph.wait_for(timeout=10000);text=paragraph.inner_text()
  assert "音高標記（high／low）" in text and "位置" not in text and "7.13保存的屬性問答基準" in text
  paragraph.screenshot(path=str(p/(label+"-changed-paragraph.png")),timeout=10000)
  html=paragraph.evaluate("e=>e.outerHTML");(p/(label+"-changed-paragraph.html")).write_text(html)
  record["viewports"].append({"label":label,"viewport":size,"http_status":response.status,"text":text,"html_sha256":hashlib.sha256(html.encode()).hexdigest(),"box":paragraph.bounding_box(),"links":paragraph.locator("a").evaluate_all("els=>els.map(x=>({text:x.textContent,href:x.getAttribute('href')}))")});page.close()
 browser.close()
record["success"]=True;(p/"changed-page-render.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n");print(json.dumps(record,ensure_ascii=False))
