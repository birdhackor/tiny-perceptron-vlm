import json
from pathlib import Path
from playwright.sync_api import sync_playwright
base=Path(__file__).resolve().parent
results={}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path="/usr/bin/chromium",args=["--no-sandbox"],headless=True)
 for name,size in [("desktop",{"width":1280,"height":900}),("mobile",{"width":390,"height":844})]:
  page=browser.new_page(viewport=size,device_scale_factor=1)
  page.goto("http://127.0.0.1:8765/11.17.html",wait_until="networkidle",timeout=30000)
  article=page.locator("main")
  text=article.inner_text()
  (base/f"{name}-main-text.txt").write_text(text)
  assert "11.17 有限物件與短字詞，如何做成可學的看圖問答？" in text
  assert "同一個包的裁切、縮放或不同配對" in text
  imgs=page.locator('main img[src*="new-11.17-"]')
  assert imgs.count()==2
  image_info=[]
  for i in range(imgs.count()):
   img=imgs.nth(i)
   img.scroll_into_view_if_needed()
   page.wait_for_timeout(100)
   image_info.append(img.evaluate("e=>({src:e.src,alt:e.alt,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,rect:e.getBoundingClientRect().toJSON()})"))
   page.screenshot(path=str(base/f"{name}-figure-{i+1}.png"))
  article.screenshot(path=str(base/f"{name}-main-full.png"))
  results[name]={"viewport":size,"image_info":image_info,"title":page.title(),"main_text_saved":f"{name}-main-text.txt","body_scroll_width":page.evaluate("document.body.scrollWidth"),"window_inner_width":page.evaluate("window.innerWidth")}
  page.close()
 browser.close()
(base/"render-metadata.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(results,ensure_ascii=False,indent=2))
