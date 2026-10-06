import hashlib,json,platform,sys,time
from pathlib import Path
from playwright.sync_api import sync_playwright
A=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-7_15-independent/reinspection-20261006')
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1280,'height':800},device_scale_factor=1)
 page.set_default_timeout(10000);response=page.goto('http://127.0.0.1:8765/7.15.html',wait_until='domcontentloaded',timeout=15000)
 page.locator('details').evaluate_all('(nodes)=>nodes.forEach(n=>n.open=true)')
 page.locator('details').first.scroll_into_view_if_needed()
 page.screenshot(path=str(A/'current-page.png'),full_page=False,animations='disabled')
 content=page.content();(A/'current-page.html').write_text(content)
 links=page.locator('a').evaluate_all('(nodes)=>nodes.map(a=>({text:a.innerText,href:a.getAttribute("href")}))')
 target=[x for x in links if '保留的直接SFT屬性模型' in x['text']]
 assert len(target)==1 and '7.13' in target[0]['href']
 text=page.locator('details').inner_text();assert '保留的直接SFT屬性模型' in text and '49筆B題更新500次' in text
 receipt={'url':page.url,'http_status':response.status,'browser':browser.version,'viewport':{'width':1280,'height':800},'details_opened':True,'actual_target_link':target,'svg_elements':page.locator('svg').count(),'article_img_elements':page.locator('main img').count(),'screenshot_path':str((A/'current-page.png').relative_to(A)),'screenshot_sha256':hashlib.sha256((A/'current-page.png').read_bytes()).hexdigest(),'html_sha256':hashlib.sha256((A/'current-page.html').read_bytes()).hexdigest(),'scope':'Current 7.15 page with supplementary record expanded; only base-checkpoint pointer wording changed substantively, no lesson image to compare. Screenshot must be personally viewed separately.'}
 (A/'page-render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
 browser.close()
