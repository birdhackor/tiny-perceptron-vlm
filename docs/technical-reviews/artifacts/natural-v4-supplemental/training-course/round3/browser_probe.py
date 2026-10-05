from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[6]
D = Path(__file__).parent
BASE = 'http://127.0.0.1:8789/'
receipt = {'base':BASE,'preview_revision':'160fd47dede5c2453c8a08dd535290ddfd0b915d','events':[]}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.goto(BASE+'training.html#T.8',wait_until='networkidle')
    prerequisite = page.get_by_role('link',name='16.1',exact=True)
    actual_href = prerequisite.get_attribute('href')
    prerequisite.click()
    page.wait_for_load_state('networkidle')
    receipt['events'].append({'action':'Navigate from actualtraining T.8 real16.1 hyperlink','href':actual_href,'url':page.url,'heading':page.locator('[id="16.1"]').inner_text()})
    link = page.get_by_role('link',name='T.8的效率量測方法',exact=True)
    link.scroll_into_view_if_needed()
    href = link.get_attribute('href')
    page.screenshot(path=str(D/'browser-16.1-method-link.png'))
    receipt['events'].append({'action':'Actual necessary16.1 methodlink visible before click','url':page.url,'href':href,'label':link.inner_text(),'screenshot':'browser-16.1-method-link.png'})
    link.click()
    page.wait_for_load_state('networkidle')
    heading = page.get_by_role('heading',name='選讀：效率實驗的量測條件',exact=False)
    assert heading.count()==1
    heading.scroll_into_view_if_needed()
    page.screenshot(path=str(D/'browser-training-method-heading.png'))
    receipt['events'].append({'action':'Actual click16.1→fulltraining.html newh3','url':page.url,'heading':heading.inner_text(),'heading_id':heading.get_attribute('id'),'heading_tag':heading.evaluate('e=>e.tagName'),'visible':heading.is_visible(),'screenshot':'browser-training-method-heading.png'})
    scope = heading.evaluate("e=>{let n=e.nextElementSibling,out=[e.innerText];while(n&&n.tagName!=='H2'&&n.tagName!=='H3'){out.push(n.innerText);n=n.nextElementSibling;}return out.join('\\n');}")
    (D/'browser-method-DOM.txt').write_text(scope)
    receipt['events'].append({'action':'Capture actual complete newh3 scope to nextH2/H3','chars':len(scope),'sha256':hashlib.sha256(scope.encode()).hexdigest(),'saved':'browser-method-DOM.txt'})
    memory = page.get_by_role('link',name='2.14的max_memory_allocated文件',exact=True)
    receipt['events'].append({'action':'Current officialAPI href remains canonical','href':memory.get_attribute('href')})
    entrance = page.get_by_role('link',name='本節入口',exact=True)
    entrance.scroll_into_view_if_needed()
    entrance.click()
    receipt['events'].append({'action':'Actual click methodsection→T.8入口','url':page.url,'heading':page.locator('[id="T.8"]').inner_text()})
    browser.close()
for name in ('browser-16.1-method-link.png','browser-training-method-heading.png'):
    receipt.setdefault('screenshots',[]).append({'path':name,'sha256':hashlib.sha256((D/name).read_bytes()).hexdigest(),'personal_view':'pending'})
(D/'browser-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
