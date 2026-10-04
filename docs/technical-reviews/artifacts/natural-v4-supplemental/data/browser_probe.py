from pathlib import Path
import hashlib,json
from playwright.sync_api import sync_playwright

OUT=Path('docs/technical-reviews/artifacts/natural-v4-supplemental/data')
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
    response=page.goto('http://127.0.0.1:8782/natural-v4-data.html',wait_until='networkidle')
    content=page.locator('main').inner_text()
    (OUT/'browser-DATA.visible-text.txt').write_text(content)
    page.screenshot(path=str(OUT/'browser-DATA.top.png'))
    records.append({'action':'goto','url':page.url,'status':response.status,'title':page.title(),'visible_assertions':['2,077' in content,'146.54 MB' in content,'9a61ecf524c9518f33f1501c28aa72997d4a82d0' in content]})
    links=page.locator('main a').evaluate_all('(as)=>as.map(a=>({text:a.innerText,href:a.getAttribute("href")}))')
    records.append({'main_links':links})
    for text in ['20.4 的貓照片','20.11','AISHELL 問句來源']:
        page.goto('http://127.0.0.1:8782/natural-v4-data.html',wait_until='networkidle')
        link=page.locator('main a').filter(has_text=text).first
        link.click()
        page.wait_for_load_state('networkidle')
        label={'20.4 的貓照片':'cat','20.11':'order','AISHELL 問句來源':'aishell'}[text]
        page.screenshot(path=str(OUT/('browser-link-'+label+'.png')))
        visible=page.locator('body').inner_text()
        (OUT/('browser-link-'+label+'.visible-text.txt')).write_text(visible)
        records.append({'action':'click','link_text':text,'url':page.url,'title':page.title(),'visible_text_saved':str(OUT/('browser-link-'+label+'.visible-text.txt'))})
    browser.close()
(OUT/'browser-receipt.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
