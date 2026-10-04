from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
    response=page.goto('http://127.0.0.1:8769/natural-v4-training.html',wait_until='networkidle')
    record={'requested_url':'http://127.0.0.1:8769/natural-v4-training.html','final_url':page.url,'status':response.status,'browser_version':browser.version,'viewport':{'width':1280,'height':900},'headings':page.locator('h1,h2,h3').evaluate_all('(els)=>els.map(e=>({tag:e.tagName,id:e.id,text:e.innerText}))'),'links':page.locator('a').evaluate_all('(els)=>els.map(e=>({label:e.innerText,href:e.getAttribute("href"),resolved:e.href}))'),'images':page.locator('img').evaluate_all('(els)=>els.map(e=>({src:e.src,alt:e.alt,loaded:e.complete&&e.naturalWidth>0,width:e.naturalWidth,height:e.naturalHeight}))'),'body_text_file':'training.browser.text.txt','screenshots':[]}
    (OUT/record['body_text_file']).write_text(page.locator('body').inner_text(),encoding='utf-8')
    for slug,pattern in [('top',None),('save-recipe','4. 保存配方'),('validation','6. 先驗證用途'),('final-test','7. 固定選定版本'),('own-model','8. 使用自己的模型')]:
        if pattern: page.get_by_role('heading',name=pattern,exact=False).scroll_into_view_if_needed()
        else: page.evaluate('window.scrollTo(0,0)')
        fn=f'training.browser.{slug}.png'
        page.screenshot(path=str(OUT/fn))
        record['screenshots'].append(fn)
    (OUT/'training.browser.receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))
    browser.close()
