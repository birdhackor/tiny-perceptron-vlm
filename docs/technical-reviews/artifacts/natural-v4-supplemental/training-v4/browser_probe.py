from pathlib import Path
import hashlib,json,datetime
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.goto('http://127.0.0.1:8782/',wait_until='networkidle')
    links=page.locator('a').evaluate_all('(xs)=>xs.map(x=>({text:x.innerText,href:x.href})).filter(x=>/TRAINING|訓練指引/.test(x.href+" "+x.text))')
    (out/'home-links.json').write_text(json.dumps(links,ensure_ascii=False,indent=2)+'\n')
    link=next((x for x in links if 'natural-assistant/v4/TRAINING' in x['href']),None)
    if link is None:
        page.goto('http://127.0.0.1:8782/natural-assistant/v4/TRAINING.html',wait_until='networkidle')
        route='direct TRAINING route, not a homepage link'
    else:
        page.locator('a[href]').filter(has_text=link['text']).first.click()
        page.wait_for_load_state('networkidle')
        route=link
    article=page.locator('article').inner_text()
    (out/'preview-training-visible.txt').write_text(article)
    page.screenshot(path=str(out/'preview-training-top.png'))
    all_links=page.locator('article a').evaluate_all('(xs)=>xs.map(x=>({text:x.innerText,href:x.href}))')
    (out/'preview-training-links.json').write_text(json.dumps(all_links,ensure_ascii=False,indent=2)+'\n')
    training_url=page.url
    page.locator('h2').filter(has_text='先驗證用途').scroll_into_view_if_needed()
    page.screenshot(path=str(out/'preview-training-validation.png'))
    student=next(x for x in all_links if 'STUDENT.html' in x['href'])
    page.locator('article a').filter(has_text=student['text']).first.click()
    page.wait_for_load_state('networkidle')
    target_title=page.locator('h1').inner_text()
    page.screenshot(path=str(out/'preview-student-link.png'))
    record={'accessed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_url':training_url,'initial_navigation':route,'training_h1':article.splitlines()[0],'all_8_sections_visible':all(str(n)+'.' in article or str(n)+'. ' in article for n in range(1,9)),'clicked':student,'target_url':page.url,'target_title':target_title,'browser':browser.version,'screenshots':['preview-training-top.png','preview-training-validation.png','preview-student-link.png'],'scope':'Executed existing preview personally; UI model serving/training was not started.'}
    (out/'browser-execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False))
    browser.close()
