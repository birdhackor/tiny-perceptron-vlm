from pathlib import Path
import json, hashlib, time
from playwright.sync_api import sync_playwright
base=Path(__file__).parent
out=base/'browser'
out.mkdir(exist_ok=True)
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    def capture(name,how):
        page.wait_for_load_state('networkidle')
        article=page.locator('article').first
        body=article.inner_text() if article.count() else page.locator('body').inner_text()
        mainlinks=article.locator('a').evaluate_all('(els)=>els.map(a=>({label:a.innerText,href:a.href}))') if article.count() else []
        headings=article.locator('h1,h2,h3,h4').evaluate_all('(els)=>els.map(a=>({text:a.innerText,id:a.id}))') if article.count() else []
        images=article.locator('img').evaluate_all('(els)=>els.map(a=>({alt:a.alt,src:a.src,naturalWidth:a.naturalWidth,naturalHeight:a.naturalHeight}))') if article.count() else []
        (out/(name+'.txt')).write_text(body,encoding='utf-8')
        (out/(name+'.html')).write_text(page.content(),encoding='utf-8')
        page.screenshot(path=str(out/(name+'-full.png')),full_page=True)
        height=page.evaluate('document.documentElement.scrollHeight')
        shots=[]
        for idx,y in enumerate(range(0,height,820)):
            page.evaluate('(y)=>window.scrollTo(0,y)',y)
            page.screenshot(path=str(out/(name+f'-viewport-{idx}.png')))
            shots.append({'file':name+f'-viewport-{idx}.png','scroll_y_requested':y,'scroll_y_actual':page.evaluate('scrollY')})
        records.append({'name':name,'url':page.url,'how':how,'title':page.title(),'headings':headings,'links':mainlinks,'images':images,'viewport_screens':shots,'body_sha256':hashlib.sha256(body.encode('utf-8')).hexdigest(),'browser':'Chromium via Playwright, /usr/bin/chromium, headless; real HTTP preview'} )
        return body
    page.goto('http://127.0.0.1:8786/')
    page.wait_for_load_state('networkidle')
    entries=page.locator('a').evaluate_all('(els)=>els.filter(a=>/validation\\.html|publishing\\.html/.test(a.href)).map(a=>({label:a.innerText,href:a.href,visible:!!(a.offsetWidth||a.offsetHeight||a.getClientRects().length)}))')
    (out/'home-entry-links.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    page.screenshot(path=str(out/'home-entry.png'))
    val=page.locator('a[href$="validation.html"]').filter(visible=True).first
    val.click()
    print('VALIDATION\n'+capture('validation','Clicked visible home/navigation link 教材實作驗證'))
    page.locator('article a[href$="publishing.html"]').click()
    print('PUBLISHING\n'+capture('publishing','Clicked final in-article 教材發布 link from validation'))
    (out/'browser-records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    browser.close()
