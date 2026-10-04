from pathlib import Path
from hashlib import sha256
from playwright.sync_api import sync_playwright
import json
from urllib.request import urlopen
base=Path(__file__).parent; out=base/'browser';out.mkdir(exist_ok=True)
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    def capture(name,how,scope='complete article'):
        page.wait_for_load_state('networkidle')
        article=page.locator('article').first
        body=article.inner_text()
        (out/(name+'.txt')).write_text(body,encoding='utf-8')
        (out/(name+'.html')).write_text(page.content(),encoding='utf-8')
        page.screenshot(path=str(out/(name+'-full.png')),full_page=True)
        height=page.evaluate('document.documentElement.scrollHeight')
        shots=[]
        for idx,y in enumerate(range(0,height,820)):
            page.evaluate('(y)=>window.scrollTo(0,y)',y)
            path=out/(name+f'-viewport-{idx}.png');page.screenshot(path=str(path))
            shots.append({'path':str(path),'requested_scroll_y':y,'actual_scroll_y':page.evaluate('scrollY')})
        rec={'name':name,'actual_url':page.url,'how':how,'title':page.title(),'read_scope':scope,'article_text_sha256':sha256(body.encode()).hexdigest(),'headings':article.locator('h1,h2,h3,h4').evaluate_all('(els)=>els.map(x=>({text:x.innerText,id:x.id}))'),'links':article.locator('a').evaluate_all('(els)=>els.map(x=>({text:x.innerText,href:x.href}))'),'images':article.locator('img').evaluate_all('(els)=>els.map(x=>({alt:x.alt,src:x.src}))'),'viewport_captures':shots}
        records.append(rec)
        print('PAGE '+name+'\n'+body+'\n')
    page.goto('http://127.0.0.1:8788/',wait_until='networkidle')
    a=page.locator('a[href$="validation.html"]').filter(visible=True).first
    entry={'actual_visible_label':a.inner_text().strip(),'href':a.get_attribute('href'),'from_url':page.url}
    a.click();capture('validation','Clicked actual home link '+entry['actual_visible_label'])
    page.locator('article a[href$="publishing.html"]').click();capture('publishing','Clicked final in-article 教材發布 from validation')
    # The new direct setup link is read as an actual route and actual section, not only an anchor check.
    a=page.locator('article a').filter(has_text='W.1').first
    setup={'actual_visible_label':a.inner_text(),'href':a.get_attribute('href'),'from_url':page.url}
    a.click();page.wait_for_load_state('networkidle')
    content=page.evaluate('''()=>{const hs=[...document.querySelectorAll('article h2')];const h=hs.find(h=>h.innerText.startsWith('W.1'));const r=document.createRange();r.setStartBefore(h);const n=hs[hs.indexOf(h)+1];if(n)r.setEndBefore(n);else r.setEndAfter(document.querySelector('article').lastElementChild);return r.toString();}''')
    (out/'W1-direct-route.txt').write_text(content,encoding='utf-8')
    (out/'W1-direct-route.html').write_text(page.content(),encoding='utf-8')
    h=page.locator('article h2').filter(has_text='W.1').first;h.scroll_into_view_if_needed()
    page.screenshot(path=str(out/'W1-direct-route.png'))
    setup.update({'actual_url':page.url,'title':page.title(),'actual_heading':h.inner_text(),'read_scope':'Complete W.1 section; no later warm-up read','article_text_sha256':sha256(content.encode()).hexdigest()})
    records.append({'name':'new-direct-W1-route',**setup});print('W1 DIRECT ROUTE\n'+content+'\n')
    page.goto('http://127.0.0.1:8788/validation.html',wait_until='networkidle')
    page.locator('article a').filter(has_text='20.12').click();page.wait_for_load_state('networkidle')
    article=page.locator('article').first
    body=article.inner_text();(out/'20-12-renewed-read.txt').write_text(body,encoding='utf-8')
    (out/'20-12-renewed-read.html').write_text(page.content(),encoding='utf-8')
    img=article.locator('img').first;img.screenshot(path=str(out/'20-12-renewed-figure.png'))
    records.append({'name':'20-12-renewed-route','actual_url':page.url,'visible_label_clicked':'20.12','title':page.title(),'read_scope':'Complete 20.12 article; fresh browser element figure capture, no CPU rerun','image_src':img.get_attribute('src'),'image_alt':img.get_attribute('alt')})
    print('RENEWED 20.12\n'+body+'\n')
    # Retain new link destinations for the necessary unchanged context, without recasting old evidence as new.
    for source,label in [('publishing.html','閱讀指南'),('publishing.html','教材資產存放'),('publishing.html','實作驗證')]:
        page.goto('http://127.0.0.1:8788/'+source,wait_until='networkidle')
        a=page.locator('article a').filter(has_text=label).first
        href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle')
        article=page.locator('article').first
        text=article.inner_text()
        scope='complete article' if label=='教材資產存放' else ('R.1 and R.2 route understanding; rest captured but not newly reviewed' if label=='閱讀指南' else 'reciprocal full document already reread above')
        fname={'閱讀指南':'reading-guide','教材資產存放':'asset-storage','實作驗證':'return-validation'}[label]
        (out/(fname+'.txt')).write_text(text,encoding='utf-8')
        page.screenshot(path=str(out/(fname+'.png')))
        records.append({'name':fname,'from_url':'http://127.0.0.1:8788/'+source,'actual_visible_label':label,'href':href,'actual_url':page.url,'title':page.title(),'read_scope':scope})
        if label=='教材資產存放': print('ASSET DESTINATION\n'+text+'\n')
        elif label=='閱讀指南': print('READING ROUTE\n'+'\n'.join(text.splitlines()[:44])+'\n')
    for name in ['validation','publishing']:
        page.goto('http://127.0.0.1:8788/'+name+'.html',wait_until='networkidle')
        for idx,code in enumerate(page.locator('article pre code').all()):
            metric=code.evaluate('(e)=>({scrollWidth:e.scrollWidth,clientWidth:e.clientWidth,overflowX:getComputedStyle(e).overflowX,text:e.innerText})')
            if metric['scrollWidth']>metric['clientWidth']:
                code.evaluate('(e)=>e.scrollLeft=e.scrollWidth')
                path=out/(name+f'-code-{idx}-right.png');code.screenshot(path=str(path));metric['screenshot']=str(path)
                metric['actual_scroll_left']=code.evaluate('(e)=>e.scrollLeft')
            records.append({'name':name+'-code-scroll','code_block_index':idx,**metric})
    browser.close()
records.insert(0,{'name':'home-entry',**entry})
svg=urlopen('http://127.0.0.1:8788/figures/natural-v4-asr-two-routes.svg',timeout=10).read()
(out/'asr-two-routes-current-preview.svg').write_bytes(svg)
records.append({'name':'current-preview-svg-byte-comparison','actual_url':'http://127.0.0.1:8788/figures/natural-v4-asr-two-routes.svg','original_path':'course/figures/natural-v4-asr-two-routes.svg','source_sha256':sha256(Path('course/figures/natural-v4-asr-two-routes.svg').read_bytes()).hexdigest(),'preview_sha256':sha256(svg).hexdigest(),'equal_bytes':svg==Path('course/figures/natural-v4-asr-two-routes.svg').read_bytes()})
(out/'browser-records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
