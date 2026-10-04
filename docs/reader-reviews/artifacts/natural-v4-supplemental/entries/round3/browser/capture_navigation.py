from pathlib import Path
import json,hashlib
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
root=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/entries/round3/browser')
base='http://127.0.0.1:8784/'
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1440,'height':1000});page.set_default_timeout(25000)
    def click(label,key,scope='.md-content'):
        source=page.url
        a=page.locator(scope).get_by_role('link',name=label,exact=True).first
        a.scroll_into_view_if_needed(); href=urljoin(page.url,a.get_attribute('href'))
        beforelabel=a.inner_text()
        a.click();page.wait_for_url(href,wait_until='networkidle')
        r=page.request.get(page.url.split('#')[0]);raw=r.body();(root/f'{key}-http.html').write_bytes(raw)
        content=page.locator('.md-content').inner_text();(root/f'{key}-content.txt').write_text(content)
        page.screenshot(path=str(root/f'{key}-viewport.png'))
        rec={'key':key,'source_url':source,'clicked_visible_label':beforelabel,'selected_scope':scope,'href':href,'actual_url':page.url,'status':r.status,'title':page.title(),'headings':page.locator('.md-content :is(h1,h2,h3)').all_text_contents(),'intro':content[:2600],'links':page.locator('.md-content a').evaluate_all('(els)=>els.map(e=>({label:e.innerText.trim(),href:e.href}))'),'visible_sidebar_links':page.locator('nav a:visible').evaluate_all('(els)=>els.map(e=>({label:e.innerText.trim(),href:e.href}))'),'raw_evidence':str(root/f'{key}-http.html'),'http_bytes':len(raw),'http_sha256':hashlib.sha256(raw).hexdigest(),'content_evidence':str(root/f'{key}-content.txt'),'screenshot':str(root/f'{key}-viewport.png'),'provenance':'Actual Chromium visible link click then wait_for_url(networkidle); raw bytes are subsequent Playwright API GET of actual navigated URL because site may intercept internal navigation','browser_version':browser.version}
        receipts.append(rec);return rec
    page.goto(base+'index.html',wait_until='networkidle');click('從第一節開始','index-to-1.1')
    page.goto(base+'course.html',wait_until='networkidle');click('暖身 W.1','course-to-W.1');click('W.2','W.1-to-W.2')
    page.goto(base+'course.html',wait_until='networkidle');click('章節目錄','course-to-index');click('第 7 章：從續寫文字到回答問題','index-to-chapter07')
    # The actual chapter link label is retained in the receipt; find the 7.4 destination rather than guess the full title.
    a=page.locator('.md-content a[href$="7.4.html"]').first; label=a.inner_text();click(label,'chapter07-to-7.4')
    page.goto(base+'readme.html',wait_until='networkidle');click('全部小節','readme-all-sections-to-index')
    page.goto(base+'index.html',wait_until='networkidle');click('第 20 章：做一位能看圖、讀字與聽問題的助理','index-to-chapter20')
    a=page.locator('.md-content a[href$="20.1.html"]').first; label=a.inner_text();click(label,'chapter20-to-20.1')
    page.goto(base+'readme.html',wait_until='networkidle');click('20.2','readme-to-20.2')
    page.goto(base+'readme.html',wait_until='networkidle');click('20.13','readme-to-20.13')
    for slug,labels in [('readme',['學生操作指引','資料','訓練']),('training-assets',['學生操作指引','新版資料說明','訓練指引'])]:
        for label in labels:
            page.goto(base+slug+'.html',wait_until='networkidle');click(label,f'{slug}-to-'+{'學生操作指引':'student','資料':'data','訓練':'training','新版資料說明':'data','訓練指引':'training'}[label])
    for label,key in [('操作指引','student'),('資料與來源','data'),('自行重訓','training')]:
        page.goto(base+'chapter-20.html',wait_until='networkidle')
        # Scope to the actual sidebar so the navigation label is personally observed.
        click(label,'sidebar-to-'+key,'nav')
    browser.close()
(root/'navigation-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2))
print(json.dumps([{'key':r['key'],'label':r['clicked_visible_label'],'url':r['actual_url'],'title':r['title'],'headings':r['headings']} for r in receipts],ensure_ascii=False,indent=2))
