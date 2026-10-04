from pathlib import Path
import json
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8769/'
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
    def capture(slug,reason,from_url,label):
        article=page.locator('article.md-content__inner')
        if article.count()==0: article=page.locator('body')
        record={'slug':slug,'from_url':from_url,'clicked_visible_label':label,'final_url':page.url,'destination_reason':reason,'title':page.title(),'headings':article.locator('h1,h2,h3,h4').evaluate_all('(els)=>els.map(e=>({id:e.id,text:e.innerText}))'),'links':article.locator('a').evaluate_all('(els)=>els.map(e=>({label:e.innerText,href:e.href}))'),'images':article.locator('img').evaluate_all('(els)=>els.map(e=>({src:e.src,alt:e.alt,loaded:e.complete&&e.naturalWidth>0,width:e.naturalWidth,height:e.naturalHeight}))'),'text_file':slug+'.browser.text.txt','screenshot':slug+'.browser.png'}
        (OUT/record['text_file']).write_text(article.inner_text(),encoding='utf-8')
        page.locator('h1').first.scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/record['screenshot']))
        receipts.append(record)
        print(json.dumps({k:record[k] for k in ['slug','final_url','headings','images','destination_reason']},ensure_ascii=False))
    page.goto(BASE+'natural-v4-training.html',wait_until='networkidle')
    origin=page.url
    page.locator('article.md-content__inner').get_by_role('link',name='學生操作指引',exact=True).click()
    page.wait_for_load_state('networkidle')
    capture('student-prerequisite','取得程式與獨立GPU環境的第1、2步，並親自核對學生頁的訓練導航。',origin,'學生操作指引')
    article=page.locator('article.md-content__inner')
    training_links=article.locator('a').evaluate_all('(els)=>els.filter(e=>e.href.includes("natural-v4-training")).map(e=>({label:e.innerText,href:e.href}))')
    print('STUDENT_TRAINING_LINKS '+json.dumps(training_links,ensure_ascii=False))
    if training_links:
        label=training_links[0]['label']; origin=page.url
        article.get_by_role('link',name=label,exact=True).first.click()
        page.wait_for_load_state('networkidle')
        capture('student-to-training','從學生頁實際点击訓練入口後回到這份GPU重做配方。',origin,label)
    labels=[('20.5 的家族切分','20-5','用家族分組理解同一原素材不可跨練習與考卷。'),('20.6 的修正分支','20-6','理解LoRA、rank、alpha與底座凍結。'),('20.7 的答案計分','20-7','理解只對最後助手答案計代價，保留輸入作條件。'),('20.8','20-8','理解完成訓練與採用候選之間的驗證決定。'),('資料說明','data-prerequisite','理解固定資料取得、清單、train/validation/test及語音角色。'),('20.9–20.12','20-9','理解照片的物件動作關係和無根據新增判尺。'),('20.13','20-13','理解完成後自述能力卡應保存哪些範圍和限制。')]
    for label,slug,reason in labels:
        page.goto(BASE+'natural-v4-training.html',wait_until='networkidle'); origin=page.url
        page.locator('article.md-content__inner').get_by_role('link',name=label,exact=True).first.click()
        page.wait_for_load_state('networkidle')
        capture(slug,reason,origin,label)
        if slug=='20-9':
            for next_slug,expected,reason2 in [('20-10','20.10','理解OCR精確、正規化、換行與順序。'),('20-11','20.11','理解逐字聽寫與字錯率。'),('20-12','20.12','理解聽寫與助理回答的兩站分離驗收。')]:
                previous=page.url
                dest=page.locator('a').filter(has_text='下一頁').first
                label2=dest.inner_text()
                dest.click(); page.wait_for_load_state('networkidle')
                capture(next_slug,reason2,previous,label2)
    page.goto(BASE+'natural-v4-training.html',wait_until='networkidle'); origin=page.url
    link=page.locator('article.md-content__inner').get_by_role('link',name='本次配方的約定判準',exact=True)
    label=link.inner_text(); href=link.get_attribute('href')
    try:
        response=page.goto(href,wait_until='domcontentloaded',timeout=20000)
        receipts.append({'slug':'validation-protocol-link','from_url':origin,'clicked_visible_label':label,'final_url':page.url,'status':response.status if response else None,'title':page.title(),'body_text_file':'validation-protocol-link.browser.text.txt','screenshot':'validation-protocol-link.browser.png','reason':'選版門檻是第6節明示的必要判準。'})
        (OUT/'validation-protocol-link.browser.text.txt').write_text(page.locator('body').inner_text(),encoding='utf-8')
        page.screenshot(path=str(OUT/'validation-protocol-link.browser.png'))
        print('PROTOCOL_LINK '+json.dumps(receipts[-1],ensure_ascii=False))
    except Exception as e:
        receipts.append({'slug':'validation-protocol-link','from_url':origin,'visible_label':label,'target_url':href,'error':str(e),'reason':'選版門檻是第6節明示的必要判準。'})
        print('PROTOCOL_LINK '+json.dumps(receipts[-1],ensure_ascii=False))
    (OUT/'prerequisites.browser.receipt.json').write_text(json.dumps({'browser_version':browser.version,'viewport':{'width':1280,'height':900},'pages':receipts},ensure_ascii=False,indent=2)+'\n')
    browser.close()
