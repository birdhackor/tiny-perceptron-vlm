from pathlib import Path
from playwright.sync_api import sync_playwright
from hashlib import sha256
import json
base=Path(__file__).parent; out=base/'browser'; records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
    page=context.new_page()
    for name,label in [('experiment-source','正式實驗入口'),('gpu-result','原始結果'),('selection-source','驗證決定')]:
        page.goto('http://127.0.0.1:8786/validation.html',wait_until='networkidle')
        a=page.locator('article a').filter(has_text=label).first
        href=a.get_attribute('href'); target=a.get_attribute('target')
        rec={'name':name,'visible_label':a.inner_text(),'href':href,'target':target}
        dest=page
        try:
            if target=='_blank':
                with page.expect_popup(timeout=12000) as popup:
                    a.click(timeout=10000)
                dest=popup.value; dest.wait_for_load_state('domcontentloaded',timeout=12000)
                rec['status']='popup navigation, HTTP status not recorded'
            else:
                with page.expect_navigation(wait_until='domcontentloaded',timeout=12000) as nav:
                    a.click(timeout=10000)
                rec['http_status']=nav.value.status if nav.value else None
            body=dest.locator('body').inner_text(timeout=8000)
            (out/(name+'-external.txt')).write_text(body,encoding='utf-8')
            dest.screenshot(path=str(out/(name+'-external.png')))
            rec.update({'actual_url':dest.url,'title':dest.title(),'read_scope':'visible destination/error body','body':body[:1800]})
        except Exception as e:
            rec.update({'actual_url':dest.url,'error':str(e),'read_scope':'navigation attempt; destination not successfully read'})
        records.append(rec)
        if dest is not page: dest.close()
    page.goto('http://127.0.0.1:8786/20.12.html',wait_until='networkidle')
    img=page.locator('article img').first
    img.screenshot(path=str(out/'20-12-browser-figure.png'))
    records.append({'name':'asr-figure','actual_url':page.url,'image_src':img.get_attribute('src'),'alt':img.get_attribute('alt'),'screenshot':'20-12-browser-figure.png'})
    download=page.locator('article a').filter(has_text='下載本節 .ipynb').first
    rec={'name':'notebook-download','visible_label':download.inner_text(),'href':download.get_attribute('href'),'from_url':page.url}
    with page.expect_download(timeout=12000) as pending:
        download.click()
    d=pending.value; path=out/'20.12-downloaded.ipynb'; d.save_as(str(path))
    nb=json.loads(path.read_text())
    rec.update({'suggested_filename':d.suggested_filename,'failure':d.failure(),'saved_path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'cells':len(nb['cells']),'first_code_cell':next(c['source'] for c in nb['cells'] if c['cell_type']=='code'),'read_scope':'download receipt and first preparation code cell; no notebook execution'})
    records.append(rec)
    page.goto('http://127.0.0.1:8786/20.12.html',wait_until='networkidle')
    a=page.locator('article a').filter(has_text='在 Colab 動手做').first
    records.append({'name':'colab-link','from_url':page.url,'visible_label':a.inner_text(),'href':a.get_attribute('href'),'limit':'Recorded real rendered link destination; did not authenticate/run Colab.'})
    page.goto('http://127.0.0.1:8786/publishing.html',wait_until='networkidle')
    page.locator('article a').filter(has_text='實作驗證').click()
    records.append({'name':'return-validation','from_url':'http://127.0.0.1:8786/publishing.html','actual_url':page.url,'title':page.title(),'visible_label':'實作驗證'})
    (out/'entry-and-external-records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    browser.close()
print(json.dumps(records,ensure_ascii=False,indent=2))
