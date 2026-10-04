from pathlib import Path
from playwright.sync_api import sync_playwright
import json
base=Path(__file__).parent; out=base/'browser'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.goto('http://127.0.0.1:8786/20.12.html',wait_until='networkidle')
    a=page.locator('article a').filter(has_text='在 Colab 動手做').first
    rec={'from_url':page.url,'actual_visible_label':a.inner_text(),'href':a.get_attribute('href'),'target':a.get_attribute('target')}
    try:
        with page.expect_popup(timeout=12000) as popup:
            a.click()
        dest=popup.value
        dest.wait_for_load_state('domcontentloaded',timeout=18000)
        try:
            dest.wait_for_load_state('networkidle',timeout=10000)
        except Exception as e:
            rec['networkidle_limit']=str(e)
        body=dest.locator('body').inner_text()
        (out/'colab-popup.txt').write_text(body,encoding='utf-8')
        (out/'colab-popup.html').write_text(dest.content(),encoding='utf-8')
        dest.screenshot(path=str(out/'colab-popup.png'))
        rec.update({'actual_url':dest.url,'title':dest.title(),'body':body,'read_scope':'actual Colab popup entry, no authentication or execution'})
    except Exception as e:
        rec.update({'error':str(e),'read_scope':'popup navigation attempt only'})
    (out/'colab-popup-record.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
    browser.close()
print(json.dumps(rec,ensure_ascii=False,indent=2))
