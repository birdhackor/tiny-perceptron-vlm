from pathlib import Path
import json
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium'); page=browser.new_page(viewport={'width':1280,'height':900})
    for name,selector,slug in [('訓練指引','article.md-content__inner','student-body-training-entry'),('自行重訓','.md-sidebar--primary','student-sidebar-training-entry')]:
        response=page.goto('http://127.0.0.1:8769/natural-v4-student.html',wait_until='networkidle'); origin=page.url
        link=page.locator(selector).get_by_role('link',name=name,exact=True).first
        link.scroll_into_view_if_needed(); href=link.get_attribute('href'); page.screenshot(path=str(OUT/(slug+'.png')))
        link.click(); page.wait_for_load_state('networkidle')
        heading=page.locator('h1').first.inner_text()
        record={'from_url':origin,'visible_label':name,'href':href,'actual_destination':page.url,'destination_heading':heading,'destination_means':'建立自己的GPU微調候選、用validation選版、固定後evaluate test，再serve自己的配置。','entry_screenshot':slug+'.png','destination_screenshot':slug+'.destination.png'}
        page.screenshot(path=str(OUT/record['destination_screenshot'])); records.append(record)
    (OUT/'student-training-navigation.receipt.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(records,ensure_ascii=False,indent=2)); browser.close()
