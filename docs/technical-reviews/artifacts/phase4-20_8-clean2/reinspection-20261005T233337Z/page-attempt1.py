"""Local rendered current-section inspection only; does not navigate external links."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
url='http://127.0.0.1:8765/20.8.html'
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height})
        response=page.goto(url,wait_until='networkidle',timeout=15000)
        assert response.status==200
        body=page.locator('body').inner_text()
        assert '全部132份驗證回答以EOS結束' in body
        assert '並非全部選版得分' in body
        page.screenshot(path=str(HERE/(label+'.png')),full_page=True)
        links=page.locator('a').evaluate_all("els => els.filter(e => e.textContent.includes('訓練指引第6節')).map(e => ({text:e.textContent,href:e.getAttribute('href')}))")
        assert len(links)==1
        results.append({'viewport':label,'width':width,'height':height,'http_status':response.status,'new_text_rendered':True,'new_reference':links,'browser_version':browser.version})
        page.close()
    browser.close()
(HERE/'page-result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
