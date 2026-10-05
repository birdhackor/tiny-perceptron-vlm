from playwright.sync_api import sync_playwright
from pathlib import Path
import json
out=Path('docs/technical-reviews/artifacts/natural-v4-supplemental/site-validation-publishing')
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1365,'height':900});page.set_default_timeout(10000);requests=[];page.on('response',lambda r:requests.append({'url':r.url,'status':r.status}) if 'search' in r.url else None);page.goto('http://127.0.0.1:8788/publishing.html',wait_until='networkidle');page.locator('.md-search__button').click();search=page.locator('input').filter(visible=True).last;search.fill('教材發布');page.wait_for_timeout(1500);visible=page.locator('body').inner_text();png=out/'search-result.png';page.screenshot(path=str(png));receipt={'query':'教材發布','search_responses':requests,'visible_text':visible,'screenshot':str(png),'browser_version':browser.version};(out/'search-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'search_responses':requests,'visible_text_start':visible[:2000]},indent=2,ensure_ascii=False));browser.close()
