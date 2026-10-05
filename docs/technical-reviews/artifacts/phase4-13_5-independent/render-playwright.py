"""Bounded Playwright driver for the system Chromium after recorded CLI timeouts."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

out=Path(__file__).resolve().parent
records=[]
started=time.monotonic()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                              args=['--no-sandbox','--disable-dev-shm-usage'],timeout=15000)
    for name,width,height in [('desktop',1280,1500),('mobile',390,1500)]:
        page=browser.new_page(viewport={'width':width,'height':height})
        page.route('**/*',lambda route:route.abort())
        page.set_content((out/'section-render.html').read_text(),wait_until='domcontentloaded',timeout=15000)
        page.screenshot(path=str(out/('playwright-'+name+'.png')),full_page=True,timeout=15000)
        records.append({'viewport':[width,height],'screenshot':'playwright-'+name+'.png',
                        'h2':page.locator('h2').all_text_contents(),'external_requests':'all aborted; HTML has inline repo CSS and no external assets',
                        'document_width':page.evaluate('document.documentElement.scrollWidth'),
                        'document_height':page.evaluate('document.documentElement.scrollHeight')})
        page.close()
    version=browser.version
    browser.close()
result={'driver':'playwright.sync_api','executable_path':'/usr/bin/chromium','chromium_version':version,
        'elapsed_seconds':time.monotonic()-started,'renders':records,'exit_code':0,
        'render_scope':'Frozen section Markdown, source disclosure expanded, local repo course.css and explicit standalone typography; no complete site-layout claim.'}
(out/'render-playwright-execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
