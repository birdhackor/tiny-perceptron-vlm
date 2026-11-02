import json
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/workspace/work/omni-colab-account/browser-connect-v2');out.mkdir(exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/workspace/work/browsers/chromium-1234/chrome-linux64/chrome',args=['--no-sandbox']);ctx=b.new_context(viewport={'width':1440,'height':1000},locale='en-US');page=ctx.new_page();page.goto('https://colab.research.google.com/notebooks/intro.ipynb',wait_until='domcontentloaded',timeout=45000)
 button=page.locator('colab-connect-button');button.wait_for(state='visible',timeout=30000);page.wait_for_timeout(2500);button.click();page.wait_for_timeout(3500)
 text=page.locator('body').inner_text();(out/'body-after.txt').write_text(text);page.screenshot(path=str(out/'after.png'));print(json.dumps({'url':page.url,'dialogs':page.get_by_role('dialog').all_text_contents(),'popup_hosts':[x.url.split('/')[2] if '://' in x.url else x.url for x in ctx.pages],'body_tail':text[-1800:]},ensure_ascii=False));b.close()
