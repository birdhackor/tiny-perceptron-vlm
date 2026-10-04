from pathlib import Path
from playwright.sync_api import sync_playwright
import json
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/training-course'); BASE='http://127.0.0.1:8769/'
r={'executable_path':'/usr/bin/chromium','actual_navigation':[],'horizontal_code_scroll':{}}
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1100});page.set_default_timeout(5000)
 page.goto(BASE+'training.html',wait_until='networkidle')
 a=page.get_by_role('link',name='開始閱讀',exact=True);label=a.inner_text();href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle')
 page.screenshot(path=str(OUT/'reading-start-route.png'))
 r['actual_navigation'].append({'from':BASE+'training.html','visible_label':label,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts(),'screenshot':'reading-start-route.png'})
 a=page.locator('.md-sidebar--primary a[href$="training.html"]').filter(visible=True).first;label=a.inner_text();href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle')
 r['actual_navigation'].append({'from':r['actual_navigation'][0]['destination'],'visible_label':label,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts()})
 previous=page.locator('.md-footer__link--prev')
 if previous.count():
  previous.scroll_into_view_if_needed();label=previous.inner_text();href=previous.get_attribute('href');page.screenshot(path=str(OUT/'training-footer-navigation.png'));previous.click();page.wait_for_load_state('networkidle')
  r['actual_navigation'].append({'from':BASE+'training.html','visible_label':label,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts(),'from_screenshot':'training-footer-navigation.png'})
 page.goto(BASE+'training.html',wait_until='networkidle')
 pre=page.locator('article pre').filter(has_text='scripts/train_simple.py --model bigram --seed 42 --device cpu --train').first
 pre.scroll_into_view_if_needed();page.screenshot(path=str(OUT/'long-command-left.png'))
 scroll=pre.evaluate('(el)=>{ const target=el; const before=target.scrollLeft; target.scrollLeft=target.scrollWidth; return {before,after:target.scrollLeft,scrollWidth:target.scrollWidth,clientWidth:target.clientWidth,text:target.innerText};}')
 page.screenshot(path=str(OUT/'long-command-right.png'))
 r['horizontal_code_scroll']={'scope':'one representative long T.3 command','source_start':'T.3 line130','actual_scroll':scroll,'screenshots':['long-command-left.png','long-command-right.png']}
 (OUT/'browser-remaining-navigation-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False,indent=2));b.close()
