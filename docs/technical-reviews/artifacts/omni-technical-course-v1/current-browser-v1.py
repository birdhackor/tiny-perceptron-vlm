"""Read current preview footer and official upload control; no login/upload/cell."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
from playwright.sync_api import sync_playwright

OUT=Path('/workspace/work/technical-course-current-v1/browser')
OUT.mkdir(parents=True,exist_ok=False)
def capture(page,name):
    png=OUT/(name+'.png');dom=OUT/(name+'.txt')
    page.screenshot(path=str(png));dom.write_text(page.locator('body').inner_text())
    return {'screenshot':{'path':str(png),'sha256':hashlib.sha256(png.read_bytes()).hexdigest()},'DOM':{'path':str(dom),'sha256':hashlib.sha256(dom.read_bytes()).hexdigest()}}

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/workspace/work/browsers/chromium-1234/chrome-linux64/chrome',args=['--no-sandbox'])
    rows=[]
    for name,width,height in [('desktop',1440,1000),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},locale='en-US')
        page=context.new_page();page.goto('http://127.0.0.1:8774/1.1.html',wait_until='domcontentloaded',timeout=30000)
        actions=page.locator('.actions').last;actions.scroll_into_view_if_needed();page.evaluate('window.scrollBy(0, 160)');page.wait_for_timeout(500)
        rows.append({'viewport':name,'width':width,'height':height,'url':page.url,
                     'colab_home_link_count':page.locator('a[href="https://colab.research.google.com/"]').count(),
                     'download_link':actions.locator('a[download]').get_attribute('href'),
                     **capture(page,name+'-footer')})
        context.close()
    context=browser.new_context(viewport={'width':1440,'height':1000},locale='en-US')
    assert context.cookies()==[]
    page=context.new_page();page.goto('https://colab.research.google.com/',wait_until='domcontentloaded',timeout=45000)
    upload=page.get_by_role('button',name='Upload notebook',exact=True);upload.wait_for(state='visible',timeout=30000);page.wait_for_timeout(600)
    official={'url':page.url,'accessible_role':'button','accessible_name':'Upload notebook','visible':upload.is_visible(),
              'root':upload.evaluate('e=>({isShadowRoot:e.getRootNode() instanceof ShadowRoot,host:e.getRootNode().host?.localName??null})'),
              **capture(page,'official-upload')}
    version=browser.version;browser.close()
result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_technical_coordinator/technical_course',
        'environment':{'python':platform.python_version(),'chromium':version,'device':'CPU browser; fresh anonymous contexts'},
        'preview_observations':rows,'official_upload_observation':official,
        'account_login_performed':False,'file_selected_or_uploaded':False,'connect_clicked':False,'cell_executed':False,
        'scope':'Current local preview desktop/mobile footer and actual official home upload control. No cloud course execution, publication or complete site visual acceptance.'}
print(json.dumps(result,ensure_ascii=False,indent=2))
