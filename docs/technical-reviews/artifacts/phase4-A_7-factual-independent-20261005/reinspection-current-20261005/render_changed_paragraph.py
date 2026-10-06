import hashlib
import json
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
source=(HERE/'current-section.md').read_bytes()
expected=source.decode('utf-8').split('\n\n')[1]
url='http://127.0.0.1:8765/A.7.html'
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        response=page.goto(url,wait_until='load',timeout=30000)
        assert response.status==200
        paragraph=page.locator('p').filter(has_text='輸入本身就達105')
        assert paragraph.count()==1
        text=paragraph.inner_text().strip()
        assert text==expected
        paragraph.screenshot(path=str(HERE/f'changed-paragraph-{name}.png'))
        results.append({'url':url,'response_status':response.status,'viewport':[width,height],
           'screenshot':f'changed-paragraph-{name}.png','rendered_paragraph':text,
           'paragraph_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
           'scope':'Only changed current A.7 opening paragraph; not unrelated whole-page rereading.'})
        page.close()
    browser.close()
receipt={'source_sha256':hashlib.sha256(source).hexdigest(),
 'browser_version':subprocess.check_output(['/usr/bin/chromium','--version'],text=True).strip(),
 'results':results}
(HERE/'changed-paragraph-render.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
