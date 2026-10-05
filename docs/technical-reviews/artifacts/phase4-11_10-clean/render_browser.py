"""Render the actual existing single-section page and verify its source text."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

out=Path(__file__).resolve().parent
original=(out/'section.md').read_text()
expected_fence=(out/'fence-1.py').read_text()
expected_paragraphs=[p.strip() for p in original.split('\n\n') if p.startswith(('16×16','這裡也數','邊長 8、','若模型上限','改切法還','既有小對照','練習圖邊長'))]
records=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
    for name,w,h in [('desktop',1280,800),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':w,'height':h},device_scale_factor=1)
        context.route('**/*',lambda route: route.continue_() if urlsplit(route.request.url).hostname in ['127.0.0.1','localhost'] else route.abort())
        page=context.new_page()
        response=page.goto('http://127.0.0.1:8765/11.10.html',wait_until='domcontentloaded',timeout=15000)
        assert response.status==200
        article=page.locator('article').first
        article.wait_for(state='visible')
        assert article.locator('pre code').first.text_content()==expected_fence
        paragraph_text=article.locator('p').all_text_contents()
        for p in expected_paragraphs:
            p=p.replace('`','')
            assert p in paragraph_text, p
        html=response.body()
        (out/'render'/(name+'-served.html')).write_bytes(html)
        (out/'render'/(name+'-article.txt')).write_text(article.inner_text())
        page.screenshot(path=str(out/'render'/('page-'+name+'.png')),full_page=True,timeout=15000)
        records.append({'name':name,'viewport':{'width':w,'height':h},'url':response.url,'status':response.status,'server_response_sha256':hashlib.sha256(html).hexdigest(),'matched_original_paragraphs':len(expected_paragraphs),'original_fence_exact_match':True,'article_bounds':article.bounding_box(),'screenshot':'render/page-'+name+'.png','browser_version':browser.version,'scope':'Existing served page; seven core original paragraphs and original Python fence checked before capture. External requests aborted; local page/CSS assets allowed; no full course export.'})
        context.close()
    browser.close()
print(json.dumps({'records':records},ensure_ascii=False,indent=2))
