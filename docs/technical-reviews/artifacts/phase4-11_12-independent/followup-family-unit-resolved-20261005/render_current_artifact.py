from pathlib import Path
import base64
import datetime
import hashlib
import json
import platform
import urllib.request
import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ART=Path(__file__).resolve().parent
ROOT=ART.parents[4]
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(ART/'current-section.md').read_bytes()
source=raw.decode()
svg=(ROOT/'course/figures/rewrite-11-glyph-labels.svg').read_bytes()
# Rendering-only details handling; the stored raw source itself is unchanged.
body=markdown.markdown(source.replace('<details>','<details markdown="1">'),extensions=['fenced_code','md_in_html'])
soup=BeautifulSoup(body,'html.parser')
images=soup.find_all('img');assert len(images)==1
images[0]['src']='data:image/svg+xml;base64,'+base64.b64encode(svg).decode()
expected_paragraphs=[p.get_text() for p in soup.find_all('p') if not p.find('img')]
expected_code=soup.find('pre').get_text()
html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{margin:0;color:#17293c;background:white;font-family:"Noto Sans CJK TC",sans-serif;font-size:18px;line-height:1.85}
main{max-width:760px;margin:24px auto;padding:0 20px}h2{font-size:28px;line-height:1.45}img{display:block;width:100%;max-width:640px;height:auto;margin:24px auto}pre{overflow:auto;background:#f5f6f7;padding:14px;font-size:14px;line-height:1.6}code{font-family:monospace}aside{padding:10px;background:#f0f5f9;font-size:14px}details{padding:10px;border:1px solid #dce5ee}a{color:#008a83}
</style><main><aside>本次修訂驗證快照（非 production 頁面）</aside>'''+str(soup)+'</main>'
(ART/'bounded-current-section.html').write_text(html)
old_page=(ART/'served-page-before-recheck.html').read_bytes()
old_text=BeautifulSoup(old_page,'html.parser').select_one('article.md-content__inner').get_text()
receipt={'reviewer_task':'/root/phase4_factual_coordinator/factual_11_12','checked_at':datetime.datetime.now(datetime.UTC).isoformat(),
         'python':platform.python_version(),'source_sha256':sha(raw),'svg_sha256':sha(svg),'artifact_html_sha256':sha(html.encode()),
         'mode':'Bounded artifact of personally read current raw11.12 + exact originalSVG bytes; not production page or production style/layout validation.',
         'production_page_parity':{'url':'http://127.0.0.1:8765/11.12.html','snapshot':'served-page-before-recheck.html','sha256':sha(old_page),
                                   'contains_new_source_family_paragraph':'每個字目前只有一個原形' in old_text,
                                   'contains_new_full_string_split_paragraph':'按完整字串切分' in old_text,
                                   'status':'stale: two revised paragraphs absent; failure retained, shared server/builder untouched'},'views':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
    receipt['chromium']=browser.version
    page=browser.new_page()
    for label,viewport in [('desktop',{'width':1280,'height':800}),('mobile',{'width':390,'height':844})]:
        page.set_viewport_size(viewport)
        page.set_content(html,wait_until='load')
        page.evaluate('document.fonts.ready')
        observed=BeautifulSoup(page.locator('main').inner_html(),'html.parser')
        assert [p.get_text() for p in observed.find_all('p') if not p.find('img')]==expected_paragraphs
        assert observed.find('pre').get_text()==expected_code
        assert base64.b64decode(observed.find('img')['src'].partition(',')[2])==svg
        screenshot=label+'-current-artifact.png'
        page.screenshot(path=str(ART/screenshot),full_page=True)
        receipt['views'].append({'name':label,'viewport':viewport,'screenshot':screenshot,'source_paragraphs_exact':len(expected_paragraphs),
                                 'original_fence_exact':True,'embedded_original_svg_exact':True})
    browser.close()
(ART/'current-artifact-render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
