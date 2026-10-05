"""Render only the current 11.11 page and compare its prose/fence to raw source."""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ART=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765/11.11.html'
with urllib.request.urlopen(URL,timeout=10) as r:raw=r.read()
(ART/'rendered-page.html').write_bytes(raw)
actual=BeautifulSoup(raw,'html.parser').find('article')
source=(ART/'section.md').read_text()
# Render the details contents independently; MkDocs enables Markdown there.
source=source.replace('<details>','').replace('</details>','')
source=re.sub(r'<summary>.*?</summary>','',source)
expected=BeautifulSoup(markdown.markdown(source,extensions=['extra']),'html.parser')
ep=[p.get_text() for p in expected.find_all('p')]
ap=[p.get_text() for p in actual.find_all('p')]
assert ep==ap[:len(ep)]
assert actual.find('pre').get_text()==(ART/'fence-1.py').read_text()
receipt={'url':URL,'accessed_on':'2026-10-05',
 'rendered_page_sha256':hashlib.sha256(raw).hexdigest(),
 'source_sha256':hashlib.sha256((ART/'section.md').read_bytes()).hexdigest(),
 'prose_paragraph_count':len(ep),'paragraphs_match_in_order':True,
 'fence_text_exact_match':True,'page_added_paragraphs':ap[len(ep):],
 'scope':'All eight source paragraphs, including details, and the Python fence match current page via Markdown-to-HTML text conversion; appended Colab navigation is separately recorded. Full book not exported.'}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu'])
    receipt['chromium_version']=browser.version
    receipt['viewports']=[]
    for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1)
        page=context.new_page()
        page.goto(URL,wait_until='domcontentloaded',timeout=15000)
        page.wait_for_selector('article pre',timeout=5000)
        page.screenshot(path=str(ART/(label+'.png')),full_page=True,timeout=10000)
        receipt['viewports'].append({'label':label,'width':width,'height':height,'screenshot':label+'.png',
          'body_scroll_width':page.evaluate('document.body.scrollWidth'),
          'article_bounds':page.locator('article').bounding_box()})
        context.close()
    browser.close()
(ART/'page-source-match.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
