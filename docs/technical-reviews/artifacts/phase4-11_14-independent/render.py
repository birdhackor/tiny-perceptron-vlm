"""Capture only the existing lesson page with installed Chromium, with bounded waits."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8765/11.14.html'
raw_html = urlopen(URL, timeout=10).read()
(HERE/'page.html').write_bytes(raw_html)
rendered = BeautifulSoup(raw_html, 'html.parser').find('article')
expected = BeautifulSoup(markdown.markdown((HERE/'original'/'section.md').read_text(),
                            extensions=['fenced_code']), 'html.parser')
normalize = lambda s: ''.join(s.split())
expected_paragraphs = [normalize(p.get_text()) for p in expected.find_all('p') if p.get_text().strip()]
actual_paragraphs = [normalize(p.get_text()) for p in rendered.find_all('p') if p.get_text().strip()]
assert all(p in actual_paragraphs for p in expected_paragraphs)
assert rendered.find('pre').find('code').get_text().encode() == (HERE/'original'/'fence-1.py').read_bytes()
expected_img = expected.find('img')
actual_img = rendered.find('img')
assert expected_img['alt'] == actual_img['alt']
assert actual_img['src'] == 'figures/practical_order.svg'
root = HERE.parents[3]
img_bytes = urlopen('http://127.0.0.1:8765/'+actual_img['src'], timeout=10).read()
assert img_bytes == (root/'course/figures/practical_order.svg').read_bytes()
results = {'url':URL, 'html_sha256':hashlib.sha256(raw_html).hexdigest(),
           'paragraphs_matched':len(expected_paragraphs), 'fence_bytes_match':True,
           'image_alt_and_source_bytes_match':True, 'viewports':[]}
os.environ['PW_TEST_SCREENSHOT_NO_FONTS_READY']='1'
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                 args=['--no-sandbox','--disable-gpu','--disable-background-networking'], timeout=15000)
    for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page = browser.new_page(viewport={'width':width,'height':height})
        page.route('**/*', lambda route: route.continue_() if route.request.url.startswith('http://127.0.0.1:8765/') else route.abort())
        page.goto(URL, wait_until='domcontentloaded', timeout=15000)
        page.locator('article img').wait_for(timeout=10000)
        assert page.locator('article img').evaluate('(i)=>i.complete && i.naturalWidth > 0')
        box=page.locator('article img').bounding_box()
        page.screenshot(path=str(HERE/f'page-{label}.png'),full_page=True, timeout=15000)
        results['viewports'].append({'label':label,'width':width,'height':height,'image_box':box,
                                    'full_page_screenshot':f'page-{label}.png'})
        page.close()
    results['browser_version']=browser.version
    browser.close()
(HERE/'render-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
