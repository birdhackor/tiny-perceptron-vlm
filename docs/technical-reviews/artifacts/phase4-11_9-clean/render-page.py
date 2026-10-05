"""Verify the local section page and capture actual desktop/mobile rendering."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import re
import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
raw = (OUT / 'frozen-section.md').read_text()
url = 'http://127.0.0.1:8765/11.9.html'
page_bytes = urlopen(url, timeout=10).read()
(OUT / 'page.html').write_bytes(page_bytes)
html = BeautifulSoup(page_bytes, 'html.parser')
source = BeautifulSoup(markdown.markdown(raw, extensions=['fenced_code']), 'html.parser')
normalize = lambda s: re.sub(r'\s+', ' ', s).strip()
expected = [normalize(p.get_text()) for p in source.select('p') if normalize(p.get_text())][:5]
actual = [normalize(p.get_text()) for p in html.select('article p') if normalize(p.get_text())][:5]
assert expected == actual, (expected, actual)
code = html.select_one('article pre code').get_text()
assert code == (OUT / 'fence/fence-1.py').read_text()
assert normalize(html.select_one('article h1').get_text()).removesuffix('¶') == raw.splitlines()[0].removeprefix('## ')
figure_url = 'http://127.0.0.1:8765/figures/rewrite-11-crop-evidence.svg'
served_figure = urlopen(figure_url, timeout=10).read()
assert served_figure == (ROOT / 'course/figures/rewrite-11-crop-evidence.svg').read_bytes()
screenshots = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', args=['--no-sandbox'], headless=True)
    for name, width, height in [('desktop', 1280, 800), ('mobile', 390, 844)]:
        page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)
        page.route('**/*', lambda route: route.continue_() if route.request.url.startswith('http://127.0.0.1:8765/') else route.abort())
        page.goto(url, wait_until='networkidle', timeout=15000)
        page.locator('article img').scroll_into_view_if_needed()
        image_info = page.locator('article img').evaluate('(img) => ({naturalWidth:img.naturalWidth,naturalHeight:img.naturalHeight,width:img.getBoundingClientRect().width,height:img.getBoundingClientRect().height,complete:img.complete})')
        assert image_info['complete'] and image_info['naturalWidth'] == 640
        page.screenshot(path=str(OUT / (name + '.png')))
        screenshots.append({'name': name, 'viewport': [width, height], 'figure': image_info})
        page.close()
    browser.close()
receipt = {'url': url, 'source_page_sha256': hashlib.sha256(page_bytes).hexdigest(),
           'source_section_sha256': hashlib.sha256((OUT / 'frozen-section.md').read_bytes()).hexdigest(),
           'body_paragraphs_exact': len(expected), 'original_fence_exact': True,
           'served_figure_original_bytes_exact': True, 'screenshots': screenshots,
           'scope': 'Own source/HTML/fence/figure equality and actual browser screenshots; no course rebuild/export.'}
(OUT / 'page-verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
