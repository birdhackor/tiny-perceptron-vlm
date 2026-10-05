from pathlib import Path
import json
import hashlib
import platform
from bs4 import BeautifulSoup
import markdown
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
section = (ART / 'original/section.md').read_text()
expected = BeautifulSoup(markdown.markdown(section, extensions=['fenced_code']), 'html.parser')
paragraphs = [p.get_text() for p in expected.find_all('p') if not p.find('img')]
code = (ART / 'original/fence-1.py').read_text()
result = {'python': platform.python_version(), 'url': 'http://127.0.0.1:8765/11.12.html',
          'source_sha256': hashlib.sha256((ART / 'original/section.md').read_bytes()).hexdigest(),
          'renderer': '/usr/bin/chromium via Playwright', 'launch_args': ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage']}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=result['launch_args'])
    result['chromium_version'] = browser.version
    page = browser.new_page(viewport={'width': 640, 'height': 530})
    page.goto('http://127.0.0.1:8765/figures/rewrite-11-glyph-labels.svg', wait_until='load', timeout=20000)
    page.screenshot(path=str(ART / 'glyphs-render.png'))
    result['views'] = []
    for label, viewport in [('desktop', {'width': 1280, 'height': 800}), ('mobile', {'width': 390, 'height': 844})]:
        page.set_viewport_size(viewport)
        page.goto(result['url'], wait_until='load', timeout=20000)
        page.locator('article.md-content__inner img.diagram').wait_for(state='visible')
        page.evaluate('document.fonts.ready')
        actual = BeautifulSoup(page.locator('article.md-content__inner').inner_html(), 'html.parser')
        observed = [q.get_text() for q in actual.find_all('p') if not q.find('img')]
        # The course adds an output display and action links; the source paragraphs themselves must be intact.
        for expected_paragraph in paragraphs:
            assert expected_paragraph in observed, repr(expected_paragraph)
        code_matches = [q.get_text() for q in actual.select('pre code') if q.get_text() == code]
        assert len(code_matches) == 1
        img = actual.select_one('img.diagram')
        assert img['src'] == 'figures/rewrite-11-glyph-labels.svg'
        browser_svg = page.request.get('http://127.0.0.1:8765/' + img['src']).body()
        source_svg = (ROOT / 'course/figures/rewrite-11-glyph-labels.svg').read_bytes()
        assert browser_svg == source_svg
        page.screenshot(path=str(ART / f'{label}-page.png'), full_page=True)
        result['views'].append({'name': label, 'viewport': viewport, 'source_paragraphs_exact': len(paragraphs),
                                'original_fence_exact': True, 'served_svg_exact': True,
                                'screenshot': f'{label}-page.png',
                                'page_html_sha256': hashlib.sha256(page.content().encode()).hexdigest()})
    browser.close()
(ART / 'render-receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
