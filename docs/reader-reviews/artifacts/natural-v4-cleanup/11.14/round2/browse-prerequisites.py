"""Reader's actual preview link clicks and section screenshots."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parent
raw = Path('/tmp/reader-final-11-14-browser-round2')
visits = []

def record(page, lesson, trigger):
    page.wait_for_load_state('networkidle', timeout=20000)
    article = page.locator('article').first
    text = article.inner_text()
    html = page.content().encode()
    (raw / (lesson + '-linked.html')).write_bytes(html)
    (out / ('browser-' + lesson + '-article.txt')).write_text(text)
    article.screenshot(path=str(out / ('browser-' + lesson + '-article.png')))
    links = article.locator('a').evaluate_all('(els) => els.map(e => ({text: e.innerText, href: e.href}))')
    item = {'lesson': lesson, 'url': page.url, 'trigger': trigger,
            'article_text': str(out / ('browser-' + lesson + '-article.txt')),
            'article_screenshot': str(out / ('browser-' + lesson + '-article.png')),
            'html_sha256': hashlib.sha256(html).hexdigest(),
            'raw_html_ignored_path': str(raw / (lesson + '-linked.html')),
            'content_links': links}
    visits.append(item)
    print(json.dumps({'lesson': lesson, 'url': page.url, 'trigger': trigger}, ensure_ascii=False))
    print(text)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                               args=['--no-sandbox', '--disable-dev-shm-usage'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
    page.goto('http://127.0.0.1:8786/11.14.html', wait_until='networkidle', timeout=20000)
    record(page, '11.14', 'direct navigation and full current article reread')
    paragraph = page.locator('article p').filter(has_text='自然照片還有光線').first
    paragraph.screenshot(path=str(out / 'browser-current-training-paragraph.png'))
    page.locator('article').get_by_role('link', name='11.4圖片問答', exact=True).click()
    record(page, '11.4', 'actual click from 11.14: 11.4圖片問答')
    page.go_back(wait_until='networkidle')
    page.locator('article').get_by_role('link', name='10.2圖片序列', exact=True).click()
    record(page, '10.2', 'actual click from 11.14: 10.2圖片序列')
    page.locator('article').get_by_role('link', name='10.1的批次與RGB軸', exact=True).click()
    record(page, '10.1', 'actual click from necessary prerequisite 10.2: 10.1的批次與RGB軸')
    page.goto('http://127.0.0.1:8786/11.14.html', wait_until='networkidle', timeout=20000)
    record(page, '11.14-return', 'return to current assigned article after prerequisite links')
    receipt = {'browser': browser.version, 'personally_read_dom_output': True,
               'visits': visits, 'scope': 'assigned article and necessary explicitly linked prerequisites only; no Colab, download, or external experiment link clicked'}
    (out / 'browser-link-navigation-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    browser.close()
