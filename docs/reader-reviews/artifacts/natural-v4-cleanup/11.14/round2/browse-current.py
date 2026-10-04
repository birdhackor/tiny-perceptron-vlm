"""Same-owner reader: real Chromium preview browse."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parent
raw = Path('/tmp/reader-final-11-14-browser-round2')
raw.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                               args=['--no-sandbox', '--disable-dev-shm-usage'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
    requests = []
    page.on('request', lambda request: requests.append(request.url))
    response = page.goto('http://127.0.0.1:8786/11.14.html', wait_until='networkidle', timeout=20000)
    page.screenshot(path=str(out / 'browser-11.14-full.png'), full_page=True)
    text = page.locator('body').inner_text()
    html = page.content().encode()
    (raw / '11.14.html').write_bytes(html)
    (out / 'browser-11.14-text.txt').write_text(text)
    links = page.locator('a').evaluate_all('(els) => els.map(e => ({text: e.innerText, href: e.getAttribute("href")}))')
    receipt = {'kind': 'actual Chromium navigation, first target visit',
               'url': page.url, 'http_status': response.status,
               'browser': browser.version, 'title': page.title(),
               'html_sha256': hashlib.sha256(html).hexdigest(),
               'raw_html_ignored_path': str(raw / '11.14.html'),
               'body_text_artifact': str(out / 'browser-11.14-text.txt'),
               'full_screenshot': str(out / 'browser-11.14-full.png'),
               'links': links, 'request_urls': requests}
    (out / 'browser-first-visit-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(text)
    browser.close()
