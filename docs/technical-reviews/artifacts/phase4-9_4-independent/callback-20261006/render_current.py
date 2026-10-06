"""Personally render the current served 9.4 page, using its actual HTTP response."""
from pathlib import Path
import hashlib
import json
import platform
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8765/9.4.html'
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                               args=['--no-sandbox', '--disable-gpu'])
    records = []
    for label, width, height in [('desktop', 1280, 800), ('mobile', 390, 844)]:
        page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)
        response = page.goto(URL, wait_until='networkidle', timeout=20000)
        assert response and response.status == 200
        body = response.body()
        if label == 'desktop':
            (BASE / 'served-page.html').write_bytes(body)
        page.locator('details').evaluate_all('(nodes) => nodes.forEach(n => n.open = true)')
        text = page.locator('body').inner_text()
        assert '9.4 何時需要安全回應？' in text
        assert '應由應用中的可靠授權流程提供資訊。' in text
        assert '六題匹配只支持可信True／False與固定句型的對應' in text
        assert '許可應由應用可靠流程提供。' not in text
        assert 'owner' in text and 'permission' in text
        code = page.locator('pre').first.inner_text()
        assert code.strip() == (BASE / 'current-fence-1.py').read_text().strip()
        (BASE / f'current-{label}-text.txt').write_text(text + '\n')
        screenshot = BASE / f'current-{label}.png'
        page.screenshot(path=str(screenshot), full_page=True, animations='disabled')
        records.append({'viewport': f'{width}x{height}', 'url': URL, 'status': response.status,
                        'served_html_sha256': hashlib.sha256(body).hexdigest(),
                        'screenshot': str(screenshot), 'screenshot_sha256': hashlib.sha256(screenshot.read_bytes()).hexdigest(),
                        'details_open': True, 'code_matches_current_fence': True,
                        'image_elements': page.locator('img').count()})
        page.close()
    receipt = {'command': 'timeout 45s .venv/bin/python docs/technical-reviews/artifacts/phase4-9_4-independent/callback-20261006/render_current.py',
               'python': platform.python_version(), 'browser': browser.version, 'executable': '/usr/bin/chromium',
               'network': 'only existing local HTTP page 127.0.0.1:8765', 'records': records}
    (BASE / 'render-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    browser.close()
print(json.dumps({'status': 'passed', 'url': URL, 'viewports': ['1280x800', '390x844'],
                  'code_matches_current_fence': True, 'receipt': str(BASE / 'render-receipt.json')}, ensure_ascii=False))
