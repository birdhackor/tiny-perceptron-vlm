from pathlib import Path
import hashlib, json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[5]
DEST = Path(__file__).parent
BASE = 'http://127.0.0.1:8783/'
figures = json.loads((DEST / 'prerequisite-receipts.json').read_text())['figures']
receipt = {'base': BASE, 'preview_revision': 'f8ca78bc3ffef82b0faa352c464909b76bfd8976', 'figure_renders': [], 'browser_events': []}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
    page.goto(BASE + 'training.html#T.4', wait_until='networkidle')
    page.locator('[id="T.4"]').scroll_into_view_if_needed()
    page.screenshot(path=str(DEST / 'browser-training-T4.png'))
    receipt['browser_events'].append({'action': 'open literal training.html#T.4 and scroll heading into view', 'url': page.url, 'heading': page.locator('[id="T.4"]').inner_text(), 'screenshot': 'browser-training-T4.png'})
    link = page.get_by_role('link', name='T.3', exact=False)
    target = page.get_by_role('link', name='T.3', exact=True).first
    receipt['browser_events'].append({'action': 'inspect actual internal link before click', 'href':target.get_attribute('href')})
    target.click()
    receipt['browser_events'].append({'action': 'click internal T.3 link from complete training page', 'url': page.url, 'heading': page.locator('[id="T.3"]').inner_text()})
    page.goto(BASE + 'training.html#T.4', wait_until='networkidle')
    table = page.locator('table').filter(has_text='TinyStories完整故事').first
    table.scroll_into_view_if_needed()
    page.screenshot(path=str(DEST / 'browser-training-realtext-table.png'))
    receipt['browser_events'].append({'action': 'inspect real-text table within T.4', 'url': page.url, 'visible_text': table.inner_text(), 'screenshot': 'browser-training-realtext-table.png'})
    # Read the actual page DOM scope and save its executed preview bytes receipt.
    scope = page.locator('[id="T.4"]').evaluate("e => {let n=e.nextElementSibling, out=[e.innerText]; while(n && !(n.tagName==='H2')) {out.push(n.innerText); n=n.nextElementSibling;} return out.join('\\n');}")
    (DEST / 'browser-T4-DOM.txt').write_text(scope, encoding='utf-8')
    receipt['browser_events'].append({'action': 'inspect full T.4 DOM until next H2', 'url': page.url, 'dom_chars': len(scope), 'dom_sha256': hashlib.sha256(scope.encode()).hexdigest()})
    for figure in figures:
        raw = (ROOT / figure).read_bytes()
        url = BASE + 'figures/' + Path(figure).name
        page.goto(url, wait_until='load')
        svg = page.locator('svg')
        box = svg.bounding_box()
        name = Path(figure).stem + '.png'
        svg.screenshot(path=str(DEST / name))
        receipt['figure_renders'].append({'source': figure, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'url': url, 'svg_box': box, 'render': name, 'renderer': 'Chromium via Playwright', 'render_sha256': hashlib.sha256((DEST/name).read_bytes()).hexdigest(), 'personal_view': 'pending'})
    browser.close()
(DEST / 'browser-render-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'events': receipt['browser_events'], 'render_count': len(figures)}, ensure_ascii=False, indent=2))
