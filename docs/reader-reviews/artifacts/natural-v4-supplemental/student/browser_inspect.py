from pathlib import Path
import hashlib
import json
import shutil
from playwright.sync_api import sync_playwright

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/reader-reviews/artifacts/natural-v4-supplemental/student'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'snapshots').mkdir(exist_ok=True)
(OUT / 'browser').mkdir(exist_ok=True)
source = ROOT / 'docs/natural-assistant/v4/STUDENT.md'
raw = source.read_bytes()
(OUT / 'snapshots/STUDENT.md').write_bytes(raw)
(OUT / 'source-receipt.json').write_text(json.dumps({
    'path': str(source.relative_to(ROOT)), 'bytes': len(raw),
    'lines': len(raw.splitlines()), 'sha256': hashlib.sha256(raw).hexdigest(),
    'snapshot': str((OUT / 'snapshots/STUDENT.md').relative_to(ROOT)),
    'encoding': 'raw UTF-8 bytes; unchanged'
}, indent=2, ensure_ascii=False) + '\n')

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
    url = 'http://127.0.0.1:8769/natural-v4-student.html'
    response = page.goto(url, wait_until='networkidle')
    page.screenshot(path=str(OUT / 'browser/student-full.png'), full_page=True)
    (OUT / 'browser/student-rendered.txt').write_text(page.locator('body').inner_text(), encoding='utf-8')
    (OUT / 'browser/student-retrieved.html').write_text(page.content(), encoding='utf-8')
    inventory = {
        'requested_url': url, 'final_url': page.url, 'status': response.status,
        'browser': browser.version, 'engine': 'Chromium', 'viewport': {'width': 1440, 'height': 1000},
        'headings': page.locator('h1,h2,h3,h4').evaluate_all('(es) => es.map(e => ({tag:e.tagName,id:e.id,text:e.innerText}))'),
        'links': page.locator('a').evaluate_all('(es) => es.map(e => ({label:e.innerText,href:e.getAttribute("href"),absolute:e.href}))'),
        'images': page.locator('img').evaluate_all('(es) => es.map(e => ({alt:e.alt,src:e.src,complete:e.complete,width:e.naturalWidth,height:e.naturalHeight}))'),
    }
    (OUT / 'browser/student-inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n')
    for i, heading in enumerate(page.locator('h1,h2').all()):
        heading.scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / f'browser/student-section-{i:02d}.png'))
    print(json.dumps(inventory, indent=2, ensure_ascii=False))
    browser.close()
