from pathlib import Path
from hashlib import sha256
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[5]
PROOF = Path(__file__).resolve().parent
names = ['capstone_tool_loop', 'tools', 'capstone_pipeline']
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 980, 'height': 700}, device_scale_factor=1, reduced_motion='reduce')
    receipts = []
    for name in names:
        source = ROOT / f'course/figures/{name}.svg'
        page.set_content(source.read_text())
        page.locator('svg').screenshot(path=str(PROOF / (name + '.png')))
        receipts.append({'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source.read_bytes()).hexdigest(), 'render': name + '.png', 'render_sha256': sha256((PROOF / (name + '.png')).read_bytes()).hexdigest()})
    print(json.dumps({'browser': browser.version, 'receipts': receipts}, indent=2))
    browser.close()
