import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

names = ['foundations_embedding', 'capstone_resources', 'natural_shared_chat', 'natural_base_adapter']
destination = Path('docs/technical-reviews/artifacts/natural-v4-factual/R.4')
receipts = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1200, 'height': 900}, device_scale_factor=1)
    for name in names:
        source = Path('course/figures') / (name + '.svg')
        page.goto(source.resolve().as_uri())
        page.locator('svg').screenshot(path=str(destination / (name + '.png')))
        receipts.append({'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'render': str(destination / (name + '.png'))})
    browser.close()
(destination / 'figure-render-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
print(json.dumps(receipts, indent=2))
