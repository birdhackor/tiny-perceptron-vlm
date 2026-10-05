from pathlib import Path
from playwright.sync_api import sync_playwright
import json

base = Path(__file__).resolve().parents[1]
svg = (base / 'figure/natural-v4-family-crops.svg').read_text()
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path='/usr/bin/chromium')
    for width, height, name in [(1280, 1100, 'desktop'), (390, 844, 'mobile')]:
        page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)
        page.set_content('<html><meta charset="utf-8"><style>body{margin:16px}svg{display:block;width:100%;max-width:760px;height:auto}</style>' + svg + '</html>')
        page.screenshot(path=str(base / f'figure/{name}.png'), full_page=True)
        print(json.dumps({'viewport': [width,height], 'browser': browser.version, 'path': f'figure/{name}.png'}))
        page.close()
    browser.close()
