"""Render the linked current 1.1 figure with the system browser, without downloads."""
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parent
root = out.parents[3]
figure = root / 'course/figures/rewrite-01-character-ids.svg'
html = ('<!doctype html><meta charset="utf-8">'
        '<style>body{margin:0}svg{display:block;max-width:100%;height:auto}</style>' + figure.read_text())
(out / 'figure-render.html').write_text(html)
with sync_playwright() as sp:
    browser = sp.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox'])
    for width in (640, 360):
        page = browser.new_page(viewport={'width': width, 'height': 360}, device_scale_factor=1)
        page.set_content(html)
        page.screenshot(path=str(out / f'character-ids-{width}.png'), full_page=True)
        page.close()
    browser.close()
print('Rendered current 1.1 character-ID SVG at 640 and 360 CSS pixels using /usr/bin/chromium.')
