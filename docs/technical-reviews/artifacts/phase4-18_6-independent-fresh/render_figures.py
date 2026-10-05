"""Render the exact reviewed SVG at desktop and mobile figure widths."""
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parent
src = Path('course/figures/rewrite-18-vocab-alignment.svg')
raw = src.read_bytes()
(out / 'figures/rewrite-18-vocab-alignment.svg').write_bytes(raw)
manifest = {'source': str(src), 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'renderer': 'Chromium via Playwright', 'viewports': {}}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                                args=['--no-sandbox'])
    manifest['browser_version'] = browser.version
    for name, width in [('desktop', 640), ('mobile', 390)]:
        page = browser.new_page(viewport={'width': width, 'height': round(width * 704 / 640)},
                                device_scale_factor=1)
        page.set_content('<html><head><style>html,body{margin:0;background:white}'
                         'svg{display:block;width:100%;height:auto}</style></head><body>'
                         + raw.decode() + '</body></html>')
        page.screenshot(path=str(out / 'figures' / f'vocab-alignment-{name}.png'), full_page=True)
        manifest['viewports'][name] = {'width': width, 'height': round(width * 704 / 640)}
    browser.close()
(out / 'figures/render.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
