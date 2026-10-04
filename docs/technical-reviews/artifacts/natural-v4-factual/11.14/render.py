from pathlib import Path
import importlib.metadata
import json
import subprocess
from playwright.sync_api import sync_playwright

base = Path(__file__).parent
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    for name, w, h in [('practical_order', 720, 1120), ('patchify', 800, 330), ('natural_photo_evidence', 720, 1140)]:
        page = browser.new_page(viewport={'width': w, 'height': h}, device_scale_factor=1, reduced_motion='reduce')
        svg = (Path('course/figures') / (name + '.svg')).read_text()
        page.set_content('<html><body style="margin:0">' + svg + '</body></html>')
        page.screenshot(path=str(base / (name + '.png')), full_page=True)
        print('rendered', name, w, h)
        page.close()
    browser.close()
print('chromium', subprocess.check_output(['/usr/bin/chromium', '--version'], text=True).strip())
print('playwright', importlib.metadata.version('playwright'))
