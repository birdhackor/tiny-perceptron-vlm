from pathlib import Path
import json
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
url = json.loads((OUT / 'own-browser-results.json').read_text())['colab_destination']['requested']
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1250, 'height': 950})
    page.goto(url, wait_until='domcontentloaded', timeout=25000)
    page.get_by_text('Runtime', exact=True).first.click(timeout=15000)
    page.wait_for_timeout(400)
    text = page.locator('body').inner_text()
    menu = [line for line in text.splitlines() if any(term in line.lower() for term in ['run all', 'restart', 'runtime type', 'session', 'shift', 'hardware'])]
    page.screenshot(path=str(OUT / 'colab-runtime-menu.png'))
    (OUT / 'own-colab-menu.json').write_text(json.dumps({'url': page.url, 'menu_lines': menu, 'authenticated': False, 'runtime_executed': False, 'chromium': browser.version}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(menu, ensure_ascii=False))
    browser.close()
