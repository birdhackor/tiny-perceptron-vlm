from pathlib import Path
from playwright.sync_api import sync_playwright
import hashlib, json, sys, torch

out = Path(__file__).parent
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1400, 'height': 900})
    receipt = {'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'browser': browser.version}, 'figures': {}}
    for name in ['posttrain_signals', 'posttrain_stages', 'tokenizer_common_scale']:
        path = Path('course/figures') / (name + '.svg')
        page.set_content('<html><head><meta charset="utf-8"></head><body>' + path.read_text() + '</body></html>')
        page.locator('svg').screenshot(path=str(out / (name + '.png')))
        receipt['figures'][str(path)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bounds': page.locator('svg').bounding_box()}
    response = page.goto('http://127.0.0.1:8788/training.html#T.4')
    page.wait_for_load_state('networkidle')
    receipt['live'] = {'url': page.url, 'status': response.status, 'title': page.title(), 'heading': page.locator('h2', has_text='把文字訓練和對話練習分開看').first.text_content()}
    visible = page.locator('h2', has_text='把文字訓練和對話練習分開看').evaluate('(el)=>{let xs=[];for(let n=el;n&&(n===el||n.tagName!=="H2");n=n.nextElementSibling)xs.push(n.innerText);return xs.join("\\n\\n")}')
    (out / 'live-section-visible.txt').write_text(visible)
    browser.close()
    (out / 'render-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))
