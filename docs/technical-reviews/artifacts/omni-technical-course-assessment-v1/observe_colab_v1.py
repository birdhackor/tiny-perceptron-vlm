"""Anonymous official Colab account-gate observation; no login or cell execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from playwright.sync_api import sync_playwright

OUT = Path('/workspace/work/technical-course-assessment-v1/colab')
OUT.mkdir(parents=True, exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/workspace/work/browsers/chromium-1234/chrome-linux64/chrome', args=['--no-sandbox'])
    context = browser.new_context(viewport={'width':1440,'height':1000}, locale='en-US')
    assert context.cookies() == []
    page = context.new_page()
    page.goto('https://colab.research.google.com/notebooks/intro.ipynb', wait_until='domcontentloaded', timeout=45000)
    button = page.locator('colab-connect-button')
    button.wait_for(state='visible', timeout=30000)
    page.wait_for_timeout(2500)
    button.click()
    page.get_by_text('You must be logged in with a Google Account to continue.', exact=True).wait_for(state='visible', timeout=15000)
    text = page.locator('body').inner_text()
    dom = OUT/'body-after.txt'; dom.write_text(text)
    screenshot = OUT/'after.png'; page.screenshot(path=str(screenshot))
    result = {
        'created_at':datetime.now(timezone.utc).isoformat(),
        'reviewer_task':'/root/omni_technical_coordinator/technical_course',
        'url':page.url,
        'version':'Anonymous fresh Chromium context, en-US, observed live 2026-10-29',
        'environment':{'python':platform.python_version(),'browser':browser.version,'device':'CPU browser; no model inference'},
        'action':'Open official Colab intro with empty context cookies; click Connect; inspect account gate; close without signing in or running a cell.',
        'dialogs':page.get_by_role('dialog').all_text_contents(),
        'observed_primary_text':'You must be logged in with a Google Account to continue.',
        'account_login_performed':False,
        'cell_executed':False,
        'successful_runtime_connection_verified':False,
        'screenshot':{'path':str(screenshot),'sha256':sha(screenshot)},
        'DOM_text':{'path':str(dom),'sha256':sha(dom)},
        'scope':'Direct evidence for account prerequisite at the hosted Connect gate in this UI; no claim that course code has run in Colab.',
    }
    browser.close()
print(json.dumps(result,ensure_ascii=False,indent=2))
