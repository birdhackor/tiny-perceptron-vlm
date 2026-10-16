"""Bound the observed Escape limitation; inspect the drawer after its animation settles."""
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import subprocess
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).parent
old_css = subprocess.run(['git', 'show', 'HEAD:course/web/course.css'], capture_output=True, check=True).stdout
result = {'at': datetime.now(UTC).isoformat(), 'scope': 'CSS-only follow-up; no model execution', 'escape_search_checked': {}}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
    result['chromium'] = browser.version
    for use_old in (False, True):
        context = browser.new_context(viewport={'width': 390, 'height': 844})
        page = context.new_page()
        if use_old:
            page.route('**/stylesheets/course*.css', lambda route: route.fulfill(status=200, content_type='text/css', body=old_css))
        page.goto('http://127.0.0.1:8896/readme.html', wait_until='networkidle')
        page.locator('.md-header label[for="__search"]').click()
        page.get_by_role('combobox').click()
        page.keyboard.press('Escape')
        page.wait_for_timeout(400)
        result['escape_search_checked']['prior_css' if use_old else 'current_css'] = page.locator('#__search').is_checked()
        context.close()
    result['escape_limitation_identical_under_prior_css'] = result['escape_search_checked']['prior_css'] == result['escape_search_checked']['current_css']
    context = browser.new_context(viewport={'width': 390, 'height': 844})
    page = context.new_page()
    page.goto('http://127.0.0.1:8896/readme.html', wait_until='networkidle')
    page.locator('.md-header label[for="__drawer"]').click()
    page.wait_for_timeout(400)
    page.mouse.click(360, 500)
    page.wait_for_timeout(500)
    result['drawer_closed'] = not page.locator('#__drawer').is_checked()
    result['toc_visibility'] = page.locator('.md-sidebar--secondary').evaluate('(e)=>getComputedStyle(e).visibility')
    screenshot = ROOT/'work/readme-backfill-callback-03/attempt03/mobile-drawer-closed-settled.png'
    page.screenshot(path=str(screenshot))
    result['settled_screenshot'] = {'path': screenshot.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(screenshot.read_bytes()).hexdigest()}
    context.close()
    browser.close()
result['passed'] = result['escape_limitation_identical_under_prior_css'] and result['drawer_closed'] and result['toc_visibility'] == 'visible'
result['judgment'] = 'Escape did not close the upstream mobile search panel under either stylesheet. Following the actual README search result was tested successfully in css-ui-results-03.json. This is an unchanged operation limitation, not evidence of a CSS regression or a promise of Escape support. The closed-drawer screenshot at180ms was mid-transition; this500ms capture checks settled presentation.'
(OUT/'css-ui-followup-results-03.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False))
