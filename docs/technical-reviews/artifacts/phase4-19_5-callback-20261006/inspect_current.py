"""Current original-reviewer callback: raw identities and local page rendering."""
import hashlib
import json
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIOR = ROOT / 'docs/technical-reviews/artifacts/phase4-19_5-independent-factual'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def section(raw, name):
    headings = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
    i = next(i for i, h in enumerate(headings) if h[0].startswith(f'## {name} '.encode()))
    return raw[headings[i].start():headings[i+1].start() if i+1 < len(headings) else len(raw)]

chapter = (ROOT / 'course/chapters/19.md').read_bytes()
current = section(chapter, '19.5')
old = (PRIOR / 'recheck/section.md').read_bytes()
assert current == (HERE / 'inputs/section.md').read_bytes()
assert digest(current) == '90d723bb85b9ccbd1ac378f32053e3994e729a55bf1dba8053d8cb9a36daba6a'
assert current == old.replace('不是隻靠'.encode(), '不是只靠'.encode())
assert (HERE / 'inputs/fence-1.py').read_bytes() == (PRIOR / 'fence-run/fence-1.py').read_bytes()
context = section(chapter, '19.12')
assert context == (HERE / 'inputs/context-19.12.md').read_bytes()
assert context == (PRIOR / 'recheck/section-19.12-context.md').read_bytes()

identities = []
for source in json.loads((PRIOR / 'sources/pinned-source-provenance.json').read_bytes()):
    name = source['url'].split('/1df335318bda03fd771807f66976953231d5a00b/', 1)[1]
    local = (ROOT / name).read_bytes()
    saved = (ROOT / source['path']).read_bytes()
    assert digest(local) == digest(saved) == source['sha256']
    identities.append({'original_path': name, 'saved_path': source['path'], 'sha256': source['sha256']})

page_facts = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox', '--disable-gpu'])
    for label, width, height in [('desktop', 1280, 800), ('mobile', 390, 844)]:
        page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)
        response = page.goto('http://127.0.0.1:8765/19.5.html', wait_until='networkidle', timeout=15000)
        assert response.status == 200
        page.screenshot(path=str(HERE / f'{label}-initial.png'), full_page=False)
        page.locator('details').evaluate_all('(nodes) => nodes.forEach((n) => n.open = true)')
        page.screenshot(path=str(HERE / f'{label}-complete.png'), full_page=True)
        page.locator('table').first.screenshot(path=str(HERE / f'{label}-scores-table.png'))
        text = page.locator('body').inner_text()
        assert '不是只靠上方程式印出標準答案' in text
        assert 'DIRECT:15' in text and '23／23' in text
        assert '整合成品的能力界定與驗收安排' in text
        page_facts.append({'viewport': [width, height], 'http_status': response.status,
                           'title': page.title(), 'body_text_sha256': digest(text.encode()),
                           'details_count': page.locator('details').count(),
                           'table_count': page.locator('table').count(),
                           'document_dimensions': page.evaluate('({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})')})
        page.close()
    browser_version = browser.version
    browser.close()

result = {
    'reviewer_task': '/root/phase4_factual_coordinator/factual_19_5',
    'current_source_sha256': digest(current), 'prior_source_sha256': digest(old),
    'current_necessary_context_sha256': digest(context),
    'only_change': '不是隻靠 → 不是只靠; no new substantive claim, command or recipe',
    'original_fence_unchanged': True, 'figures': {},
    'prior_code_and_raw_measurements': identities,
    'retained_support_scope': 'Prior personally executed original fence/encoding/parser and raw measurement audit remain valid for identical code/data and unchanged substantive claims; no pipeline/model re-evaluation.',
    'rendered_current_page': page_facts,
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'cpu',
                    'playwright': version('playwright'), 'chromium': browser_version},
    'exit_result': 'all assertions passed',
}
(HERE / 'current-inspection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
