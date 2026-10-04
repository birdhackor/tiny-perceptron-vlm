"""Actually inspect the new 8783 training guide in Chromium; no model service."""
from pathlib import Path
import datetime, hashlib, json, subprocess
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
URL = 'http://127.0.0.1:8783/natural-v4-training.html'
REVISION = 'f8ca78bc3ffef82b0faa352c464909b76bfd8976'
pin = subprocess.run(['git', 'show', REVISION + ':docs/natural-assistant/v4/TRAINING.md'], cwd=ROOT, capture_output=True, check=True).stdout
current = (ROOT / 'docs/natural-assistant/v4/TRAINING.md').read_bytes()
assert pin == current
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
    response = page.goto(URL, wait_until='networkidle')
    assert response.status == 200
    article = page.locator('article').inner_text()
    (OUT / 'preview-training-visible.txt').write_text(article)
    headings = page.locator('article h2').all_text_contents()
    assert len(headings) == 8
    row = page.locator('article tr').filter(has_text='整次訓練的外層計時')
    assert row.count() == 1
    displayed = row.inner_text()
    assert '開啟日誌、啟動Python訓練前開始' in displayed
    assert '訓練結束並讀入結果後停止' in displayed
    assert '外層日誌與結果讀取的少量開銷' in displayed
    page.locator('article table').first.scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / 'preview-current-timing.png'))
    links = page.locator('article a').evaluate_all('(xs)=>xs.map(x=>({text:x.innerText,href:x.href}))')
    (OUT / 'preview-training-links.json').write_text(json.dumps(links, ensure_ascii=False, indent=2) + '\n')
    page.locator('article h2').filter(has_text='先驗證用途').scroll_into_view_if_needed() if any('先驗證用途' in x for x in headings) else page.locator('article h2').filter(has_text='先驗證').scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / 'preview-current-validation.png'))
    student = next(x for x in links if 'natural-v4-student.html' in x['href'])
    page.locator('article a').filter(has_text=student['text']).first.click()
    page.wait_for_load_state('networkidle')
    target_title = page.locator('article h1').inner_text()
    page.screenshot(path=str(OUT / 'preview-current-student-link.png'))
    record = {'accessed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'training_url': URL, 'http_status': response.status, 'literal_revision': REVISION, 'literal_source_sha256': hashlib.sha256(pin).hexdigest(), 'current_source_matches_literal': True, 'browser': browser.version, 'actual_headings': headings, 'corrected_timing_row_visible': displayed, 'clicked': student, 'target_url': page.url, 'target_title': target_title, 'screenshots': ['preview-current-timing.png', 'preview-current-validation.png', 'preview-current-student-link.png'], 'scope': 'Real new book preview and actual prerequisite link navigation. No UI model server, GPU training, or model inference started.'}
    (OUT / 'browser-execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    browser.close()
