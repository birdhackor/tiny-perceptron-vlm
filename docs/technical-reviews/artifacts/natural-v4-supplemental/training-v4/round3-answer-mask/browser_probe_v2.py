"""Personally inspect actual8790 lesson and changed SVG served by it."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess
from playwright.sync_api import sync_playwright
OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
URL = 'http://127.0.0.1:8790/20.7.html'
revision = subprocess.run(['git', 'rev-parse', '9e963'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
source = subprocess.run(['git', 'show', revision + ':course/chapters/20.md'], cwd=ROOT, capture_output=True, check=True).stdout.decode()
body = re.search(r'^## 20\.7 .*?(?=^## |\Z)', source, re.M | re.S).group()
assert body.encode() == (OUT / '20.7.current.raw.md').read_bytes()
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1400}, device_scale_factor=1)
    response = page.goto(URL, wait_until='networkidle')
    assert response.status == 200
    figure = page.locator('article img[src*="natural-v4-answer-mask"]')
    assert figure.count() == 1
    figure.scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / 'preview-current-answer-mask.png'))
    svg_url = figure.evaluate('(x)=>x.src')
    svg_response = page.request.get(svg_url)
    svg_bytes = svg_response.body()
    assert svg_response.status == 200
    digest = hashlib.sha256(svg_bytes).hexdigest()
    assert digest == 'ffc11a178aef0ba81ed1416c10ddeb90f8b233927cc5bdee4a7d5b620a04707b'
    article = page.locator('article').inner_text()
    assert '第一個答案A由它前面的助理邊界位置預測' in article
    assert '輸入與下一項目標的對齊方式' in article
    (OUT / 'preview-current-visible.txt').write_text(article)
    record = {'url': URL, 'http_status': response.status, 'accessed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'browser': browser.version, 'literal_revision': revision, 'current_20_7_source_matches_literal': True, 'current_20_7_source_sha256': hashlib.sha256(body.encode()).hexdigest(), 'rendered_img_url': svg_url, 'served_svg_sha256': digest, 'figure_bounding_box': figure.bounding_box(), 'article_title': page.locator('article h1').inner_text(), 'screenshot': 'preview-current-answer-mask.png', 'scope': 'Actual current authoring preview, necessary lesson20.7 and servedSVG only; no new model training, fullbook/release rerun or publication approval. Authoring preview timing is pending.'}
    (OUT / 'browser-execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    browser.close()
