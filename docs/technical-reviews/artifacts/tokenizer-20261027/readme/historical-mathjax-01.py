"""Read-only asset/closure verification; no downloads, staging, or old-record rewrites."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/readme'
sys.path.insert(0, str(ROOT))
from scripts.stage_mathjax import validate_manifest, read_asset

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest = validate_manifest(json.loads((ROOT / 'course/web/mathjax-assets.json').read_bytes()))
site = ROOT / 'outputs/tokenizer-20261027/site/javascripts/mathjax'
assets = []
for item in manifest['files']:
    path = site / item['path']
    raw = read_asset(path, item)
    assets.append({'path': item['path'], 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
disposition_path = ROOT / 'docs/course-review-20261019/dispositions-continuity-01.json'
disposition = json.loads(disposition_path.read_bytes())
items = []
for index, item in enumerate(disposition['items']):
    if not item['issue_id'].startswith('TECH-readme-mathjax_network'):
        continue
    original_path = ROOT / item['report']['path']
    assert sha(original_path) == item['report']['sha256']
    original = json.loads(original_path.read_bytes())
    issue = original['issues'][0]
    assert issue['claim_id'] == 'mathjax_network'
    fix = item['verification'][0]
    assert sha(ROOT / fix['report']['path']) == fix['report']['sha256']
    items.append({'issue_id': item['issue_id'], 'disposition_json_pointer': f'/items/{index}', 'original_report': item['report'], 'original_json_pointer': item['json_pointer'], 'original_issue': issue, 'historical_fix_report': fix['report'], 'historical_fix_json_pointer': fix['json_pointer']})
assert len(items) == 5
source = (ROOT / 'README.md').read_text()
assert '公式由網站隨附的 MathJax 在瀏覽器排版。' in source
assert 'MathJax 連網排版' not in source
result = {'python': sys.version, 'command': 'uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/readme/historical-mathjax-01.py', 'manifest_version': manifest['version'], 'upstream_revision': manifest['revision'], 'site_assets_verified': len(assets), 'site_asset_bytes': sum(x['bytes'] for x in assets), 'assets': assets, 'disposition_path': str(disposition_path.relative_to(ROOT)), 'disposition_sha256': sha(disposition_path), 'historical_items': items, 'current_source_sha256': sha(ROOT / 'README.md'), 'scope': 'All 94 already-built local MathJax assets byte/hash checked using current validator. No prepare_assets call, no fetch, no writes to old reports, no full kernel/build/training.'}
(OUT / 'historical-mathjax-output-01.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('assets','historical_items')},ensure_ascii=False,indent=2))
