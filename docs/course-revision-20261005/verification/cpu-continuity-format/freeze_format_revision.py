import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from scripts.export_course import DOCUMENTS, home_introduction
from scripts.reading_time import build_inventory

root = Path('/workspace/tiny-perceptron-vlm')
base = root / 'docs/course-revision-20261005/continuity'
target = base / 'revised-02'
assert not target.exists(), 'Never overwrite a prior freeze'
index = json.loads((root / 'course/lesson-index.json').read_text())
inventory = build_inventory(root, index, DOCUMENTS, home_introduction(index, True), target / 'source-pages')
prior = json.loads((base / 'revised-01/inventory.json').read_text())
prior_pages = {p['page_id']: p for p in prior['pages']}
changed = []
figures = {}
for page in inventory['pages']:
    snapshot = root / page['snapshot']
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == page['source_sha256']
    old = prior_pages[page['page_id']]
    if (page['source_sha256'], page['figures_sha256']) != (old['source_sha256'], old['figures_sha256']):
        changed.append(page['page_id'])
    for name, digest in page['figures_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        assert name not in figures or figures[name] == digest
        figures[name] = digest
assert set(changed) == {'13.14', '16.12'}
for name, digest in figures.items():
    destination = target / 'source-figures' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
(target / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
receipt = {'recorded_at': datetime.now(UTC).isoformat(), 'inventory_path': str((target / 'inventory.json').relative_to(root)),
    'inventory_sha256': hashlib.sha256((target / 'inventory.json').read_bytes()).hexdigest(),
    'source_count': len(inventory['pages']), 'figure_count': len(figures), 'changed_from_revised_01': changed,
    'reason': 'Required Ruff Markdown Python-fence formatting: 13.14 trailing blank removed; 16.12 two function-separator blanks added. No semantic or numerical edits. Historical evidence excluded from formatter to preserve its original bytes.',
    'changed_versions': [{ 'page_id': p, 'previous_sha256': prior_pages[p]['source_sha256'], 'current_sha256': next(x['source_sha256'] for x in inventory['pages'] if x['page_id']==p)} for p in changed],
    'scope': 'Root actual source/figure freeze only; no reader or scientific judgment.'}
(target / 'freeze-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False))
