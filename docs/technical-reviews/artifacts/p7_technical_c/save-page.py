"""Add frozen metadata to the owner's manually supplied page judgment."""
import json
import hashlib
import sys
from pathlib import Path

row = json.load(sys.stdin)
library_path = Path('docs/technical-reviews/artifacts/p7_technical_c/evidence-library.json')
library = json.loads(library_path.read_text()) if library_path.exists() else {}
for plural in ('sources', 'artifacts'):
    selected = row.pop(plural + '_ids', [])
    row.setdefault(plural, []).extend(library[plural][key] for key in selected)
for ref in row.pop('own_visual_receipts', []):
    p = Path(ref['receipt']); receipt = json.loads(p.read_text())
    row.setdefault('visual_checks', []).append({'figure': ref['figure'], 'source_sha256': receipt['source_sha256'], 'status': 'verified', 'details': '本人实际tools.view_image看过640与360。', 'observation': receipt['observation'], 'receipt': {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}, 'artifacts': receipt['artifacts']})
page_receipt = row.pop('own_page_receipt', None)
if page_receipt:
    p = Path(page_receipt); receipt = json.loads(p.read_text())
    row['page_visual_check'] = {'status': 'verified', 'required': True, 'source_sha256': receipt['source_sha256'], 'details': receipt['details'], 'observation': receipt['observation'], 'receipt': {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}, 'artifacts': receipt['artifacts']}
manifest = json.loads(Path('docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json').read_text())
meta = next(page for page in manifest['pages'] if page['page_id'] == row['page_id'])
for key in ('source_sha256', 'figures_sha256'):
    row[key] = meta[key]
row.setdefault('trace_file', 'docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/c/2bcb00175a2949dd82090328f77c573f.jsonl')
row['question_refs'] = []
path = Path('docs/technical-reviews/artifacts/p7_technical_c/pages') / f'{row["page_id"]}.json'
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    raise SystemExit(f'Preserve existing judgment: {path}')
path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
print(path)
