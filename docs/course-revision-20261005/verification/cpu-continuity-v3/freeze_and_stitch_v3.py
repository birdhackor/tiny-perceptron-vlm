import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.export_course import DOCUMENTS, home_introduction
from scripts.reading_time import build_inventory

root = Path.cwd()
base = root / 'docs/course-revision-20261005/continuity'
target = base / 'revised-03'
assert not target.exists()
prior_inventory = json.loads((base / 'revised-02/inventory.json').read_text())
prior_pages = {p['page_id']:p for p in prior_inventory['pages']}
index = json.loads((root / 'course/lesson-index.json').read_text())
inventory = build_inventory(root, index, DOCUMENTS, home_introduction(index, True))
changed = []
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for page in inventory['pages']:
    old = prior_pages[page['page_id']]
    assert page['figures_sha256'] == old['figures_sha256']
    if page['source_sha256'] != old['source_sha256']:
        changed.append(page['page_id'])
        assert page['page_id'] == '16.4'
        snapshot = target / 'source-pages/16.4.md'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        from scripts.reading_time import lesson_slices
        snapshot.write_text(lesson_slices((root / page['source']).read_text())['16.4'])
        page['snapshot'] = str(snapshot.relative_to(root))
    else:
        page['snapshot'] = old['snapshot']
    assert sha(root / page['snapshot']) == page['source_sha256']
    for path,digest in page['figures_sha256'].items():
        assert sha(root / path) == digest
assert changed == ['16.4']
(target / 'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
receipt = {'recorded_at':datetime.now(UTC).isoformat(),'inventory_path':str((target / 'inventory.json').relative_to(root)),
    'inventory_sha256':sha(target / 'inventory.json'),'source_count':321,'changed_from_revised_02':['16.4'],
    'changed_versions':[{'page_id':'16.4','previous_sha256':prior_pages['16.4']['source_sha256'],
        'current_sha256':next(p['source_sha256'] for p in inventory['pages'] if p['page_id']=='16.4')}],
    'snapshot_policy':'Only16.4 receives a new snapshot;320 identical-version original V2 snapshots are referenced and byte-verified. All actual referenced figures remain identical.',
    'scope':'Root source freeze only, not reader or scientific acceptance.'}
(target / 'freeze-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
output_base = root / 'outputs/course-revision-20261005'
prior = output_base / 'cpu-kernels-after-continuity-format'
destination = output_base / 'cpu-kernels-after-continuity-v3'
assert not destination.exists()
prior_receipt = json.loads((prior / 'assembly-receipt.json').read_text())
prior_rows = {r['lesson']:r for r in prior_receipt['pages']}
rows = []
for notebook in sorted((root / 'notebooks').rglob('*.ipynb')):
    rel = notebook.relative_to(root / 'notebooks')
    current = json.loads(notebook.read_text())
    old_path = prior / rel
    assert sha(old_path) == prior_rows[notebook.stem]['assembled_output_sha256']
    old = json.loads(old_path.read_text())
    assert len(current['cells']) == len(old['cells'])
    executed = copy.deepcopy(current)
    for cell,previous in zip(executed['cells'],old['cells'],strict=True):
        assert cell['cell_type'] == previous['cell_type']
        if cell['cell_type'] == 'code':
            assert cell['source'] == previous['source'],notebook.stem
            cell['outputs'] = copy.deepcopy(previous.get('outputs',[]))
            cell['execution_count'] = previous['execution_count']
            if ''.join(cell['source']).strip():
                assert cell['execution_count'] is not None
                assert not any(o['output_type']=='error' for o in cell['outputs'])
    output = destination / rel
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(executed,ensure_ascii=False,indent=1)+'\n')
    rows.append({'lesson':notebook.stem,'notebook_sha256':sha(notebook),'actual_execution_origin':str(old_path.relative_to(root)),
        'actual_execution_origin_sha256':sha(old_path),'assembled_output':str(output.relative_to(root)),'assembled_output_sha256':sha(output),
        'mode':'reused_actual_prior_code_and_svg_execution_with_current_markdown'})
assert len(rows)==283
assembly = {'recorded_at':datetime.now(UTC).isoformat(),'total':283,'new_independent_kernels':0,'reused_actual_prior_outputs':283,
    'code_changed':[],'prior_assembly_receipt':str((prior / 'assembly-receipt.json').relative_to(root)),
    'prior_assembly_receipt_sha256':sha(prior / 'assembly-receipt.json'),'pages':rows,
    'scope':'No new CPU/GPU run. All283 code cells and figures match actual prior executions. Only new16.4 prose is stitched; previous aggregates preserved.'}
(destination / 'assembly-receipt.json').write_text(json.dumps(assembly,ensure_ascii=False,indent=2)+'\n')
permanent = root / 'docs/course-revision-20261005/verification/cpu-continuity-v3'
permanent.mkdir(parents=True,exist_ok=True)
(permanent / 'assembly-receipt.json').write_bytes((destination / 'assembly-receipt.json').read_bytes())
(permanent / 'freeze_and_stitch_v3.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(receipt,ensure_ascii=False))
print('Actual283 prior identical-code/SVG CPU outputs reused; zero new kernels or GPU training.')
