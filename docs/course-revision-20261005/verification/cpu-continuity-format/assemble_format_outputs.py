import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

root = Path('/workspace/tiny-perceptron-vlm')
base = root / 'outputs/course-revision-20261005'
prior = base / 'cpu-kernels-after-continuity'
target = base / 'cpu-kernels-after-continuity-format'
assert not target.exists(), 'Never overwrite actual prior output records'
prior_receipt = json.loads((prior / 'assembly-receipt.json').read_text())
prior_rows = {r['lesson']: r for r in prior_receipt['pages']}
old_inventory = json.loads((root / 'docs/course-revision-20261005/continuity/revised-01/inventory.json').read_text())
old_pages = {p['page_id']: p for p in old_inventory['pages']}
rows = []
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def code(nb):
    return [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
changed = []
for path in sorted((root / 'notebooks').rglob('*.ipynb')):
    rel = path.relative_to(root / 'notebooks')
    current = json.loads(path.read_text())
    old_path = prior / rel
    old = json.loads(old_path.read_text())
    assert sha(old_path) == prior_rows[path.stem]['assembled_output_sha256']
    for figure, digest in old_pages[path.stem]['figures_sha256'].items():
        assert sha(root / figure) == digest
    if code(current) != code(old):
        changed.append(path.stem)
        assert path.stem == '16.12'
        actual = base / 'cpu-kernel-continuity-format-16_12' / rel
        validation = json.loads((actual.parents[1] / 'validation-kernel.json').read_text())
        assert validation['total'] == validation['passed'] == 1 and validation['failed'] == 0
        executed = json.loads(actual.read_text())
        assert [c['source'] for c in current['cells']] == [c['source'] for c in executed['cells']]
        mode = 'actual_new_independent_kernel_after_required_whitespace_format'
    else:
        actual = old_path
        executed = copy.deepcopy(current)
        assert len(current['cells']) == len(old['cells'])
        for c, p in zip(executed['cells'], old['cells'], strict=True):
            assert c['cell_type'] == p['cell_type']
            if c['cell_type'] == 'code':
                assert c['source'] == p['source']
                c['outputs'] = copy.deepcopy(p.get('outputs', []))
                c['execution_count'] = p['execution_count']
        mode = 'reused_real_prior_aggregate_execution_for_identical_code_and_svg'
    for cell in executed['cells']:
        if cell['cell_type'] == 'code' and ''.join(cell['source']).strip():
            assert cell['execution_count'] is not None
            assert all(o['output_type'] != 'error' for o in cell.get('outputs', []))
    destination = target / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(executed, ensure_ascii=False, indent=1) + '\n')
    rows.append({'lesson': path.stem, 'mode': mode, 'notebook_sha256': sha(path), 'actual_execution_origin': str(actual.relative_to(root)),
        'actual_execution_origin_sha256': sha(actual), 'assembled_output': str(destination.relative_to(root)), 'assembled_output_sha256': sha(destination),
        'prior_aggregate_mode': prior_rows[path.stem]['mode'], 'prior_real_kernel_origin': prior_rows[path.stem]['actual_execution_origin']})
assert len(rows) == 283 and changed == ['16.12']
receipt = {'recorded_at': datetime.now(UTC).isoformat(), 'total': 283, 'new_independent_kernels': 1, 'reused_prior_aggregate': 282,
    'code_changed': changed, 'previous_assembly_receipt': str((prior / 'assembly-receipt.json').relative_to(root)),
    'previous_assembly_receipt_sha256': sha(prior / 'assembly-receipt.json'),
    'scope': 'One actual new CPU kernel after function-separator whitespace changes. Other 282 use exact unchanged-code and SVG executions with current Markdown. Not a second full283 run or model training.', 'pages': rows}
(target / 'assembly-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
permanent = root / 'docs/course-revision-20261005/verification/cpu-continuity-format'
permanent.mkdir(parents=True, exist_ok=True)
(permanent / 'assembly-receipt.json').write_bytes((target / 'assembly-receipt.json').read_bytes())
(permanent / '16.12-validation-kernel.json').write_bytes((base / 'cpu-kernel-continuity-format-16_12/validation-kernel.json').read_bytes())
print('Current 283 outputs: one actual targeted kernel, 282 unchanged-code prior executions reused; original aggregates retained.')
