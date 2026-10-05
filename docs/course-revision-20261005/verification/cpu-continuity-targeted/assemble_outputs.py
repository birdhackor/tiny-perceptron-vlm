"""Reuse actual unchanged CPU outputs, preserving their original provenance."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'outputs/course-revision-20261005'
OLD = BASE / 'cpu-kernels-before-continuity'
NEW = BASE / 'cpu-kernels-after-continuity'
RERUN = {'2.5', '6.5', '7.5', '9.10', '10.2'}
CODE_CHANGED = {'7.5', '10.2'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

old_validation = json.loads((OLD / 'validation-kernel.json').read_text())
assert old_validation['total'] == old_validation['passed'] == 283
assert old_validation['failed'] == 0
initial_inventory = json.loads((ROOT / 'docs/course-revision-20261005/continuity/inventory.json').read_text())
initial_pages = {p['page_id']: p for p in initial_inventory['pages']}
rows = []
changed = set()
for path in sorted((ROOT / 'notebooks').rglob('*.ipynb')):
    rel = path.relative_to(ROOT / 'notebooks')
    current = json.loads(path.read_text())
    original = json.loads((OLD / rel).read_text())
    if code(current) != code(original):
        changed.add(path.stem)
    if path.stem in RERUN:
        run = BASE / ('cpu-kernel-continuity-' + path.stem.replace('.', '_'))
        result = json.loads((run / 'validation-kernel.json').read_text())
        assert result['total'] == result['passed'] == 1 and result['failed'] == 0
        actual = run / rel
        executed = json.loads(actual.read_text())
        assert [c['source'] for c in current['cells']] == [c['source'] for c in executed['cells']]
        mode = 'actual_new_independent_kernel'
        reason = 'code_changed' if path.stem in CODE_CHANGED else 'referenced_svg_changed'
    else:
        actual = OLD / rel
        assert code(current) == code(original)
        for figure, digest in initial_pages[path.stem]['figures_sha256'].items():
            assert sha(ROOT / figure) == digest
        assert len(current['cells']) == len(original['cells'])
        executed = copy.deepcopy(current)
        for c, prior in zip(executed['cells'], original['cells'], strict=True):
            assert c['cell_type'] == prior['cell_type']
            if c['cell_type'] == 'code':
                assert c['source'] == prior['source']
                c['execution_count'] = prior['execution_count']
                c['outputs'] = copy.deepcopy(prior.get('outputs', []))
        mode = 'reused_actual_prior_code_execution_with_current_markdown'
        reason = 'unchanged_code_and_unchanged_referenced_svg; no new kernel run claimed'
    for c in executed['cells']:
        if c['cell_type'] == 'code' and ''.join(c['source']).strip():
            assert c['execution_count'] is not None
            assert all(o['output_type'] != 'error' for o in c.get('outputs', []))
    target = NEW / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(executed, ensure_ascii=False, indent=1) + '\n')
    rows.append({'lesson': path.stem, 'mode': mode, 'reason': reason,
                 'notebook': str(path.relative_to(ROOT)), 'notebook_sha256': sha(path),
                 'actual_execution_origin': str(actual.relative_to(ROOT)),
                 'actual_execution_origin_sha256': sha(actual),
                 'assembled_output': str(target.relative_to(ROOT)),
                 'assembled_output_sha256': sha(target)})
assert len(rows) == 283 and changed == CODE_CHANGED
receipt = {'created_at': datetime.now(timezone.utc).isoformat(),
           'scope': 'Current-code CPU outputs only; five actual targeted kernels, 278 prior executions reused with current Markdown. Not a second full283 kernel run, model training, or a readability/scientific acceptance.',
           'total': 283, 'new_independent_kernels': 5, 'reused_prior_executions': 278,
           'failed': 0, 'code_changed': sorted(CODE_CHANGED), 'figure_dependency_changed': ['2.5','6.5','9.10'],
           'original_validation_path': str((OLD / 'validation-kernel.json').relative_to(ROOT)),
           'original_validation_sha256': sha(OLD / 'validation-kernel.json'), 'pages': rows}
(NEW / 'assembly-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
permanent = ROOT / 'docs/course-revision-20261005/verification/cpu-continuity-targeted'
permanent.mkdir(parents=True, exist_ok=True)
(permanent / 'assembly-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
for lesson in sorted(RERUN):
    run = BASE / ('cpu-kernel-continuity-' + lesson.replace('.', '_'))
    (permanent / (lesson + '-validation-kernel.json')).write_bytes((run / 'validation-kernel.json').read_bytes())
print('Assembled current 283 outputs: 5 actual targeted kernels, 278 unchanged-code prior executions; zero errors.')
