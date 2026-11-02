"""Bounded fix14 dependency checks for this same README reviewer, without broad tool signoff."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import re
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
WORK = Path('/workspace/work/omni-technical-readme-v1/current-fix14')
AD = ROOT / 'docs/technical-reviews/artifacts/omni-technical-readme-v1'
sys.path.insert(0, str(ROOT))
from scripts.export_course import checked_notebook

sha = lambda raw: hashlib.sha256(raw).hexdigest()
file_sha = lambda path: sha(path.read_bytes())
original_path = ROOT / 'docs/course-revision-20261029-omni/reviews/technical/technical_readme/technical-original-v1-schema-corrected.json'
original = json.loads(original_path.read_text())
freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json'
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == 'readme')
assert file_sha(ROOT / 'README.md') == page['source_sha256'] == original['source_sha256']
assert page['figures_sha256'] == original['figure_sha256']
for name, expected in page['figures_sha256'].items():
    assert file_sha(ROOT / name) == expected

old_freeze = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json').read_text())
old_pages = {p['page_id']: p for p in old_freeze['pages']}
changed_pages = [p['page_id'] for p in freeze['pages'] if p['source_sha256'] != old_pages[p['page_id']]['source_sha256']]
assert changed_pages == ['12.2']

source_drift = []
unchanged_sources = 0
for source in original['sources']:
    if source['kind'] != 'repository_code':
        continue
    current_hash = file_sha(ROOT / source['path'])
    if current_hash != source['sha256']:
        source_drift.append({'source_id': source['id'], 'path': source['path'], 'original_sha256': source['sha256'], 'current_sha256': current_hash})
    else:
        unchanged_sources += 1
assert [p['path'] for p in source_drift] == ['scripts/build_visuals.py']
old_script = ROOT / 'docs/course-repair-20261008/reviews/freeze-01/freeze/implementation/scripts/build_visuals.py'
assert file_sha(old_script) == source_drift[0]['original_sha256']

previous_audit = json.loads((AD / 'source-audit.json').read_text())
slices = []
for item in previous_audit['dependency_slices']:
    text = (ROOT / item['path']).read_text()
    headings = list(re.finditer(r'^## (\S+) .+$', text, re.M))
    for i, heading in enumerate(headings):
        if heading[1] != item['selector']:
            continue
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.start():end]
        assert sha(body.encode()) == item['section_sha256'], item['selector']
        for inspected in item['inspected_slices']:
            assert sha(body[inspected['start_character']:inspected['end_character']].encode()) == inspected['sha256']
        slices.append({'path': item['path'], 'selector': item['selector'],
                       'section_sha256': item['section_sha256'], 'current_file_sha256': file_sha(ROOT / item['path']),
                       'same_inspected_slice': True})
assert len(slices) == 31

index = json.loads((ROOT / 'course/lesson-index.json').read_text())
comparison = []
for item in index:
    notebook = ROOT / item['notebook']
    relative = notebook.relative_to(ROOT / 'notebooks')
    if item['id'] == '12.2':
        executed = WORK / 'kernel-own' / relative
        method = 'same_reviewer_new_kernel_this_callback'
    else:
        executed = Path('/workspace/work/omni-technical-readme-v1/sandbox/outputs/notebooks-readme-all') / relative
        method = 'same_reviewer_original_kernel_reuse_matching_cells'
    checked_notebook(json.loads(notebook.read_text()), executed)
    comparison.append({'lesson': item['id'], 'method': method, 'notebook_sha256': file_sha(notebook),
                       'executed_path': str(executed), 'executed_sha256': file_sha(executed)})
assert len(comparison) == 331
inventory = WORK / 'notebook-comparison-rows.json'
inventory.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')

# Independently verify the supplied cumulative copy too; do not promote its author's report to our execution.
cumulative_path = ROOT / 'docs/course-revision-20261029-omni/evidence/notebooks-technical-fix14-cumulative.json'
cumulative = json.loads(cumulative_path.read_text())
assert cumulative['source_freeze']['sha256'] == file_sha(freeze_path)
bundle_checks = 0
for record in cumulative['records']:
    item = next(i for i in index if i['id'] == record['lesson'])
    notebook = ROOT / item['notebook']
    executed = Path(record['bundle_path'])
    assert file_sha(notebook) == record['notebook_sha256']
    assert file_sha(executed) == record['executed_sha256']
    assert Path(record['source']).read_bytes() == executed.read_bytes()
    checked_notebook(json.loads(notebook.read_text()), executed)
    bundle_checks += 1
assert bundle_checks == 331

test_raw = WORK / 'targeted-tests.txt'
assert int((WORK / 'targeted-tests.exit.txt').read_text()) == 0
assert '217 passed' in test_raw.read_text()
kernel_result = json.loads((WORK / 'kernel-own/validation-kernel.json').read_text())
assert kernel_result['total'] == kernel_result['passed'] == 1 and kernel_result['failed'] == 0
gates = json.loads((WORK / 'gate-command-records.json').read_text())
result = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer_task': '/root/omni_technical_coordinator/technical_readme',
    'environment': {'python': platform.python_version(), 'device': 'cpu', 'torch': '2.14.1+cpu'},
    'source_freeze': {'path': freeze_path.relative_to(ROOT).as_posix(), 'sha256': file_sha(freeze_path)},
    'readme_same_bytes': True, 'readme_figure_map_same': True, 'changed_canonical_sources_since_fix13': changed_pages,
    'original_registered_code_sources_unchanged': unchanged_sources, 'changed_registered_code_sources': source_drift,
    'original_source_snapshot': {'source': 'scripts/build_visuals.py', 'snapshot': {'path': old_script.relative_to(ROOT).as_posix(), 'sha256': file_sha(old_script)}},
    'rechecked_necessary_slices': slices,
    'own_current_notebook_coverage': {'checked_current_cells': 331, 'new_kernels_this_callback': ['12.2'], 'reused_own_matching_original_kernels': 330,
        'inventory_path': str(inventory), 'inventory_sha256': file_sha(inventory),
        'scope': 'Only 12.2 newly executed in this callback. Other 330 are this reviewer\'s preserved original runs with exact current cells; not 331 fresh runs or a new scientific review of 12.2.'},
    'provided_cumulative_bundle_check': {'path': cumulative_path.relative_to(ROOT).as_posix(), 'sha256': file_sha(cumulative_path),
        'recorded_by': cumulative['recorded_by'], 'checked_cell_and_hash_records': bundle_checks,
        'new_by_author': cumulative['reexecuted_this_fix'], 'reused_by_author': cumulative['reused_matching_cells'],
        'scope': 'Independently checked hashes/cells of the supplied raw bundle. These remain author execution records, not our fresh 331-kernel execution.'},
    'targeted_tests': {'command': 'PYTHONPATH=/workspace/tiny-perceptron-vlm UV_PROJECT_ENVIRONMENT=/workspace/work/omni-technical-readme-v1/venv uv run --extra cpu --no-sync pytest -q tests/test_build_visuals_index.py tests/test_windows_student_commands.py tests/test_reading_time.py',
        'exit_code': 0, 'stdout': test_raw.read_text(), 'raw_path': str(test_raw), 'raw_sha256': file_sha(test_raw),
        'scope': 'Linux tests of current image-index rules, Windows command/quote alignment and reading-time/Git-pin fixture. Not a Windows runtime test or full current pytest suite. The reading-time fixture makes temporary test-only Git objects; no project commit/ref change.'},
    'own_new_kernel': {'command': 'PYTHONPATH=/workspace/tiny-perceptron-vlm UV_PROJECT_ENVIRONMENT=/workspace/work/omni-technical-readme-v1/venv uv run --extra cpu --no-sync python scripts/check_notebooks.py --mode kernel --workers 1 --lesson 12.2 --output /workspace/work/omni-technical-readme-v1/current-fix14/kernel-own',
        'exit_code': int((WORK / 'kernel-own.exit.txt').read_text()), 'result': kernel_result,
        'raw_path': str(WORK / 'kernel-own.txt'), 'raw_sha256': file_sha(WORK / 'kernel-own.txt')},
    'current_gate_commands': gates['records'],
    'scope_boundary': 'Bounded current for README original 80 claims and c75. No signoff on the 210-path general tool audit, no full new reader completion, no activation/publication, no GPU/formal training, no model/API calls.'
}
out = AD / 'current-dependency-probe-v1.json'
with out.open('x') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(out), 'sha256': file_sha(out), 'unchanged_registered_sources': unchanged_sources,
                  'necessary_slices': len(slices), 'own_current_cell_coverage': len(comparison),
                  'provided_bundle_verified': bundle_checks, 'targeted_tests': '217 passed',
                  'current_gate_exit_codes': [r['exit_code'] for r in gates['records']]}, ensure_ascii=False, indent=2))
