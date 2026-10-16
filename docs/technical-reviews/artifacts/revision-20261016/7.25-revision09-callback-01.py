"""Known-context callback verification; no notebook kernel or training is launched."""
import ast
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from scripts.reading_time import lesson_slices
from scripts.export_course import checked_notebook

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
artdir = root / 'docs/technical-reviews/artifacts/revision-20261016'
original_path = root / 'docs/course-revision-20261016/reviews/technical-7.25-original.json'
original = json.loads(original_path.read_text())
freezes = {}
for version in ('08', '09'):
    manifest_path = root / f'docs/course-revision-20261016/freeze-{version}.json'
    manifest = json.loads(manifest_path.read_text())
    page = next(p for p in manifest['pages'] if p['page_id'] == '7.25')
    path = root / page['snapshot']
    assert sha(path) == page['source_sha256']
    freezes[version] = {'page': page, 'body': path.read_bytes().decode('utf-8')}
old, new = (freezes[v]['body'] for v in ('08', '09'))
assert hashlib.sha256(old.encode()).hexdigest() == original['source_sha256']
current = lesson_slices((root / 'course/chapters/07.md').read_bytes().decode('utf-8'))['7.25']
assert current == new
pattern = r'```python\n(.*?)```'
old_code, new_code = (re.findall(pattern, body, re.S)[0] for body in (old, new))
assert len(re.findall(pattern, new, re.S)) == 1
prose_equal = re.sub(pattern, '```python\n<CODE>```', old, flags=re.S) == re.sub(pattern, '```python\n<CODE>```', new, flags=re.S)
assert prose_equal
old_ast = ast.dump(ast.parse(old_code), include_attributes=False)
new_ast = ast.dump(ast.parse(new_code), include_attributes=False)
assert old_ast == new_ast
assert freezes['08']['page']['figures_sha256'] == freezes['09']['page']['figures_sha256'] == {}
assert not re.findall(r'!\[[^\]]*\]\(([^)]+)\)', new)

results = {}
for version, code in (('08', old_code), ('09', new_code)):
    namespace = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, f'freeze{version}-7.25', 'exec'), namespace)
    logs = namespace['logs']
    assert all(len(v['ok']) == len(v['work']) == 4 for v in logs.values())
    rows = {k: {'correct_total': sum(v['ok']), 'accuracy': sum(v['ok']) / 4,
                'work_total': sum(v['work']), 'mean_work': sum(v['work']) / 4}
            for k, v in logs.items()}
    allowed = [k for k, v in rows.items() if v['mean_work'] <= 3]
    oracle = [max(v['ok'][i] for v in logs.values()) for i in range(4)]
    assert allowed == ['low', 'medium'] and oracle == [1, 1, 1, 1]
    assert max([0, 0, 0]) == 0
    results[version] = {'stdout': out.getvalue(), 'rows': rows, 'allowed_at_mean_3': allowed,
                        'oracle': oracle, 'all_wrong_oracle': 0}
assert results['08'] == results['09']
assert results['09']['stdout'] == json.loads((artdir / '7.25-execution.json').read_text())['stdout']

receipts = json.loads((artdir / '7.25-source-receipts.json').read_text())
source_checks = []
for source in receipts['sources']:
    path = Path(source['local_fulltext_not_committed'])
    assert sha(path) == source['raw_sha256']
    text = path.with_suffix('.txt').read_text()
    assert all(e['text'] in text for e in source['excerpts'])
    source_checks.append({'id': source['id'], 'url': source['url'], 'raw_sha256': sha(path),
                          'raw_bytes_unchanged': True, 'receipt_excerpts_present': True})

kernel_path = artdir / 'kernel-execution-09.json'
author_kernel = json.loads(kernel_path.read_text())
author_row = next(r for r in author_kernel['rows'] if r['id'] == '7.25')
assert author_row['status'] == 'passed'
input_path = root / freezes['09']['page']['notebook']
assert sha(input_path) == author_row['input_sha256'] == freezes['09']['page']['notebook_sha256']
output_path = Path(author_row['output_path'])
assert sha(output_path) == author_row['output_sha256']
notebook = checked_notebook(json.loads(input_path.read_text()), output_path)
join = lambda value: ''.join(value) if isinstance(value, list) else value
cells = [c for c in notebook['cells'] if c['cell_type'] == 'code' and 'logs = {' in join(c['source'])]
assert len(cells) == 1
assert ast.dump(ast.parse(join(cells[0]['source'])), include_attributes=False) == new_ast
notebook_stdout = ''.join(join(o['text']) for o in cells[0]['outputs'] if o['output_type'] == 'stream')
assert notebook_stdout == results['09']['stdout']

result = {
    'reviewer_task': '/root/tech_20261016_7_25',
    'initial_context': 'fresh', 'this_review_context': 'informed callback; not a new blind review',
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/7.25-revision09-callback-01.py',
    'environment': {'python': sys.version, 'executable': sys.executable,
                    'platform': platform.platform(), 'device': 'CPU; pure Python arithmetic; no training or new notebook kernel'},
    'original_report_sha256': sha(original_path),
    'freeze08_sha256': freezes['08']['page']['source_sha256'],
    'freeze09_sha256': freezes['09']['page']['source_sha256'],
    'current_source_matches_freeze09': True, 'prose_outside_python_identical': prose_equal,
    'python_ast_identical_without_location_attributes': old_ast == new_ast,
    'ast_sha256': hashlib.sha256(new_ast.encode()).hexdigest(),
    'figure_hashes': {},
    'diff': ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='freeze08', tofile='freeze09')),
    'own_execution': results,
    'primary_source_receipt_recheck': source_checks,
    'author_kernel_receipt_inspection': {
        'path': kernel_path.relative_to(root).as_posix(), 'sha256': sha(kernel_path),
        'author_task': author_kernel['author_task'], 'recorded_total': author_kernel['total'],
        'recorded_passed': author_kernel['passed'], 'inspected_lesson_only': '7.25', 'row': author_row,
        'input_output_hashes_checked': True, 'all_cells_sources_counts_errors_checked': True,
        'lesson_stdout_matches_own_execution': True,
        'scope': 'Read and verified author-saved 7.25 notebook; reviewer did not launch or rerun the 12 kernels.'},
    'assertions': 'all passed',
}
print(json.dumps(result, ensure_ascii=False, indent=2))
