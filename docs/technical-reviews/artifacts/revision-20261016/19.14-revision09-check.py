"""19.14 freeze-09 informed callback: compare snapshots and execute both code versions."""
from pathlib import Path
import ast
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
import sys
from scripts.reading_time import lesson_slices

root = Path.cwd()
review_dir = root / 'docs/course-revision-20261016'
artifact_dir = root / 'docs/technical-reviews/artifacts/revision-20261016'
reviewer = '/root/tech_20261016_19_14'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def page_record(freeze):
    return next(x for x in freeze['pages'] if x['page_id'] == '19.14')

old_freeze = json.loads((review_dir / 'freeze-08.json').read_text())
new_freeze = json.loads((review_dir / 'freeze-09.json').read_text())
old_record, new_record = page_record(old_freeze), page_record(new_freeze)
old_bytes = (root / old_record['snapshot']).read_bytes()
new_bytes = (root / new_record['snapshot']).read_bytes()
old_body, new_body = old_bytes.decode('utf-8'), new_bytes.decode('utf-8')
assert sha(old_bytes) == old_record['source_sha256']
assert sha(new_bytes) == new_record['source_sha256']
current = lesson_slices((root / 'course/chapters/19.md').read_bytes().decode('utf-8'))['19.14']
assert current == new_body
assert old_record['figures_sha256'] == new_record['figures_sha256'] == {}
pattern = r'```python\n(.*?)```'
old_match = re.search(pattern, old_body, re.S)
new_match = re.search(pattern, new_body, re.S)
old_code, new_code = old_match.group(1), new_match.group(1)
old_prose = old_body[:old_match.start(1)] + '<CODE>' + old_body[old_match.end(1):]
new_prose = new_body[:new_match.start(1)] + '<CODE>' + new_body[new_match.end(1):]
assert old_prose == new_prose
old_nonblank = [line for line in old_code.splitlines() if line.strip()]
new_nonblank = [line for line in new_code.splitlines() if line.strip()]
assert old_nonblank == new_nonblank
old_ast = ast.dump(ast.parse(old_code), include_attributes=False)
new_ast = ast.dump(ast.parse(new_code), include_attributes=False)
assert old_ast == new_ast
old_namespace, new_namespace = {}, {}
old_out, new_out = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(old_out):
    exec(compile(old_code, 'freeze-08:19.14', 'exec'), old_namespace)
with contextlib.redirect_stdout(new_out):
    exec(compile(new_code, 'freeze-09:19.14', 'exec'), new_namespace)
expected = '原句 train\n改問資訊是否充分 train\n候選倒序 train\n'
assert old_out.getvalue() == new_out.getvalue() == expected
split_of = new_namespace['split_of']
family_id = 'state:start2:set5:set1'
digest = hashlib.sha256(family_id.encode()).hexdigest()
bucket = int(digest[:8], 16) % 10
assert bucket == 3
representatives = {}
for i in range(1000):
    fid = f'family:{i}'
    remainder = int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16) % 10
    representatives.setdefault(remainder, fid)
assert set(representatives) == set(range(10))
branch_checks = []
for remainder, fid in sorted(representatives.items()):
    wanted = 'train' if remainder < 7 else 'validation' if remainder < 9 else 'test'
    observed = split_of(fid)
    previous_observed = old_namespace['split_of'](fid)
    assert wanted == observed == previous_observed
    branch_checks.append({'family_id': fid, 'bucket': remainder, 'expected': wanted,
                          'observed': observed, 'previous_observed': previous_observed})
records = [{'family_id': family_id, 'component': component, 'variant': variant,
            'split': split_of(family_id)}
           for component in ['B', 'C', 'Auto']
           for variant in ['original', 'sufficiency-question', 'reversed-candidates']]
assert len({r['split'] for r in records}) == 1
canonical_ab = lambda a, b: tuple(sorted((a, b)))
assert canonical_ab(2, 3) == canonical_ab(3, 2) == (2, 3)
preservation = {}
for path in ['docs/course-revision-20261016/reviews/technical-19.14-original.json',
             'docs/course-revision-20261016/reviews/technical-19.14-revise.json',
             'docs/technical-reviews/artifacts/revision-20261016/19.14-execution.json',
             'docs/technical-reviews/artifacts/revision-20261016/19.14-source-receipts.json',
             'docs/technical-reviews/artifacts/revision-20261016/19.14-source-receipts-corrected.json']:
    preservation[path] = sha((root / path).read_bytes())
result = {
    'reviewer_task': reviewer,
    'callback_context': 'informed; initial reviewer task began fresh; this is not a new blind review',
    'source': 'course/chapters/19.md#19.14',
    'source_sha256': new_record['source_sha256'],
    'old_source_sha256': old_record['source_sha256'],
    'old_snapshot': old_record['snapshot'], 'new_snapshot': new_record['snapshot'],
    'command': 'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.14-revision09-check.py',
    'environment': {'python': sys.version, 'platform': platform.platform(),
                    'device': 'CPU; pure Python only; no model training or inference'},
    'comparison': {'current_source_matches_freeze09_snapshot': True,
                   'prose_outside_code_identical': True,
                   'nonblank_code_lines_identical': True,
                   'ast_identical_excluding_location_attributes': True,
                   'canonical_ast_sha256': sha(new_ast.encode()),
                   'figures_unchanged': {},
                   'unified_diff': ''.join(difflib.unified_diff(old_body.splitlines(keepends=True),
                        new_body.splitlines(keepends=True), fromfile=old_record['snapshot'],
                        tofile=new_record['snapshot']))},
    'exact_lesson_code': new_code,
    'old_lesson_stdout': old_out.getvalue(), 'exact_lesson_stdout': new_out.getvalue(),
    'family_sha256': digest, 'family_bucket': bucket,
    'branch_checks': branch_checks, 'shared_family_records': records,
    'canonical_addends': {'2+3': canonical_ab(2, 3), '3+2': canonical_ab(3, 2)},
    'preserved_prior_evidence_sha256': preservation,
    'author_execution_scope': 'Author kernel-execution09 is separate evidence reported as 12 new kernels. It was not run or used as my own execution here.',
    'result': 'All assertions passed. The only differences are two added blank lines around the Python function; prose, figure set, nonblank code, and AST are unchanged. Both snippets print the same three train lines. All ten bucket branches, nine shared-family variants and unordered-addend canonicalization agree. No model ability, actual data split ratio, or complete family canonicalization is established.'
}
path = artifact_dir / '19.14-revision09-execution.json'
with path.open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({k: result[k] for k in ['source_sha256', 'comparison', 'exact_lesson_stdout', 'family_bucket', 'result']}, ensure_ascii=False, indent=2))
