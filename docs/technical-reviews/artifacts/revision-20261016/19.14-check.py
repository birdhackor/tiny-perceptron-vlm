from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from scripts.reading_time import lesson_slices

root = Path.cwd()
body = lesson_slices((root / 'course/chapters/19.md').read_text(encoding='utf-8'))['19.14']
code = re.search(r'```python\n(.*?)```', body, re.S).group(1)
namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(code, 'course/chapters/19.md#19.14', 'exec'), namespace)
split_of = namespace['split_of']
family_id = 'state:start2:set5:set1'
digest = hashlib.sha256(family_id.encode()).hexdigest()
bucket = int(digest[:8], 16) % 10
observed_lines = output.getvalue().splitlines()
expected_split = 'train' if bucket < 7 else 'validation' if bucket < 9 else 'test'
assert observed_lines == [f'{variant} {expected_split}' for variant in ['原句', '改問資訊是否充分', '候選倒序']]

# Branch coverage with actual family strings; all ten possible remainders.
representatives = {}
for i in range(1000):
    fid = f'family:{i}'
    remainder = int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16) % 10
    representatives.setdefault(remainder, fid)
assert set(representatives) == set(range(10))
branch_checks = []
for remainder, fid in sorted(representatives.items()):
    expected = 'train' if remainder < 7 else 'validation' if remainder < 9 else 'test'
    observed = split_of(fid)
    assert observed == expected
    branch_checks.append({'family_id': fid, 'bucket': remainder, 'expected': expected, 'observed': observed})

# Same event, three model-record types, three wording/candidate variants.
records = [
    {'family_id': family_id, 'component': component, 'variant': variant, 'split': split_of(family_id)}
    for component in ['B', 'C', 'Auto']
    for variant in ['original', 'sufficiency-question', 'reversed-candidates']
]
assert len({r['split'] for r in records}) == 1
# The exercise's stated equivalence relation is the unordered pair of addends.
canonical_ab = lambda a, b: tuple(sorted((a, b)))
assert canonical_ab(2, 3) == canonical_ab(3, 2) == (2, 3)
result = {
    'reviewer_task': '/root/tech_20261016_19_14',
    'source': 'course/chapters/19.md#19.14',
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'command': 'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.14-check.py',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU; pure Python only; no training or model inference'},
    'exact_lesson_code': code,
    'exact_lesson_stdout': output.getvalue(),
    'family_sha256': digest,
    'family_bucket': bucket,
    'branch_checks': branch_checks,
    'shared_family_records': records,
    'canonical_addends': {'2+3': canonical_ab(2, 3), '3+2': canonical_ab(3, 2)},
    'result': 'All assertions passed. Same family is assigned one split; ten remainder branches match 0..6/train, 7..8/validation, 9/test. This is a deterministic allocation demonstration, not an empirical split ratio, model-quality result, or proof of correct family canonicalization.'
}
artifact = root / 'docs/technical-reviews/artifacts/revision-20261016/19.14-execution.json'
artifact.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ['source_sha256', 'exact_lesson_stdout', 'family_sha256', 'family_bucket', 'result']}, ensure_ascii=False, indent=2))
