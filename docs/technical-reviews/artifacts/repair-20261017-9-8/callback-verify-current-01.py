"""Freeze03 revision callback: exact source comparison and pure-Python example."""

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
PACK02 = Path('/workspace/work/textbook-repair-20261017/input-freeze02/pages')
PACK03 = Path('/workspace/work/textbook-repair-20261017/input-freeze03/pages')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
report_path = ROOT / 'docs/course-repair-20261017/reviews/technical-9-8-original-01.json'
original_bytes = report_path.read_bytes()
assert sha(original_bytes) == '939aef82ec78c2022a1f1f03393e8a48c4c473dee2c2d3c6ef6ef8794b263b85'
original = json.loads(original_bytes)
freeze_path = ROOT / 'docs/course-repair-20261017/freeze-03.json'
freeze_bytes = freeze_path.read_bytes()
assert sha(freeze_bytes) == 'c2651ec9c2b63fed22d68733451a2418282bd8ef3d35733b9690619b17627c89'
freeze = json.loads(freeze_bytes)
entry = next(item for item in freeze['pages'] if item['page_id'] == '9.8')
old = (PACK02 / '9.8.md').read_text()
current = (PACK03 / '9.8.md').read_text()
assert sha(old.encode()) == original['source_sha256']
assert sha(current.encode()) == entry['source_sha256']
old_quote = original['optional_observations'][0]['original_quote']
new_quote = '假設' + old_quote
assert old.count(old_quote) == current.count(new_quote) == 1
assert current == old.replace(old_quote, new_quote, 1)
spec = importlib.util.spec_from_file_location('technical_checker', ROOT / 'scripts/check_technical_reviews.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
actual = dict(checker.sections(ROOT / 'course/chapters/09.md'))['9.8']
assert actual == current
snapshot = ROOT / entry['snapshot']
assert snapshot.read_bytes() == current.encode()
code02 = re.search(r'```python\n(.*?)```', old, re.S)[1]
code03 = re.search(r'```python\n(.*?)```', current, re.S)[1]
assert code02 == code03
ns = {}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(code03, 'freeze03/9.8/example', 'exec'), ns)
original_audit = json.loads((ARTIFACTS / 'verification-original-01.json').read_text())
assert capture.getvalue() == original_audit['lesson_example_stdout']
original_stdout = capture.getvalue()
train_before = list(ns['families']['train'])
ns['families']['test'].append('只看外觀能知道盒內球數嗎')
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    for split, prompts in ns['families'].items():
        for prompt in prompts:
            print(split, prompt, '→', ns['expected'])
assert capture.getvalue() == original_audit['exercise_stdout']
assert train_before == ns['families']['train']
verified = []
for collection in ['artifacts', 'sources', 'historical_inputs']:
    for item in original[collection]:
        if 'path' not in item:
            continue
        path = ROOT / item['path']
        digest = sha(path.read_bytes())
        assert digest == item['sha256'], path
        verified.append({'collection': collection, 'path': item['path'], 'sha256': digest, 'bytes_unchanged': True})
receipts = json.loads((ARTIFACTS / 'source-receipts-original-01.json').read_text())
for receipt in receipts:
    path = Path(receipt['full_local_path'])
    assert sha(path.read_bytes()) == receipt['sha256']
    verified.append({'collection': 'original_primary_source_full_copy', 'path': str(path), 'sha256': receipt['sha256'], 'bytes_unchanged': True})
prerequisites = []
for lesson_id in ['8.3', '9.2', '9.6']:
    a = (PACK02 / (lesson_id + '.md')).read_bytes()
    b = (PACK03 / (lesson_id + '.md')).read_bytes()
    assert a == b
    prerequisites.append({'lesson_id': lesson_id, 'sha256': sha(b), 'bytes_unchanged': True})
assert not re.findall(r'!\[[^\]]*\]\(([^)]+)\)', current)
assert entry['figures_sha256'] == {}
out = {
    'reviewer_task': '/root/repair_tech_9_8',
    'current_context': 'informed_original_owner_source_revision_callback',
    'checked_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-9-8/callback-verify-current-01.py',
    'environment': {'python': platform.python_version(), 'torch_installed_metadata_only': importlib.metadata.version('torch'), 'platform': platform.platform(), 'device': 'cpu; pure-Python execution only'},
    'original_report_sha256': sha(original_bytes),
    'freeze03_sha256': sha(freeze_bytes),
    'original_source_sha256': sha(old.encode()),
    'current_source_sha256': sha(current.encode()),
    'only_source_edit': {'old_quote': old_quote, 'new_quote': new_quote, 'exactly_prefix_added': '假設', 'other_bytes_unchanged': True},
    'current_source_equals_pack_and_snapshot': True,
    'python_block_unchanged': True,
    'current_example_stdout': original_stdout,
    'current_exercise_stdout': capture.getvalue(),
    'reused_evidence_hash_checks': verified,
    'explicit_prerequisites': prerequisites,
    'figures': {},
    'no_model_execution': 'No torch import, model inference, optimizer step, training, GPU, downloads, or installation in this callback. Historical numerical results remain the prior saved-output audit.',
    'all_assertions_passed': True,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
