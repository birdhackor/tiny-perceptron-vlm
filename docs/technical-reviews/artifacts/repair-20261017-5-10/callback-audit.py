"""Informed source-revision callback; compare bytes without rerunning models."""

import hashlib
import json
import platform
import re
from pathlib import Path

root = Path(__file__).resolve().parents[4]
outside = Path('/workspace/work/textbook-repair-20261017/technical-5-10')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(path, heading):
    raw = path.read_bytes()
    start = raw.index(heading.encode())
    end = raw.find(b'\n## ', start + 1)
    return raw[start:end + 1] if end >= 0 else raw[start:]


old_path = root / 'docs/course-repair-20261017/reviews/technical-5-10-original-01.json'
original_hash = 'cb02a39565e9d2fd9d43a1d1a47312ad6cebb8b76a433adedc21ab930892bf89'
assert sha(old_path) == original_hash
original = json.loads(old_path.read_text())
freeze_path = root / 'docs/course-repair-20261017/freeze-03.json'
freeze_hash = 'c2651ec9c2b63fed22d68733451a2418282bd8ef3d35733b9690619b17627c89'
assert sha(freeze_path) == freeze_hash
frozen = json.loads(freeze_path.read_text())
entry = next(x for x in frozen['pages'] if x.get('page_id') == '5.10')
before_path = root / 'docs/course-repair-20261017/freeze-02/sources/5.10.md'
after_path = root / entry['snapshot']
before, after = before_path.read_bytes(), after_path.read_bytes()
old_phrase, new_phrase = 'TinyStories官方測試集', 'TinyStories官方驗證資料'
assert before.count(old_phrase.encode()) == 1
assert before.replace(old_phrase.encode(), new_phrase.encode()) == after
assert sha(after_path) == entry['source_sha256']
assert body(root / 'course/chapters/05.md', '## 5.10 ') == after
code = lambda raw: re.findall(rb'```python\n(.*?)```', raw, re.S)
assert code(before) == code(after)
assert not re.findall(rb'!\[[^\]]*\]\([^)]*\)', after)
assert entry['figures_sha256'] == {}

artifacts = []
for item in original['artifacts']:
    actual = sha(root / item['path'])
    assert actual == item['sha256']
    artifacts.append({'id': item['id'], 'path': item['path'], 'sha256': actual, 'unchanged': True})
sources = []
for item in original['sources']:
    if item['kind'] == 'repository_code':
        actual = sha(root / item['path'])
        assert actual == item['sha256']
        sources.append({'id': item['id'], 'path': item['path'], 'sha256': actual, 'unchanged': True})
assert sha(root / original['review_scope']['historical_result_path']) == original['review_scope']['historical_result_sha256']
receipt = json.loads((root / original['artifacts'][2]['path']).read_text())
external = []
for item in receipt['sources']:
    actual = sha(Path(item['local_path']))
    assert actual == item['sha256']
    external.append({'path': item['local_path'], 'sha256': actual, 'unchanged': True})
readme = outside / 'callback-tinystories-README-f54c09f.md'
expected_readme = next(x for x in receipt['sources'] if x['url'].endswith('/README.md'))
assert sha(readme) == expected_readme['sha256']
assert b'the file tinystories-valid.txt can be used for validation loss' in readme.read_bytes()

print(json.dumps({
    'reviewer_task': '/root/repair_tech_5_10',
    'current_context': 'informed_original_owner_source_revision_callback',
    'environment': {'python': platform.python_version(), 'device': 'cpu', 'platform': platform.platform()},
    'freeze_path': str(freeze_path.relative_to(root)), 'freeze_sha256': freeze_hash,
    'original_report_path': str(old_path.relative_to(root)), 'original_report_sha256': original_hash,
    'old_source_sha256': sha(before_path), 'current_source_sha256': sha(after_path),
    'only_one_declared_phrase_changed': True, 'python_blocks_identical': True,
    'current_live_body_matches_snapshot': True, 'current_figures': {},
    'reused_artifact_hashes': artifacts, 'reused_repository_source_hashes': sources,
    'reused_original_external_source_hashes': external,
    'reused_historical_result_sha256': original['review_scope']['historical_result_sha256'],
    'refetched_readme': {'url': expected_readme['url'], 'version': 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
        'path': str(readme), 'sha256': sha(readme), 'bytes': readme.stat().st_size,
        'status_observed_in_fetch': '200', 'contains_explicit_valid_file_statement': True},
    'cpu_selection_and_data_audits_rerun_this_callback': False,
    'historical_model_experiments_rerun_this_callback': False,
    'model_updates_this_callback': 0,
    'scope': 'Current full 5.10 was reread; this execution checks byte identity and old evidence applicability only. Original CPU execution is reused transparently, not claimed as a new execution.'
}, ensure_ascii=False, indent=2))
