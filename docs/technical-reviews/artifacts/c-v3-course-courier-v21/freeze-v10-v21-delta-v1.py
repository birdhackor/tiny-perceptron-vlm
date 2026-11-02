"""Read-only freeze metadata and four prior-input identities, no semantic verdict."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('/tmp/tpb')

def read(name, expected=None):
    data = (root / name).read_bytes()
    ref = {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if expected:
        assert ref['sha256'] == expected, ref
    return data, ref

raw, oldref = read('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json', 'ddcc500db74f5df9ab3bb6fddcdc511ca177c98ce64e1a94160d1a7ccc1c35be')
old = {p['page_id']: p for p in json.loads(raw)['pages']}
raw, newref = read('docs/course-revision-20261101-c-v3/source-freeze-v21.json', 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea')
new = {p['page_id']: p for p in json.loads(raw)['pages']}
fields = ['source', 'selector', 'source_sha256', 'figures_sha256', 'notebook', 'notebook_sha256']
changed, equal = [], []
for pid in sorted(old.keys() & new.keys()):
    delta = {f: {'before': old[pid].get(f), 'current': new[pid].get(f)} for f in fields if old[pid].get(f) != new[pid].get(f)}
    if delta:
        changed.append({'page_id': pid, 'changed_fields': delta})
    else:
        equal.append(pid)
raw, fixref = read('docs/course-revision-20261029-omni/reviews/P5-actual/legacy-own-fix-omni_technical_coordinator--technical_course/current-fix-verification-v1.json', 'c2704c480678c96617608bfe91f2241afd0cea8e9178b01bee5e2a5720d96472')
fix = json.loads(raw)
inputs = []
for prior in [*fix['read_scope']['helper_slices'], fix['read_scope']['full_noncanonical_prerequisite']]:
    _, ref = read(prior['path'])
    inputs.append({**ref, 'prior_sha256': prior['sha256'], 'same_sha256': ref['sha256'] == prior['sha256']})
raw, course_ref = read(new['course']['snapshot'], new['course']['source_sha256'])
result = {'schema_version': 1, 'executed_at': datetime.now(timezone.utc).isoformat(),
          'courier_task': '/root/omni_p6_technical_tools', 'physical_root': str(root),
          'command': 'python docs/technical-reviews/artifacts/c-v3-course-courier-v21/freeze-v10-v21-delta-v1.py',
          'before_freeze': oldref, 'current_freeze': newref, 'old_fix': fixref,
          'compared_fields': fields, 'before_count': len(old), 'current_count': len(new),
          'added': [{k: new[pid].get(k) for k in ['page_id', *fields, 'snapshot']} for pid in sorted(new.keys() - old.keys())],
          'removed': [{k: old[pid].get(k) for k in ['page_id', *fields, 'snapshot']} for pid in sorted(old.keys() - new.keys())],
          'changed': changed, 'equal_compared_fields_count': len(equal), 'equal_compared_fields_page_ids': equal,
          'current_prior_helper_and_guide_identities': inputs,
          'current_course_snapshot': {**course_ref, 'lines': len(raw.splitlines()), 'matches_current_freeze': True},
          'scope': 'Exact authenticated freeze JSON field comparison and four actual current file SHA checks only; not 370-page semantic reading or all-live-figure/notebook validation. No build/test/model/network/Git operation.'}
payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
with Path(__file__).with_name('freeze-v10-v21-delta-stdout-v1.json').open('x', encoding='utf8', newline='\n') as f:
    f.write(payload)
print(payload, end='')
