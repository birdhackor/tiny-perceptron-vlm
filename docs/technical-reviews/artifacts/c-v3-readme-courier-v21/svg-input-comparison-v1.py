"""Neutral read-only identity comparison, not a review or build execution."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/tmp/tpb')

def file_ref(name, expected=None):
    raw = (ROOT / name).read_bytes()
    value = {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if expected:
        assert value['sha256'] == expected, value
    return raw, value

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

raw, old_probe = file_ref('docs/technical-reviews/artifacts/omni-technical-readme-v1/current-c75-probe-v1.json', 'c5f617e97fc3c00e8fe1ec44ea775605a6fda2d368b82f71035e663c4a7e2574')
before = json.loads(raw)['verified_original_svgs']
raw, old_comparison = file_ref('docs/technical-reviews/artifacts/omni-P5-1.10/courier-readme-inventory-stdout-v1.json', 'df842f4479b80ee96e312f062ab28c10185dfda88eb5b7d14ad0ba800b770f11')
old = json.loads(raw)
assert digest(before) == old['prior_digest_recomputed']
for item in old['changed_common']:
    assert before[item['name']] == item['before']
    before[item['name']] = item['current']
assert digest(before) == old['current_digest']
assert not old['added'] and not old['removed']
raw, old_fix = file_ref('docs/course-revision-20261029-omni/reviews/P5-actual/legacy-own-fix-omni_technical_coordinator--technical_readme/current-fix-verification-v1.json', '77f1d4d66671b204df79c71ae1f198ec54dc29d8c211e658f63237d626798834')
fix = json.loads(raw)
after = {}
for p in sorted((ROOT / 'course/figures').glob('*.svg')):
    raw = p.read_bytes()
    after[p.name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
inputs = []
for item in fix['current_source_evidence']['source_inputs']:
    _, current = file_ref(item['path'])
    inputs.append({**current, 'prior_sha256': item['sha256'], 'prior_bytes': item['bytes'],
                   'sha_equal': current['sha256'] == item['sha256'], 'bytes_equal': current['bytes'] == item['bytes']})
common = sorted(before.keys() & after.keys())
changed = [{'name': n, 'before': before[n], 'current': after[n]} for n in common if before[n] != after[n]]
result = {'schema_version': 1, 'executed_at': datetime.now(timezone.utc).isoformat(),
          'courier_task': '/root/omni_p6_technical_tools', 'physical_root': str(ROOT),
          'command': 'python docs/technical-reviews/artifacts/c-v3-readme-courier-v21/svg-input-comparison-v1.py',
          'environment': {'python': sys.version, 'platform': platform.platform(), 'compute': 'CPU metadata only'},
          'old_probe': old_probe, 'old_comparison': old_comparison, 'old_fix': old_fix,
          'prior_215_inventory_reconstruction': 'Old verified_original_svgs plus exactly old_comparison.changed_common current values; both old digests verified, no added/removed names.',
          'prior_digest': digest(before), 'current_digest': digest(after),
          'prior_count': len(before), 'current_count': len(after), 'unchanged_common_count': len(common) - len(changed),
          'changed_common': changed, 'added': [{'name': n, **after[n]} for n in sorted(after.keys() - before.keys())],
          'removed': [{'name': n, **before[n]} for n in sorted(before.keys() - after.keys())],
          'source_inputs': inputs,
          'scope': 'Only names/bytes/SHA identity. No build, model, project import, image viewing, rejection-policy execution or semantic verdict.'}
payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
dest = Path(__file__).with_name('svg-input-comparison-stdout-v1.json')
with dest.open('x', encoding='utf8', newline='\n') as stream:
    stream.write(payload)
print(payload, end='')
