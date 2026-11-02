"""Read-only courier identity comparison; no project imports or build execution."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
PRIOR = ROOT / 'docs/technical-reviews/artifacts/omni-technical-readme-v1/current-c75-probe-v1.json'

def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

raw = PRIOR.read_bytes()
prior = json.loads(raw)
before = prior['verified_original_svgs']
after = {}
for path in sorted((ROOT / 'course/figures').glob('*.svg')):
    data = path.read_bytes()
    after[path.name] = {'sha256': sha256(data).hexdigest(), 'bytes': len(data)}

inputs = []
for relative, expected in prior['source_inputs'].items():
    data = (ROOT / relative).read_bytes()
    actual = sha256(data).hexdigest()
    inputs.append({'path': relative, 'bytes': len(data), 'prior_sha256': expected,
                   'current_sha256': actual, 'equal': expected == actual})

common = sorted(before.keys() & after.keys())
changed = [{'name': name, 'before': before[name], 'current': after[name]}
           for name in common if before[name] != after[name]]
print(json.dumps({
    'executed_at': datetime.now(timezone.utc).isoformat(),
    'writer_task': '/root/omni_p5_technical_1_10',
    'prior': {'path': str(PRIOR.relative_to(ROOT)), 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()},
    'algorithm': 'Same name/sha256/bytes inventory and compact sorted-key digest as current_c75_probe_v1.py lines24-28; current root read only.',
    'prior_count': len(before), 'current_count': len(after),
    'prior_digest_recomputed': digest(before),
    'prior_recorded_before_digest': prior['before_inventory_sha256'],
    'prior_digest_matches_record': digest(before) == prior['before_inventory_sha256'],
    'current_digest': digest(after),
    'exact_whole_inventory_equal': before == after,
    'common_count': len(common), 'unchanged_common_count': len(common) - len(changed),
    'changed_common': changed,
    'added': [{'name': name, **after[name]} for name in sorted(after.keys() - before.keys())],
    'removed': [{'name': name, **before[name]} for name in sorted(before.keys() - after.keys())],
    'source_inputs': inputs,
    'scope': 'Mechanical identity only; no build_visuals execution, no figure viewing, no semantic assessment or still-fixed decision.'
}, ensure_ascii=False, indent=2))
