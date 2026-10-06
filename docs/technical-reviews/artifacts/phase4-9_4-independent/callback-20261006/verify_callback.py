"""Same-owner callback: exact hashes/provenance, no prior report reads or model runs."""
from pathlib import Path
import hashlib
import json
import platform
import re
import sys

ROOT = Path.cwd()
BASE = Path('docs/technical-reviews/artifacts/phase4-9_4-independent')
CB = BASE / 'callback-20261006'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (CB / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


env = {'python': platform.python_version(), 'python_executable': sys.executable,
       'cwd': str(ROOT), 'device': 'CPU file/hash operations only',
       'model_inference': 'none', 'training': 'none', 'network': 'none',
       'original_cpu_verification': 'retained original CPython3.13.5/PyTorch2.14.1+cpu run; not rerun'}
save('callback-environment.json', env)
raw = (ROOT / 'course/chapters/09.md').read_bytes()
headings = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
i = next(i for i, h in enumerate(headings) if h[0].startswith(b'## 9.4 '))
current = raw[headings[i].start():headings[i+1].start()]
assert current == (CB / 'current-section.md').read_bytes()
assert digest(current) == 'f0ac2d91c859767d1f54db4f70b5f158d261ac1fa9a72bac9d3699e211a3730b'
fences = re.findall(rb'(?ms)^```python\n(.*?)^```\s*$', current)
assert len(fences) == 1
assert fences[0] == (BASE / 'original-fence/fence-1.py').read_bytes()
assert not re.findall(rb'!\[[^\]]*\]\([^)]+\)', current)
assert b'\xe8\xa8\xb1\xe5\x8f\xaf\xe6\x87\x89\xe7\x94\xb1\xe6\x87\x89\xe7\x94\xa8\xe5\x8f\xaf\xe9\x9d\xa0\xe6\xb5\x81\xe7\xa8\x8b\xe6\x8f\x90\xe4\xbe\x9b\xe3\x80\x82' not in current
assert '應由應用中的可靠授權流程提供資訊。' in current.decode()
manifest = json.loads((BASE / 'artifact-manifest.json').read_bytes())
frozen = {item['path']: item['sha256'] for item in manifest['files']}
reuse = json.loads((CB / 'reuse-verification.json').read_bytes())
for item in reuse['proof_files']:
    assert digest(Path(item['path']).read_bytes()) == item['frozen_sha256'] == frozen[item['path']]
original_path = ROOT / 'docs/course-experiments/results/safety.json'
assert original_path.read_bytes() == (BASE / 'current/docs/course-experiments/results/safety.json').read_bytes()
record = json.loads(original_path.read_bytes())
assert record['revision'] == 'a864a60bbf72583afc9bbaf45e052bd4fe076c62'
for path in ['scripts/course_experiments/behavior.py', 'scripts/course_experiments/common.py',
             'scripts/course_experiments/text.py', 'tiny_perceptron/data.py']:
    assert digest((BASE / 'run-version' / path).read_bytes()) == record['code_sha256'][path]
targeted = json.loads((CB / 'targeted-original-json-leaves.json').read_bytes())
assert digest(original_path.read_bytes()) == targeted['full_raw_sha256']
permission = targeted['leaves']['permission_test']
for name, samples in permission.items():
    assert len(samples) == 6
    assert all(sample['exact'] and sample['eos'] for sample in samples)
    assert {int(re.search(r'盒子(\d+)', s['messages'][0]['content'])[1]) for s in samples} == {1, 9, 17}
    for s in samples:
        assert 'owner' not in s['messages'][0]['content'] and '公開' not in s['messages'][0]['content']
        assert s['generated'] == s['expected']
        # Original tokenizer encoding contract already reviewed/executed; exact raw recorded IDs.
        ids = s['generated_ids']; content_ids = ids[:ids.index(2)]
        assert content_ids == [b + 8 for b in s['expected'].encode('utf-8')]
save('callback-verification.json', {
    'current_section_sha256': digest(current), 'current_fence_sha256': digest(fences[0]),
    'unchanged_hashes_verified': len(reuse['proof_files']), 'raw_measurement_json_sha256': digest(original_path.read_bytes()),
    'permission_counts': {k: {'records': len(v), 'raw_id_matches': len(v), 'eos': len(v)} for k, v in permission.items()},
    'changed_claims': 'No substantive claims changed. Orthographic normalization and deletion of a repeated authorization sentence; operative requirement still present.',
    'reused_execution_scope': 'Original fence execution, exercise variants, split/hash reconstruction retained from this same reviewer; no new execution of fence or models claimed.',
    'targeted_pointers_file': str(CB / 'targeted-original-json-leaves.json'), 'figure_sha256': {},
    'prior_report_policy': 'opaque preserved bytes, not parsed or read during callback', 'status': 'passed'})
print(json.dumps({'status': 'passed', 'source_sha256': digest(current), 'fence_byte_identical': True,
                  'reused_proof_hashes': len(reuse['proof_files']), 'permission_id_matches': {k: '6/6' for k in permission},
                  'new_fence_execution': False, 'model_execution': False}, ensure_ascii=False))
