"""Read only the fixed v2 HF metadata; fetch no weights or training data."""
import hashlib
import json
import pathlib
import platform
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[4]
REVISION = '979cdfacc588ad0536f1c64fff96f264571cf054'
BASE = 'https://huggingface.co'
URL = BASE + '/api/models/birdhackor/tiny-perceptron-course-models/tree/' + REVISION + '/selftrained/v2?recursive=true&expand=false'
with urllib.request.urlopen(URL, timeout=40) as response:
    raw = response.read()
files = [x for x in json.loads(raw) if x['type'] == 'file']
release = json.loads((ROOT / 'docs/selftrained/v2-jobs/release-final-four-exports.json').read_text())['release']
assert len(files) == 16
assert len(release['exports']) == 4
rows = []
for spec in release['exports']:
    name = spec['name']
    url = BASE + '/birdhackor/tiny-perceptron-course-models/resolve/' + REVISION + '/selftrained/v2/' + name + '/inference-manifest.json'
    with urllib.request.urlopen(url, timeout=40) as response:
        manifest_raw = response.read()
    manifest = json.loads(manifest_raw)
    declared = {x['path']: x for x in spec['files']}
    assert hashlib.sha256(manifest_raw).hexdigest() == declared['inference-manifest.json']['sha256']
    for entry in spec['files']:
        assert entry['license'] == 'MIT' and entry['redistribution_approved'] is True
        target = next(x for x in files if x['path'] == 'selftrained/v2/' + name + '/' + entry['path'])
        assert target['size'] == entry['bytes']
        if entry['path'] != 'inference-manifest.json':
            assert manifest['files'][entry['path']] == entry['sha256']
    rows.append({'name': name, 'url': url, 'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(), 'selected_step': manifest['selected_step'], 'files': len(spec['files']), 'declared_licenses': sorted({x['license'] for x in spec['files']})})
result = {'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/readme-hf-check.py', 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU; metadata only; no model execution'}, 'tree_url': URL, 'tree_sha256': hashlib.sha256(raw).hexdigest(), 'revision': REVISION, 'accessed_on': '2026-10-16', 'anonymous': True, 'file_count': len(files), 'exports': rows, 'result': 'PASS: fixed original remote tree/manifests match all 16 release file sizes and manifest-declared SHA; all author release entries explicitly MIT. No model weight bytes fetched.'}
(ROOT / 'docs/technical-reviews/artifacts/revision-20261016/readme-hf-check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
