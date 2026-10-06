"""Original reviewer reinspection: read-only hashes and archive member counts."""
from pathlib import Path
import collections, hashlib, json, re, subprocess, tarfile

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
INITIAL = BASE / 'history/initial-report.json'
CHANGED = ['docs/asset-storage.md', 'assets/training/README.md']
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
def load(p):
    return json.loads(Path(p).read_bytes())
initial = load(INITIAL)
assert sha(INITIAL) == '1cf5f7fef1587d4e9c20126d60f565ea3c5f1307884f1450a70ad212ac206f1e'
result = {'initial_report_path': str(INITIAL.relative_to(ROOT)),
    'initial_report_sha256': sha(INITIAL), 'changed_pages': {}, 'unchanged_pages': {},
    'prior_evidence_dependency_hashes': [], 'figure_hashes': {}, 'archive_hashes': []}
for path, old_hash in initial['source_files'].items():
    current = sha(ROOT / path)
    if path in CHANGED:
        frozen = BASE / 'revised-input' / path
        assert sha(frozen) == current
        text = frozen.read_text()
        assert '8,950個其他檔案（包含圖片、錄音與來源說明）' in text
        assert '8,950個素材' not in text
        if path == 'assets/training/README.md':
            assert '發布狀態（2026-10-02）' not in text
            assert '36946663981' not in text
            assert '不提供額外認證重新下載' not in text
        result['changed_pages'][path] = {'initial_sha256': old_hash, 'current_sha256': current,
            'frozen_path': str(frozen.relative_to(ROOT)), 'full_current_page_personally_read': True,
            'changed_count_sentence': next(line for line in text.splitlines() if '8,950個其他檔案' in line)}
    else:
        assert current == old_hash
        result['unchanged_pages'][path] = {'initial_sha256': old_hash, 'current_sha256': current,
            'unchanged': True, 'prior_full_read_scope_retained': True, 'full_page_reread_claimed': False}
for group in ['sources', 'artifacts']:
    for item in initial[group]:
        current = sha(ROOT / item['path'])
        assert current == item['sha256'], item['path']
        result['prior_evidence_dependency_hashes'].append({'group': group, 'id': item['id'],
            'path': item['path'], 'initial_sha256': item['sha256'], 'current_sha256': current,
            'unchanged': True})
for path, h in initial['figure_sha256'].items():
    assert sha(ROOT / path) == h
    result['figure_hashes'][path] = {'sha256': h, 'unchanged': True,
        'initial_640_360_render_and_personal_view_retained': True, 'new_render_claimed': False}
manifest = load('docs/selftrained/v2-manifest.json')
package = manifest['package']
tar_path = ROOT / package['path']
assert tar_path.stat().st_size == package['bytes']
assert sha(tar_path) == package['sha256']
expected = {x['path']: {'bytes': x['bytes'], 'sha256': x['sha256']}
    for x in manifest['records'] + manifest['assets']}
actual = {}
extensions = collections.Counter()
split_counts = collections.Counter()
support = []
with tarfile.open(tar_path, 'r:gz') as tar:
    for m in tar:
        assert m.isfile(), m.name
        b = tar.extractfile(m).read()
        actual[m.name] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
        suffix = Path(m.name).suffix
        extensions[suffix] += 1
        if suffix == '.jsonl':
            for line in b.splitlines():
                if line.strip():
                    split_counts[json.loads(line)['split']] += 1
        elif suffix not in ['.png', '.wav']:
            support.append({'path': m.name, **actual[m.name]})
assert actual == expected
assert extensions['.png'] == 8335 and extensions['.wav'] == 603
assert extensions['.jsonl'] == 12 and len(support) == 12
assert len(actual) == 8962 and len(actual) - extensions['.jsonl'] == 8950
assert sum(v['bytes'] for v in actual.values()) == 127161811
assert dict(split_counts) == {'test': 3734, 'train': 28876, 'validation': 2435}
result['v2_original_archive_reinspection'] = {
    'manifest_path': 'docs/selftrained/v2-manifest.json',
    'manifest_sha256': sha('docs/selftrained/v2-manifest.json'),
    'manifest_inspected_pointers': ['/package', '/records', '/assets'],
    'package': package, 'archive_sha256_recomputed': sha(tar_path),
    'all_member_bytes_and_sha256_match_manifest': True, 'file_count': len(actual),
    'unpacked_bytes': sum(v['bytes'] for v in actual.values()),
    'extension_counts': dict(extensions), 'jsonl_files': extensions['.jsonl'],
    'media_files': extensions['.png'] + extensions['.wav'],
    'support_files': support, 'other_files': len(actual) - extensions['.jsonl'],
    'record_counts': dict(split_counts)}
# Existing original LFS payloads only: no fetch, hydration, or download.
git = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip())
def existing_payload(path, oid, size):
    p = Path(path)
    if p.stat().st_size != size:
        pointer = p.read_text()
        assert 'oid sha256:' + oid in pointer and 'size ' + str(size) in pointer
        p = git / 'lfs/objects' / oid[:2] / oid[2:4] / oid
    assert p.stat().st_size == size and sha(p) == oid
    result['archive_hashes'].append({'manifest_path': path, 'actual_path': str(p),
        'bytes': size, 'sha256': oid, 'existing_payload_unchanged': True})
for asset in load('assets/training/manifest.json')['assets']:
    existing_payload(asset['archive'], asset['archive_sha256'], asset['archive_bytes'])
for archive in load('docs/natural-assistant/v4/manifest.json')['archives']:
    existing_payload(archive['path'], archive['sha256'], archive['bytes'])
result['executed_command'] = '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-a/verify_reinspection.py'
result['execution_scope'] = 'Hash and immutable archive classification rechecks only. No training, generation, new answer grading, installations, uploads, GPU, or network download.'
(BASE / 'reinspection-measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'initial_report_sha256': sha(INITIAL),
    'full_changed_pages_read': CHANGED, 'unchanged_pages': len(result['unchanged_pages']),
    'unchanged_source_artifact_dependencies': len(result['prior_evidence_dependency_hashes']),
    'unchanged_figures': len(result['figure_hashes']), 'existing_other_archive_payloads': len(result['archive_hashes']),
    'v2': {k: result['v2_original_archive_reinspection'][k] for k in
        ['file_count', 'unpacked_bytes', 'jsonl_files', 'media_files', 'other_files', 'record_counts']},
    'support_files': [x['path'] for x in support], 'assertions': 'passed'}, ensure_ascii=False, indent=2))
