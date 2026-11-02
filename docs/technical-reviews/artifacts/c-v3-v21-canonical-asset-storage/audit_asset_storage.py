"""Bounded asset-storage audit; never extracts real weights or copies a release ZIP."""

import hashlib
import importlib.util
import io
import json
import platform
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(relative):
    return json.loads((ROOT / relative).read_bytes())


def main():
    started = datetime.now(UTC).isoformat()
    report = {'started_at': started, 'actor': '/root/c_v3_technical_coord/canonical_asset_storage',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'device': 'CPU; standard-library storage and transport audit only'},
              'limits': 'No training, final evaluation, model inference, real-weight extraction, archive copy, clone, install or publication.'}
    paths = ['.gitattributes', '.gitignore', 'scripts/course_product.py', 'scripts/fetch_training_assets.py',
             'tiny_perceptron/assets.py', 'assets/training/manifest.json',
             'docs/selftrained/v2-manifest.json', 'docs/natural-assistant/v4/manifest.json',
             'docs/omni-products/v1/public-release.json', 'docs/natural-assistant/v4/public-release.json',
             'docs/decision-product/v3/public-release.json',
             'docs/decision-product/v3/metadata/acquisition.json',
             'docs/decision-product/v3/metadata/qualified-bundle-manifest.json']
    report['source_fingerprints'] = {p: file_digest(ROOT / p) for p in paths}
    base = load('assets/training/manifest.json')
    v2 = load('docs/selftrained/v2-manifest.json')
    v4 = load('docs/natural-assistant/v4/manifest.json')
    packs = [{'path': a['archive'], 'bytes': a['archive_bytes'], 'sha256': a['archive_sha256']} for a in base['assets']]
    packs.append({'path': v2['package']['path'], 'bytes': v2['package']['bytes'], 'sha256': v2['package']['sha256']})
    packs.extend({k: a[k] for k in ('path', 'bytes', 'sha256')} for a in v4['archives'])
    for pack in packs:
        path = ROOT / pack['path']
        pack['observed_bytes'] = path.stat().st_size
        pack['observed_sha256'] = file_digest(path)
        pack['matches'] = pack['bytes'] == pack['observed_bytes'] and pack['sha256'] == pack['observed_sha256']
    report['training_packages'] = packs
    report['table_sizes'] = {'base_count': len(base['assets']),
                             'base_bytes': sum(p['observed_bytes'] for p in packs[:len(base['assets'])]),
                             'v2_count': 1, 'v2_bytes': packs[len(base['assets'])]['observed_bytes'],
                             'v4_count': len(v4['archives']),
                             'v4_bytes': sum(p['observed_bytes'] for p in packs[len(base['assets']) + 1:]),
                             'MB_denominator': 1000000}
    report['v2_manifest_file_counts'] = {'jsonl': len(v2['records']), 'other': len(v2['assets']),
                                         'total': len(v2['records']) + len(v2['assets']),
                                         'note': 'Enumerated manifest, not a new inspection of historical training/evaluation records.'}
    spec = importlib.util.spec_from_file_location('asset_storage_course_product', ROOT / 'scripts/course_product.py')
    product = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(product)
    descriptor = load('docs/decision-product/v3/public-release.json')
    c3 = {'local_path': str(ARCHIVE), 'observed_bytes': ARCHIVE.stat().st_size,
          'observed_sha256': file_digest(ARCHIVE)}
    c3['archive_identity_matches'] = c3['observed_bytes'] == descriptor['bytes'] and c3['observed_sha256'] == descriptor['sha256']
    with zipfile.ZipFile(ARCHIVE) as source:
        c3['member_names'] = source.namelist()
        c3['inference_files'] = {}
        for name, expected in descriptor['bundle'].items():
            member = 'tiny-perceptron-c-v3/bundle/' + name
            raw = source.read(member)
            c3['inference_files'][name] = {'bytes': len(raw), 'sha256': digest(raw),
                                          'matches': digest(raw) == expected,
                                          'occurrences': source.namelist().count(member)}
        acquisition = load('docs/decision-product/v3/metadata/acquisition.json')
        c3['metadata_identity'] = []
        for item in acquisition['files']:
            raw = source.read(item['archive_member'])
            retained = (ROOT / item['path']).read_bytes()
            c3['metadata_identity'].append({'path': item['path'], 'bytes': len(raw),
                                            'sha256': digest(raw), 'matches_retained': raw == retained,
                                            'matches_manifest': digest(raw) == item['sha256'] and len(raw) == item['bytes']})
    c3['resume_state_named_members'] = [n for n in c3['member_names']
                                        if any(t in Path(n).name.lower() for t in ('optimizer', 'rng', 'sampler', 'latest.pt'))]
    report['c3_local_zip'] = c3
    # Exercise the installer with tiny synthetic bytes, never the real weights.
    synthetic = {name: ('synthetic audit fixture: ' + name).encode() for name in sorted(product.BUNDLE_FILES)}
    with tempfile.TemporaryDirectory(prefix='synthetic-', dir=HERE) as temporary:
        temp = Path(temporary)
        fixture = temp / 'synthetic.zip'
        with zipfile.ZipFile(fixture, 'w') as target:
            for name, raw in synthetic.items():
                target.writestr('tiny-perceptron-c-v3/bundle/' + name, raw)
            target.writestr('tiny-perceptron-c-v3/evidence/not-installed.txt', b'fixture extra')
        fake = {'bytes': fixture.stat().st_size, 'sha256': file_digest(fixture),
                'bundle': {n: digest(raw) for n, raw in synthetic.items()}}
        output = temp / 'installed'
        product.install_archive(fixture, output, fake)
        installed = sorted(p.name for p in output.iterdir())
        product.verify_bundle(output, fake)
        bad = dict(fake, sha256='0' * 64)
        try:
            product.install_archive(fixture, temp / 'bad-output', bad)
        except ValueError as error:
            rejected = str(error)
        else:
            raise AssertionError('bad archive identity accepted')
        report['installer_synthetic_execution'] = {'installed': installed,
                'exact_four_files': set(installed) == product.BUNDLE_FILES,
                'extra_archive_file_installed': (output / 'not-installed.txt').exists(),
                'bad_archive_rejected': rejected,
                'scope': 'Actual install_archive/verify_bundle execution on tiny synthetic file bytes; real fixed ZIP inspected read-only.'}
    report['training_asset_list_stdout'] = __import__('subprocess').run(
        [sys.executable, str(ROOT / 'scripts/fetch_training_assets.py'), '--list'],
        cwd=ROOT, text=True, check=True, capture_output=True).stdout
    # Exactly one unauthenticated application-level GET; preserve proxy and TLS settings.
    transport = {'requested_at': datetime.now(UTC).isoformat(), 'url': descriptor['url'],
                 'request_method': 'GET', 'application_headers': {'User-Agent': 'asset-storage-fresh-review/1'},
                 'authorization_header_supplied': False, 'cookie_header_supplied': False,
                 'attempts': 1, 'body_saved': False, 'automatic_redirects': True}
    request = urllib.request.Request(descriptor['url'], headers=transport['application_headers'])
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(descriptor['bytes'] + 1)
            transport.update({'http_status': response.status, 'bytes': len(raw), 'sha256': digest(raw),
                              'content_type': response.headers.get('Content-Type'),
                              'identity_matches': len(raw) == descriptor['bytes'] and digest(raw) == descriptor['sha256']})
            if transport['identity_matches']:
                with zipfile.ZipFile(io.BytesIO(raw)) as source:
                    transport['member_names_match_local'] = source.namelist() == c3['member_names']
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
        transport.update({'success': False, 'error_type': type(error).__name__, 'error': str(error)})
    transport['completed_at'] = datetime.now(UTC).isoformat()
    report['c3_anonymous_transport'] = transport
    report['completed_at'] = datetime.now(UTC).isoformat()
    (HERE / 'audit-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed_at': report['completed_at'], 'table_sizes': report['table_sizes'],
                      'all_training_archive_hashes_match': all(p['matches'] for p in packs),
                      'c3_archive_matches': c3['archive_identity_matches'],
                      'c3_bundle_files_match': all(p['matches'] for p in c3['inference_files'].values()),
                      'metadata_matches': all(p['matches_retained'] and p['matches_manifest'] for p in c3['metadata_identity']),
                      'installer': report['installer_synthetic_execution'], 'transport': transport}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
