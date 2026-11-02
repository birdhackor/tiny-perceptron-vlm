"""Read-only source-identity, archive, and CLI probe for lesson 19.19.

No network, extraction, installation, model inference, training or final evaluation.
"""

import difflib
import hashlib
import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_json(path):
    raw = (ROOT / path).read_bytes()
    return json.loads(raw), {'path': path, 'sha256': digest(raw)}


def main():
    import torch
    from scripts.course_product import verify_bundle

    freeze, freeze_ref = checked_json('docs/course-revision-20261101-c-v3/source-freeze-v21.json')
    assert freeze_ref['sha256'] == 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'
    page = next(row for row in freeze['pages'] if row['page_id'] == '19.19')
    frozen = (ROOT / page['snapshot']).read_bytes()
    body = re.search(r'^## 19\.19 .*?(?=^## |\Z)', (ROOT / 'course/chapters/19.md').read_text(), re.M | re.S)[0].encode()
    assert frozen == body and digest(frozen) == page['source_sha256']
    prior, prior_ref = checked_json('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json')
    assert prior_ref['sha256'] == 'd74eb6e0fc958fbd410c7588dbe4d96a7ced72ff512c7ea8f8863c87f68c9108'
    before_page = next(row for row in prior['pages'] if row['page_id'] == '19.19')
    before = (ROOT / before_page['snapshot']).read_bytes()
    assert digest(before) == before_page['source_sha256']
    added = '\n本節固定下載原B／文字C版本來教配套核對；現行裝置C-v3另用[19.25](19.md#19.25)的固定Release與loader，不覆寫這份HF版本。\n'
    reverted = frozen.decode().replace(added, '', 1).replace('RNG（Random Number Generator，隨機數生成器）的狀態', 'RNG', 1).encode()
    assert reverted == before

    legacy, legacy_ref = checked_json('tiny_perceptron/lesson_products/release.json')
    displayed = json.loads(re.search(r'```json\n(.*?)\n```', frozen.decode(), re.S)[1])
    assert displayed == legacy
    release, release_ref = checked_json('docs/decision-product/v3/public-release.json')
    acquisition, acquisition_ref = checked_json('docs/decision-product/v3/metadata/acquisition.json')
    assert ARCHIVE.stat().st_size == release['bytes'] == 6168350
    assert digest(ARCHIVE.read_bytes()) == release['sha256']
    members = []
    metadata = []
    with zipfile.ZipFile(ARCHIVE) as source:
        names = source.namelist()
        for name, expected in release['bundle'].items():
            member = 'tiny-perceptron-c-v3/bundle/' + name
            assert names.count(member) == 1
            raw = source.read(member)
            assert digest(raw) == expected
            members.append({'member': member, 'bytes': len(raw), 'sha256': digest(raw)})
        for row in acquisition['files']:
            assert names.count(row['archive_member']) == 1
            raw = source.read(row['archive_member'])
            assert len(raw) == row['bytes'] and digest(raw) == row['sha256']
            assert (ROOT / row['path']).read_bytes() == raw
            metadata.append({'path': row['path'], 'archive_member': row['archive_member'], 'sha256': digest(raw)})
    verify_bundle(ROOT / 'checkpoints/decision-product-v3', release)

    cli = []
    for module in ['scripts.course_product', 'tiny_perceptron.lesson_products']:
        command = [sys.executable, '-m', module, '--help']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        cli.append({'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    assert 'download,judge' in cli[0]['stdout']
    assert 'solve,judge,auto,serve,download' in cli[1]['stdout']
    print(json.dumps({
        'created_at': datetime.now(UTC).isoformat(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'platform': platform.platform()},
        'source_freeze': freeze_ref,
        'source': {'path': page['snapshot'], 'sha256': digest(frozen), 'live_equals_snapshot': True, 'figures': page['figures_sha256']},
        'prior_freeze': prior_ref,
        'prior_source': {'path': before_page['snapshot'], 'sha256': digest(before)},
        'delta': ''.join(difflib.unified_diff(before.decode().splitlines(keepends=True), frozen.decode().splitlines(keepends=True))),
        'only_delta_is_version_split_and_rng_expansion': True,
        'legacy_displayed_release_equals_repository': True,
        'legacy_release': legacy_ref,
        'c_v3_release': release_ref,
        'archive': {'path': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'sha256': release['sha256'], 'bundle_members': members},
        'metadata_acquisition': acquisition_ref,
        'metadata_byte_identity': metadata,
        'existing_c_v3_bundle_verified': 'checkpoints/decision-product-v3',
        'cli_help': cli,
        'scope': 'Read-only local source/archive/CLI verification. No anonymous network transfer, no new install, no model inference, no training, no final evaluation. Metadata byte identity is not an execution receipt.'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
