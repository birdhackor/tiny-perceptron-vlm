"""Independent Windows ZIP-normalization emulation; not native Windows CI."""
from pathlib import Path
from unittest.mock import patch
import datetime
import hashlib
import importlib.util
import inspect
import json
import sys
import zipfile

import pytest

ROOT = Path('/workspace/work/omni-models-v1')
E = Path('/workspace/work/omni-public-delivery-review-current-v2')
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import public_download as delivery

spec = importlib.util.spec_from_file_location('reviewed_public_tests', ROOT / 'tests/test_omni_public_download.py')
authored = importlib.util.module_from_spec(spec)
spec.loader.exec_module(authored)
fixtures = E / 'portability-fixtures-recheck'
fixtures.mkdir(exist_ok=False)
original_info = zipfile.ZipInfo
stdlib_init = inspect.getsource(original_info.__init__)
stdlib_sanitizer = inspect.getsource(zipfile._sanitize_filename) if hasattr(zipfile, '_sanitize_filename') else stdlib_init
assert 'filename.replace(os.sep, "/")' in stdlib_sanitizer


class WindowsNormalizedZipInfo(original_info):
    def __init__(self, filename='NoName', date_time=(1980, 1, 1, 0, 0, 0)):
        # Exact effective transformation in CPython ZipInfo on Windows, where
        # os.sep is backslash. Keep the local filesystem and platform Linux.
        super().__init__(filename.replace('\\', '/'), date_time)


unsafe = ['C:/escaped.txt', 'C:escaped.txt', 'name:stream', 'CON', 'NUL.txt', '../x', 'x/./y', 'x//y', 'tail.', 'folder\\x']
replayed = []
for mode, info_type in [('POSIX_actual', original_info), ('Windows_ZipInfo_normalization_emulated', WindowsNormalizedZipInfo)]:
    for i, member in enumerate(unsafe):
        folder = fixtures / f'{mode}-{i}'
        folder.mkdir()
        with patch.object(zipfile, 'ZipInfo', info_type):
            authored.test_cross_platform_unsafe_members_rejected_before_install(folder, member)
        assert not (folder / 'rejected').exists()
        replayed.append({'mode': mode, 'member': member, 'raw_portable_rejected': not delivery.portable_member('core/codec/' + member), 'no_writes': True})

# Recreate the real initial failure, including old over-narrow error regexp.
folder = fixtures / 'original-regexp-control'
folder.mkdir()
with patch.object(zipfile, 'ZipInfo', WindowsNormalizedZipInfo):
    archive, asset, raw_files = authored.package(folder, {'core/codec/folder\\x': b'outside or ambiguous'})
    with zipfile.ZipFile(archive) as reader:
        names = reader.namelist()
        inner = json.loads(reader.read('package-file-manifest.json'))
    assert 'core/codec/folder/x' in names and 'core/codec/folder\\x' not in names
    assert 'core/codec/folder\\x' in inner and 'core/codec/folder/x' not in inner
    try:
        delivery.install(archive, asset, folder / 'rejected', expected_files=authored.selected_files(raw_files))
    except ValueError as error:
        actual_error = str(error)
    else:
        raise AssertionError('normalized-but-inconsistent archive accepted')
    assert actual_error == '成品內manifest缺漏' and not (folder / 'rejected').exists()
    try:
        with pytest.raises(ValueError, match='路徑'):
            delivery.install(archive, asset, folder / 'rejected', expected_files=authored.selected_files(raw_files))
    except AssertionError:
        old_test_regex_failure_reproduced = True
    else:
        raise AssertionError('old regexp unexpectedly accepted manifest exception')

# A coherent normalized package remains legal; no arbitrary rejection added.
folder = fixtures / 'valid-normalized-package'
folder.mkdir()
with patch.object(zipfile, 'ZipInfo', WindowsNormalizedZipInfo):
    archive, asset, raw_files = authored.package(folder, {'core/codec/folder/x': b'valid ASCII fixture'})
    delivery.install(archive, asset, folder / 'valid', expected_files=authored.selected_files(raw_files))
assert (folder / 'valid/core/codec/folder/x').read_bytes() == b'valid ASCII fixture'

# The new raw assertion catches a deliberately weakened validator even when
# zipfile could normalize the name; installer acceptance still fails the test.
folder = fixtures / 'raw-validator-mutation'
folder.mkdir()
with patch.object(delivery, 'portable_member', lambda _: True):
    try:
        authored.test_cross_platform_unsafe_members_rejected_before_install(folder, 'folder\\x')
    except AssertionError:
        weakened_validator_caught = True
    else:
        raise AssertionError('raw validator mutation not caught')
assert not (folder / 'codec.zip').exists()
folder = fixtures / 'acceptance-mutation'
folder.mkdir()
with patch.object(delivery, 'install', lambda *args, **kwargs: {}):
    try:
        authored.test_cross_platform_unsafe_members_rejected_before_install(folder, 'folder\\x')
    except pytest.fail.Exception:
        weakened_installer_caught = True
    else:
        raise AssertionError('installer acceptance mutation not caught')

log = ROOT / 'outputs/omni-PR9-Windows-initial-failure-v1.log'
assert hashlib.sha256(log.read_bytes()).hexdigest() == '7ecc402645a3db66c1076775d61f88bc1d325b0039bdd10fa3bea1a966e497b2'
lines = log.read_text().splitlines()
assert any('1 failed, 2576 passed, 5 skipped' in line for line in lines)
excerpt = [line for line in lines if 'Actual message:' in line or 'test_cross_platform_unsafe_members_rejected_before_install[folder' in line or '1 failed, 2576 passed, 5 skipped' in line]
report = {
    'reviewer_task': '/root/omni_final_definition_review',
    'completed_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
    'source_test_sha256': hashlib.sha256((ROOT / 'tests/test_omni_public_download.py').read_bytes()).hexdigest(),
    'original_native_CI_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
    'original_native_CI_excerpt': excerpt,
    'CPython_version': sys.version,
    'CPython_ZipInfo_normalization_source_excerpt': [line.strip() for line in stdlib_sanitizer.splitlines() if 'os.sep' in line or 'filename.replace' in line],
    'platform_executed': sys.platform, 'native_Windows_executed': False,
    'replayed_authored_unsafe_cases': replayed,
    'initial_failure_regexp_mismatch_reproduced': old_test_regex_failure_reproduced,
    'Windows_normalized_archive_error': actual_error,
    'valid_consistent_normalized_archive_accepted': True,
    'weakened_raw_validator_mutation_caught': weakened_validator_caught,
    'weakened_installer_acceptance_mutation_caught': weakened_installer_caught,
    'new_exact_test_bytes_native_Windows_CI_status': 'not_verified_pending_post_commit_native_CI',
    'source_implementation_changed': False, 'new_network_NN_or_paid': False,
}
output = E / 'portability-evidence-current-v2.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'replayed_cases': len(replayed), 'native_Windows_pass_claimed': False, 'passed': True}, indent=2))
