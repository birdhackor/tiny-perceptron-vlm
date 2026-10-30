"""Remaining original controls, isolated fixture-only current recheck."""
from pathlib import Path
from unittest.mock import patch
import datetime
import hashlib
import json
import sys
import urllib.error
import warnings
import zipfile

ROOT = Path('/workspace/work/omni-models-v1')
E = Path('/workspace/work/omni-public-delivery-review-current-v1')
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import public_download as loader, public_demo as demo


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('expected rejection')


fixtures = E / 'supplemental-fixtures'
fixtures.mkdir(exist_ok=False)
files = {
    'core/codec/training/weights.safetensors': b'opaque original control',
    'core/codec/training/training-result.json': b'{"fixture":true}',
    'core/codec/job-result.json': b'{"fixture":true}',
    'core/codec/training/latest-continuation.pt': b'opaque pt bytes',
}
expected = {
    name: {'sha256': digest(raw), **({'bytes': len(raw)} if name.endswith('weights.safetensors') else {})}
    for name, raw in files.items() if not name.endswith('.pt')
}


def package(label, extra=None, symlink=None):
    path = fixtures / (label + '.zip')
    members = {**files, **(extra or {})}
    manifest = {name: {'sha256': digest(raw), 'bytes': len(raw)} for name, raw in members.items()}
    with zipfile.ZipFile(path, 'w') as archive:
        for name, raw in members.items():
            if name == symlink:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (0o120777 << 16)
                archive.writestr(info, raw)
            else:
                archive.writestr(name, raw)
        archive.writestr('package-file-manifest.json', json.dumps(manifest))
    return path, {'job': 'codec', 'name': 'codec.zip', 'sha256': loader.file_sha(path), 'bytes': path.stat().st_size}


# Replay the original count limit, and verify the valid bound remains accepted.
path, row = package('valid64', {f'core/codec/extra-{i}': b'x' for i in range(59)})
loader.install(path, row, fixtures / 'valid64-output', expected_files=expected)
path, row = package('invalid65', {f'core/codec/extra-{i}': b'x' for i in range(60)})
rejected(lambda: loader.install(path, row, fixtures / 'invalid65-output', expected_files=expected))
assert not (fixtures / 'invalid65-output').exists()

# ZIP link members and pre-existing subdirectory/root links reject before writes.
path, row = package('zip-link', {'core/codec/link': b'../target'}, symlink='core/codec/link')
rejected(lambda: loader.install(path, row, fixtures / 'zip-link-output', expected_files=expected))
assert not (fixtures / 'zip-link-output').exists()
path, row = package('ordinary')
outside = fixtures / 'link-target'
outside.mkdir()
root_link = fixtures / 'root-link'
root_link.symlink_to(outside, target_is_directory=True)
rejected(lambda: loader.install(path, row, root_link, expected_files=expected))
target = fixtures / 'subdirectory-link'
target.mkdir()
(target / 'core').symlink_to(outside, target_is_directory=True)
rejected(lambda: loader.install(path, row, target, expected_files=expected))
assert list(outside.iterdir()) == []

# Exercise expansion refusal without allocating the 256MiB production limit.
with patch.object(loader, 'MAX_EXPANDED', 1):
    rejected(lambda: loader.install(path, row, fixtures / 'expanded-rejected', expected_files=expected))
assert not (fixtures / 'expanded-rejected').exists()

with patch.object(loader.urllib.request, 'urlopen', side_effect=urllib.error.URLError('https://signed.example/?private=query')):
    try:
        loader.fetch({'release_tag': 'fixture-v1'}, {'name': 'codec.zip', 'bytes': 1, 'sha256': '0' * 64}, fixtures / 'fetch-error')
    except RuntimeError as error:
        assert 'signed.example' not in str(error) and 'private' not in str(error)
    else:
        raise AssertionError('network exception not raised')
rejected(lambda: demo.public_question({**demo.choice_example(), 'state': '\ud800'}))

report = {
    'reviewer_task': '/root/omni_final_definition_review',
    'completed_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
    'original_remaining_controls_replayed': {
        'member_count64_valid65_invalid': True,
        'ZIP_symlink_rejected': True,
        'existing_root_and_intermediate_symlink_rejected': True,
        'expanded_limit_mechanism_rejected_using_small_patched_limit': True,
        'redirect_query_not_logged_in_sanitized_exception': True,
        'surrogate_question_rejected': True,
    },
    'real_network_or_NN_or_paid': False,
}
output = E / 'original-controls-evidence-current-v1.json'
with output.open('x') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(output), 'sha256': loader.file_sha(output), 'passed': True}, indent=2))
