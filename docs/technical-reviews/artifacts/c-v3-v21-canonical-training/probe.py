"""本頁 source-first 小型核對；不訓練、不下載、不讀 final 題目。"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.course_product import dispatch_request
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents

ROOT = Path('/tmp/tpb').resolve()
HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
BUNDLE = ARCHIVE.parent / 'tiny-perceptron-c-v3/bundle'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(arguments):
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, text=True, capture_output=True, check=False)
    return {'command': [sys.executable, *arguments], 'returncode': completed.returncode,
            'stdout': completed.stdout, 'stderr': completed.stderr}


result = {
    'created_at': datetime.now(UTC).isoformat(),
    'actor': '/root/c_v3_technical_coord/canonical_training',
    'scope': 'Deterministic page examples and three public example JSON requests only; no training, final replay, network download or weight extraction/copy.',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu',
                    'executable': sys.executable, 'root': str(ROOT), 'threads_for_course_product': '2'},
}
descriptor = json.loads((ROOT / 'docs/decision-product/v3/public-release.json').read_bytes())
result['archive'] = {'path': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'sha256': sha(ARCHIVE.read_bytes())}
assert result['archive']['bytes'] == descriptor['bytes'] == 6168350
assert result['archive']['sha256'] == descriptor['sha256'] == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
result['bundle_identity'] = []
result['metadata_identity'] = []
with zipfile.ZipFile(ARCHIVE) as archive:
    for name, digest in descriptor['bundle'].items():
        member = 'tiny-perceptron-c-v3/bundle/' + name
        assert archive.namelist().count(member) == 1
        assert sha(archive.read(member)) == digest == sha((BUNDLE / name).read_bytes())
        result['bundle_identity'].append({'member': member, 'bytes': archive.getinfo(member).file_size,
                                         'sha256': digest, 'existing_read_only_file_matches': True})
    acquisition = json.loads((ROOT / 'docs/decision-product/v3/metadata/acquisition.json').read_bytes())
    for item in acquisition['files']:
        raw = archive.read(item['archive_member'])
        assert raw == (ROOT / item['path']).read_bytes()
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
        result['metadata_identity'].append({'path': item['path'], 'archive_member': item['archive_member'],
                                           'sha256': sha(raw), 'bytes': len(raw)})
    result['zip_has_optimizer_or_rng_named_member'] = [name for name in archive.namelist()
                                                     if any(word in name.lower() for word in ('optimizer', 'rng', 'latest.pt'))]
result['implementation_identity'] = [
    {'path': path, 'sha256': sha((ROOT / path).read_bytes()), 'descriptor_match': sha((ROOT / path).read_bytes()) == digest}
    for path, digest in descriptor['implementation'].items()
]
assert all(row['descriptor_match'] for row in result['implementation_identity'])
result['policy'] = json.loads((BUNDLE / 'policy.json').read_bytes())
result['public_transport_verified'] = False
result['public_transport_note'] = 'No network request was attempted; the public asset is pending release by root.'

result['cpu_requests'] = []
for name in ('record-ready', 'record-unknown', 'answer-absent'):
    entry = run(['-m', 'scripts.course_product', 'judge', '--bundle', str(BUNDLE),
                 '--request', str(ROOT / 'examples/decision_product' / (name + '.json'))])
    assert entry['returncode'] == 0, entry
    entry['request_name'] = name
    entry['parsed_output'] = json.loads(entry['stdout'])
    result['cpu_requests'].append(entry)
ready, unknown, absent = (item['parsed_output']['results'] for item in result['cpu_requests'])
assert ready['choice']['status'] == 'accepted' and ready['choice']['selected'] == 'record'
assert ready['info']['selected'] == 'yes' and ready['rubric']['selected'] == '2'
assert unknown['choice']['status'] == 'abstained' and unknown['info']['selected'] == 'no'
assert unknown['rubric']['status'] == 'abstained'
assert absent['choice']['status'] == 'abstained'
try:
    dispatch_request(None, {'text': '2+3=?'})
except ValueError as error:
    result['legacy_text_rejection'] = str(error)
else:
    raise AssertionError('typed entry accepted legacy text')
result['entry_tests'] = run(['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_course_product_entry.py'])
assert result['entry_tests']['returncode'] == 0, result['entry_tests']

tok = ByteTokenizer()
result['transparent_examples'] = {
    'byte_A': tok.encode('A'), 'EOS': tok.eos_id, 'common_Chinese_UTF8_bytes': len('中'.encode('utf-8')),
    'simple_document_splits': {key: len(value) for key, value in split_documents(toy_documents(), 42).items()},
    'quantization_hand_example': {'old_bytes': 80000, 'new_bytes': 60000, 'byte_change': 60000-80000,
                                  'old_correct': 4, 'new_correct': 3, 'examples': 5,
                                  'meets_smaller': 60000 < 80000, 'meets_no_fewer_correct': 3 >= 4},
}
for kind in ('toy-text', 'attributes-sft'):
    records = generate_records(kind)
    families = sorted({row['family'] for row in records})
    random.Random(42).shuffle(families)
    a, b = int(.8 * len(families)), int(.9 * len(families))
    assignment = {family: 'train' if i < a else 'validation' if i < b else 'test' for i, family in enumerate(families)}
    result['transparent_examples'][kind] = {
        'records': len(records), 'families': len(families),
        'split_records': {split: sum(assignment[row['family']] == split for row in records)
                          for split in ('train', 'validation', 'test')},
        'split_families': {split: sum(value == split for value in assignment.values())
                           for split in ('train', 'validation', 'test')},
        'blue_square_low_partition': assignment.get('blue:square:low'),
    }
    if kind == 'attributes-sft':
        sample = next(row for row in records if row['family'] == 'red:circle:low' and row['messages'][0]['content'].endswith('shape?'))
        inputs, labels = render_chat(sample['messages'])
        result['transparent_examples']['attribute_supervision'] = {
            'answer': sample['messages'][-1]['content'], 'effective_target_ids': labels[labels != -100].tolist(),
            'expected_target_ids': tok.encode('circle') + [tok.eos_id],
        }
        assert labels[labels != -100].tolist() == tok.encode('circle') + [tok.eos_id]
result['completed_at'] = datetime.now(UTC).isoformat()
with (HERE / 'probe-result.json').open('x', encoding='utf-8') as output:
    json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
    output.write('\n')
print(json.dumps({'status': 'completed', 'result': str(HERE / 'probe-result.json'),
                  'tests': result['entry_tests']['stdout'].strip(), 'cpu_requests': len(result['cpu_requests'])}, ensure_ascii=False))
