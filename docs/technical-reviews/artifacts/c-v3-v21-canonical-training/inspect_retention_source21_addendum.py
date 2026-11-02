"""分清 prepared-candidate 快照、正式包及 source22 對 source21 的漂移。"""

import hashlib
import json
import platform
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


audit = json.loads((HERE / 'retention-source21-v1.json').read_bytes())
with zipfile.ZipFile(ARCHIVE) as archive:
    prefix = 'tiny-perceptron-c-v3/'
    mapping_raw = archive.read(prefix + 'evidence/review-attachments-index.json')
    mapping = json.loads(mapping_raw)
    files = mapping['files']
    for item in files:
        raw = archive.read(prefix + item['delivered_path'])
        assert sha(raw) == item['sha256'] and len(raw) == item['bytes']
    prepared = {}
    for name in ('reviewed-candidate-bundle', 'reviewed-qualification-summary'):
        member = 'evidence/' + name + '.json'
        raw = archive.read(prefix + member)
        prepared[sha(raw)] = member
    mismatches = [row for row in audit['c3_original_stored_measurements']['review_references'] if not row['identical']]
    assert all(row['expected'] in prepared for row in mismatches)
    result = {
        'created_at': datetime.now(UTC).isoformat(), 'actor': '/root/c_v3_technical_coord/canonical_training',
        'environment': {'python': platform.python_version(), 'device': 'CPU ZIP/hash only'},
        'scope': 'Identity addendum only; no model/research/training/final/network/current creation.',
        'basis': {'path': str((HERE/'retention-source21-v1.json').relative_to(ROOT)), 'sha256': sha((HERE/'retention-source21-v1.json').read_bytes())},
        'attachment_mapping': {'member': 'evidence/review-attachments-index.json', 'sha256': sha(mapping_raw),
                               'original_reference_entries': len(files), 'unique_delivered_payloads': len({row['delivered_path'] for row in files}),
                               'all_delivered_bytes_and_hashes_identical': True,
                               'actors': sorted({row['actor'] for row in files})},
        'recursive_report_reference_scope': 'The earlier audit recursively counted172 path/SHA occurrences and73 unique path/SHA pairs, including repeated nested historic links. These are not the delivery attachment map57 entries/52 payloads.',
        'prepared_candidate_identity': prepared,
        'live_path_mismatch_occurrences': len(mismatches),
        'mismatch_resolution': 'Every mismatching live path refers to a prepared-candidate lifecycle manifest or qualification-summary SHA. The exact earlier reviewed bytes remain in the fixed ZIP as reviewed-candidate-bundle.json or reviewed-qualification-summary.json. They must retain their earlier identity; qualified live manifests are different bytes, not drift silently accepted.',
        'source21_dependency_hold': [
            {'lesson_id': row['lesson_id'], **item}
            for row in audit['source21_numbered_bundle']['rows'] for item in row['dependencies'] if not item['identical_live']
        ],
        'hold_resolution_pending': 'W.7 pins source21 chapter19 full file6fb672..., while the live file is nowd72c66... after the explicitly announced19.25 source22 formatting patch. Do not claim source21 live identity or close a current. Await real source22 R/T mini-wave, exact AST/delta evidence and final26 callback; original and old bundle stay immutable.',
        'status': 'bounded source21 evidence retained; final collector current not written',
    }
    assert len(files) == 57 and len({row['delivered_path'] for row in files}) == 52
out = HERE / 'retention-source21-addendum-v1.json'
with out.open('x', encoding='utf-8') as target:
    json.dump(result, target, ensure_ascii=False, indent=2)
    target.write('\n')
print(json.dumps({'path': str(out.relative_to(ROOT)), 'sha256': sha(out.read_bytes()), 'attachment_map': [57,52],
                  'prepared_lifecycle_mismatch_occurrences_resolved': len(mismatches), 'collector_current_written': False}, ensure_ascii=False))
