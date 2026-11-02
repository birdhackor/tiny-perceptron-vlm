"""Catalog exact historical findings separately from later bounded judgments."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PROOFS = Path(__file__).resolve().parent
SCOPE = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-tools/scope-inventory-v7.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def findings(value, pointer=''):
    result = []
    if isinstance(value, dict):
        identity = value.get('id', value.get('issue_id', value.get('finding_id')))
        if isinstance(identity, str):
            result.append({'pointer': pointer, 'id': identity,
                           'exact_original_object_sha256': sha(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()),
                           'recorded_fields': {key: item for key, item in value.items()
                                               if key in {'status', 'severity', 'necessity', 'title', 'summary', 'resolution', 'disposition'}}})
        for key, item in value.items():
            result.extend(findings(item, pointer + '/' + key))
    elif isinstance(value, list):
        for key, item in enumerate(value):
            result.extend(findings(item, pointer + '/' + str(key)))
    return result


records = []
scope = json.loads(SCOPE.read_bytes())
for binding in scope['checked_file_scope']:
    path = Path(binding['path'])
    if '/infrastructure/' not in str(path) or 'review' not in path.name or 'locations' in path.name:
        continue
    raw = (ROOT / path).read_bytes()
    assert sha(raw) == binding['sha256'] and len(raw) == binding['bytes']
    data = json.loads(raw)
    records.append({**binding, 'historical_role': 'original' if 'original' in path.name else 'bounded_current_or_evidence_index',
                    'recorded_verdict': data.get('verdict', data.get('status')),
                    'findings_as_recorded': findings(data),
                    'interpretation': 'Preserved source judgment. Later evidence is a separate record; this catalog grants no fresh closure or permission.'})

old_report_hashes = {
    'technical-original-v1.json': '8b1aea2a2dc58a420b8d5a0a27dd4d0e43d8c64cb48761a4792a613f6b6e1e4e',
    'technical-current-v1.json': 'ce87d40b7b285f4c5ae400455d783aac9af0ba51a4281dae2964e1fc999bc1d6',
    'technical-current-v2.json': 'f91fb56c3bcac4f59acc819803e65e9700e0c9d36bbd61485025e704fc257fc9',
    'coverage-appendix-v1.json': 'd2e26b33f377c5eb03e3e1db65c8879e005eb6a43709101bd7e0d6984c52e774',
    'coverage-appendix-v2.json': '9adc01cf01cc7278ca30f07c0f537388b14f0e56f4c5222e716c40fabcf3b892',
    'coverage-appendix-v3.json': 'e7cbe35e870b3d8b40cd74233172807fa2804fa11ccaed5eb405cd443f885c8a',
    'technical-bounded-original-v2.json': '1ee345294b61d037236ebb15da416ed15eab6b6dd800fcf0c5f59d252955073c',
}
review_dir = PROOFS.parent
for name, digest in old_report_hashes.items():
    assert sha((review_dir / name).read_bytes()) == digest, name

preview_log = PROOFS / 'scope460-existing-preview-links-v1.log'
preview_errors = preview_log.read_text().split('ValueError: ', 1)[1].strip().splitlines()
assert preview_errors == ['下載與目前教材不同：' + lesson for lesson in ('19.21', '19.22', '19.23', '19.24')]
result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'scope_inventory_sha256': sha(SCOPE.read_bytes()),
    'historical_bounded_review_records': records,
    'original_and_prior_own_report_hashes_unchanged': old_report_hashes,
    'existing_preview_v5_check': {
        'passed': False, 'exit_code': 1, 'log': str(preview_log.relative_to(ROOT)),
        'sha256': sha(preview_log.read_bytes()), 'errors': preview_errors,
        'interpretation': 'Expected current integration rejection after source/Notebook corrections. The old preview was not rebuilt by this reviewer and is not current publication evidence.',
    },
    'necessary_issue_closures_from_catalog': [],
    'limits': ['Each original finding keeps its recorded severity/status/identity and exact object binding.',
               'An old revise/blocked original is never rewritten to pass by discovery of a later current report.',
               'Historical eligibility statements, including the earlier unused Qwen attempt, do not override actual consumed terminal ledger evidence.',
               'This is evidence preservation and provenance review, not fresh model capability approval, paid authority, current369 gate, stage collection or activation.'],
}
out = PROOFS / 'scope460-evidence-preservation-result-v1.json'
with out.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'bounded_review_records': len(records), 'findings_preserved': sum(len(row['findings_as_recorded']) for row in records),
                  'own_immutable_reports_unchanged': len(old_report_hashes), 'preview_pending': True}))
