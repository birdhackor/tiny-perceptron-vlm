"""Exact ordinary successor/author notation delta and genuine OPEN-C06 receipts."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
base_source = (HERE / 'audit-base-v1.py').read_text().split("\ncheck('current_source_implementation_original_identity'", 1)[0]
ns = {'__file__': str(HERE / 'audit-base-v1.py'), '__name__': 'own_final_delta'}
exec(compile(base_source, str(HERE / 'audit-base-v1.py'), 'exec'), ns)
root, wave, read, ref, raw, api = (ns[key] for key in ('ROOT', 'WAVE', 'doc', 'ref', 'raw', 'api'))
old_ref, old, old_dp = ns['manifest_ref'], ns['manifest'], ns['dp']
new_ref = ref(wave / 'final-assembly-v3/manifest-author-received-v1.json')
assert new_ref['sha256'] == 'cfdd421ca03e71e19daf2368a7a9b0e4d39debe68a19935c8e029ec32425ef98'
new = read(new_ref)
assert new['dispositions']['sha256'] == '04cf41a87c387675f7dae1e3aa0b18b82c9fc2e2d5ddc45b81ec68b188551d61'
new_dp = read(new['dispositions'])
allowed = {'actual_final_author_intake', 'dispositions_collection', 'created_at', 'dispositions',
           'intake_note', 'previous_working_candidate', 'author_source_category_notation_addendum'}
manifest_changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
assert manifest_changed == allowed
assert new['previous_working_candidate'] == {'path': old_ref['path'], 'sha256': old_ref['sha256']}
assert len(new_dp['items']) == len(old_dp['items']) == 554 and new_dp['items'][:552] == old_dp['items'][:552]
changed = []
for index in (552, 553):
    before, after = dict(old_dp['items'][index]), dict(new_dp['items'][index])
    assert before.pop('original_severity_preserved') == 'necessary'
    assert after.pop('original_necessity') == 'necessary'
    assert before == after
    source = api.node(read(after['report']), after['json_pointer'])
    assert source['necessity'] == 'necessary' and 'severity' not in source
    changed.append({'index': index, 'issue_id': after['issue_id'], 'source_necessity': source['necessity'],
                    'source_has_severity': False, 'all_actual_grade_action_verification_fields_unchanged': True})
top_changed = {key for key in set(old_dp) | set(new_dp) if old_dp.get(key) != new_dp.get(key)}
assert top_changed == {'notation_previous_dispositions', 'at', 'kind', 'items', 'author_input_reports'}
assert new_dp['notation_previous_dispositions'] == new['previous_working_candidate'] or new_dp['notation_previous_dispositions'] == old['dispositions']
assert new_dp['notation_previous_dispositions'] == old['dispositions']
assert new_dp['author_input_reports'][:-1] == old_dp['author_input_reports']
assert new_dp['author_input_reports'][-1] == new['author_source_category_notation_addendum']
addendum = read(new['author_source_category_notation_addendum'])
assert addendum['author_task'] == '/root' and addendum['changed_item_indices'] == [552, 553]
assert addendum['items'] == new_dp['items'][552:]
for author in new_dp['author_input_reports']:
    read(author, 'ordinary actual author input provenance path and SHA')

latest_ref = ref(wave / 'continuity-collection-intake-v10.json')
assert latest_ref['sha256'] == 'a6a4c3e0809057ecba9da32ada1c6e9e2064d07d88e8d1a1df26b1bdb6dd9619'
latest = read(latest_ref)
declared = read(new['actual_final_author_intake'])
assert new['actual_final_author_intake']['sha256'] == '92a4801e2423653ff92d172c753622798c97c024b7f435c0817ee81845cada98'
assert latest['collection_index'][:len(ns['intake']['collection_index'])] == ns['intake']['collection_index']
assert latest['collection_index'][:len(declared['collection_index'])] == declared['collection_index']
open_report = read(ns['open_ref'])
assert open_report['status'] == 'open_actual_work_pending' and 'completed_at' not in open_report
actual_rows = []
for index, row in enumerate(latest['collection_index']):
    report = read(row, 'actual latest OPEN-C06 receipt anchor')
    field, made = ns['created'](report)
    assert made is not None and api.at(open_report['started_at']) <= made <= api.at(row['received_at']) <= datetime.now(UTC)
    assert row['wave'] == open_report['wave'] and row['collection_role'] == 'continuity'
    actual_rows.append({'index': index, 'path': row['path'], 'sha256': row['sha256'],
                        'reviewer_task': row['reviewer_task'], 'created_field': field,
                        'created_at': made.isoformat(), 'received_at': row['received_at']})
dp_rows = [row for row in actual_rows if row['path'] == new['dispositions']['path'] and row['sha256'] == new['dispositions']['sha256']]
assert len(dp_rows) == 1 and dp_rows[0]['received_at'] == new['dispositions_collection']['received_at'] == '2026-11-01T06:29:11.928686+00:00'
new_preclose_ref = ref(wave / 'final-assembly-v3/native-preclose-parts-v2.json')
assert new_preclose_ref['sha256'] == '9a968ffbb241f27c9baa585e3ff6ff82bd971636290e0783b875a23ba1d270c0'
new_preclose = read(new_preclose_ref)
assert new_preclose['manifest'] == {'path': new_ref['path'], 'sha256': new_ref['sha256']} and new_preclose['status'] == 'passed'
assert any(row['sha256'] == new_preclose_ref['sha256'] for row in actual_rows)

# This is the existing numbered/source/receipt/disposition metadata function,
# not audit_manifest, baseline replay, a test suite, builder or whole-book gate.
finish = []


def observe(frame, event, argument):
    if event == 'return' and frame.f_code.co_name == 'finish' and frame.f_code.co_filename.endswith('/historical_fix_rechecks.py'):
        value = frame.f_locals['self']
        finish.append({'entries': len(value.entries), 'used': len(value.used), 'same': set(value.entries) == value.used})


sys.setprofile(observe)
try:
    numbered = ns['setup'].check_numbered(new, intake=False, manifest_path=root / new_ref['path'])
finally:
    sys.setprofile(None)
assert numbered['status'] == 'passed_metadata_checks' and numbered['sections'] == 26
assert finish == [{'entries': 61, 'used': 61, 'same': True}]
api.check_tool_audit(root, new, ns['roles'], ns['sealed'])
assert all(not (verify['report']['path'].startswith('outputs/') or verify['report']['sha256'] == 'eabb3d0035397a8b4cac4eeaf32d2f18cb01657775b17ad08ef6d89567f2c108')
           for item in new_dp['items'] for verify in item.get('verification', []) if verify.get('kind') == 'tool_audit')

base_observed = read(ref(HERE / 'base-observations-v1.json'))
raw_details = json.loads(Path(base_observed['details_external_optional']['path']).read_text())
formal = [row for row in raw_details['verified_files'] if any('current formal technical artifact' in use for use in row['uses'])]
assert all(row['path'].startswith('docs/technical-reviews/artifacts/') for row in formal)
assert not any(Path(row['path']).is_absolute() or row['path'].startswith(('outputs/', 'data/', 'checkpoints/')) for row in raw_details['verified_files'])
for path, expected in new['implementation'].items():
    assert ns['digest']((root / path).read_bytes()) == expected
assert ref(root / new_ref['path'])['sha256'] == new_ref['sha256']
assert ref(root / new['dispositions']['path'])['sha256'] == new['dispositions']['sha256']
record = {
    'schema_version': 1, 'reviewer_task': '/root/omni_p6_technical_tools', 'created_at': datetime.now(UTC).isoformat(),
    'kind': 'own exact final ordinary successor delta and latest actual receipt check, not final report or future closure',
    'audited_candidate_manifest': {'path': new_ref['path'], 'sha256': new_ref['sha256']},
    'base_candidate': old_ref, 'manifest_changed_keys': sorted(manifest_changed), 'author_DP': new['dispositions'],
    'author_DP_top_metadata_changed_keys': sorted(top_changed), 'unchanged_first_author_rows': 552,
    'two_notation_deltas': changed, 'author_notation_addendum': new['author_source_category_notation_addendum'],
    'notation_discovery_attribution': 'Root self-detected; independently confirmed actual source necessity vs absent severity. Not a new independent discovery by this auditor.',
    'actual_latest_C06_intake': latest_ref, 'actual_DP_receipt': dp_rows[0], 'actual_C06_receipts': actual_rows,
    'native_numbered_metadata_on_exact_v3': numbered, 'native_historical_finish_on_exact_v3': finish,
    'native_current_tool_primary_on_exact_v3': 'passed unchanged cc3602 full460 primary',
    'native_kind_tool_verification_uses_old_outputs_supplement': False,
    'current_formal_technical_artifact_anchors_available': len(formal),
    'base_verified_relative_anchors': len(raw_details['verified_files']),
    'source_implementation_note': 'All relevant candidate fields/proofs/source freeze/661 actors/61 sidecar/4004 issue report entries/26 numbered/aux/dependency/primary byte-identical to independently checked base;460 source bytes rehashed again at end.',
    'C06_state': 'Actually OPEN, no completed_at or continuity seal in candidate. No own receipt or future suffix manifest audited.',
    'evidence_boundary': 'Ordinary relative metadata/source/artifact anchors are locally available and hash-bound. External original papers/UI/raw logs are acquisition receipts with hashes and method; no claim that an ordinary clone includes full external bytes or physical access has been reverified universally.',
}
destination = HERE / 'final-delta-observations-v1.json'
with destination.open('x') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'saved': ref(destination), 'candidate': new_ref['sha256'], 'numbered': numbered,
                  'native_finish': finish, 'C06_receipts': len(actual_rows), 'final_report_saved': False}), flush=True)
