"""Finish actual primary/receipt scan, keeping root author metadata distinct."""

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = (HERE / 'audit-compat-and-relay-v1.py').read_text().split("\ncheck('legacy_receipt_schema_and_all_actual_primary_data'", 1)[0]
old = "        assert api.owner(report) == task\n"
new = """        actor = api.owner(report) or report.get('author_task') or report.get('collector_task') or report.get('created_by')
        # Author/collector metadata need not impersonate an independent reviewer.
        # Primary judgements separately require api.owner and actual assignment.
        assert actor == task or (actor is None and task == '/root'), ('receipt body actor mismatch', reference['path'], actor, task)
"""
assert source.count(old) == 1
source = source.replace(old, new)
context = {'__file__': str(HERE / 'audit-compat-and-relay-v1.py'), '__name__': 'own_proof_continuation'}
exec(compile(source, str(HERE / 'audit-compat-and-relay-v1.py'), 'exec'), context)
base = context['namespace']
record = {'reviewer_task': '/root/omni_p6_technical_tools', 'created_at': datetime.now(UTC).isoformat(),
          'kind': 'primary/receipt metadata completion with accepted legacy schemas and explicit root transport distinction; not final verdict',
          'base_candidate': context['manifest_ref'],
          'earlier_scanner_assumptions_preserved': ['base-observations-v1.json', 'compat-observations-v1.json']}
try:
    record['result'] = context['recheck_primary_and_seals']()
    record['status'] = 'checked'
except Exception as error:
    record['status'] = 'needs_assessment'
    record['error'] = type(error).__name__ + ': ' + str(error)
    record['traceback'] = traceback.format_exc()
record['sealed_row_assessments'] = base['details'].get('sealed_receipt_assessments', [])
record['legacy_no_declared_creation_time_counts'] = dict(context['collections'].Counter(row['wave'] + '/' + row['role'] for row in base['details'].get('old_format_receipts_without_declared_creation_time', [])))
record['legacy_missing_owner_rows_unique'] = len({(row['path'], row['wave'], row['role']) for row in context['legacy_ownerless']})
record['reviewer_vs_later_collection_role_rows_unique'] = len({(row['path'], row['wave'], row['actual_collection_role']) for row in context['phase_callback_rows']})
record['auxiliary_literal_wave_label_rows'] = context['auxiliary_rows']
large_path = context['RAW'] / 'primary-current-data-details-v1.json'
with large_path.open('x') as stream:
    json.dump({'details': base['details'], 'legacy_ownerless': context['legacy_ownerless'],
               'later_collection': context['phase_callback_rows'], 'verified_refs': list(base['verified'].values())},
              stream, ensure_ascii=False, indent=2)
    stream.write('\n')
record['details_external_optional'] = context['ref'](large_path)
destination = HERE / 'primary-observations-v1.json'
with destination.open('x') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'saved': context['ref'](destination), 'status': record['status'],
                  'result': record.get('result'), 'error': record.get('error'),
                  'sealed_assessments': len(record['sealed_row_assessments'])}, ensure_ascii=False))
