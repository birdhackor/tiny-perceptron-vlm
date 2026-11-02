from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import statistics

root = Path('/workspace/work/omni-book-integrated-v1')
raw_root = Path('/workspace/work/omni-final-completed-41004229262/verified')
summary_path = root / 'docs/omni-products/v1/results-summary.json'
summary = json.loads(summary_path.read_bytes())
raw = {name: json.loads((raw_root / 'measurements' / name / 'results.json').read_bytes())
       for name in ('av', 'sound', 'choice', 'integration')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in ('rows', 'records')}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


checks = []
expected = compact(raw['av'])
assert summary['AV'] == expected
checks.append('AV: all retained values equal original, per-family records referenced in raw ZIP')

expected = compact(raw['sound'])
timing = raw['sound']['actual_playable_timing']
timing_summary = {'observations': len(timing), 'seconds': {}}
for key in ('first_playable_seconds', 'whole_code_generation_seconds', 'whole_completion_seconds'):
    values = [r[key] for r in timing]
    timing_summary['seconds'][key] = {'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values)}
assert summary['sound']['actual_playable_timing']['observations'] == timing_summary['observations']
assert summary['sound']['actual_playable_timing']['seconds'] == timing_summary['seconds']
expected['actual_playable_timing'] = summary['sound']['actual_playable_timing']
assert summary['sound'] == expected
checks.append('Sound: retained values exact; all nine min/median/max timing values recomputed from ten original rows')

expected = compact(raw['choice'])
grouped = defaultdict(list)
for row in raw['choice']['candidate_count_secondary']['rows']:
    grouped[row['total_choices_including_none']].append(row)
expected['candidate_count_secondary']['accuracy_by_total_choices_including_none'] = {
    str(count): {'correct': sum(r['answer'] == r['gold'] for r in rows),
                 'observations': len(rows), 'parent_families': len({r['family'] for r in rows})}
    for count, rows in grouped.items()
}
assert summary['C'] == expected
checks.append('C: retained values exact; seven candidate-count diagnostics recomputed from original 420 rows')

expected = compact(raw['integration'])
expected['legacy_binding']['old_A_manifest_sha256'] = expected['legacy_binding'].pop('old_A')['manifest_sha256']
expected['old_A_retention']['counterfactual_diagnostics']['example'] = raw['integration']['old_A_retention']['counterfactual_diagnostics']['rows'][0]
groups = defaultdict(list)
for row in raw['integration']['auto']['rows']:
    groups[row['gold_route']].append(row)
expected['auto']['by_gold_route'] = {
    route: {'families': len(rows), **{key: sum(r[key] for r in rows) for key in
             ('route_correct', 'gold_route_end_to_end_correct', 'real_route_end_to_end_correct')}}
    for route, rows in groups.items()
}
assert len(groups['A']) == 20
expected['auto']['A_subgroups'] = {
    'calculator_families': sum(r['family'].startswith('auto-A-calculator-') for r in groups['A']),
    'AV_sound_families': sum(not r['family'].startswith('auto-A-calculator-') for r in groups['A'])
}
assert summary['integration'] == expected
checks.append('Integration: retained values exact; route-group counters, A subgroups and counterfactual example verified against original rows')

assert summary['capability_verdict'] == json.loads((raw_root / 'final-report.json').read_bytes())
selection = json.loads((root / 'docs/omni-products/v1/model-selection.json').read_bytes())
assert summary['selection'] == {k: selection[k] for k in ('selected', 'temperature', 'threshold', 'policy_feasible')}
assert sha(root / 'docs/omni-products/v1/model-selection.json') == summary['final']['selection_sha256']
for name, binding in summary['final']['raw_results'].items():
    assert sha(raw_root / binding['archive_path']) == binding['sha256']
verification = summary['independent_actual_execution_verification']['report']
assert sha(root / verification['path']) == verification['sha256']
checks.append('Full capability verdict, frozen model selection/T/tau, four original raw SHA bindings, and independent execution review SHA are exact')


# Verify current mature-summary references and failure record without loading any weights.
for model in ('kev', 'qwen25'):
    link = summary['mature_research'][model]['summary']
    assert sha(root / link['path']) == link['sha256']
    result_summary = json.loads((root / link['path']).read_bytes())
    if model == 'kev':
        mature_root = Path('/workspace/work/omni-kev-recovery-completed-41011677439/verified')
        for raw_link in result_summary['raw_files'].values():
            assert sha(mature_root / raw_link['archive_path']) == raw_link['sha256']
    else:
        original_receipt = json.loads(Path('/workspace/work/omni-qwen25-original-attempt-41015365689/artifact/research-receipt.json').read_bytes())
        assert result_summary['status'] == 'failed_stopped_waiting_for_task_id_before_fixed_input_upload'
        assert result_summary['actual_execution']['last_phase'] == original_receipt['phase']
        assert result_summary['actual_execution']['error_type'] == original_receipt['error_type']
        assert result_summary['termination']['compute_stopped'] == original_receipt['compute_stopped'] is True
        verification = result_summary['independent_actual_execution_verification']['report']
        assert sha(root / verification['path']) == verification['sha256']
publication = summary['publication']['author_live_verification']
assert sha(root / publication['path']) == publication['sha256']
checks.append('Mature summaries/raw Kev digests, exact Qwen failure/no-result/stop record, independent failure verification and actual public receipt references match without neural inference.')

receipt = {'created_at': datetime.now(UTC).isoformat(), 'kind': 'author no-NN summary fact check; not independent textbook review',
           'summary_sha256': sha(summary_path), 'checks': checks, 'NN_forward_calls': 0,
           'scope': 'summary intentionally omits per-family rows; all retained numerical fields and added derived tables checked',
           'prior_author_probe': 'A whole-raw-dictionary equality assertion first rejected intentional row omission and timing summarization; no source or original evidence changed.',
           'publication_and_mature_fields': 'actual mature outcomes and public receipts recorded separately; full textbook review status belongs to active manifest'}
out = root / 'outputs/omni-P5-structured-summary-author-check-v2.json'
with out.open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'groups_checked': len(checks), 'no_NN_calls': True, 'all_retained_numeric_fields_exact': True}))
