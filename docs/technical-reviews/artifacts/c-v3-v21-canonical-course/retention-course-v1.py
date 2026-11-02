"""Read-only immutable retention and final26 identity check after original intake."""

import hashlib
import importlib.util
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
import difflib
import zipfile

ROOT = Path('/tmp/tpb').resolve()
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-course/retention-course-v1.json'
COORD = ROOT / 'docs/course-revision-20261101-c-v3/technical-coordination-v21'
PREFIX = 'tiny-perceptron-c-v3/'
ZIP = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_ref(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'sha256': sha(path.read_bytes())}


def load_ref(ref):
    raw = (ROOT / ref['path']).read_bytes()
    assert sha(raw) == ref['sha256'], ref['path']
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


started = datetime.now(UTC).isoformat()
record = {'started_at': started, 'actor': '/root/c_v3_technical_coord/canonical_course', 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU metadata/hash/saved-logit calculations only; no neural re-execution'}, 'scope': 'After genuine own original intake only. Retain unchanged mature methods and existing scientific/product measurements with original actor/date; independently check actual dependencies, source deltas, v21 output identity, final26 contracts and saved C3 evidence. No new final, NN17, training, kernel, installation, site build or network.'}
original_receipt_ref = {'path': 'docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-course-original-v1.json', 'sha256': 'c49076ad9b1301f7656bff3393ad76c0207cc241d927233943683ef832bd13a2'}
receipt = load_ref(original_receipt_ref)
own_original = load_ref(receipt['report'])
assert receipt['reviewer_task'] == record['actor'] and receipt['received_at'] > own_original['created_at']
assert started > receipt['received_at']
record['original_intake'] = {'receipt': original_receipt_ref, 'report': receipt['report'], 'created_at': own_original['created_at'], 'actually_received_at': receipt['received_at']}

mapping = json.loads((COORD / 'prior-evidence-reference-map-resume-v1.json').read_text())
prior = next(row for row in mapping['pages'] if row['page_id'] == 'course')
proof = prior['prior_technical_proof']
old = load_ref(proof['report'])
old_original = load_ref(proof['original_report'])
old_freeze = load_ref(proof['original_source_freeze'])
assert all(row == next(x for x in old['claims'] if x['id'] == row['id']) for row in old_original['claims'])
dependency_rows = []
for item in old['sources'] + old['artifacts']:
    if item.get('path') and item.get('sha256'):
        path = ROOT / item['path']
        observed = sha(path.read_bytes())
        dependency_rows.append({'id': item['id'], 'path': item['path'], 'expected': item['sha256'], 'observed': observed, 'identical': observed == item['sha256']})
assert all(row['identical'] for row in dependency_rows)
inspection = json.loads((ROOT / 'docs/technical-reviews/artifacts/omni-p6-wave04-course/inspection.json').read_text())
external = []
for item in inspection['external_originals']:
    path = Path(item['local_original'])
    observed = sha(path.read_bytes())
    assert observed == item['sha256'] and path.stat().st_size == item['bytes']
    external.append({'source_id': item['id'], 'url': item['url'], 'version': item['version'], 'original_accessed_on': item['accessed_on'], 'sha256': observed, 'bytes': path.stat().st_size, 'same_original_bytes': True, 'new_research_read': False})
old_result = json.loads((ROOT / 'docs/technical-reviews/artifacts/omni-p6-wave04-course/probe-result.json').read_text())
drifts = []
for path, item in old_result['file_receipts'].items():
    p = Path(path)
    observed = sha(p.read_bytes()) if p.is_file() else None
    if observed != item['sha256']:
        drifts.append({'path': path, 'recorded_sha256': item['sha256'], 'current_sha256': observed, 'disposition': 'Current source/navigation/Notebook changed; excluded from unchanged raw retention. Current links checked by own a1, relevant raw deltas and v21 output identity below.'})
assert len(drifts) == 13
page18 = next(row for row in old_freeze['pages'] if row['page_id'] == 'course')
old_body = (ROOT / page18['snapshot']).read_bytes()
assert sha(old_body) == old['source_sha256'] == page18['source_sha256']
new_body = (ROOT / 'course/README.md').read_bytes()
delta = '\n'.join(difflib.unified_diff(old_body.decode().splitlines(), new_body.decode().splitlines(), fromfile='source18/course', tofile='source21/course', n=2))
reading = module('course_retention_reading', 'scripts/reading_time.py')
section_deltas = []
for source in ['course/chapters/09.md', 'course/chapters/19.md', 'course/training.md']:
    live_slices = reading.lesson_slices((ROOT / source).read_text())
    for page in old_freeze['pages']:
        if page['source'] != source or page['page_id'] not in live_slices:
            continue
        live = live_slices[page['page_id']]
        if sha(live.encode()) != page['source_sha256']:
            frozen = (ROOT / page['snapshot']).read_text()
            assert sha(frozen.encode()) == page['source_sha256']
            section_deltas.append({'lesson_id': page['page_id'], 'source': source, 'prior_snapshot': file_ref(ROOT / page['snapshot']), 'current_section_sha256': sha(live.encode()), 'delta': '\n'.join(difflib.unified_diff(frozen.splitlines(), live.splitlines(), n=1))})
record['prior_whole_page_retention'] = {'original_owner': old['reviewer_task'], 'original_created_at': old_original['created_at'], 'prior_current_created_at': old['created_at'], 'original_ref': proof['original_report'], 'current_ref': proof['report'], 'original_received_at': proof['original_collection']['received_at'], 'current_received_at': proof['current_collection']['received_at'], 'original36_claims_identical_in_current': True, 'source18_snapshot': file_ref(ROOT / page18['snapshot']), 'current_source': file_ref(ROOT / 'course/README.md'), 'unchanged_dependency_identities': dependency_rows, 'external_original_identities': external, 'raw_receipts_checked': len(old_result['file_receipts']), 'raw_receipts_identical': len(old_result['file_receipts']) - len(drifts), 'changed_receipts_not_retained': drifts, 'whole_page_raw_delta': delta, 'relevant_section_raw_deltas': section_deltas, 'retained_raw_results': {k: old_result[k] for k in ['transparent_mechanisms', 'auto_stub_mechanism', 'legacy', 'formal_jobs', 'raw_recomputation', 'download_entrypoint_checks']}, 'limits': inspection['limitations'], 'retention_scope': 'Only source-identical mature definitions/mechanisms, original ABC lineage/finite-task/Auto, ten old training completion facts, fixed old AV/sound/C-v2/B observations and unchanged implementations. Current C3 recommendation/default and new navigation are independently checked; old historical default language is not retained as a current C3 claim.'}

# Static source/output identity only; never execute any Notebook kernel.
kernel_root = Path('/workspace/work/omni-book-integrated-v1/outputs/omni-c-v3-kernel-bundle-v21')
kernel_receipt = json.loads((ROOT / 'docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json').read_text())
kernel_rows = []
for source in sorted((ROOT / 'notebooks').rglob('*.ipynb')):
    relative = source.relative_to(ROOT / 'notebooks')
    executed = kernel_root / relative
    assert executed.is_file(), relative
    plain, output = json.loads(source.read_text()), json.loads(executed.read_text())
    cells = lambda notebook: [(cell['cell_type'], cell['source']) for cell in notebook['cells']]
    assert cells(plain) == cells(output), relative
    errors = [x for cell in output['cells'] for x in cell.get('outputs', []) if x.get('output_type') == 'error']
    assert not errors, relative
    kernel_rows.append({'lesson_id': source.stem, 'current_source_sha256': sha(source.read_bytes()), 'executed_sha256': sha(executed.read_bytes()), 'cell_source_exact': True})
assert len(kernel_rows) == kernel_receipt['bundle_count'] == 332
for row in kernel_receipt['actual_executions']:
    matched = next(x for x in kernel_rows if x['lesson_id'] == row['lesson'])
    assert matched['current_source_sha256'] == row['source_sha256'] and matched['executed_sha256'] == row['executed_sha256']
record['kernel_v21_identity'] = {'root': str(kernel_root), 'tracked_receipt': file_ref(ROOT / 'docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json'), 'count': len(kernel_rows), 'retained_count': 328, 'refresh_count': 4, 'all_cell_type_and_source_exact': True, 'error_outputs': 0, 'new_kernel_executions_by_this_actor': 0, 'bounded_refreshed_rows': [x for x in kernel_rows if x['lesson_id'] in {y['lesson'] for y in kernel_receipt['actual_executions']}], 'first_lesson': next(x for x in kernel_rows if x['lesson_id'] == '1.1'), 'limit': 'Source/executed-bundle identity and existing CPU outputs, not fresh332 executions or hosted-site/Colab runtime validation; pre-existing preview evidence stays bounded to its own date.'}

bundle_ref = {'path': 'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-final-v1.json', 'sha256': '6b0849e632c43964359de7deb0f2abf30cb6224cea42de0003f6f8efbfd69f04'}
bundle = load_ref(bundle_ref)
section_freeze = load_ref(bundle['section_freeze'])
load_ref(bundle['canonical_source_freeze'])
index = load_ref(bundle['issue_inventory'])
retention = load_ref(bundle['retention_metadata'])
assert bundle['issue_inventory']['sha256'] == '53f6ee399f447b7e8f75a604a6e16e5f511650137a418dd1927d309e311e7625'
setup = module('course_retention_setup', 'scripts/omni_revision/review_setup.py')
setup.ROOT = ROOT
validator = module('course_retention_validator', 'scripts/check_technical_reviews.py')
api = module('course_retention_api', 'docs/review-tools/incremental_revision.py')
reports = {row['lesson_id']: load_ref(row['report']) for row in bundle['sections']}
owners = defaultdict(list)
for lesson_id, report in reports.items():
    owners[report['reviewer_task']].append(lesson_id)
assert len(owners) == 26
numbered_rows = []
fresh = {'R.2', 'R.3', 'R.4', 'T.11'}
for entry in bundle['sections']:
    lesson_id = entry['lesson_id']
    report, original = reports[lesson_id], load_ref(entry['original_report'])
    original_freeze = load_ref(entry['original_section_freeze'])
    original_page = next(row for row in original_freeze['pages'] if row['page_id'] == lesson_id)
    parent_id = original_page.get('canonical_page_id', lesson_id)
    setup.check_original(original, entry['original_section_freeze'], lesson_id, parent_id, original['reviewer_task'], validator, api, entry.get('original_source_snapshots'))
    current_page = next(row for row in section_freeze['pages'] if row['page_id'] == lesson_id)
    body = setup.frozen_body(current_page)
    errors = validator._validate(ROOT, ROOT / current_page['source'], lesson_id, body, report, owners, set())
    assert not errors, (lesson_id, errors)
    assert report['reviewer_task'] == original['reviewer_task']
    for i, issue in enumerate(original['issues']):
        resolution = next((x for x in report.get('original_issue_resolutions', []) if x.get('json_pointer') == '/issues/' + str(i)), None)
        assert resolution and resolution.get('resolution') and resolution.get('status') == 'resolved', (lesson_id, i)
    dependencies = []
    for item in report['sources'] + report['artifacts']:
        if item.get('path') and item.get('sha256'):
            observed = sha((ROOT / item['path']).read_bytes())
            assert observed == item['sha256'], (lesson_id, item['path'])
            dependencies.append({'path': item['path'], 'sha256': observed})
    group = 'fresh_source21' if lesson_id in fresh else 'same_owner_bounded_callback' if lesson_id == 'W.7' else 'unchanged_retained'
    if group == 'unchanged_retained':
        retained = next(row for row in retention['sections'] if row['lesson_id'] == lesson_id)
        assert retained['entry'] == entry and retained['status'] == 'retained_unchanged_evidence'
    matching_receipts = []
    if group != 'unchanged_retained':
        for path in (COORD / 'receipts').glob('*.json'):
            item = json.loads(path.read_text())
            if item.get('lesson_id') == lesson_id and item.get('report') in [entry['report'], entry['original_report']]:
                matching_receipts.append({'receipt': file_ref(path), 'intake_kind': item.get('intake_kind'), 'report': item['report'], 'received_at': item['received_at']})
        assert any(x['report'] == entry['report'] and x['received_at'] == entry['current_collection']['received_at'] for x in matching_receipts), lesson_id
        assert datetime.fromisoformat(entry['current_collection']['received_at']) >= datetime.fromisoformat(report['created_at'])
        if group == 'fresh_source21':
            assert any(x['report'] == entry['original_report'] and x['received_at'] == entry['original_collection']['received_at'] for x in matching_receipts)
            assert entry['current_dispatch_window']['not_before'] >= entry['original_collection']['received_at'][:19] + '+00:00'
    numbered_rows.append({'lesson_id': lesson_id, 'scope': group, 'reviewer_task': report['reviewer_task'], 'original_ref': entry['original_report'], 'current_ref': entry['report'], 'original_created_at': original['created_at'], 'current_created_at': report['created_at'], 'original_collection': entry['original_collection'], 'current_collection': entry['current_collection'], 'original_check_original': 'passed', 'current_native_errors': errors, 'all_original_issue_resolutions_present': True, 'current_dependencies': dependencies, 'matching_actual_receipts': matching_receipts})
assert Counter(row['scope'] for row in numbered_rows) == {'unchanged_retained': 21, 'same_owner_bounded_callback': 1, 'fresh_source21': 4}
for row in index['reports']:
    load_ref(row)
for row in index['issues']:
    report = load_ref(row['report'])
    i = int(row['json_pointer'].rsplit('/', 1)[1])
    assert i < len(report['issues'])
record['final26'] = {'bundle': bundle_ref, 'section_freeze': bundle['section_freeze'], 'canonical_source_freeze': bundle['canonical_source_freeze'], 'issue_inventory': bundle['issue_inventory'], 'actual_reports_count': 26, 'unique_reviewers': 26, 'groups': dict(Counter(row['scope'] for row in numbered_rows)), 'rows': numbered_rows, 'index_report_refs_checked': len(index['reports']), 'index_issue_pointers_checked': len(index['issues']), 'not_global_T_seal': True, 'no_collector_report_hash_cycle': all(row['report']['path'] != 'docs/course-revision-20261101-c-v3/technical-v21/canonical_course/current-v1.json' for row in bundle['sections'])}

shared_ref = {'path': 'docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/execution.json', 'sha256': '375deeebc7d2107171fd38ed80ffbe127823b7865640ca3118a8fa479c6b1959'}
shared = load_ref(shared_ref)
assert ZIP.stat().st_size == 6168350 and sha(ZIP.read_bytes()) == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
with zipfile.ZipFile(ZIP) as archive:
    member_refs = []
    def member(name):
        raw = archive.read(PREFIX + name)
        member_refs.append({'member': name, 'bytes': len(raw), 'sha256': sha(raw)})
        return json.loads(raw)
    for name, item in shared['archive']['members_read'].items():
        raw = archive.read(PREFIX + name)
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    observations = member('evidence/final-observations.json')
    families = member('evidence/final-families.json')
    opening = member('evidence/final-opening.json')
    selection = member('evidence/selection.json')
    final = member('evidence/final-results.json')
    manual = member('evidence/product-checks.json')
    legacy = member('evidence/legacy-regression.json')
    registry = member('evidence/registry-before-final.json')
    policy = member('bundle/policy.json')
    assert sha(archive.read(PREFIX + 'evidence/selection.json')) == opening['selection_sha256']
    assert sha(archive.read(PREFIX + 'evidence/product-checks.json')) == opening['product_checks_sha256']
    assert sha(archive.read(PREFIX + 'evidence/legacy-regression.json')) == opening['legacy_regression_sha256']
    assert sha(archive.read(PREFIX + 'evidence/registry-before-final.json')) == opening['previous_registry_sha256']
    assert selection['created_at'] < opening['created_at'] < final['created_at']
    assert selection['temperatures'] == policy['temperatures'] and selection['selected']['threshold'] == policy['threshold']
    for name, digest in opening['implementation'].items():
        assert sha((ROOT / 'tiny_perceptron/decision_product' / name).read_bytes()) == digest == sha(archive.read(PREFIX + 'tiny_perceptron/decision_product/' + name))
    keys = {row['key'] for row in families}
    excluded = {key for row in registry['entries'] for key in row['family_keys']}
    assert len(keys) == 1500 and not keys & excluded
    grouped = defaultdict(Counter)
    metrics = defaultdict(lambda: [0, 0])
    accepted = errors = 0
    for row in observations:
        grouped[row['family']][row['kind']] += 1
        # Saved JSON uses null for masked -inf logits, notably info's none.
        logits = [-math.inf if value is None else value for value in row['logits']]
        best = max(range(len(logits)), key=logits.__getitem__)
        metrics[row['kind']][0] += int(best == row['label'])
        metrics[row['kind']][1] += 1
        if row['kind'] == 'choice':
            scaled = [x / policy['temperatures']['choice'] for x in logits]
            maximum = max(scaled)
            confidence = 1 / sum(math.exp(x - maximum) for x in scaled)
            take = row['ids'][best] != 'none' and row['eligible'][best] and confidence >= policy['threshold']
            accepted += int(take)
            errors += int(take and best != row['label'])
    assert len(observations) == 4500 and set(grouped) == keys
    assert all(counts == {'choice': 1, 'info': 1, 'rubric': 1} for counts in grouped.values())
    assert all(value == [1500, 1500] for value in metrics.values()) and accepted == 1025 and errors == 0
    assert shared['retained_measurement_recalculation']['choice_accepted'] == accepted
    local_states = {(row['task'], row['permission'], row['ready']) for row in manual['observations'] if row['kind'] != 'removed'}
    manual_counts = Counter(row['kind'] for row in manual['observations'])
    legacy_counts = Counter(row['kind'] for row in legacy['observations'])
    assert len(local_states) == 27 and manual_counts == {'choice': 972, 'info': 972, 'rubric': 972, 'removed': 432}
    assert legacy_counts == {'choice': 60, 'info': 60, 'rubric': 60}
    assert len({row['original_family'] for row in legacy['observations']}) == 60
    review_index = member('evidence/reviews/index.json')
    attachments = member('evidence/review-attachments-index.json')
    review_rows = []
    for row in review_index['reviews']:
        raw = archive.read(PREFIX + 'evidence/reviews/' + row['path'])
        assert sha(raw) == row['sha256']
        report = json.loads(raw)
        assert report['reviewer_task'] == row['actor'] and report['unresolved_necessary'] == 0
        assert report['implementation'] == review_index['implementation'] == opening['implementation']
        review_rows.append({'kind': row['kind'], 'actor': row['actor'], 'created_at': report['created_at'], 'member': 'evidence/reviews/' + row['path'], 'sha256': row['sha256'], 'original_status_at_its_date': report['status'], 'preserved_scope': report.get('scope', report.get('actual_read_scope')), 'remaining_optional_issues': report.get('issues', []), 'not_new_scientific_review': True})
    for row in attachments['files']:
        raw = archive.read(PREFIX + row['delivered_path'])
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    assert len(attachments['files']) == 57 and len({row['delivered_path'] for row in attachments['files']}) == 52
    record['C3_immutable_qualification'] = {'archive': {'path': str(ZIP), 'bytes': ZIP.stat().st_size, 'sha256': sha(ZIP.read_bytes())}, 'shared_original_execution': shared_ref, 'shared_actor': shared['reviewer_task'], 'shared_started_at': shared['started_at'], 'shared_completed_at': shared['completed_at'], 'members_directly_checked': member_refs, 'all_shared_raw_member_identities_match': True, 'opening_17_implementation_identities_match': True, 'selection_before_opening_before_results': True, 'final_families': 1500, 'final_kind_observations': dict(metrics), 'total_correlated_observations': len(observations), 'independent_units_not_pooled': True, 'device_counts': sorted({len(row['facts']) for row in families}), 'overlap_with_previous_exposed': 0, 'accepted': accepted, 'accepted_errors': errors, 'coverage_all_1500': accepted / 1500, 'all27_manual_local_states_present': True, 'manual_observation_counts': dict(manual_counts), 'legacy_already_exposed_worlds': 60, 'legacy_observation_counts': dict(legacy_counts), 'retained_independent_source_and_label_check': shared['retained_measurement_recalculation'], 'three_reviews': review_rows, 'attachments_references_verified': len(attachments['files']), 'unique_attachment_paths_verified': len({row['delivered_path'] for row in attachments['files']}), 'limits': shared['limits'], 'retention_scope': 'Preserve old qualified predictor and three original actors/date/scope; directly hash exact raw4500 saved rows/families/opening/selection/manual27states/60exposed and recalculate saved logits/counts only. No original NN17/4500-logit replication, training or final reopening; current same weights/config/policy identity and three genuine CPU smoke examples are independently in own probe-course-v1.json.'}

record['completed_at'] = datetime.now(UTC).isoformat()
with OUT.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'artifact': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT.read_bytes()), 'old_dependency_count': len(dependency_rows), 'old_raw786_identical': 786 - len(drifts), 'old_changed_receipts': len(drifts), 'section_deltas': [x['lesson_id'] for x in section_deltas], 'kernel_count': len(kernel_rows), 'final26_groups': record['final26']['groups'], 'C3_accepted': accepted, 'three_review_actors_dates': [(row['actor'], row['created_at']) for row in review_rows], 'attachment_references': 57, 'unique_paths': 52}, ensure_ascii=False, indent=2))
