"""Continue own data audit with explicit legacy schema/callback distinctions.

The first audit's raw observations remain immutable. Its receipt scanner assumed
all old rows repeated owner/phase fields, and did not distinguish reviewer role
from actual collection_role. This corrects that auditor assumption, not sources.
"""

from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name('audit-base-v1.py')
prefix = SCRIPT.read_text().split("\ncheck('current_source_implementation_original_identity'", 1)[0]
namespace = {'__file__': str(SCRIPT), '__name__': 'own_base_definition_reuse'}
exec(compile(prefix, str(SCRIPT), 'exec'), namespace)
globals().update({key: value for key, value in namespace.items() if key not in {'__file__', '__name__'}})

legacy_ownerless, phase_callback_rows, auxiliary_rows = [], [], []


def receipt_compatible(reference, wave, role, task=None, creation_required=False):
    if wave == opened['wave'] and role == 'continuity':
        phase, rows, end = opened, intake['collection_index'], START
    else:
        phase = sealed[wave][role]
        rows, end = phase.get('collection_index', []), api.at(phase['completed_at'])
        assert phase['status'] == 'completed' and end <= START
        assert reference['sha256'] in phase['report_sha256s']
    matches = [row for row in rows if identity(row) == identity(reference)]
    assert len(matches) == 1, ('not unique receipt', reference['path'])
    row = matches[0]
    when = api.at(row['received_at'])
    assert api.at(phase['started_at']) <= when <= end <= START
    report = doc(reference)
    if task is not None:
        assert api.owner(report) == task
        if 'reviewer_task' in row:
            assert row['reviewer_task'] == task
        else:
            legacy_ownerless.append({'path': reference['path'], 'wave': wave, 'role': role,
                                     'body_owner': task, 'unique_sha_bound_receipt': row})
    actual_collection_role = row.get('collection_role', row.get('role', role))
    assert actual_collection_role == role
    if 'role' in row and row['role'] != role:
        assert row['role'] in api.ROLES and api.ROLES.index(row['role']) <= api.ROLES.index(role)
        phase_callback_rows.append({'path': reference['path'], 'reviewer_role': row['role'],
                                     'actual_collection_role': role, 'wave': wave})
    if row.get('wave', wave) != wave:
        # One explicitly auxiliary original is catalogued inside R03. Its
        # genuine received_at is inside that phase, and no canonical proof is
        # relabeled to the auxiliary wave. Preserve the literal old label.
        assert reference['path'] == 'docs/course-revision-20261029-omni/reviews/P5-actual/omni_p5_reader_auxiliary/reader-original-v1.json'
        assert reference['sha256'] == 'ec04ab46b3621c279b14b2ca037d6d76233e3c727d7fe18e7223d69352930cf0'
        assert row['wave'] == 'omni-mainline-03-actual-auxiliary' and wave == 'omni-mainline-03-actual'
        assert set(row['page_ids']) == {'readme-en', 'model-guide'} and not set(row['page_ids']).intersection(catalog)
        auxiliary_rows.append({'actual_container_wave': wave, 'preserved_auxiliary_attribution': row})
    field, made = created(report)
    assert not creation_required or made is not None
    if made is not None:
        assert made <= when, ('created after receipt', reference['path'], field)
    return row


namespace['receipt'] = receipt_compatible
receipt = receipt_compatible


def recheck_primary_and_seals():
    phase_results = namespace['actors_and_stages']()
    assert not namespace['details']['sealed_receipt_assessments']
    primary_results = namespace['primary_proofs']()
    details.update(namespace['details'])
    return {'phases': phase_results, 'primary': primary_results,
            'legacy_ownerless_primary_receipt_calls': len(legacy_ownerless),
            'explicit_later_collection_role_callback_calls': len(phase_callback_rows),
            'distinct_auxiliary_label_records': len(auxiliary_rows)}


def relay_and_owner_evidence():
    owner_report_ref = {'path': 'docs/course-revision-20261029-omni/reviews/technical/technical_review_tools_history_v2/bounded-tool-fix-verification-current-v2.json',
                        'sha256': '4ba0e98c9af44ec2b40db4478c2a643ddeb0ebb7e67f9f0a9fc2c2f2514bbba4'}
    owner_report = doc(owner_report_ref)
    owner = api.owner(owner_report)
    relay = doc(owner_report['source_review']['source_relay_receipt'])
    assert relay['original_owner'] == owner and relay['relay_actor'] == '/root/omni_p8_technical_chapter14'
    assert relay['technical_judgment_by_relay'] is False and relay['new_reviewer_assignment_created'] is False
    assert owner_report['execution_method']['direct_filesystem_read_or_cpu_execution_by_this_task'] is False
    for row in relay['source_entries']:
        data = raw(row, 'actual neutral relay source identity')
        assert len(data) == row['bytes'] and len(data.decode().splitlines()) == row['utf8_lines']
    coverage = {}
    for row in relay['source_parts_delivered']:
        source = (ROOT / row['path']).read_text().splitlines(keepends=True)
        first, last = row['line_range']
        selected = ''.join(source[first - 1:last]).encode()
        assert len(selected) == row['bytes'] and digest(selected) == row['sha256']
        coverage.setdefault(row['path'], []).extend(range(first, last + 1))
    for name, lines in coverage.items():
        assert lines == list(range(1, len((ROOT / name).read_text().splitlines()) + 1))
    checks_data = owner_report['software_checks']
    probe = doc(checks_data['original_mutations']['execution'])
    command = doc(checks_data['original_mutations']['command_receipt'])
    original = doc(checks_data['original_mutations']['original_script_artifact'])
    old_prefix = original['mutation_reproduction']['script'].split('probes = [', 1)[0]
    assert digest(old_prefix.encode()) == checks_data['original_mutations']['original_prefix_sha256']
    driver = raw(checks_data['original_mutations']['driver'], 'owner exact mediation driver; read without execution').decode()
    assert "prefix.replace(old, new)" in driver and "prefix.count(old) == 1" in driver
    assert command['requested_by'] == command['probe_design_by'] == probe['probe_design_by'] == owner
    assert probe['reviewer_task'] == owner and command['executed_by'] == probe['executed_by'] == relay['relay_actor']
    assert len(probe['cases']) == 6 and all(row['matched'] for row in probe['cases'])
    assert collections.Counter(row['observed'] for row in probe['cases']) == {'accept': 2, 'reject': 4}
    assert command['exit_code'] == 0 and command['before_sha256'] == command['after_sha256']
    later = doc(checks_data['targeted_pytest']['receipt'])
    short = doc(checks_data['earlier_actual_command_preserved']['receipt'])
    assert '--no-sync' in later['argv'] and later['explicit_environment']['UV_OFFLINE'] == '1'
    assert '--no-sync' not in short['argv']
    assert later['logs']['stderr']['bytes'] == 0 and short['logs']['stderr']['bytes'] == 221
    log_checks = []
    for recorded in (command, later, short):
        for label, item in recorded['logs'].items():
            path = Path(item['path'])
            if path.exists():
                data = path.read_bytes()
                assert digest(data) == item['sha256'] and len(data) == item['bytes']
                if label == 'stdout' and recorded in (later, short):
                    assert b'81 passed' in data
                log_checks.append({'path': item['path'], 'sha256': item['sha256'], 'checked_here': True})
            else:
                log_checks.append({'path': item['path'], 'sha256': item['sha256'], 'checked_here': False,
                                   'interpretation': 'External raw log retained by acquisition receipt, not claimed included in ordinary clone.'})
    windows_ref = {'path': 'docs/course-revision-20261029-omni/reviews/P5-tools/bounded-windows-current-source18-v1.json',
                   'sha256': 'd8611016acb9fb2a15d050121c6b3e18f29a926ff85add9dd493f960ef953c26'}
    windows = doc(windows_ref)
    assert api.owner(windows) == '/root/omni_p5_full_tools_review'
    windows_result = doc(windows['executed_checks'][1]['result'])
    for item in windows['evidence']:
        raw(item, 'current003 original owner evidence, not own execution')
    details['relay_raw_log_observations'] = log_checks
    details['windows_result_structure'] = {'keys': list(windows_result), 'report': windows_ref,
                                           'owner_test_counts': windows['executed_checks']}
    return {'original_owner': owner, 'neutral_relay': relay['relay_actor'],
            'source_segments_exact_complete': {key: len(value) for key, value in coverage.items()},
            'owner_synthetic_cases': 6, 'owner_mutations_rejected': 4, 'owner_controls_accepted': 2,
            'owner_relay_tests': 81, 'old_short_command_stderr_preserved': 221,
            'current003_other_owner': api.owner(windows), 'new_own_tests_or_probes_run': 0}


def auxiliary_and_global_scope():
    aux = doc(manifest['auxiliary_release_inventory'])
    assert len(aux['pages']) == 2 and aux['actual_issue_occurrence_count'] == len(aux['author_disposition_occurrences']) == 7
    canonical_occurrences = {(row['report']['sha256'], row['json_pointer']) for row in dp['items']}
    for page in aux['pages']:
        assert page['outside_canonical'] is True and page['page_id'] not in catalog
        raw({'path': page['source'], 'sha256': page['source_sha256']}, 'separate auxiliary current source')
        for role, pointer in page['proofs'].items():
            proof = api.node(doc(pointer['report']), pointer['json_pointer'])
            current = doc(proof['report'])
            value = api.node(current, proof.get('json_pointer', ''))
            assert value.get('page_id', value.get('lesson_id')) == page['page_id']
            assert value['source_sha256'] == page['source_sha256']
            assert value.get('figures_sha256', value.get('figure_sha256')) == page['figures_sha256']
            assert roles[api.owner(current)] == role
    for pointer in aux['author_disposition_occurrences']:
        decision = api.node(doc(pointer['report']), pointer['json_pointer'])
        assert (decision['report']['sha256'], decision['json_pointer']) not in canonical_occurrences
        assert decision['issue_id'] == pointer['issue_id']
    global_report = doc(manifest['dependency_audit'])
    assert api.owner(global_report) == '/root/omni_p5_coherence_entries' and global_report['verdict'] == 'pass'
    assert set(global_report['inventory_page_ids']) == set(live)
    retained = global_report['retained_semantic_scope']
    assert retained['initial_classification'] == {'full_including_context': 8, 'bounded': 14, 'mechanical_only': 347}
    assert retained['new_reader_actors'] == retained['new_C_semantic_actors'] == 0
    assert retained['all369_new_semantic_read'] is False
    previous_ref = global_report['original_owner_two_issue_verification_retained']['report']
    previous = doc(previous_ref)
    assert previous_ref['sha256'] == '213742f695ea2b9b22619b81efcd92200709985a5b6c1c33ff1b32f37ed65695'
    assert 'verification_pages' not in global_report
    for i in (0, 1):
        assert previous['verification_pages'][i]['source_sha256'] == live[previous['verification_pages'][i]['page_id']]['source_sha256']
    receipt_compatible(manifest['dependency_audit'], opened['wave'], 'continuity', api.owner(global_report), True)
    peer_ref = catalog['12.2']['technical']['report']
    audio = doc(peer_ref)
    peer = doc(audio['peer_comparison_ref'])
    exposure = peer['incidental_exposure']
    assert exposure['adopted_inputs_distinguished'] and peer['issues'][0]['issue_id'] == 'audio122-peer-optional-01'
    author_items = [row for row in dp['items'] if row['issue_id'] == 'audio122-peer-optional-01']
    assert len(author_items) == 1 and author_items[0]['final_necessity'] == 'optional' and author_items[0]['action'] == 'defer'
    assert author_items[0].get('verification', []) == []
    c05 = doc(ref(ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/wave05/continuity-phase-proof-v1.json'))
    details['C05_scope_fields'] = {key: value for key, value in c05.items() if key not in {'collection_index', 'report_sha256s'}}
    details['audio_exposure_and_optional'] = {'exposure': exposure, 'author_disposition': author_items[0]}
    return {'auxiliary_pages': 2, 'auxiliary_author_occurrences': 7, 'main_canonical_scope_unchanged': True,
            'global_retained_read_counts': retained['initial_classification'],
            'old_global_verification_pages_preserved_byref': [0, 1], 'new_C_semantic_claims': 0,
            'audio_post_original_exposure_disclosed': True, 'audio_optional_still_unimplemented_defer': True}


check('legacy_receipt_schema_and_all_actual_primary_data', recheck_primary_and_seals)
check('same_owner_neutral_relay_and_windows_bounded_evidence', relay_and_owner_evidence)
check('auxiliary_global_retention_and_incidental_exposure_scope', auxiliary_and_global_scope)
details['legacy_ownerless_receipt_calls'] = legacy_ownerless
details['later_phase_callback_calls'] = phase_callback_rows
details['auxiliary_attribution_records'] = auxiliary_rows
large = {'reviewer_task': TASK, 'base_candidate': manifest_ref, 'details': details, 'verified_files': list(verified.values())}
large_path = RAW / 'compat-current-data-details-v1.json'
with large_path.open('x') as stream:
    json.dump(large, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
summary = {'schema_version': 1, 'reviewer_task': TASK, 'created_at': datetime.now(UTC).isoformat(),
           'kind': 'own scanner schema correction and current evidence followup, not final verdict',
           'base_candidate': manifest_ref, 'checks': checks, 'details_external_optional': ref(large_path),
           'first_harness_observation_preserved': ref(OUT / 'base-observations-v1.json'),
           'correction': 'Distinct reviewer role vs collection_role and body-bound owner for older receipts; preserved one explicitly auxiliary wave label is outside canonical scope. No source/checker/report/actual receipt changed.'}
with (OUT / 'compat-observations-v1.json').open('x') as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'summary': ref(OUT / 'compat-observations-v1.json'), 'checks': checks,
                  'final_verdict_saved': False}), flush=True)
