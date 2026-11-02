"""Independent exact data checks on immutable preclose inputs; no final verdict.

Only native metadata parts run. No baseline replay, all-book gate, tests, model,
kernel, training, Git or network operation is invoked by this script.
"""

import collections
import hashlib
import importlib.util
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path('/workspace/work/omni-book-integrated-v1')
WAVE = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/wave06'
OUT = Path(__file__).resolve().parent
RAW = Path('/workspace/work/omni-p6-final-current-data-c06-v1')
RAW.mkdir(exist_ok=True)
TASK = '/root/omni_p6_technical_tools'
START = datetime.now(UTC)
MANIFEST = WAVE / 'final-assembly-v2/manifest-author-received-v1.json'
EXPECTED = '62bdcda3d5d7003b0bb04280d55e0413a7f0a0db9003345a772273fae5df2d97'
verified, cache, checks, details = {}, {}, {}, {}


def blocked(*args, **kwargs):
    raise RuntimeError('network prohibited in exact current data audit')


socket.create_connection = blocked
socket.getaddrinfo = blocked
socket.socket.connect = blocked


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': digest(raw), 'bytes': len(raw)}


def raw(reference, reason='exact metadata/evidence anchor'):
    name, expected = reference['path'], reference['sha256']
    assert isinstance(name, str) and not Path(name).is_absolute(), ('ordinary relative path required', name)
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT), ('path escapes root', name)
    data = path.read_bytes()
    assert digest(data) == expected, ('artifact SHA differs', name, expected, digest(data))
    key = (name, expected)
    row = verified.setdefault(key, {'path': name, 'sha256': expected, 'bytes': len(data), 'uses': []})
    if reason not in row['uses']:
        row['uses'].append(reason)
    return data


def doc(reference, reason='complete JSON parse, structure/fingerprint, selected metadata'):
    key = (reference['path'], reference['sha256'])
    if key not in cache:
        cache[key] = json.loads(raw(reference, reason))
    else:
        raw(reference, reason)
    return cache[key]


def module(name, relative):
    path = ROOT / relative
    raw(ref(path), 'current unchanged native source fingerprint; previously read source contract')
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def check(name, function):
    begin = datetime.now(UTC)
    try:
        result = function()
        checks[name] = {'status': 'checked', 'result': result}
    except Exception as error:
        checks[name] = {'status': 'needs_assessment', 'error': type(error).__name__ + ': ' + str(error)}
    checks[name]['started_at'] = begin.isoformat()
    checks[name]['completed_at'] = datetime.now(UTC).isoformat()
    print(json.dumps({'check': name, 'status': checks[name]['status'], 'error': checks[name].get('error')}, ensure_ascii=False), flush=True)


def identity(reference):
    return reference['path'], reference['sha256']


def created(report):
    for field in ('created_at', 'at', 'confirmed_at', 'saved_at', 'reviewed_at', 'started_at'):
        value = report.get(field) if isinstance(report, dict) else None
        if isinstance(value, str):
            try:
                return field, api.at(value)
            except (TypeError, ValueError):
                pass
    return None, None


assert ref(MANIFEST)['sha256'] == EXPECTED
manifest_ref = ref(MANIFEST)
manifest = doc(manifest_ref)
api = module('own_final_current_native', 'docs/review-tools/incremental_revision.py')
composite = module('own_final_current_inventory', 'docs/review-tools/composite_review.py')
technical = module('own_final_current_technical', 'scripts/check_technical_reviews.py')
setup = module('own_final_current_setup', 'scripts/omni_revision/review_setup.py')
live = {row['page_id']: row for row in composite.current_inventory(ROOT)['pages']}
catalog = {row['page_id']: row for row in manifest['pages']}
freeze = doc(manifest['source_freeze'])
frozen = {row['page_id']: row for row in freeze['pages']}
assignments = doc(manifest['assignments'])
assigned = {row['task']: row for row in assignments['tasks']}
roles = {task: row['role'] for task, row in assigned.items()}
sealed = api.review_seals(ROOT, manifest['review_waves'], ('reader', 'technical'))
open_ref = ref(WAVE / 'continuity-stage-open-v1.json')
opened = doc(open_ref)
intake_ref = ref(WAVE / 'continuity-collection-intake-v8.json')
assert intake_ref['sha256'] == '27d2c006622c360cdd1fbf29f8c5e053c436aa661d8e4dbb2090d17dc9936aab'
intake = doc(intake_ref)
dp = doc(manifest['dispositions'])
patches = api.author_patches(ROOT, manifest, live)
earlier_files = api.file_versions(ROOT, patches)
earlier = {pid: api.page_versions(ROOT, frozen[pid], page, patches) for pid, page in live.items()}


def receipt(reference, wave, role, task=None, creation_required=False):
    if wave == opened['wave'] and role == 'continuity':
        phase, rows = opened, intake['collection_index']
        end = START
    else:
        phase = sealed[wave][role]
        rows, end = phase.get('collection_index', []), api.at(phase['completed_at'])
        assert phase['status'] == 'completed' and end <= START
        assert reference['sha256'] in phase['report_sha256s']
    matches = [row for row in rows if identity(row) == identity(reference)]
    assert len(matches) == 1, ('not a unique actual receipt', reference['path'], wave, role, len(matches))
    row = matches[0]
    when = api.at(row['received_at'])
    assert api.at(phase['started_at']) <= when <= end <= START
    if task is not None:
        assert row.get('reviewer_task') == task, ('receipt task differs', reference['path'], task, row.get('reviewer_task'))
    for field in ('role', 'collection_role'):
        if field in row:
            assert row[field] == role, ('actual collection role differs', reference['path'], field)
    if 'wave' in row:
        assert row['wave'] == wave
    report = doc(reference, 'report with actual unique phase receipt')
    field, made = created(report)
    assert not creation_required or made is not None, ('new report missing real creation time', reference['path'])
    if made is not None:
        assert made <= when, ('creation after actual receipt', reference['path'], field, made.isoformat(), row['received_at'])
    return row


def current_identities():
    own_primary = doc(manifest['tool_audit']['report'])
    assert manifest['tool_audit']['report']['sha256'] == 'cc3602c7134a75f37ec319309d85f2c232db3d99864c99128a600c6f34f51fd2'
    original = doc(own_primary['original'])
    assert own_primary['original']['sha256'] == '2ebf862da7355fdc11d94ffdda7fc097e35569df56294629aab5501d2a84d20e'
    files = {row['path']: row['sha256'] for row in own_primary['checked_files']}
    assert len(files) == 460 and files == manifest['implementation']
    assert files == {row['path']: row['sha256'] for row in original['checked_files']}
    assert original['schema_version'] == 1 and original['review_stage'] == 'technical'
    assert original['reviewer_context'] == 'fresh' and api.owner(original) == TASK
    assert original['checks'] and isinstance(original['issues'], list) and original['issues']
    for name, expected in files.items():
        raw({'path': name, 'sha256': expected}, 'current460 fingerprint only; no renewed semantic read')
    for name, expected in manifest['criteria'].items():
        raw({'path': name, 'sha256': expected}, 'current criteria fingerprint')
    fields = ('source', 'selector', 'source_sha256', 'figures_sha256', 'notebook', 'notebook_sha256')
    assert len(live) == len(catalog) == len(frozen) == 369 and set(live) == set(catalog) == set(frozen)
    for pid, page in live.items():
        assert all(page.get(key) == frozen[pid].get(key) == catalog[pid].get(key) for key in fields), pid
        raw({'path': frozen[pid]['snapshot'], 'sha256': page['source_sha256']}, 'current canonical snapshot identity')
        if page.get('notebook'):
            raw({'path': page['notebook'], 'sha256': page['notebook_sha256']}, 'current generated Notebook identity only')
    figures = {name: value for page in live.values() for name, value in page['figures_sha256'].items()}
    assert len(figures) == len(freeze['figure_snapshots']) == 126
    for name, expected in figures.items():
        raw({'path': name, 'sha256': expected}, 'current figure fingerprint only')
        raw(freeze['figure_snapshots'][name], 'current frozen figure fingerprint')
    raw_root = Path(original['scope']['raw_snapshot_root'])
    raw_count = 0
    if raw_root.is_dir():
        for name, expected in files.items():
            assert digest((raw_root / name).read_bytes()) == expected, ('own original raw snapshot differs', name)
            raw_count += 1
    details['own_tool_original'] = {'original': own_primary['original'], 'primary': manifest['tool_audit']['report'],
                                     'current_files': len(files), 'original_raw_files_rehashed_if_resident': raw_count,
                                     'original_raw_root': str(raw_root)}
    return {'implementation': len(files), 'canonical': len(live),
            'notebooks': sum(bool(row.get('notebook')) for row in live.values()), 'figures': len(figures)}


def actors_and_stages():
    assert len(assigned) == len(assignments['tasks']) == 661
    old_candidate = doc(manifest['previous_working_candidate'])
    old_assignments = doc(old_candidate['assignments'])
    assert len(old_assignments['tasks']) == 659 and assignments['tasks'][:659] == old_assignments['tasks']
    for task, row in assigned.items():
        assert task not in manifest['author_tasks'] and task.startswith('/root/')
        assert row['context'] == 'fresh' and row['role'] in api.ROLES
        phase = sealed[row['wave']][row['role']]
        window = row['dispatch_window']
        assert api.at(phase['started_at']) <= api.at(window['not_before']) <= api.at(window['not_after']) <= api.at(phase['completed_at']) <= START
    dispatch = doc(ref(WAVE / 'technical-dispatch-intake-v2.json'))
    assert assignments['tasks'][-2:] == dispatch['tasks']
    assert all(row['fork_turns'] == 'none' for row in dispatch['tasks'])
    assert {row['lesson_id'] for row in dispatch['tasks']} == {'12.2', 'chapter-14'}
    assert opened['status'] == 'open_actual_work_pending' and 'completed_at' not in opened
    assert api.at(sealed[opened['wave']]['technical']['completed_at']) <= api.at(opened['started_at']) <= START
    assert 'continuity' not in manifest['review_waves'][-1]['seals']
    all_rows, anomalies, no_time = [], [], []
    for wave, phases in sealed.items():
        for role, phase in phases.items():
            assert phase['status'] == 'completed' and api.at(phase['completed_at']) <= START
            rows = phase.get('collection_index', [])
            assert len({identity(row) for row in rows}) == len(rows)
            assert {row['sha256'] for row in rows} == set(phase['report_sha256s'])
            for row in rows:
                try:
                    receipt(row, wave, role, row.get('reviewer_task'))
                    report = doc(row)
                    if created(report)[1] is None:
                        no_time.append({'wave': wave, 'role': role, 'path': row['path']})
                except Exception as error:
                    anomalies.append({'wave': wave, 'role': role, 'path': row['path'], 'error': str(error)})
                all_rows.append({'wave': wave, 'role': role, **row})
    for row in intake['collection_index']:
        receipt(row, opened['wave'], 'continuity', row['reviewer_task'], creation_required=True)
    details['all_sealed_receipts'] = all_rows
    details['sealed_receipt_assessments'] = anomalies
    details['old_format_receipts_without_declared_creation_time'] = no_time
    return {'actors': len(assigned), 'unchanged_prior_tasks': 659, 'new_fresh_T': 2,
            'completed_phases': sum(len(x) for x in sealed.values()),
            'sealed_rows': len(all_rows), 'open_C06_actual_rows': len(intake['collection_index']),
            'receipt_assessments': len(anomalies), 'legacy_missing_time': len(no_time)}


def primary_proofs():
    previous = doc(manifest['previous_working_candidate'])
    old_pages = {row['page_id']: row for row in previous['pages']}
    baseline = doc(manifest['baseline_freeze'])
    baseline_pages = {row['page_id']: row for row in baseline['pages']}
    rows, inherited, anomalies = [], [], []
    reader_native = 0
    for pid, page in catalog.items():
        for role in api.ROLES:
            proof = page[role]
            if proof['mode'] == 'inherit':
                assert pid in baseline_pages
                assert all(live[pid].get(key) == baseline_pages[pid].get(key) for key in ('source', 'selector', 'source_sha256', 'figures_sha256'))
                assert proof['baseline_page_id'] == pid and proof['dependency_note'].strip()
                if live[pid].get('notebook'):
                    assert baseline_pages[pid]['notebook']['live_sha256'] == live[pid]['notebook_sha256']
                inherited.append({'page_id': pid, 'role': role})
                continue
            assert proof['mode'] == 'review'
            current = doc(proof['report'])
            task = api.owner(current)
            assert roles[task] == role and assigned[task]['wave'] == proof['wave']
            assert (assigned[task]['lesson_id'] == pid if role == 'technical' else pid in assigned[task]['pages'])
            value = api.page_member(current, proof.get('json_pointer', ''), pid, task)
            assert value.get('verdict') in {'pass', '已足夠'}
            assert api.patched_version(value['source_sha256'], live[pid]['source_sha256'], earlier[pid]) is not None
            assert value.get('figures_sha256', value.get('figure_sha256')) == live[pid]['figures_sha256']
            current_row = receipt(proof['report'], proof['wave'], proof.get('collection_role', role), task)
            original = doc(proof['original_report'])
            original_value = api.page_member(original, proof.get('original_json_pointer', ''), pid, task)
            original_row = receipt(proof['original_report'], proof['wave'], role, task)
            assert api.at(original_row['received_at']) <= api.at(current_row['received_at'])
            original_made, current_made = created(original)[1], created(current)[1]
            if original_made is not None and current_made is not None:
                assert original_made <= current_made
            if role == 'reader':
                reader_phase = sealed[proof['wave']]['reader']
                for session in proof.get('sessions', []):
                    expected_pid, _ = api.reader_session(ROOT, session, task)
                    assert expected_pid == pid
                api.reader_closure(ROOT, proof, value, task, pid, reader_phase, sealed)
                reader_native += 1
            rows.append({'page_id': pid, 'role': role, 'owner': task, 'wave': proof['wave'],
                         'original': proof['original_report'], 'current': proof['report'],
                         'original_received_at': original_row['received_at'], 'current_received_at': current_row['received_at'],
                         'declared_creation_present': original_made is not None and current_made is not None})
    for pid in ('12.2', 'chapter-14'):
        for role in ('reader', 'continuity'):
            assert catalog[pid][role] == old_pages[pid][role]
    details['primary_proofs'] = rows
    details['baseline_inherit_proofs'] = inherited
    return {'current_review_proofs': len(rows), 'modes': {role: dict(collections.Counter(page[role]['mode'] for page in catalog.values())) for role in api.ROLES},
            'legitimate_baseline_inherit': len(inherited), 'native_reader_sessions_closures': reader_native,
            'two_page_R_C_retained_exact_old_modes_waves': True}


def native_parts_and_occurrences():
    histories = api.historical_rechecks(ROOT, manifest, live, roles, sealed)
    assert len(histories.entries) == 61
    sidecar = doc(manifest['historical_fix_rechecks'])
    previous = doc(manifest['previous_working_candidate'])
    old_sidecar = doc(previous['historical_fix_rechecks'])
    assert len(sidecar['chains']) == 3 and sidecar['chains'][:2] == old_sidecar['chains']
    first_chain = doc(sidecar['chains'][0])
    pointer = doc(first_chain['prior_active_pointer'])
    old_manifest = doc({'path': pointer['manifest'], 'sha256': pointer['manifest_sha256']})
    old_dp = doc(old_manifest['dispositions'])
    assert len(old_dp['items']) == 257 and dp['items'][:257] == old_dp['items']
    assert len(dp['items']) == 554 and dp['pending_count'] == 0
    assert all(item['action'] == 'fixed' for item in dp['items'] if item['final_necessity'] == 'necessary')
    dp_row = receipt(manifest['dispositions'], opened['wave'], 'continuity', '/root', True)
    assert dp_row['received_at'] == manifest['dispositions_collection']['received_at'] == '2026-11-01T06:19:40.861177+00:00'
    finished = []

    def observe(frame, event, argument):
        if event == 'return' and frame.f_code.co_name == 'finish' and frame.f_code.co_filename.endswith('/historical_fix_rechecks.py'):
            actual = frame.f_locals['self']
            finished.append({'entries': len(actual.entries), 'used': len(actual.used), 'same': set(actual.entries) == actual.used})

    sys.setprofile(observe)
    try:
        relied = api.check_dispositions(ROOT, manifest, api.ROLES, live, roles, sealed, earlier)
    finally:
        sys.setprofile(None)
    assert finished == [{'entries': 61, 'used': 61, 'same': True}]
    api.check_tool_audit(ROOT, manifest, roles, sealed)
    # Compute expected occurrences independently rather than relying on DP count.
    reports = {r['sha256']: r for r in manifest['issue_reports']}
    notes = {}
    for page in catalog.values():
        for session in page['reader'].get('sessions', []):
            for note in session['notes']:
                notes[note['sha256']] = note
    expected = set()
    for sha, reference in (reports | notes).items():
        report = doc(reference, 'all ordinary issue-report/note anchors, complete parse; mechanical issue enumeration')
        expected.update((sha, pointer) for pointer in api.reported_issues(report))
    for extra in manifest.get('additional_issue_inventories', []):
        array = api.node(doc(extra['report']), extra['json_pointer'])
        expected.update((extra['report']['sha256'], extra['json_pointer'] + '/' + str(i)) for i in range(len(array)))
    actual = {(item['report']['sha256'], item['json_pointer']) for item in dp['items']}
    assert len(actual) == len(expected) == 554 and actual == expected
    details['native_parts'] = {'historical_finish_observation': finished, 'ordinary_issue_report_entries': len(manifest['issue_reports']),
                                'unique_issue_report_hashes': len(reports), 'expected_occurrences': len(expected),
                                'author_actions': dict(collections.Counter(row['action'] for row in dp['items'])),
                                'author_necessities': dict(collections.Counter(row['final_necessity'] for row in dp['items'])),
                                'author_patch_reliance_nonempty': {pid: value for pid, value in relied.items() if value},
                                'actual_prior_dispositions': old_manifest['dispositions']}
    return {'constructor': 61, 'historical_finish_used': 61, 'native_check_dispositions': 'passed',
            'expected_and_actual_occurrences': 554, 'first257_exact': True, 'tool_primary': 'passed'}


def new_T_and_six_history():
    sidecar = doc(manifest['historical_fix_rechecks'])
    chain = doc(sidecar['chains'][2])
    assert len(chain['rechecks']) == 2
    output = []
    for group in chain['rechecks']:
        pid = group['page_id']
        proof = catalog[pid]['technical']
        original, current, appendix = doc(proof['original_report']), doc(proof['report']), doc(group['appendix'])
        task = api.owner(current)
        assert api.owner(original) == api.owner(appendix) == task
        assert original['schema_version'] == 1 and original['review_stage'] == 'technical' and original['reviewer_context'] == 'fresh'
        setup.check_original(original, proof['original_source_freeze'], pid, pid, task, technical, api,
                             original_source_snapshots=proof.get('original_source_snapshots', []))
        own_row = receipt(proof['original_report'], proof['wave'], 'technical', task, True)
        now_row = receipt(proof['report'], proof['wave'], 'technical', task, True)
        appendix_row = receipt(group['appendix'], group['collection']['wave'], 'technical', task, True)
        request_ref = appendix.get('prompted_history_request') or appendix.get('history_request')
        if not request_ref:
            request_ref = ref(WAVE / ('technical-audio122' if pid == '12.2' else 'technical-chapter14') / 'prompted-history-request-v1.json')
        request = doc(request_ref)
        request_created = created(request)[1]
        assert api.at(own_row['received_at']) <= api.at(now_row['received_at']) < request_created <= created(appendix)[1] <= api.at(appendix_row['received_at'])
        assert len(appendix['historical_issue_rechecks']) == 3
        assert all(row['necessity'] == 'necessary' and row['status'] == 'still_fixed' and row['verdict'] == 'pass' for row in appendix['historical_issue_rechecks'])
        unavailable = doc(group['unavailability']['report'])
        record = api.node(unavailable, group['unavailability']['json_pointer'])
        assert record['unavailable'] is True
        assert any(event.get('tool') == 'collaboration.followup_task' and event.get('request', {}).get('target') == record['reviewer_task'] for event in record['events'])
        assert any(event.get('tool') == 'collaboration.list_agents' and isinstance(event.get('response'), dict) and event['response'].get('agents') == [] and record['reviewer_task'].startswith(event['request']['path_prefix']) for event in record['events'])
        output.append({'page_id': pid, 'task': task, 'original_receipt': own_row, 'current_receipt': now_row,
                       'prompted_request': request_ref, 'request_created_at': request_created.isoformat(),
                       'appendix_receipt': appendix_row, 'unavailability': group['unavailability'],
                       'three_original_issue_ids': [row['issue_id'] for row in appendix['historical_issue_rechecks']]})
    old = doc(ref(WAVE / 'technical-chapter14/historical-group-15-appendix-v1.json'))
    new = doc(ref(WAVE / 'technical-chapter14/historical-group-15-appendix-v2.json'))
    old_normal, new_normal = dict(old), dict(new)
    for key in ('created_at', 'previous_appendix', 'format_repair'):
        old_normal.pop(key, None)
        new_normal.pop(key, None)
    new_normal = json.loads(json.dumps(new_normal))
    for row in new_normal['historical_issue_rechecks']:
        assert row.pop('reason') == row['current_reason']
    assert old_normal == new_normal
    details['new_T_original_current_history'] = output
    return {'new_whole_originals_validated': 2, 'new_history_rows': 6, 'old_owner_unavailability_records': 2,
            'C14_v1_to_v2_only_reason_and_timestamp_metadata': True}


def seven_tool_refreshes():
    previous = doc(ref(WAVE / 'final-assembly-v1/manifest-author-received-v1.json'))
    old_dp = doc(previous['dispositions'])
    author = doc(ref(WAVE / 'author-tool-verification-current-decisions-v1.json'))
    assert len(old_dp['items']) == 552 and len(author['items']) == 9
    changed = [i for i, (old, new) in enumerate(zip(old_dp['items'], dp['items'])) if old != new]
    assert changed == [315, 316, 319, 320, 338, 339, 455]
    rows = []
    reports = {}
    for index in changed:
        item, before = dp['items'][index], old_dp['items'][index]
        assert item['report'] == before['report'] and item['json_pointer'] == before['json_pointer']
        assert item['final_necessity'] == before['final_necessity'] and item['action'] == before['action'] == 'fixed'
        assert item['affected_files'] == before['affected_files']
        source_owner = api.owner(doc(item['report']))
        assert len(item['verification']) == 1 and item['verification'][0]['kind'] == 'tool_audit'
        verification_ref = item['verification'][0]['report']
        verification = doc(verification_ref)
        task = api.owner(verification)
        assert task == source_owner or source_owner in manifest['author_tasks']
        assert roles[task] == 'technical' and verification['verdict'] == 'pass'
        assert item['issue_id'] in verification['resolved_issue_ids']
        actual = {row['path']: row['sha256'] for row in verification['checked_files']}
        assert set(item['affected_files']) <= set(actual)
        for path in item['affected_files']:
            raw({'path': path, 'sha256': actual[path]}, 'new per-issue same-owner current affected-file binding')
        actual_receipt = receipt(verification_ref, opened['wave'], 'continuity', task, True)
        rows.append({'index': index, 'issue_id': item['issue_id'], 'issue_owner': source_owner,
                     'verification_owner': task, 'verification': verification_ref,
                     'actual_C06_receipt': actual_receipt, 'affected_files': item['affected_files']})
        reports[verification_ref['sha256']] = verification_ref
    assert len(reports) == 2
    for item in dp['items'][552:]:
        assert item in author['items'] and item['action'] == 'fixed' and item['final_necessity'] == 'necessary'
    details['seven_current_tool_verifications'] = rows
    details['two_resolved_echo_occurrences'] = [{'index': i, 'issue_id': item['issue_id'], 'source': item['report'], 'pointer': item['json_pointer']} for i, item in enumerate(dp['items'][552:], 552)]
    return {'changed_post257_verifications': 7, 'same_original_owner_current_reports': 2,
            'new_resolved_echoes': 2, 'independent_new_defects_claimed': 0,
            'original_necessity_actions_affected_files_preserved': True}


def current_technical_sources():
    reports, owners = api.technical_reports(manifest, ROOT)
    readers = {task for task, role in roles.items() if role == 'reader'}
    failures = []
    for pid, page in catalog.items():
        proof = page['technical']
        if proof['mode'] != 'review':
            continue
        current = doc(proof['report'])
        text = raw({'path': frozen[pid]['snapshot'], 'sha256': live[pid]['source_sha256']}).decode()
        errors = technical._validate(ROOT, ROOT / live[pid]['source'], pid, text, current, owners, readers,
                                     earlier_versions={'source_sha256': earlier[pid], 'files': earlier_files})
        if errors:
            failures.append({'page_id': pid, 'errors': errors})
        for artifact in current.get('artifacts', []):
            if isinstance(artifact, dict) and isinstance(artifact.get('path'), str) and artifact.get('sha256'):
                raw(artifact, 'current formal technical artifact availability; no result recomputation')
    details['current_technical_source_metadata_errors'] = failures
    assert not failures, failures[:3]
    return {'current_canonical_technical_native_schema_source_artifact_checks': len(reports),
            'scientific_recomputations': 0, 'baseline_or_fullbook_gate_runs': 0}


check('current_source_implementation_original_identity', current_identities)
check('actual_actors_sealed_periods_open_C06', actors_and_stages)
check('all_selected_primary_proof_data', primary_proofs)
check('native61_exact554_and_tool_primary', native_parts_and_occurrences)
check('new_T06_original_current_prompted_history_and_unavailability', new_T_and_six_history)
check('seven_current_tool_references_and_two_echoes', seven_tool_refreshes)
check('current_technical_positive_source_artifact_availability', current_technical_sources)

# Persist only mechanical detail outside Git. Small report and rerunnable script
# bind the ordinary immutable inputs; this optional index is not a new runtime
# native tool-verification dependency or a substitute for external original text.
large = {'reviewer_task': TASK, 'audited_candidate_manifest': manifest_ref,
         'verified_files': list(verified.values()), 'details': details}
large_path = RAW / 'base-current-data-details-v1.json'
with large_path.open('x') as stream:
    json.dump(large, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
summary = {'schema_version': 1, 'reviewer_task': TASK, 'started_at': START.isoformat(),
           'created_at': datetime.now(UTC).isoformat(), 'kind': 'independent intermediate immutable base observations, NOT final data-audit verdict',
           'audited_candidate_manifest': manifest_ref, 'checks': checks,
           'verified_unique_path_sha_anchors': len(verified), 'details_external_optional': ref(large_path),
           'needs_assessment_count': sum(row['status'] == 'needs_assessment' for row in checks.values()),
           'C06_state': 'actually OPEN; no future seal/own receipt/final successor manifest verified or invented',
           'root_provenance_notation_correction_pending': 'Root reported original_severity_preserved misnaming on two new echoes; final report waits for true immutable successor and receipt. Necessary classification/action/evidence are unchanged; root discovery is not attributed to this auditor.'}
with (OUT / 'base-observations-v1.json').open('x') as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'summary': ref(OUT / 'base-observations-v1.json'), 'needs_assessment': summary['needs_assessment_count'],
                  'details': ref(large_path), 'final_verdict_saved': False}), flush=True)
