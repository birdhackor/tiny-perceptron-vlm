"""Independent no-network admission fixtures; never reserve, run or upload real work."""
import copy
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts.omni_revision import cpu_final_execution as runner
from scripts.omni_revision import research_execution

ROOT = Path(__file__).resolve().parents[5]
SNAPSHOT = Path('/workspace/work/omni-P5-full-tools-review-source-v3')
REVISION = 'a' * 40
ATTEMPT = 'cpu-final-123-1'
ENVIRONMENT = {
    'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': runner.REPOSITORY,
    'GITHUB_REF': 'refs/heads/main', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
    'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': REVISION,
    'OMNI_ARTIFACT_OUTCOME': 'success',
}
controls = []


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rejected(name, function):
    try:
        function()
    except ValueError as error:
        controls.append({'case': name, 'rejected': True, 'error': str(error)})
    else:
        raise AssertionError('accepted negative: ' + name)


with tempfile.TemporaryDirectory(prefix='p5-cpu-admission-') as directory:
    private = Path(directory)
    for name in runner.source_paths():
        target = private / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SNAPSHOT / name, target)
    audit_relative = runner.AUDIT.relative_to(ROOT)
    (private / audit_relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SNAPSHOT / audit_relative, private / audit_relative)
    git_objects = {path.relative_to(private).as_posix(): path.read_bytes() for path in private.rglob('*') if path.is_file()}

    def fake_git(arguments, cwd=None, **kwargs):
        assert Path(cwd) == private
        if arguments == ['git', 'rev-parse', 'HEAD']:
            return (REVISION + '\n').encode()
        assert arguments[:2] == ['git', 'show']
        revision, name = arguments[2].split(':', 1)
        assert revision == REVISION
        return git_objects[name]

    with patch.dict(os.environ, ENVIRONMENT), patch.object(runner, 'ROOT', private), patch.object(runner, 'AUDIT', private / audit_relative), patch.object(research_execution, 'ROOT', private), patch.object(research_execution, 'FROZEN', private / 'docs/omni-products/v1'), patch.object(runner.subprocess, 'check_output', fake_git):
        sources = runner.checked_sources()
        assert len(sources['files']) == 98
        controls.append({'case': 'all98_sources_exact_mock_Git_objects_positive', 'accepted': True})
        with patch.dict(os.environ, {'GITHUB_SHA': 'b' * 40}):
            rejected('checkout_SHA_different', runner.checked_sources)
        original_audit = (private / audit_relative).read_bytes()
        audit = json.loads(original_audit)
        audit['files'].pop(next(iter(audit['files'])))
        write(private / audit_relative, audit)
        rejected('partial98_audit', runner.checked_sources)
        (private / audit_relative).write_bytes(original_audit)
        audit = json.loads(original_audit)
        audit['reviewer_task'] = ''
        write(private / audit_relative, audit)
        rejected('empty_independent_reviewer', runner.checked_sources)
        (private / audit_relative).write_bytes(original_audit + b'\n')
        rejected('uncommitted_audit_bytes', runner.checked_sources)
        (private / audit_relative).write_bytes(original_audit)
        target = private / 'scripts/omni_revision/cpu_final_execution.py'
        original_source = target.read_bytes()
        target.write_bytes(original_source + b'\n')
        rejected('source_differs_from98_audit', runner.checked_sources)
        target.write_bytes(original_source)
        actual_file = private / 'preserved-source.py'
        target.rename(actual_file)
        target.symlink_to(actual_file)
        rejected('same_bytes_symlink_source', runner.checked_sources)
        target.unlink()
        actual_file.rename(target)
        assert runner.checked_sources() == sources


def finish_case(name, change=None, expected='completed'):
    with tempfile.TemporaryDirectory(prefix='p5-finish-admission-') as directory:
        output = Path(directory)
        source = {'revision': REVISION, 'files': dict(sources['files']), 'audit_sha256': sources['audit_sha256']}
        tag, body = runner.release_identity(ATTEMPT, REVISION)
        reserved = {
            'attempt_id': ATTEMPT, 'revision': REVISION, 'status': 'reserved',
            'components': dict(runner.transport.COMPONENTS), 'bound_usd': '0.10',
            'purpose': json.dumps({'job': runner.JOB, 'ownership': {'attempt': ATTEMPT}, 'sources_sha256': runner.digest(source)}),
        }
        asset = {'id': 31, 'name': 'final-results.zip', 'bytes': 1024, 'sha256': 'c' * 64, 'roundtrip_verified': True}
        receipt = {
            'schema': 'omni-cpu-final-execution-v1', 'attempt': ATTEMPT,
            'source_revision': REVISION, 'release_tag': tag, 'release_id': 41,
            'assets': [asset], 'success': True, 'all_training_processes_exited': True,
            'no_modal_compute_created': True,
        }
        ownership = {'tag': tag, 'body': body, 'release_id': 41}
        provider = {'id': 41, 'draft': True, 'tag_name': tag, 'body': body, 'target_commitish': REVISION,
                    'assets': [{'id': 31, 'name': 'final-results.zip', 'size': 1024}]}
        artifact = {'total_count': 1, 'artifacts': [{'id': 71, 'name': 'omni-final-123-1', 'size_in_bytes': 2048,
                    'created_at': '2026-10-30T03:00:00Z', 'expires_at': '2026-11-06T03:00:00Z', 'expired': False}]}
        state = {'attempts': [copy.deepcopy(reserved)], 'reserved_total_usd': '0.10'}
        fixture = {'reserved': reserved, 'receipt': receipt, 'ownership': ownership, 'provider': provider, 'artifact': artifact, 'state': state}
        if change:
            change(fixture)
        for filename, value in [('reservation.json', {'attempt': reserved}), ('cpu-final-receipt.json', receipt), ('release-ownership.json', ownership)]:
            write(output / filename, value)
        calls, writes = [], []

        class Ledger:
            def read(self):
                return state, 'exact-observed-blob'

            def write(self, following, observed):
                assert observed == 'exact-observed-blob'
                assert following['reserved_total_usd'] == '0.10'
                writes.append(following)
                return {'explicit_offline_fixture': True}

        def api(endpoint, *args, **kwargs):
            assert not args and not kwargs
            calls.append(endpoint)
            if endpoint.startswith('releases/'):
                return provider
            assert endpoint == 'actions/runs/123/artifacts?per_page=100'
            return artifact

        with patch.dict(os.environ, ENVIRONMENT), patch.object(runner, 'checked_sources', lambda: source), patch.object(runner, 'GitHubLedger', Ledger), patch.object(runner, 'api', api):
            if expected == 'reject_before_provider':
                rejected(name, lambda: runner.finish_final(ATTEMPT, output))
                assert calls == [] and writes == []
            else:
                runner.finish_final(ATTEMPT, output)
                assert len(writes) == 1 and writes[0]['attempts'][0]['status'] == expected
                controls.append({'case': name, 'status': expected, 'bound_retained': True, 'all_provider_and_ledger_operations_mocked': True})


finish_case('exact_attempt_source_owned_release_artifact_positive')
for name, key, value in [('foreign_receipt_attempt', 'attempt', 'cpu-final-999-1'), ('foreign_source', 'source_revision', 'b' * 40), ('foreign_schema', 'schema', 'other')]:
    finish_case(name, lambda f, key=key, value=value: f['receipt'].__setitem__(key, value), 'reject_before_provider')
finish_case('local_live_reservation_identity_differs', lambda f: f['reserved'].__setitem__('bound_usd', '0.01'), 'reject_before_provider')
finish_case('duplicate_live_reservation', lambda f: f['state']['attempts'].append(copy.deepcopy(f['reserved'])), 'reject_before_provider')
finish_case('boolean_ownership_id', lambda f: f['ownership'].__setitem__('release_id', True), 'reject_before_provider')


def foreign_consistent_draft(fixture):
    tag, body = runner.release_identity('cpu-final-999-1', REVISION)
    fixture['receipt']['release_tag'] = tag
    fixture['ownership'].update(tag=tag, body=body)
    fixture['provider'].update(tag_name=tag, body=body)


finish_case('foreign_consistent_owned_release', foreign_consistent_draft, 'reject_before_provider')
for name, key, value in [('provider_draft_false', 'draft', False), ('provider_wrong_source', 'target_commitish', 'b' * 40), ('provider_wrong_ID', 'id', 42), ('provider_wrong_body', 'body', 'other')]:
    finish_case(name, lambda f, key=key, value=value: f['provider'].__setitem__(key, value), 'unresolved')
finish_case('provider_extra_asset', lambda f: f['provider']['assets'].append({'id': 32, 'name': 'foreign.zip', 'size': 1}), 'unresolved')
for name, key, value in [('expired_artifact', 'expired', True), ('oversized_artifact', 'size_in_bytes', runner.transport.MAX_ARTIFACT + 1), ('wrong_artifact_name', 'name', 'other'), ('too_long_retention', 'expires_at', '2027-01-01T00:00:00Z')]:
    finish_case(name, lambda f, key=key, value=value: f['artifact']['artifacts'][0].__setitem__(key, value), 'unresolved')
finish_case('child_process_still_unresolved', lambda f: f['receipt'].__setitem__('all_training_processes_exited', False), 'unresolved')
finish_case('known_stopped_failed_measurement', lambda f: f['receipt'].__setitem__('success', False), 'failed-stopped')

result = {'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
          'controls': controls, 'control_count': len(controls), 'all_expected': True,
          'production_sources_and_tests_edited': False, 'real_provider_or_ledger_mutation': False,
          'model_forward_or_final_run': False,
          'initial_harness_failure_preserved': {'script': 'scope333-cpu-admission-controls-v1.py', 'log': 'scope333-cpu-admission-controls-v1.log', 'reason': 'Reviewer moved fixture ROOT but did not move matching FROZEN; failed before source positive. v2 moves both consistently; production unchanged.'},
          'limits': ['Explicit offline Git/provider/ledger fixtures establish source/admission rejection behavior only. No production workflow, reservation, NN measurement, upload, publication, invoice, or native Windows operation.', 'Source admission used exact98 copied real source bytes and actual committed() code against explicit Git objects. Finish admission uses that checked source result, real finance.finish and actual artifact_bound, with in-memory provider/store.']}
path = Path(__file__).with_name('scope333-cpu-admission-controls-result-v2.json')
with path.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
