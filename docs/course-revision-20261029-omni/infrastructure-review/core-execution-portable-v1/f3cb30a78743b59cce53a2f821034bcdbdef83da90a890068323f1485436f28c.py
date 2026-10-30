"""獨立唯讀核對；輸入為本 reviewer 實取的 provider 回應和原始 ZIP。"""

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/work/omni-models-v1')
INPUT = Path('/workspace/work/omni-core-completed-40988311100')
ATTEMPT = 'cpu-core-40988311100-1'
REVISION = 'f9310888eaa95e79156ffddea122a8c9b1d6beb5'
AUDIT_PATH = 'docs/course-revision-20261029-omni/infrastructure/formal-core-audit-v1.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def response(name, wrapped=False):
    value = read(HERE / name)
    if wrapped:
        assert value['status'] == 'fulfilled'
        value = value['value']
    assert not value.get('isError')
    return value['structuredContent']


def utc(value):
    return datetime.fromisoformat(value)


started = datetime.now(UTC).isoformat()
run = json.loads(response('provider-run-response.json', True)['content'])
jobs = response('provider-jobs-response.json')['jobs']
artifacts = response('provider-artifacts-response.json')['artifacts']
assert run['id'] == 40988311100 and run['run_attempt'] == 1
assert run['head_sha'] == REVISION and run['head_branch'] == 'main'
assert run['event'] == 'workflow_dispatch' and run['status'] == 'completed' and run['conclusion'] == 'success'
assert len(jobs) == 1 and jobs[0]['id'] == 123844407710
assert jobs[0]['status'] == 'completed' and jobs[0]['conclusion'] == 'success'
assert all(s['status'] == 'completed' and s['conclusion'] == 'success' for s in jobs[0]['steps'])
ledger_response = response('provider-ledger-response.json')
ledger_bytes = ledger_response['content'].encode()
assert hashlib.sha1(b'blob ' + str(len(ledger_bytes)).encode() + b'\0' + ledger_bytes).hexdigest() == ledger_response['sha']
ledger = json.loads(ledger_bytes)
attempts = [r for r in ledger['attempts'] if json.loads(r['purpose']).get('job') == 'cpu-core-formal']
assert len(attempts) == 1
attempt = attempts[0]
assert attempt['attempt_id'] == ATTEMPT and attempt['revision'] == REVISION and attempt['status'] == 'completed'
assert attempt['reserved_usd'] == '0.10' and ledger['reserved_total_usd'] == '2.20'
assert sum(Decimal(r['reserved_usd']) for r in ledger['attempts']) == Decimal('2.20')
receipt = attempt['receipt']
assert all(receipt[k] is True for k in ('success', 'resources_verified', 'compute_stopped', 'all_training_processes_exited', 'no_modal_compute_created'))
assert receipt['final_opened'] is False and all(receipt['cleanup_checks'].values())
assert receipt['source_revision'] == REVISION and receipt['attempt'] == ATTEMPT
assert len(artifacts) == 1
artifact = artifacts[0]
assert artifact['id'] == 12952460790 and artifact['name'] == 'omni-core-40988311100-1'
assert not artifact['expired'] and artifact['workflow_run']['head_sha'] == REVISION
retention = utc(artifact['expires_at']) - utc(artifact['created_at'])
assert timedelta(days=6, hours=23) < retention <= timedelta(days=7)
assert artifact['size_in_bytes'] <= 256 * 1024**2
assert {k: artifact[k] for k in receipt['artifact_api_records'][0]} == receipt['artifact_api_records'][0]
release = read(HERE / 'provider-release-gh.json')
assert release['id'] == receipt['release_id'] == 428573191
assert release['tag_name'] == receipt['release_tag'] == 'omni-core-candidate-' + ATTEMPT
assert release['draft'] is True and release['prerelease'] is True and release['published_at'] is None
assert release['target_commitish'] == REVISION
assert release['body'] == f'Owned unpublished core weights and full optimizer/RNG; {ATTEMPT}; source {REVISION}'
assert len(release['assets']) == len(receipt['assets']) == 11
official_assets = {r['id']: r for r in release['assets']}
assert len(official_assets) == 11
for r in receipt['assets']:
    official = official_assets[r['id']]
    assert r['roundtrip_verified'] is True and official['state'] == 'uploaded'
    assert (official['name'], official['size'], official['digest']) == (r['name'], r['bytes'], 'sha256:' + r['sha256'])
aggregate = next(r for r in receipt['assets'] if r['name'] == 'core-checkpoints.zip')
aggregate_bytes = (INPUT / 'core-checkpoints.zip').read_bytes()
assert len(aggregate_bytes) == aggregate['bytes'] == 54751619
assert sha(aggregate_bytes) == aggregate['sha256'] == '107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee'
assert receipt['package'] == {'bytes': len(aggregate_bytes), 'sha256': sha(aggregate_bytes)}
assert sha((INPUT / 'retrieval-receipt-v1.json').read_bytes()) == 'db266e3f7111ca9040833b41043f771156cedd9236c2d36183e5ac343c20cfbd'

actions_bytes = (HERE / 'actions-artifact.zip').read_bytes()
assert len(actions_bytes) == artifact['size_in_bytes']
assert 'sha256:' + sha(actions_bytes) == artifact['digest']
with zipfile.ZipFile(HERE / 'actions-artifact.zip') as az:
    assert len(az.infolist()) == len(set(az.namelist())) == 25
    assert az.read('core-checkpoints.zip') == aggregate_bytes
    sources = json.loads(az.read('sources.json'))
    reserved = json.loads(az.read('reservation.json'))
    protection = json.loads(az.read('core-protection.json'))
    cost = json.loads(az.read('cost-proof.json'))
    worker_receipt = json.loads(az.read('cpu-receipt.json'))
    ownership = json.loads(az.read('release-ownership.json'))
    assert all(worker_receipt[k] == receipt[k] for k in worker_receipt)
    assert ownership['release_id'] == release['id'] and ownership['body'] == release['body']
    assert sources['revision'] == REVISION
    audit = read(ROOT / AUDIT_PATH)
    expected = {**audit['reviewed_files_sha256'], AUDIT_PATH: sha((ROOT / AUDIT_PATH).read_bytes())}
    assert len(audit['reviewed_files_sha256']) == 36 and sources['files'] == expected
    source_rows = []
    for name, digest in expected.items():
        current = (ROOT / name).read_bytes()
        committed = subprocess.check_output(['git', 'show', REVISION + ':' + name], cwd=ROOT)
        assert not (ROOT / name).is_symlink() and sha(current) == sha(committed) == digest
        source_rows.append({'path': name, 'sha256': digest, 'bytes': len(current)})
    purpose = json.loads(attempt['purpose'])
    assert purpose['sources_sha256'] == sha(canonical(sources))
    assert purpose['cost_proof_sha256'] == sha(canonical(cost))
    assert reserved['attempt']['reservation_sha256'] == attempt['reservation_sha256']
    assert reserved['attempt']['status'] == 'reserved'
    assert protection['guarded_total_usd'] == '58.63000000' and protection['cap_usd'] == '60'
    assert Decimal(protection['future_bounds_usd']) >= Decimal('22.80')
    assert cost['public_standard_runner_compute']['repository_private'] is False
    assert cost['storage_and_cleanup_bound'] == '0.10' and cost['retention_days'] == 7
    reservation_blob_response = response('provider-reservation-blob.json')
    reservation_blob = reservation_blob_response['content'].encode()
    blob_hash = hashlib.sha1(b'blob ' + str(len(reservation_blob)).encode() + b'\0' + reservation_blob).hexdigest()
    assert blob_hash == reserved['transaction']['blob'] == '2f942d8f0a8ca2ea3503e991affbf56e8799099c'
    early_ledger = json.loads(reservation_blob)
    early_attempt = next(r for r in early_ledger['attempts'] if r['attempt_id'] == ATTEMPT)
    assert early_attempt == reserved['attempt']
    commit = json.loads(response('provider-reservation-commit.json')['content'])
    assert commit['sha'] == reserved['transaction']['commit'] == 'a2e08e577a28ab394d00f96afecc1ae0d30abe9a'
    assert utc(attempt['reserved_at']) <= utc(receipt['started_at'])
    control = [{'path': i.filename, 'bytes': i.file_size, 'sha256': sha(az.read(i))} for i in az.infolist() if not i.filename.endswith('.zip')]

archive_rows = []
with zipfile.ZipFile(INPUT / 'core-checkpoints.zip') as archive:
    infos = archive.infolist()
    assert len(infos) == len({x.filename for x in infos}) == 80
    manifest = json.loads(archive.read('package-file-manifest.json'))
    assert set(manifest) == set(archive.namelist()) - {'package-file-manifest.json'}
    for item in infos:
        name = item.filename
        assert not name.startswith('/') and '..' not in Path(name).parts
        raw = archive.read(item)
        actual = {'bytes': len(raw), 'sha256': sha(raw)}
        extracted = INPUT / 'verified' / name
        assert not extracted.is_symlink() and extracted.read_bytes() == raw
        if name != 'package-file-manifest.json':
            assert manifest[name] == actual
            if name.startswith('core/'):
                installed = ROOT / 'outputs/omni-formal-core-v1' / name.removeprefix('core/')
            elif name.startswith('frozen/'):
                installed = ROOT / 'docs/omni-products/v1' / name.removeprefix('frozen/')
            else:
                assert name.startswith('selection-evidence/')
                installed = ROOT / 'outputs/omni-core-selection-v1' / name.removeprefix('selection-evidence/')
            assert not installed.is_symlink() and installed.read_bytes() == raw
        archive_rows.append({'path': name, **actual})

log = (HERE / 'provider-job-full.log').read_text()
pending, logged_jobs, training_excerpt = [], {}, []
for line in log.splitlines():
    _, _, payload = line.partition(' ')
    if not payload.startswith('{'):
        continue
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        continue
    if 'update' in value and 'dev_score' in value:
        pending.append(value)
        training_excerpt.append(line)
    elif 'job' in value and 'training_result_sha256' in value:
        assert value['job'] not in logged_jobs and value['success'] is True
        name = value['job']
        tr = read(ROOT / 'outputs/omni-formal-core-v1' / name / 'training/training-result.json')
        assert value['training_result_sha256'] == sha((ROOT / 'outputs/omni-formal-core-v1' / name / 'training/training-result.json').read_bytes())
        assert pending == read(ROOT / 'outputs/omni-formal-core-v1' / name / 'training/training-log.json')
        logged_jobs[name] = {'rows': len(pending), 'last_update': pending[-1]['update'], 'training_result_sha256': value['training_result_sha256']}
        pending = []
        training_excerpt.append(line)
assert not pending and set(logged_jobs) == set(read(ROOT / 'docs/omni-products/v1/formal-recipes.json')['jobs'])
(HERE / 'own-training-log-excerpt.log').write_text('\n'.join(training_excerpt) + '\n')
result = {
    'schema': 'omni-core-provider-artifact-independent-verification-v1',
    'reviewer_task': '/root/omni_development_execution_review',
    'started_at': started,
    'finished_at': datetime.now(UTC).isoformat(),
    'all_assertions_passed': True,
    'run': {k: run[k] for k in ('id', 'run_attempt', 'head_sha', 'head_branch', 'status', 'conclusion', 'created_at', 'updated_at', 'html_url')},
    'job': jobs[0],
    'ledger': {'blob_sha': ledger_response['sha'], 'sha256': sha(ledger_bytes), 'attempt_status': attempt['status'], 'reservation_commit': commit['sha'], 'reservation_blob': blob_hash, 'reserved_usd_retained': attempt['reserved_usd'], 'all_new_bounds_retained_usd': ledger['reserved_total_usd'], 'guarded_total_usd': protection['guarded_total_usd'], 'future_bounds_usd': protection['future_bounds_usd']},
    'release': {k: release[k] for k in ('id', 'tag_name', 'draft', 'prerelease', 'target_commitish', 'body', 'published_at')},
    'assets': [{k: r[k] for k in ('id', 'name', 'size', 'digest', 'state')} for r in release['assets']],
    'actions_artifact': {k: artifact[k] for k in ('id', 'name', 'size_in_bytes', 'digest', 'expired', 'created_at', 'expires_at')},
    'aggregate': {'bytes': len(aggregate_bytes), 'sha256': sha(aggregate_bytes), 'entries': len(archive_rows), 'expanded_bytes': sum(r['bytes'] for r in archive_rows), 'all_entries_and_installed_bytes_match': True, 'equals_independently_downloaded_actions_inner_zip': True},
    'original_source36_plus_audit': source_rows,
    'actions_control_file_manifest': control,
    'provider_training_log_matches_all10_job_logs': logged_jobs,
    'provider_full_log_sha256': sha(log.encode()),
    'limits': ['No independent invoice settlement is inferred.', 'Ten individual draft ZIP bodies were not downloaded again; official SHA256 digests and ledger bindings match all 11 assets. The independently downloaded Actions inner aggregate equals the local aggregate byte for byte.', 'Completed hosted job plus captured exit receipts establish operational completion; no post-job host process inspection is available.'],
}
(HERE / 'provider-archive-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'all_assertions_passed': True, 'provider_run': run['id'], 'assets': 11, 'archive_entries': 80, 'source_files_plus_audit': len(source_rows), 'training_log_rows': sum(r['rows'] for r in logged_jobs.values())}))
