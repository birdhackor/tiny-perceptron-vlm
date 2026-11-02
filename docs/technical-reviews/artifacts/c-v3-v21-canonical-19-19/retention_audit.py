"""Audit released immutable evidence; never rerun its model/network/HTTP commands."""

import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
ART = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-19'
OLD = 'docs/technical-reviews/artifacts/omni-technical-19-19-v1/'
OWNER = '/root/omni_technical_coordinator/technical_19_19'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    target = ROOT / path
    digest = sha(target)
    if expected:
        assert digest == expected, (path, digest)
    return json.loads(target.read_bytes()), {'path': path, 'sha256': digest}


def main():
    original, original_ref = read('docs/course-revision-20261101-c-v3/technical-v21/canonical_19_19/original-v1.json', '6a4d251d4f908b70fd0210d7d3621edeb6b8bb28ff3c71966c9371368fe8ad98')
    intake, intake_ref = read('docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-19-19-original-v1.json', '5d3bc327ec771505eb6087242656c885cceda915ec8223894c48df005affbf94')
    assert intake['report'] == original_ref and intake['native_validation'] == 'check_original'
    prior, prior_ref = read('docs/course-revision-20261029-omni/reviews/technical/technical_19_19/technical-original-v1.json', '807e3437e8410e743f141a4ee7702d71e3f4450886c157ceb78cd5c35f57baa4')
    assert prior['reviewer_task'] == OWNER
    execution, execution_ref = read(OLD + 'execution-v1.json', 'f9ccb3539716d27feb60f04592fa351aa356255e5b9100b0fe8109ffd9f13b3f')
    additional, additional_ref = read(OLD + 'additional-probes-v1.json', '464f52f09e9ad86f3b7e83503a1c91b5889a4c5b98b2fe24d2bf07c308d24452')
    inspection, inspection_ref = read(OLD + 'inspection-receipts-v1.json', '65477805b1928d307b8a5cb5fb96e4b21a5b4b903104c44887692d4ce4df8e2f')
    assert inspection['reviewer_task'] == OWNER
    assert execution['source_sha256'] == prior['source_sha256'] == '84f0b11ed6b202131613912f3377508f9b35ddd7456b51347ab19f0e25e54175'
    previous_snapshot = ROOT / prior['source_snapshot']
    assert sha(previous_snapshot) == execution['source_sha256']
    new_probe, _ = read('docs/technical-reviews/artifacts/c-v3-v21-canonical-19-19/probe-result.json')
    assert new_probe['prior_source']['sha256'] == execution['source_sha256']
    assert new_probe['only_delta_is_version_split_and_rng_expansion']

    identities = []
    for row in prior['artifacts'] + [s for s in prior['sources'] if s['kind'] == 'repository_code']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
        identities.append({'path': row['path'], 'sha256': row['sha256']})
    installed = []
    for row in inspection['installed_runtime_source_reads']:
        assert sha(row['path']) == row['sha256']
        installed.append(row)

    release, release_ref = read('tiny_perceptron/lesson_products/release.json')
    commands = re.search(r'```bash\n(.*?)```', previous_snapshot.read_text(), re.S)[1].strip().splitlines()
    assert commands == [r['command'] for r in execution['embedded_commands']]
    assert len(commands) == 5 and all(r['exit_code'] == 0 for r in execution['embedded_commands'])
    assert not any(execution['preexisting_default_packages'].values())
    assert execution['embedded_release_matches']
    package_identity = []
    for i, product in enumerate(('b', 'c')):
        result = execution['embedded_commands'][i]['json']
        entry = release['products'][product]
        assert result == {'repo': release['repo'], 'revision': release['revision'], 'prefix': entry['prefix'], 'product': product, 'manifest_sha256': entry['manifest_sha256'], 'anonymous': True, 'files_verified': 5}
        package = execution['packages'][product]
        directory = Path(package['directory'])
        assert sha(directory / 'artifact-manifest.json') == entry['manifest_sha256']
        manifest = json.loads((directory / 'artifact-manifest.json').read_bytes())
        assert manifest['files'] == package['files'] and len(package['files']) == 5
        for name, row in package['files'].items():
            assert sha(directory / name) == row['sha256']
            assert (directory / name).stat().st_size == row['bytes']
        training = json.loads((directory / 'training-receipt.json').read_bytes())
        assert training['random_initialization'] is True
        assert training['selected_step'] == package['training_receipt_selected_fields']['selected_step']
        assert package['all_values_tensors'] and not package['has_optimizer_or_rng_keys']
        package_identity.append({'product': product, 'directory': str(directory), 'manifest_sha256': entry['manifest_sha256'], 'files': package['files'], 'training_origin': package['training_receipt_selected_fields']})

    solve, judge, auto = [r['json'] for r in execution['embedded_commands'][2:]]
    assert (solve['answer'], solve['loops'], solve['stress_length']) == (2, 3, False)
    assert (judge['question'], judge['selected'], judge['temperature']) == ('info', 'no', 0.25)
    assert judge['probabilities'] == {'yes': 9.26070332716428e-16, 'no': 1.0}
    assert (auto['status'], auto['requested_loops'], auto['b_loops'], auto['choice']['selected'], auto['info']['selected'], auto['solution']['answer']) == ('answered', 3, 3, 'boxes', 'yes', 2)
    offline = execution['offline_guard']
    assert sha(offline['path']) == offline['sha256']
    assert offline['probe']['exit_code'] != 0 and 'REVIEW_OFFLINE_NETWORK_BLOCKED' in offline['probe']['stderr']
    assert len(execution['network_blocked_inference']) == 3
    assert all(r['exit_code'] == 0 for r in execution['network_blocked_inference'])
    cap = execution['extra_cli'][0]['json']
    assert (cap['requested_loops'], cap['b_loops'], cap['status']) == (3, 2, 'budget_limited')
    assert [r['json']['question'] for r in execution['extra_cli'][1:3]] == ['choice', 'rubric']
    assert execution['extra_cli'][3]['exit_code'] != 0
    http = execution['http']
    assert [r['http_status'] for r in http['requests']] == [200, 200, 200, 200, 400]
    assert http['listening_sockets'] and all('0100007F:223D' in line for line in http['listening_sockets'])
    assert sha(http['server_log_path']) == http['server_log_sha256']

    summary, summary_ref = read('docs/course-revision-20261029-omni/evidence/legacy-bc-public-observed-outputs.json')
    historical, historical_ref = read('docs/course-revision-20261029-omni/evidence/legacy-bc-public-examples.json')
    assert summary['execution_receipt'] == historical_ref
    assert summary['release'] == historical['release'] == release
    streams = []
    for row in historical['executions']:
        assert row['exit_code'] == 0
        for name in ('stdout', 'stderr'):
            record = row[name]
            assert sha(record['path']) == record['sha256']
            streams.append({'case': row['case'], 'stream': name, **record})
        assert json.loads(Path(row['stdout']['path']).read_bytes()) == summary['outputs'][row['case']]
    for row in additional['inference_only'].values():
        assert row['eval_mode'] and row['state_keys_equal_parameter_keys']
        assert row['no_registered_buffers'] and row['parameters_unchanged_by_inference']
    assert additional['eval_not_sampling_control']['number_of_distinct_samples'] == 2
    assert '成品檔案指紋不符' in additional['manifest_tamper_rejected']['error']

    excerpt_ranges = {
        'torch-save-load.py': [(54, 65), (166, 178), (289, 300)],
        'torch-randomness-v2.9.0.rst': [(6, 15), (29, 45)],
        'torch-serialization-v2.9.0.rst': [(260, 275)],
        'torch-module-v2.9.0.py': [(2884, 2910)],
        'torch-mps-v2.9.0.rst': [(3, 13)],
        'hf-download.md': [(44, 48), (60, 65)],
        'python-hashlib.rst': [(37, 43), (113, 115)],
        'safetensors-README.md': [(76, 86)],
        'rfc9110.txt': [(384, 388), (756, 762)],
    }
    excerpts = []
    for row in inspection['original_source_reads']:
        if row['status'] != 'ok':
            continue
        assert sha(row['path']) == row['sha256']
        lines = Path(row['path']).read_text().splitlines()
        excerpts.append({'name': row['name'], 'url': row['url'], 'original_sha256': row['sha256'], 'original_accessed_at': row['accessed_at'], 'local_reread_at': datetime.now(UTC).isoformat(), 'excerpts': [{'start_line': start, 'end_line': end, 'text': '\n'.join(lines[start - 1:end])} for start, end in excerpt_ranges[row['name']]]})
    excerpt_path = ART / 'retained-original-excerpts-v1.json'
    with excerpt_path.open('x') as file:
        json.dump({'reviewer_task': original['reviewer_task'], 'created_at': datetime.now(UTC).isoformat(), 'scope': 'Small excerpts reread from SHA-verified local original-source snapshots; no new web fetch; full texts remain outside Git.', 'sources': excerpts}, file, ensure_ascii=False, indent=2)
        file.write('\n')
    result = {
        'created_at': datetime.now(UTC).isoformat(),
        'reviewer_task': original['reviewer_task'],
        'environment': {'python': platform.python_version(), 'device': 'cpu (file inspection)', 'runtime': 'stdlib only; no model import or inference'},
        'original': original_ref, 'intake': intake_ref,
        'prior_report_index_only': prior_ref,
        'retained_execution_owner': OWNER,
        'retained_execution_started_at': execution['started_at'],
        'retained_execution_completed_at': execution['completed_at'],
        'retained_additional_probes_at': additional['created_at'],
        'retained_refs': [execution_ref, additional_ref, inspection_ref],
        'identity_checks': identities, 'installed_source_identity': installed,
        'legacy_release': release_ref,
        'legacy_package_identity': package_identity,
        'five_commands_identity_and_exits': 'Exact match to source18 retained bash fence; five exit=0; original default packages absent.',
        'retained_cpu_outputs_match_current_quoted_fields': True,
        'retained_offline_scope': 'Python audit hook; connect/getaddrinfo/sendto rejected; no OS network namespace claim.',
        'retained_cap_result': {'requested_loops': cap['requested_loops'], 'b_loops': cap['b_loops'], 'status': cap['status']},
        'retained_http_statuses': [r['http_status'] for r in http['requests']],
        'retained_http_bind': '127.0.0.1:8765',
        'author_summary': summary_ref, 'author_receipt': historical_ref,
        'author_raw_stream_identity': streams,
        'original_excerpt_artifact': {'path': str(excerpt_path.relative_to(ROOT)), 'sha256': sha(excerpt_path)},
        'scope': 'New work only parses/hashes retained evidence and rereads source. All historical download, inference, HTTP and sampling executions remain attributed to the 2026-10-29 owner. No network, installation, model inference, HTTP server, training, final or publication was performed. Legacy HF anonymous success does not establish C-v3 public transport.'
    }
    target = ART / 'retention-audit-result-v1.json'
    with target.open('x') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'path': str(target.relative_to(ROOT)), 'sha256': sha(target), 'official_excerpt': result['original_excerpt_artifact'], 'retention_identity_checks': len(identities)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
