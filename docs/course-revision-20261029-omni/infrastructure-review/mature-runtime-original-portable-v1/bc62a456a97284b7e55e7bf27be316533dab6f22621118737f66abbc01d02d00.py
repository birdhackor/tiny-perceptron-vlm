"""真 ee799d finish_run + finish；真 owned foreign core 成功 receipt/GET資料，只 mock遠端邊界。"""

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path('/workspace/work/omni-models-v1')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'original'
ACTUAL = Path('/workspace/work/omni-kev-first-preparation-failure-41004743884')
import torch
from scripts.omni_revision import research_execution as runner
from scripts.omni_revision.finance import digest, reservation_identity, validate_state

assert hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest() == '12fed26657e9eee4de565de96948e00f312e30bf99aa11d4ab7c449b4a2f4b36'
started = datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


ledger = json.loads(read(ORIGINAL / 'provider-ledger-response.json')['structuredContent']['content'])
target_reservation = read(ACTUAL / 'reservation.json')['attempt']
target = target_reservation['attempt_id']
core_row = next(row for row in ledger['attempts'] if row['attempt_id'] == 'cpu-core-40988311100-1')
assert core_row['status'] == 'completed' and core_row['receipt']['success'] is True
foreign = copy.deepcopy(core_row['receipt'])
core_release = read(HERE / 'provider-foreign-core-release.json')
kev_release = read(ORIGINAL / 'provider-release.json')
assert core_release['id'] == foreign['release_id'] == 428573191
assert core_release['draft'] is True and core_release['tag_name'] == foreign['release_tag']
assert {(a['id'], a['name'], a['size']) for a in core_release['assets']} == {(a['id'], a['name'], a['bytes']) for a in foreign['assets']}
# CPU receipt原本不重複保存body；僅加入真正 provider owned draft 的原body，沒有改成功／停止／資產旗標。
foreign['release_body'] = core_release['body']
assert {k: v for k, v in foreign.items() if k != 'release_body'} == core_row['receipt']
artifact_records = read(HERE / 'provider-target-artifacts.json')
assert artifact_records['total_count'] == len(artifact_records['artifacts']) == 1
assert artifact_records['artifacts'][0]['name'] == 'omni-' + target

# 重建真 Kev 執行「預約已持久、尚未finish」的狀態；不新增／刪除預約或改保留額。
before = copy.deepcopy(ledger)
row = next(row for row in before['attempts'] if row['attempt_id'] == target)
assert reservation_identity(row) == reservation_identity(target_reservation)
row.clear()
row.update(copy.deepcopy(target_reservation))
assert row['status'] == 'reserved' and row['reservation_sha256'] == digest(reservation_identity(row))
validate_state(before, datetime.now(UTC))
assert before['reserved_total_usd'] == '6.60'

network_attempts, nn_attempts = [], []


def no_network(*args, **kwargs):
    network_attempts.append('forbidden')
    raise AssertionError('all provider/ledger side effects forbidden in this probe')


def no_forward(module, inputs):
    nn_attempts.append(type(module).__name__)
    raise AssertionError('NN forward forbidden')


def run_case(name, supplied_receipt, tamper_reservation=False):
    folder = HERE / name
    folder.mkdir(exist_ok=False)
    private_reservation = copy.deepcopy(target_reservation)
    if tamper_reservation:
        private_reservation['revision'] = '0' * 40
    (folder / 'reservation.json').write_text(json.dumps({'attempt': private_reservation}) + '\n')
    (folder / 'research-receipt.json').write_text(json.dumps(supplied_receipt) + '\n')
    calls, writes = [], []

    def get_api(endpoint, *args, **kwargs):
        assert not args and not kwargs, 'unexpected API mutation arguments'
        calls.append(endpoint)
        if endpoint == 'actions/runs/41004743884/artifacts?per_page=100':
            return copy.deepcopy(artifact_records)
        if endpoint == 'releases/428573191':
            return copy.deepcopy(core_release)
        if endpoint == 'releases/428663042':
            return copy.deepcopy(kev_release)
        raise AssertionError('unrecognized provider GET')

    class LedgerRecorder:
        def read(self):
            return copy.deepcopy(before), 'private-observed-blob'

        def write(self, following, blob):
            assert blob == 'private-observed-blob'
            validate_state(following, datetime.now(UTC))
            writes.append(copy.deepcopy(following))
            return {'commit': 'private-recorder-only', 'blob': 'private-recorder-only'}

    error = None
    environment = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'birdhackor/tiny-perceptron-vlm', 'GITHUB_REF': 'refs/heads/main', 'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_RUN_ID': '41004743884', 'GITHUB_RUN_ATTEMPT': '1', 'MODAL_SANDBOX_V2': '0', 'MODAL_ENVIRONMENT': 'main', 'OMNI_ARTIFACT_OUTCOME': 'success'}
    with patch.dict(os.environ, environment), patch.object(runner, 'api', side_effect=get_api), patch.object(runner, 'GitHubLedger', LedgerRecorder), patch.object(socket.socket, 'connect', new=no_network), patch.object(subprocess, 'run', new=no_network):
        try:
            runner.finish_run('kev', target, folder)
        except ValueError as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
    completed = next(row for row in writes[0]['attempts'] if row['attempt_id'] == target) if writes else None
    return {'case': name, 'provider_gets': calls, 'real_provider_calls': 0, 'ledger_recorder_write_calls': len(writes), 'real_ledger_write_calls': 0, 'error': error, 'result_status': completed['status'] if completed else None, 'result_outer_attempt': completed['attempt_id'] if completed else None, 'result_receipt_attempt': completed['receipt'].get('attempt') if completed else None, 'result_receipt_schema': completed['receipt'].get('schema') if completed else None, 'result_receipt_source_revision': completed['receipt'].get('source_revision') if completed else None, 'result_receipt_release_tag': completed['receipt'].get('release_tag') if completed else None, 'result_receipt_release_body': completed['receipt'].get('release_body') if completed else None, 'reserved_total_usd_unchanged': writes[0]['reserved_total_usd'] == before['reserved_total_usd'] if writes else True, 'result_receipt_success': completed['receipt']['success'] if completed else None}


guard = torch.nn.modules.module.register_module_forward_pre_hook(no_forward)
try:
    exploit = run_case('foreign-genuine-core-success', foreign)
    assert exploit['error'] is None and exploit['ledger_recorder_write_calls'] == 1 and exploit['result_status'] == 'completed'
    assert exploit['result_outer_attempt'] == target and exploit['result_receipt_attempt'] != target
    assert exploit['result_receipt_schema'] == 'omni-cpu-core-execution-v1'
    assert exploit['result_receipt_source_revision'] != target_reservation['revision']
    own_failure = run_case('matching-genuine-kev-failure-control', read(ACTUAL / 'research-receipt.json'))
    assert own_failure['error'] is None and own_failure['result_status'] == 'failed-stopped'
    bad_reservation = run_case('reservation-tamper-control', foreign, True)
    assert bad_reservation['error']['type'] == 'ValueError' and bad_reservation['ledger_recorder_write_calls'] == 0
finally:
    guard.remove()
assert not network_attempts and not nn_attempts
result = {'schema': 'omni-mature-runtime-mr07-original-bounded-probe-v1', 'created_at': datetime.now(UTC).isoformat(), 'started_at': started, 'reviewer_task': '/root/omni_development_execution_review', 'source_revision': 'ee799d24fd1ea26108af709ec41e49db63d58e31', 'all_assertions_passed': True, 'confirmed_foreign_genuine_owned_success_receipt_can_complete_another_reserved_row': True, 'foreign_receipt_provenance': {'receipt_attempt': core_row['attempt_id'], 'genuine_durable_status': core_row['status'], 'genuine_success_resources_assets_flags_changed': False, 'only_added_field': 'release_body copied exactly from the genuine owned draft GET', 'original_receipt_canonical_sha256': digest(core_row['receipt']), 'genuine_owned_release_id': core_release['id'], 'release_response_sha256': hashlib.sha256((HERE / 'provider-foreign-core-release.json').read_bytes()).hexdigest()}, 'target_reserved_attempt': target, 'target_source_revision': target_reservation['revision'], 'target_reservation_sha256': target_reservation['reservation_sha256'], 'target_state': 'private reconstruction of actual reserved pre-finish row; live ledger remains failed-stopped', 'cases': [exploit, own_failure, bad_reservation], 'patched_boundaries_only': ['runner.api returns real recorded GitHub GET metadata', 'runner.GitHubLedger reads a private pre-finish state and only records writes'], 'original_code_executed': ['research_execution.identity', 'research_execution.bound', 'research_execution.finish_run', 'finance.reservation_identity', 'finance.finish', 'finance.validate_state before and on recorder writes'], 'network_attempts': network_attempts, 'nn_forward_attempts': nn_attempts, 'real_modal_rpc_calls': 0, 'real_dispatch_calls': 0, 'real_ledger_writes': 0, 'real_release_writes': 0, 'limits': ['No actual mature-model success receipt exists; the foreign successful receipt is the genuine completed own CPU-core execution, accepted without schema/kind/attempt/source binding by mature finish_run.', 'No live ledger transition was attempted. The positive transition is captured only by the local ledger recorder.']}
(HERE / 'mr07-probe-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'all_assertions_passed': True, 'foreign_genuine_core_success_target_status': exploit['result_status'], 'outer_attempt': exploit['result_outer_attempt'], 'foreign_receipt_attempt': exploit['result_receipt_attempt'], 'matching_failed_control': own_failure['result_status'], 'tamper_control_ledger_writes': bad_reservation['ledger_recorder_write_calls'], 'real_ledger_writes': 0}))
