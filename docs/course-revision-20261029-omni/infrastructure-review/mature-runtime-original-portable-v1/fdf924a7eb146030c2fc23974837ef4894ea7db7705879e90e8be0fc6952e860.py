"""新 runtime original 原判與可攜索引；絕不改既有 audit／模型／controller。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/work/omni-models-v1')
ACTUAL = Path('/workspace/work/omni-kev-first-preparation-failure-41004743884')
REPORT = ROOT / 'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-v1.json'
INDEX = Path('/workspace/work/omni-mature-runtime-review-portable-evidence-original-v1.json')


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def save_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


assert not REPORT.exists() and not INDEX.exists()
snapshot = read(HERE / 'snapshot-manifest.json')
sdk = read(HERE / 'real-sdk-probe-results.json')
deadlines = read(HERE / 'real-sdk-deadline-results.json')
bootstrap = read(HERE / 'bootstrap-marker-probe-results.json')
inputs = read(HERE / 'input-bindings-results.json')
assert sdk['all_assertions_passed'] and deadlines['all_assertions_passed'] and bootstrap['all_assertions_passed']
for row in snapshot['files']:
    assert binding(row['original_path'])['sha256'] == binding(row['snapshot_path'])['sha256'] == row['sha256']
    committed = subprocess.check_output(['git', 'show', 'ee799d24fd1ea26108af709ec41e49db63d58e31:' + row['path']], cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest() == row['sha256']
for row in read(HERE / 'preserved-prior-reports-manifest.json')['files']:
    assert binding(row['path']) == row
response = read(HERE / 'provider-ledger-response.json')['structuredContent']
ledger = json.loads(response['content'])
rows = [row for row in ledger['attempts'] if row['attempt_id'] == 'mature-kev-41004743884-1']
assert len(rows) == 1
attempt = rows[0]
native_row = read(ROOT / 'outputs/omni-kev-first-preparation-failure-row-v1.json')
assert attempt == native_row and attempt['status'] == 'failed-stopped' and attempt['reserved_usd'] == '4.30'
assert ledger['reserved_total_usd'] == '6.60'
assert sum(Decimal(row['reserved_usd']) for row in ledger['attempts']) == Decimal('6.60')
receipt = attempt['receipt']
assert receipt['phase'] == 'uploading_fixed_input_payload' and receipt['error_type'] == 'TypeError'
assert receipt['success'] is False and receipt['compute_stopped'] is True and receipt['resources_verified'] is True
assert receipt['stopped_sandboxes'] == {'sb-yxbPtKwCRI3u2pImYXT1OC': 3}
assert receipt['remaining_owned_sandboxes'] == [] and receipt['stop_errors'] == {} and all(receipt['cleanup_checks'].values())
assert receipt['app_id'] == 'ap-tb4m5Jl1b8Zwbi3wwVonu7'
assert read(ACTUAL / 'runtime-receipt.json')['error_type'] == 'TypeError'
assert read(ACTUAL / 'runtime-progress.json')['phase'] == receipt['phase']
apps = read(ACTUAL / 'apps-after.json')
assert len(apps) == 1 and apps[0]['app_id'] == receipt['app_id'] and apps[0]['state'] == 'stopped'
release = read(HERE / 'provider-release.json')
assert release['id'] == receipt['release_id'] == 428663042
assert release['tag_name'] == receipt['release_tag'] and release['body'] == receipt['release_body']
assert release['target_commitish'] == attempt['revision'] == 'ee799d24fd1ea26108af709ec41e49db63d58e31'
assert release['draft'] and release['prerelease'] and release['published_at'] is None
assert len(release['assets']) == len(receipt['assets']) == 1
asset = release['assets'][0]
assert asset['id'] == 690960942 and asset['name'] == 'input.zip' and asset['size'] == 434670
assert asset['digest'] == 'sha256:' + inputs['payload_sha256']
assert receipt['assets'][0]['roundtrip_verified'] is True
run = json.loads(read(HERE / 'provider-run-response.json')['structuredContent']['content'])
jobs = read(HERE / 'provider-jobs-response.json')['structuredContent']['jobs']
assert run['id'] == 41004743884 and run['head_sha'] == attempt['revision'] and run['run_attempt'] == 1
assert run['status'] == 'completed' and run['conclusion'] == 'failure'
assert len(jobs) == 1 and jobs[0]['id'] == 123896974095 and jobs[0]['conclusion'] == 'failure'
artifacts = read(HERE / 'provider-artifacts-response.json')['structuredContent']['artifacts']
assert len(artifacts) == 1
artifact = artifacts[0]
assert artifact['id'] == 12954523057 and artifact['size_in_bytes'] == 517015
assert artifact['digest'] == 'sha256:' + binding(ACTUAL / 'original-provider-artifact.zip')['sha256']
native_log = binding(ROOT / 'outputs/omni-kev-first-preparation-failure-41004743884.log')
assert native_log['bytes'] == 41448 and native_log['sha256'] == '107a36a2dfad9673914696d39b5619f762f1c53784b252271e6054cbb01e9d5f'
closed = datetime.now(UTC).isoformat()
projection = {
    'schema': 'omni-mature-runtime-failure-provider-projection-v1', 'created_at': closed,
    'reviewer_task': '/root/omni_development_execution_review', 'all_assertions_passed': True,
    'run': {k: run[k] for k in ('id', 'head_sha', 'run_attempt', 'status', 'conclusion', 'created_at', 'updated_at', 'html_url')},
    'job': {'id': jobs[0]['id'], 'conclusion': jobs[0]['conclusion'], 'steps': jobs[0]['steps']},
    'ledger': {'blob_sha': response['sha'], 'attempt': attempt['attempt_id'], 'status': attempt['status'], 'reserved_usd_fully_retained': attempt['reserved_usd'], 'all_new_bounds_retained_usd': ledger['reserved_total_usd']},
    'receipt': {k: receipt[k] for k in ('phase', 'error_type', 'paid_call_started', 'paid_call_at', 'started_at', 'finished_at', 'app_id', 'sandbox_id', 'success', 'compute_stopped', 'resources_verified', 'cleanup_started_at', 'cleanup_deadline_at', 'cleanup_checks', 'stopped_sandboxes', 'remaining_owned_sandboxes', 'stop_errors')},
    'official_app_stop_projection_in_actual_artifact': apps,
    'draft_release': {k: release[k] for k in ('id', 'tag_name', 'target_commitish', 'draft', 'body', 'published_at')},
    'only_asset': {k: asset[k] for k in ('id', 'name', 'size', 'digest', 'state')},
    'artifact': {k: artifact[k] for k in ('id', 'name', 'size_in_bytes', 'digest', 'created_at', 'expires_at', 'expired')},
    'native_log_fingerprint_only': native_log,
    'no_stdout_result_or_summary_files_retained': True,
    'fresh_modal_app_cli_query': {'exit_code': 1, 'recognized_token_missing': True, 'state_refreshed': False, 'stderr_not_copied_into_portable_evidence': True},
    'limitations': ['Fresh GitHub GET rechecks are independent. Modal stop status is read from the exact original Actions artifacts/ledger, not from a newly authenticated Modal inventory query.', 'A zero current task count or absent stdout/result does not establish historical zero neural inference.'],
}
save_new(HERE / 'provider-diagnosis-results.json', projection)
save_new(HERE / 'final-source-recheck.json', {'created_at': closed, 'reviewer_task': '/root/omni_development_execution_review', 'source_count': 92, 'all92_current_equal_snapshot_prior_audit_and_failed_run_commit': True, 'old5_mature_judgments_unchanged': True, 'reviewer_wrote_author_sources': False, 'reviewer_dispatches_or_ledger_writes': 0})
portable_paths = [
    HERE / 'snapshot-manifest.json', HERE / 'actual-artifact-manifest.json', HERE / 'input-bindings-results.json',
    HERE / 'preserved-prior-reports-manifest.json', HERE / 'probe_real_modal_filesystem.py', HERE / 'real-sdk-probe-results.json', HERE / 'real-sdk-probe.log',
    HERE / 'probe_real_modal_deadlines.py', HERE / 'real-sdk-deadline-results.json', HERE / 'real-sdk-deadline.log',
    HERE / 'probe_bootstrap_marker_gate.py', HERE / 'bootstrap-marker-probe-results.json', HERE / 'bootstrap-marker-probe.log',
    HERE / 'provider-diagnosis-results.json', HERE / 'final-source-recheck.json', HERE / 'seal_original_review.py',
]
portable_paths += sorted((HERE / 'controller-original-private-fixture').glob('*.json'))
portable = [binding(path) for path in portable_paths]
save_new(INDEX, {'schema': 'omni-mature-runtime-original-portable-evidence-v1', 'created_at': closed, 'reviewer_task': '/root/omni_development_execution_review', 'report_path': str(REPORT), 'files': portable, 'file_count': len(portable), 'total_bytes': sum(row['bytes'] for row in portable), 'excluded': ['original/sources full92 source snapshot including 3MB data', 'original/dependency-snapshot third-party SDK full text', 'actual input/Actions ZIP and all original control/full legacy files', 'native original log, credentials, environment dumps, raw provider responses', 'private bootstrap extracted source/data fixtures'], 'location_map_instruction': '保留原 path/SHA，以 location-map 對應小型自製證據的新保存位置；不把 full snapshot／第三方全文／大型 ZIP 加入普通 Git。'})
plan_path = Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261029-omni/PLAN.md')
save_new(REPORT, {
    'schema': 'omni-mature-research-runtime-review-original-v1', 'created_at': closed,
    'author_task': '/root', 'reviewer_task': '/root/omni_development_execution_review',
    'status': 'changes_required', 'verdict': '真 Modal1.6.0 介面參數順序缺陷已重現；現版不得再用於付費研究。原失敗完整保留，不替換舊原判／audit。',
    'scope': {'failure_attempt': attempt['attempt_id'], 'run_id': run['id'], 'job_id': jobs[0]['id'], 'failed_revision': attempt['revision'], 'source_count': 92, 'source_audit_reference': snapshot['source_audit'], 'science_sources_changed': False, 'model_data_eval57_training25_implementation8_changed': False, 'reviewer_paid_calls': 0, 'reviewer_ledger_writes': 0, 'reviewer_nn_predictions': 0, 'exclusions': ['自家真 final 結果判定（由另一 owner 審）', '成熟 teacher 能力評估', '任何新派工授權', 'P5 教材審閱']},
    'source_snapshot': binding(HERE / 'snapshot-manifest.json'),
    'files': {row['path']: row['sha256'] for row in snapshot['files']},
    'necessary_findings': [{
        'id': 'MR06', 'severity': 'blocking',
        'summary': 'staged input 的兩次 write_bytes 把 data 與 remote_path 位置傳反；真正 SDK 在第一個 filesystem exec/RPC 前 TypeError。',
        'locations': ['scripts/omni_revision/staged_execution.py:139', 'scripts/omni_revision/staged_execution.py:141', 'tests/test_omni_mature_research.py:183'],
        'actual_modal_contract': {'internal_type': '_SandboxFilesystem', 'write_signature': '(data: bytes | bytearray | memoryview, remote_path: str) -> None', 'read_signature': '(remote_path: str) -> bytes', 'internal_async_methods': True, 'internal_aio_attribute': False, 'public_namespace': 'SandboxFilesystem 同步介面；非同步呼叫使用其 .aio。'},
        'production_path': 'research_execution.execute 傳 internal _Sandbox，run 以 synchronizer.create_blocking(execute) 進同一 SDK loop；filesystem raw async 方法可直接 await/hard_wait。',
        'observed_original_failure': {'phase': receipt['phase'], 'error_type': receipt['error_type'], 'input_payload_uploaded_present': False},
        'real_sdk_reproduction': sdk['interface_and_roundtrip']['inverted_original_write'],
        'real_original_controller_reproduction': sdk['original_controller_reproduction'],
        'required_fix': ['兩次寫入明示 data=payload_or_digest、remote_path=absolute_path；不要在 internal raw 方法盲加 .aio。', '保留 raw read_bytes 的正確 await；若改 public Sandbox 型別，必須另核所有 .aio/同步介面與 deadline，不能混用。', '加入真正 modal==1.6.0 _Sandbox._new_hydrated/property/body 在 production matching synchronizer loop 的 offline contract regression；只 mock exec/RPC 邊界。', '原 fake.write_bytes(self,path,payload) 的簽名也與 SDK 相反；須修其簽名，使位置寫反的 original 在回歸測試中必敗。先前 source audit 未驗到這個真 SDK 契約缺口，本新原判保留此事實。', '修正後另建 current 與新 audit／來源 freeze；舊 mature-research-audit-v1 與所有 original/current 原 bytes 留存。'],
        'recheck_tests': ['原兩次位置參數的第一寫 TypeError／zero FS exec 負例', '真 SDK named data/remote_path 寫2讀2內容與 SHA 正例', 'raw async 與 public .aio 型別對照', 'write/read timeout 與取消、non-bytes／relative path 在 RPC 前拒絕', '原 controller 首寫失敗仍逐已知 ID 停止，無 marker／stdout/result 成功旗標'],
    }],
    'optional_findings': [{
        'id': 'O-runtime-01', 'summary': 'uploading_fixed_input_payload 同時涵蓋 payload 與 SHA marker 寫入，且只保存 error_type；對未來不同失敗不足以直接確認哪個外部動作已完成。',
        'suggestion': '可在各寫入前後保存不同 phase、已知 payload SHA／大小、marker_acknowledged 或明示 unknown；錯誤只存安全類型／已遮罩短訊息，勿存 credentials 或 payload全文。任何 marker 回應不明仍不得聲稱 worker 未開始。',
    }],
    'diagnosis': {
        'proven': ['原92檔在 snapshot、audit51f2、failed commit ee799d24 與封存前 current 逐bytes相同。', '官方 Actions run/job completed/failure；artifact ZIP 官方 digest、18原 entry 與取回檔相同；耐久帳唯一此 attempt failed-stopped，4.30全留，總6.60。', '第一個 write_bytes 的 remote_path 實參是 bytes payload；真 SDK PurePosixPath 驗證必在任何 _Sandbox.exec 之前 TypeError。', '真原 controller + 真 internal SDK 的受控 RPC-boundary 重現停在同一 phase，filesystem exec 0次，沒有到第二次 SHA marker 寫入或 stdout await；已知 ID 停止仍完整通過。', 'bootstrap 在有效 marker、精確 payload SHA/bytes 和安全 ZIP 驗證之後才呼叫 research_worker；私有 empty／只有payload／錯marker 負例均不呼叫 worker，完整 identity 正例只記錄 stub 呼叫。', 'owned draft 官方目前只有 input.zip；沒有 result.zip 或 worker output 被保存。'],
        'not_proven': ['沒有直接讀到已停止 provider Sandbox 的初始 /tmp 狀態或歷史 research_worker／teacher NN 執行事件。', '因此不把無 stdout/result、current tasks=0、phase 或本機 fixture 單獨稱為 provider 的實測零 NN。'],
        'sha_marker_write_reached_by_reviewed_controller_path': False,
        'filesystem_exec_reached_by_inverted_first_write': False,
        'bootstrap_process_requested_at_SandboxCreate': True,
        'research_worker_path_requires_valid_payload_marker': True,
        'provider_research_worker_started_directly_observed': None,
        'provider_teacher_nn_executed_directly_observed': None,
        'causal_classification': 'deterministic pre-marker preparation failure of the reviewed source path; not an empirical provider zero-NN telemetry measurement',
        'read_bytes_same_bug': False,
        'read_bytes_explanation': '此 production 使用 internal _Sandbox，read_bytes 是 raw coroutine、參數正確，真 SDK 讀回正例及 deadline 負例均過；沒有理由因本次 TypeError 去改兩個 read_bytes。',
    },
    'actual_failure_evidence': binding(HERE / 'provider-diagnosis-results.json'),
    'dependency_snapshot': {'offline_sdk': sdk['sdk'], 'python': sdk['python'], 'production_matching_loop': sdk['production_matching_loop_entry'], 'sdk_sources': sdk['source_manifest'], 'path_validator': deadlines['path_validator_source'], 'full_thirdparty_sdk_snapshot_ordinary_git': False, 'native_transitive_versions_independently_observed': False},
    'execution_boundaries': {'real_sdk_corrected_named_args_roundtrip': True, 'real_original_controller_negative': True, 'real_sdk_write_read_timeout_cancel_negative': True, 'simulated_exec_cancellation_is_not_remote_termination_proof': True, 'paid_SandboxCreate_before_this_error': True, 'unknown_or_failed_bounds_refunded': False},
    'controlled_recovery_assessment': {
        'status': 'conditional_design_only_no_dispatch_approval',
        'authorization_reference': {**binding(plan_path), 'short_excerpts': [{'line': 100, 'text': '失敗無自動重試，新attempt另占已批准上界，不刪核心final換成功。'}, {'line': 150, 'text': '失敗／修正／收尾保留15；不能用保留款偷偷加工作範圍。'}], 'session_context': 'root 指示承接使用者已批准的新60美元 round；本文不重設舊 round 或把保留款擴成新科學工作。'},
        'eligibility': '可提出精確限縮的準備修復新 attempt；必須在任何新預約／派工前由獨立 current/audit 驗收修正、舊失敗前置與一次性 permit。現有 general once-only 拒絕繼續生效。',
        'required_controls': [
            '舊 attempt 原 bytes、4.30 bound、全停止證據、原source audit／runtime original與owned input release 全保留；不刪、不改 completed/failed 標籤、不退款、不以舊 ID 重派。',
            '預設同 job 任何既有 attempt 一律拒絕。只另加明示、獨立簽核、只對 mature-kev-41004743884-1 的一次性 preparation-repair permit；不能移除一般 once-only 或允許任意 force/retry。',
            'permit 逐項綁舊 failed-stopped status、source92／SDK證據、原phase/error、已知App/Sandbox／完整停止、原receipt與input asset SHA、沒有任何新嘗試；若出現 worker開封／marker送達回應不明等新證據，先 fail closed。前置無 worker 的判斷必須明示是本次確定的 pre-marker source-path 證明及其限制，不能用單純無輸出替代。',
            '建議從原owned asset精確取回並重用 input.zip（434670B/8652c384...）及其 job.json（73f751...），逐entry綁 original worker62、Kev pin、data/model-selection/final-opening、全部規格；只修 controller 準備層。現 prepare 使用 ZIP mtime，新打包即使內容同也不保證同SHA，因此不得假稱新 ZIP 必然同bytes。',
            '新來源閉包完整凍結且已提交，另建新 audit 路徑並核所有新 permit/helper/workflow/test 的 transitive source；原92科學條件、57eval/25training/8impl、資料／模型／參數／Kev FP32 L40S4CPU48GiB／全部case與生成次數不變。不能重新挑 T／tau、換 head/model/pin 或重跑自家final。',
            '新唯一 run/attempt、App/ID/tags/draft，不覆蓋舊資產。耐久 purpose 保持 job=mature-kev，增加 recovery_of／permit SHA／新audit SHA／原input與科學條件 SHA；查全 ledger 防同permit與第二次修復，即使第一次修復失敗也不能再自動例外。',
            '先新鮮唯讀 billing/rates/inventory＋strict core/selection/opening 與來源核對，再按原4.30 full gross bound 做原CAS；6.60舊總額不減，若原費率／保護條件仍成立新保留總額至少10.90。官方increment/舊carry與Qwen3.10、final/publication contingency仍按原守門重新計算，不把此數字當當前報價。',
            '沿用唯一 raw SandboxCreate/no retry、已知ID先停、未知保持unresolved、所有deadline、先匯出再停、官方App停止與owned draft真roundtrip／大小保存界；不能把正確 FS coroutine 的 timeout/cancel 當作已遠端停止。',
            'Qwen 尚未嘗試，因共用同一 staging 缺陷也須先綁新已審 staging 來源；本原判不派 Qwen、不准雙跑，也不把修復準備成功等同研究完成或能力通過。',
        ],
    },
    'portable_evidence': binding(INDEX),
    'evidence': [binding(HERE / name) for name in ('real-sdk-probe-results.json', 'real-sdk-deadline-results.json', 'bootstrap-marker-probe-results.json', 'actual-artifact-manifest.json', 'input-bindings-results.json', 'preserved-prior-reports-manifest.json', 'final-source-recheck.json')],
    'limitations': ['原 native log 有 masked credential bindings，只核41448B／SHA107a36...，沒有將全文、環境或token ID放進報告或portable證據。', '本機新唯讀Modal app list因token missing失敗；新鮮遠端核對限GitHub run／job／artifact／ledger／draft metadata，原Modal停止資料以實際artifact與耐久帳來源限定。', '所有 SDK transport/exec 與 bootstrap worker positive 都是明示 private fixtures；不是實際 GPU/NN 正控制。', '此新 original changes_required 不覆寫或變造任何舊 MR 原判/current/audit；新 paid 使用須修正後另回核。'],
})
print(json.dumps({'report': binding(REPORT), 'portable_index': binding(INDEX), 'portable_file_count': len(portable), 'portable_total_bytes': sum(row['bytes'] for row in portable), 'necessary_findings': ['MR06'], 'status': 'changes_required'}, ensure_ascii=False))
