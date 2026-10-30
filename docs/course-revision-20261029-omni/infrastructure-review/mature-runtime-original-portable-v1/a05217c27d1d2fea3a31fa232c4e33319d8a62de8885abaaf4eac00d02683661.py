"""封存新的 bounded MR07 original；不改 MR06 或其他舊判定。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-models-v1')
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'original'
REPORT = ROOT / 'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-mr07-v1.json'
INDEX = Path('/workspace/work/omni-mature-runtime-review-portable-evidence-mr07-original-v1.json')


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def save_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


assert not REPORT.exists() and not INDEX.exists()
result = read(HERE / 'mr07-probe-results.json')
assert result['all_assertions_passed'] and result['confirmed_foreign_genuine_owned_success_receipt_can_complete_another_reserved_row']
snapshot = read(ORIGINAL / 'snapshot-manifest.json')
for item in snapshot['files']:
    assert binding(item['original_path'])['sha256'] == item['sha256']
    if item['path'] in ('scripts/omni_revision/research_execution.py', 'scripts/omni_revision/finance.py'):
        raw = subprocess.check_output(['git', 'show', 'ee799d24fd1ea26108af709ec41e49db63d58e31:' + item['path']], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
mr06 = ROOT / 'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-v1.json'
assert binding(mr06)['sha256'] == 'd947cf06c95c28c60f34ab02f70a0bb4a3ec6f5d3348e6af2b08945e68082d40'
assert binding(ROOT / 'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v1.json')['sha256'] == '51f2e24cb3307f69c4989ef3737a6c2ad987d0654152fd3c6db1ea01f5d4b995'
closed = datetime.now(UTC).isoformat()
save_new(HERE / 'mr07-final-source-recheck.json', {'checked_at': closed, 'reviewer_task': '/root/omni_development_execution_review', 'all_original92_still_equal_prior_snapshot': True, 'relevant_controller_finance_equal_failed_commit': True, 'mr06_original_unchanged': binding(mr06), 'original_mature_audit_unchanged': True, 'author_sources_or_old_reports_written_by_reviewer': []})
portable_paths = [HERE / name for name in ('source-provenance.json', 'probe_foreign_owned_receipt.py', 'mr07-probe.log', 'mr07-probe-results.json', 'mr07-final-source-recheck.json', 'seal_mr07_original.py')]
portable_paths.append(HERE / 'foreign-genuine-core-success/finish-transaction.json')
portable = [binding(path) for path in portable_paths]
save_new(INDEX, {'schema': 'omni-mature-runtime-mr07-original-portable-evidence-v1', 'created_at': closed, 'reviewer_task': '/root/omni_development_execution_review', 'report_path': str(REPORT), 'files': portable, 'file_count': len(portable), 'total_bytes': sum(item['bytes'] for item in portable), 'excluded': ['full original module source snapshot research_execution-ee799d.py', 'all full92 source/data/third-party snapshots already retained outside ordinary Git', 'raw provider release/artifact GET responses and full legacy ledger', 'duplicate private receipt and reservation inputs; provenance plus compact result/transition retained instead'], 'location_map_instruction': '保留原path/SHA，用location-map攜帶此7個小型自製證據；舊MR06 index／report不改。'})
relevant = {item['path']: item['sha256'] for item in snapshot['files'] if item['path'] in ('scripts/omni_revision/research_execution.py', 'scripts/omni_revision/finance.py')}
save_new(REPORT, {
    'schema': 'omni-mature-research-runtime-bounded-original-mr07-v1',
    'created_at': closed, 'author_task': '/root', 'reviewer_task': '/root/omni_development_execution_review',
    'status': 'changes_required',
    'verdict': 'MR07 反例成立：真 foreign owned 成功 receipt 可將另一筆 reserved 成熟研究列標為 completed；只發生於本地 recorder，沒有寫真帳。',
    'scope': {'actual_function_name': 'scripts.omni_revision.research_execution.finish_run', 'author_candidate_name': 'finish_results', 'source_revision': 'ee799d24fd1ea26108af709ec41e49db63d58e31', 'scope_kind': 'bounded original receipt-to-reservation binding review only', 'models_data_eval57_training25_implementation8_changed': False, 'actual_mature_success_claimed': False, 'real_modal_rpc_calls': 0, 'real_dispatch_calls': 0, 'real_ledger_writes': 0, 'real_release_writes': 0, 'nn_forward_attempts': 0},
    'relevant_files_sha256': relevant,
    'original_source_provenance': binding(HERE / 'source-provenance.json'),
    'prior_full92_snapshot_manifest': binding(ORIGINAL / 'snapshot-manifest.json'),
    'preserved_mr06_original': binding(mr06),
    'necessary_findings': [{
        'id': 'MR07', 'severity': 'blocking',
        'summary': 'finish_run 只核本地 reservation 與 ledger row 身分；接受 receipt 自帶的任何真 owned draft，沒有把其 attempt/schema/job/source/tag/body/target/lineage 綁回本次預約。',
        'locations': ['scripts/omni_revision/research_execution.py:574', 'scripts/omni_revision/research_execution.py:597', 'scripts/omni_revision/research_execution.py:602', 'scripts/omni_revision/research_execution.py:626', 'scripts/omni_revision/research_execution.py:628'],
        'mechanism': ['identity 只驗 CLI job/attempt 與 Actions 環境。', 'Actions artifact 名稱正確屬於 target run 即可；release GET 完全由 receipt.release_id 驅動，tag/body/assetset 只與同一 receipt 自述相符。', 'reservation_identity 的正確比較不包含任何 receipt-to-target 檢查；finance.finish 只依 success/resource flags 將傳入的 receipt 附到 target reserved row。', '因此 foreign release 真存在且被原 owner 真擁有，也不足以證明屬於這一筆 reservation。'],
        'actual_counterexample': {
            'foreign_receipt': 'genuine completed cpu-core-40988311100-1 receipt from original official ledger',
            'foreign_schema': 'omni-cpu-core-execution-v1',
            'foreign_source_revision': 'f9310888eaa95e79156ffddea122a8c9b1d6beb5',
            'foreign_release_id': 428573191,
            'flags_altered': False,
            'only_added_field': 'release_body copied exactly from that genuine draft GET because CPU receipt stores it separately',
            'target': 'actual mature-kev-41004743884-1 reserved pre-finish row privately reconstructed from genuine reservation/history',
            'target_revision': 'ee799d24fd1ea26108af709ec41e49db63d58e31',
            'genuine_target_artifact_check_passed': True,
            'genuine_foreign_owned_draft11_asset_check_passed': True,
            'local_transition': 'reserved -> completed',
            'result_outer_attempt': 'mature-kev-41004743884-1',
            'result_receipt_attempt': 'cpu-core-40988311100-1',
            'reserved_total_usd_before_and_after': '6.60',
            'live_ledger_target_status': 'failed-stopped; untouched',
        },
        'required_fix': [
            '在任何 release lookup 或 ledger write 前，先驗 current source/audit 與 immutable reserved row，並重建本次 attempt/job/revision、expected tag/body；不能以 receipt 自述作 expected identity。',
            '新成熟 receipt 必須有明確 schema、exact attempt/job、source_revision、audit SHA，並逐項比對 reservation purpose 的 sources/input/job_config/core-selection/final-opening／recovery lineage。資料來源在 source/permit/reservation 間要有完整可重算連結。',
            'owned release 的 tag/body/target_commitish 必須等於本次重建值，並核本次 release ownership 與 asset 身分；保持真 GET、bounded storage、CAS、不可返還額與停止檢查。',
            'success completion 必須連到本次已驗 worker/result/input/config／original opening 的完整證據；一般 foreign 真成功 receipt 仍須在 read-only provider lookup 及任何 writer 前拒絕。',
            '準備修復 permit 必須綁 exact failed predecessor；不得把可匹配其他 owned release 的 receipt 當成其停止／成功 lineage。舊MR06原失敗與bounds不改。',
        ],
        'required_regressions': ['本次真 foreign core成功receipt＋真draft metadata必須early拒絕，writer0且盡量provider lookup0。', '不同 attempt/source/audit/job/tag/body/target_commitish/input/config/opening/recovery_of逐項漂移拒絕。', 'matching 成熟成功fixture能完成、matching failed-stopped 能收尾，unknown仍unresolved。', '原 reservation_identity 篡改拒絕仍成立，保留CAS和所有bound。'],
    }],
    'optional_findings': [],
    'execution_evidence': {
        'probe_results': binding(HERE / 'mr07-probe-results.json'),
        'probe_log': binding(HERE / 'mr07-probe.log'),
        'foreign_transition_record': binding(HERE / 'foreign-genuine-core-success/finish-transaction.json'),
        'cases': result['cases'],
        'genuine_receipt_provenance': result['foreign_receipt_provenance'],
        'original_functions_executed': result['original_code_executed'],
        'only_mocked_boundaries': result['patched_boundaries_only'],
    },
    'portable_evidence': binding(INDEX),
    'limitations': ['沒有真成熟模型成功執行；使用的 foreign 成功收據是已真完成的自家 CPU core，沒有捏造其 success/resources/assets。', 'target reserved state 是私有歷史重建；live Kev row 現為 failed-stopped，沒有嘗試改它，也不主張已完成 terminal row 能被此函式再次finish。', 'provider GET在測試中回傳已實取的真 metadata；ledger writer 只記憶錄取並另用真 validate_state 驗，不進GitHub API。', '本判定僅證實身分綁定缺口與bounded反例；不改MR06，不簽新source audit／準備修復permit，也不授權派工。'],
})
print(json.dumps({'report': binding(REPORT), 'portable_index': binding(INDEX), 'portable_file_count': len(portable), 'portable_total_bytes': sum(item['bytes'] for item in portable), 'necessary_findings': ['MR07']}, ensure_ascii=False))
