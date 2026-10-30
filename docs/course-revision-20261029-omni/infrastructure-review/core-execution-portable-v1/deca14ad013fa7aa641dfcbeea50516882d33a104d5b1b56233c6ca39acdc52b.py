"""只封存此 reviewer 的新 execution 驗證紀錄；不寫任何模型來源或既有報告。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/work/omni-models-v1')
REPORT = ROOT / 'docs/course-revision-20261029-omni/infrastructure/core-execution-verification-v1.json'
INDEX = Path('/workspace/work/omni-core-execution-portable-evidence-v1.json')


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def write_new(path, document):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


assert not REPORT.exists() and not INDEX.exists()
provider = read(HERE / 'provider-archive-results.json')
models = read(HERE / 'checkpoint-selection-results.json')
assert provider['all_assertions_passed'] and models['all_assertions_passed']
initial = read(HERE / 'initial-input-manifest.json')
for row in initial['input_files'] + initial['current_frozen_files']:
    assert binding(row['path']) == row
sources = read(HERE / 'source-recheck.json')
for row in sources['sources']:
    assert binding(ROOT / row['path'])['sha256'] == row['expected'] == row['current_sha256'] == row['run_revision_sha256']
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
frozen = []
for row in initial['current_frozen_files']:
    path = Path(row['path'])
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(['git', 'show', head + ':' + relative], cwd=ROOT)
    assert raw == path.read_bytes()
    frozen.append({**row, 'relative_path': relative, 'committed_at_current_head': True})
assert not (ROOT / 'docs/omni-products/v1/final-opening.json').exists()
closed = datetime.now(UTC).isoformat()
write_new(HERE / 'final-state-recheck.json', {
    'schema': 'omni-core-execution-verification-final-state-v1',
    'reviewer_task': '/root/omni_development_execution_review',
    'checked_at': closed,
    'initial_input_capture_at': initial['captured_at_utc'],
    'all_initial_input_files_bytes_sha_unchanged': True,
    'initial_input_file_count': len(initial['input_files']),
    'source36_unchanged_again': True,
    'current_head': head,
    'five_frozen_manifests_committed_and_bytes_equal': frozen,
    'final_opening_exists': False,
    'model_or_controller_files_written': [],
    'old_audits_or_reports_written': [],
})
portable_names = (
    'initial-input-manifest.json',
    'source-recheck.json',
    'archive-entry-verification.json',
    'verify_provider_archive.py',
    'provider-archive-probe.log',
    'provider-archive-results.json',
    'own-training-log-excerpt.log',
    'verify_checkpoints_selection.py',
    'checkpoint-selection-probe.log',
    'checkpoint-selection-results.json',
    'final-state-recheck.json',
    'closeout_record.py',
)
portable = [binding(HERE / name) for name in portable_names]
raw_provider_names = sorted(p.name for p in HERE.iterdir() if p.name.startswith('provider-') and p.name not in portable_names)
write_new(INDEX, {
    'schema': 'omni-core-execution-review-portable-evidence-v1',
    'created_at': closed,
    'reviewer_task': '/root/omni_development_execution_review',
    'report_path': str(REPORT),
    'preservation': 'Copy these small self-authored probes, results, excerpts and SHA manifests with a location-map. No source snapshot, original dataset, model tensor, ZIP or third-party full text belongs in ordinary Git.',
    'files': portable,
    'file_count': len(portable),
    'total_bytes': sum(row['bytes'] for row in portable),
    'excluded': {
        'large_original_zip_and_checkpoints': '/workspace/work/omni-core-completed-40988311100',
        'independently_downloaded_actions_zip': str(HERE / 'actions-artifact.zip'),
        'full_decoded_official_job_log': str(HERE / 'provider-job-full.log'),
        'raw_provider_response_files': [str(HERE / name) for name in raw_provider_names],
        'source_snapshots_and_dataset_copies': 'No full source snapshot or dataset copy was created by this execution verification.',
    },
})
audit_path = ROOT / 'docs/course-revision-20261029-omni/infrastructure/formal-core-audit-v1.json'
audit = read(audit_path)
assert binding(audit_path)['sha256'] == '47a6f91e0b8a4362def935c783495e8a9e527d25cefbb1afc429fcabc9a31aa3'
job_rows = {}
for name, value in models['jobs'].items():
    actual_training = read(ROOT / 'outputs/omni-formal-core-v1' / name / 'training/training-result.json')
    job_rows[name] = {key: value[key] for key in ('kind', 'updates', 'complete', 'source_unchanged_during_training', 'optimizer_updates_observed', 'trainable_parameters', 'initial_parameter_sha256', 'initial_sha_reproduced_without_forward', 'best_update', 'best_dev_score', 'best_dev_metrics', 'job_origin_sha256', 'job_result_sha256', 'training_result_sha256')}
    job_rows[name]['saved_training_files'] = actual_training['files']
write_new(REPORT, {
    'schema': 'omni-core-execution-verification-v1',
    'created_at': closed,
    'author_task': '/root',
    'reviewer_task': '/root/omni_development_execution_review',
    'status': 'pass_for_reviewed_scope',
    'verdict': '實際十個 CPU core 工作及其來源綁定保存產物核對通過；沒有能力或 final 通過判定。',
    'review_started_at': initial['captured_at_utc'],
    'scope': {
        'kind': 'actual completed execution and persisted artifacts; separate from immutable source audits',
        'run_id': 40988311100,
        'attempt_id': 'cpu-core-40988311100-1',
        'execution_revision': provider['run']['head_sha'],
        'local_recheck_head': head,
        'formal_job_count': 10,
        'actual_model_loads': 10,
        'actual_continuation_files_read_with_weights_only': 20,
        'neural_forward_attempts': 0,
        'training_or_optimizer_updates_by_reviewer': 0,
        'new_predictions_by_reviewer': 0,
        'paid_dispatches_by_reviewer': 0,
        'final_opening_created': False,
        'excluded': ['final NN evaluation', 'capability acceptance', 'mature teacher model execution', 'curriculum three-round review', 'a new training/resume experiment', 'invoice settlement'],
    },
    'necessary_findings': [],
    'optional_findings': [],
    'source_audit_reference': binding(audit_path),
    'immutable_reviewed_files_sha256': audit['reviewed_files_sha256'],
    'source_verification': {
        'original_audit_source36_equal_run_commit_and_current': True,
        'actual_actions_sources_json_equals_source36_plus_audit': True,
        'reservation_purpose_sources_digest_matches_actual_actions_sources': True,
        'all_initial_input_files_still_exact_at_closeout': True,
        'five_frozen_manifests_committed_at_current_head': frozen,
        'previous_reports_or_audits_modified': False,
    },
    'provider_and_durable_state': {
        'run': provider['run'],
        'job_id': provider['job']['id'],
        'all_job_steps_completed_successfully': True,
        'ledger': provider['ledger'],
        'unique_core_purpose_completed': True,
        'durable_receipt_success': True,
        'no_modal_compute_created_recorded': True,
        'all_training_processes_exited_recorded': True,
        'owned_unpublished_draft': provider['release'],
        'official_asset11_ids_sizes_sha256_equal_ledger': True,
        'assets': provider['assets'],
        'actions_artifact': provider['actions_artifact'],
        'official_training_log_112_rows_equal_all_saved_job_logs': True,
        'acquisition': {'actions_run_jobs_ledger_logs_artifacts': 'independent GitHub connector GET', 'draft_release': 'independent authenticated gh api GET after connector draft GET returned404', 'actions_zip_bytes': 'GitHub connector artifact download and read-only temporary File Service download'},
    },
    'aggregate_archive': provider['aggregate'],
    'author_retrieval_receipt': binding(Path('/workspace/work/omni-core-completed-40988311100/retrieval-receipt-v1.json')),
    'jobs': job_rows,
    'continuation_verification': {
        'all20_files_real_sha_and_size_verified_before_weights_only_load': True,
        'latest_steps_equal_all_prescribed_updates': True,
        'best_steps_equal_earliest_maximum_dev_log_rows': True,
        'all_parameter_model_tensors_finite_shape_correct': True,
        'full_adamw_state_for_every_parameter_with_exact_step_lr_moments': True,
        'python_rng_torch_rng_sampler_rng_and_cpu_cuda_empty_state_valid': True,
        'metadata_config_runtime_training_revision_implementation_exact': True,
        'latest_embeds_exact_original_best_checkpoint_bytes': True,
        'best_checkpoint_has_no_recursive_best_bytes': True,
        'exported_safetensors_exactly_equal_best_model_tensors': True,
        'latest_parameter_sha_matches_saved_training_result': True,
        'weights_differ_from_recorded_initial_parameters': True,
    },
    'fairness': models['fairness'],
    'codec_talker': models['codec_talker'],
    'selection': {
        'codec_selection_sha256': models['codec_selection_sha256'],
        'model_selection_sha256': models['selection_sha256'],
        'selected_av': 'av-time-rope-s29',
        'rule': 'maximum dev score = paired accuracy + 1e-6 mean missing-cue accuracy; then earlier update; then prescribed job order',
        'all6_av_ranking': models['av_dev_ranking'],
        'core_final_checked_selection_actual_positive': models['common_checked_selection'],
        'calibration_policy': models['calibration_policy'],
        'data_then_metrics_then_recipe_then_actual_training_chronology_verified': True,
        'codec_success_then_freeze_then_talker_and_all10_complete_then_model_selection_verified': True,
        'current_final_source_audit57_binding_matches': True,
        'final_source_audit_reference_sha256': models['current_final_audit_sha256'],
        'final_opening_absent_at_closeout': True,
    },
    'evidence': {
        'portable_index': binding(INDEX),
        'provider_archive_result': binding(HERE / 'provider-archive-results.json'),
        'checkpoint_selection_result': binding(HERE / 'checkpoint-selection-results.json'),
        'initial_input_manifest': binding(HERE / 'initial-input-manifest.json'),
        'archive_entry_manifest': binding(HERE / 'archive-entry-verification.json'),
        'final_state_recheck': binding(HERE / 'final-state-recheck.json'),
    },
    'limitations': [
        '這是 provider／帳本／實際檔案／權重載入與保存紀錄一致性的核對；不把下載成功或 dev 分數當成能力通過。',
        *provider['limits'],
        *models['limits'],
        '模型 loader 儀器只記錄後原樣呼叫真 load_trained；checked_selection 內所有實際來源、收據與檔案驗證均照原函式執行，沒有 stub 核心 gate 或替換權重。',
        '實際校準／policy 的 logits 已完成原始 metadata/gold/候選 ID 及算術一致性核對；本 reviewer 沒有重新產生神經預測，也沒有重算需 codec NN inference 的 Talker target codes。',
    ],
})
print(json.dumps({'report': binding(REPORT), 'portable_index': binding(INDEX), 'portable_file_count': len(portable), 'portable_total_bytes': sum(row['bytes'] for row in portable), 'necessary_findings': 0}, ensure_ascii=False))
