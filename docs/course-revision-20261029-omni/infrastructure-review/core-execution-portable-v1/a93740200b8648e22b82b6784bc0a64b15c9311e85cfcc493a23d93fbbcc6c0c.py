"""實際自製 core 產物唯讀核對；神經 module forward 一律即刻拒絕。"""

import hashlib
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path('/workspace/work/omni-models-v1')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
CORE = ROOT / 'outputs/omni-formal-core-v1'
FROZEN = ROOT / 'docs/omni-products/v1'
EVIDENCE = ROOT / 'outputs/omni-core-selection-v1'

import torch
from safetensors.torch import load_file

from scripts.omni_revision import core_final
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.av import AVConfig, AVModel
from tiny_perceptron.omni_products.choice import ChoiceConfig, ChoiceModel
from tiny_perceptron.omni_products.reliability import select_policy, select_temperature
from tiny_perceptron.omni_products.routing import training_routes


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def digest_tensor(value):
    return sha(value.contiguous().view(torch.uint8).numpy().tobytes())


def softmax(logits, temperature=1):
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exponentials = [math.exp(x - top) for x in scaled]
    return [x / sum(exponentials) for x in exponentials]


def winner(record):
    return min(range(len(record['ids'])), key=lambda i: (-record['logits'][i], record['ids'][i]))


def verify_prediction(record, question, gold=True):
    ids = [row['id'] for row in question['candidates']] + (['none'] if question['kind'] == 'choice' else [])
    assert record['ids'] == ids and len(record['logits']) == len(ids)
    assert all(type(x) in (int, float) and math.isfinite(x) for x in record['logits'])
    assert record['answer'] == ids[winner(record)]
    probabilities = softmax(record['logits'])
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(record['probabilities'], probabilities, strict=True))
    assert math.isclose(record['p_max'], max(probabilities), rel_tol=1e-12)
    if gold:
        assert record['gold'] == question['gold']


started = datetime.now(UTC).isoformat()
assert read(HERE / 'provider-archive-results.json')['all_assertions_passed'] is True
torch.set_num_threads(2)
assert str(torch.__version__) == '2.14.1+cpu'
recipes = read(FROZEN / 'formal-recipes.json')
data = read(FROZEN / 'data-definition.json')
selection = read(FROZEN / 'model-selection.json')
codec_selection = read(FROZEN / 'codec-selection.json')
assert trainer.file_sha(FROZEN / 'model-selection.json') == '935594413eda401f4fddc81b57f0245cd6a28219dda19e4e27b9d0f751385e20'
assert trainer.file_sha(FROZEN / 'codec-selection.json') == '1fdaba26dbd57f854244318d4b16992fc013fc63e4fba1d144570826d223ef22'
forward_attempts = []


def forbid_forward(module, inputs):
    forward_attempts.append(type(module).__name__)
    raise AssertionError('Independent artifact review forbids neural-network forward')


guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid_forward)
load_calls, loaded = [], {}
real_loader = core_final.load_trained


def record_real_load(directory, kind, *args, **kwargs):
    model = real_loader(directory, kind, *args, **kwargs)
    name = Path(directory).parent.name
    load_calls.append({'job': name, 'kind': kind})
    loaded[name] = model
    return model


try:
    # 全部實際來源檢查、十份真權重 strict load 與共同 gate 原樣執行；只記錄真 loader 呼叫。
    with patch.object(core_final, 'load_trained', side_effect=record_real_load):
        actual_selection = core_final.checked_selection(CORE, trainer.file_sha(FROZEN / 'model-selection.json'))
    assert actual_selection == selection
    assert len(load_calls) == 10 and set(loaded) == set(recipes['jobs'])
    final_audit = read(core_final.AUDIT)
    final_sources = core_final.evaluation_sources()
    assert len(final_sources) == 57
    assert all(final_audit['reviewed_files_sha256'][name] == digest for name, digest in final_sources.items())
    assert not (FROZEN / 'final-opening.json').exists()

    jobs = {}
    for name in recipes['job_order']:
        directory = CORE / name / 'training'
        origin = read(CORE / name / 'job-origin.json')
        result = read(CORE / name / 'job-result.json')
        training = read(directory / 'training-result.json')
        logs = read(directory / 'training-log.json')
        configured = recipes['jobs'][name]['training']
        model = loaded[name]
        parameters = list(model.named_parameters())
        assert training['complete'] and training['optimizer_updates_observed'] and training['source_unchanged_during_training']
        assert training['updates'] == configured['updates']
        assert training['runtime'] == {'device': 'cpu', 'torch': str(torch.__version__), 'threads': 2}
        assert training['implementation_sha256'] == trainer.source_binding()
        assert training['training_revision'] == trainer.TRAINING_REVISION
        assert training['trainable_parameters'] == sum(p.numel() for _, p in parameters)
        assert origin['execution_host'] == {'github_actions': True, 'run_id': '40988311100', 'run_attempt': '1'}
        assert origin['Modal_dispatched'] is False and result['Modal_dispatched'] is False
        assert origin['new_Modal_reservation'] == result['new_Modal_reservation'] == '0.00'
        assert origin['attempt'] == result['attempt'] == 1
        assert datetime.fromisoformat(result['finished_at']) > datetime.fromisoformat(origin['started_at'])
        assert result['elapsed_seconds'] <= configured['training_seconds'] + 300
        assert training['elapsed_seconds'] <= configured['training_seconds']
        assert all(training['config'][key] == value for key, value in configured.items())
        expected_updates = list(range(configured['validate_every'], configured['updates'] + 1, configured['validate_every']))
        if configured['updates'] not in expected_updates:
            expected_updates.append(configured['updates'])
        assert [row['update'] for row in logs] == expected_updates
        assert all(math.isfinite(row['loss']) and math.isfinite(row['dev_score']) for row in logs)
        assert all(a['elapsed_seconds'] < b['elapsed_seconds'] for a, b in zip(logs, logs[1:]))
        best = max(logs, key=lambda row: (row['dev_score'], -row['update']))
        assert training['best'] == best
        if recipes['jobs'][name]['kind'] == 'av':
            for row in logs:
                metrics = row['dev_metrics']
                assert row['dev_score'] == metrics['paired_accuracy'] + 1e-6 * (metrics['remove_image_accuracy'] + metrics['remove_audio_accuracy']) / 2
            frequencies = [v for k, v in model.state_dict().items() if k.endswith('.frequencies')]
            assert len(frequencies) == 2
            assert all(torch.equal(v, 2 ** torch.linspace(-1, 6, 8)) for v in frequencies)
        weights = load_file(str(directory / 'weights.safetensors'), device='cpu')
        checkpoints = {}
        for filename, expected_step in [('best-continuation.pt', best['update']), ('latest-continuation.pt', configured['updates'])]:
            path = directory / filename
            binding = training['files'][filename]
            assert path.stat().st_size == binding['bytes'] and trainer.file_sha(path) == binding['sha256']
            saved = torch.load(path, weights_only=True, map_location='cpu')
            assert set(saved) == {'model', 'optimizer', 'step', 'python_rng', 'torch_rng', 'cuda_rng', 'sampler_rng', 'metadata', 'auxiliary', 'numpy_rng'}
            assert saved['step'] == expected_step
            assert saved['metadata'] == {'config': training['config'], 'training_revision': trainer.TRAINING_REVISION, 'implementation_sha256': trainer.source_binding(), 'runtime': {'device': 'cpu', 'torch': str(torch.__version__)}}
            state = saved['model']
            assert set(state) == set(model.state_dict()) and all(torch.isfinite(v).all() for v in state.values())
            assert all(state[k].shape == model.state_dict()[k].shape for k in state)
            optimizer = saved['optimizer']
            assert set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1
            group = optimizer['param_groups'][0]
            assert group['params'] == list(range(len(parameters)))
            assert set(optimizer['state']) == set(group['params'])
            assert group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0.01
            assert group['decoupled_weight_decay'] is True
            assert group['lr'] == trainer.learning_rate(expected_step - 1, configured['updates'], configured['learning_rate'])
            for key, value in optimizer['state'].items():
                assert set(value) == {'step', 'exp_avg', 'exp_avg_sq'}
                assert value['step'].item() == expected_step
                assert value['exp_avg'].shape == value['exp_avg_sq'].shape == parameters[key][1].shape
                assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
                assert (value['exp_avg_sq'] >= 0).all()
            assert isinstance(saved['python_rng'], tuple) and saved['python_rng'][0] == 3 and len(saved['python_rng'][1]) == 625
            random.Random().setstate(saved['python_rng'])  # 私有 RNG，沒有抽樣。
            assert saved['cuda_rng'] == []
            for rng in (saved['torch_rng'], saved['sampler_rng']):
                assert rng.dtype == torch.uint8 and rng.shape == torch.Size([5056])
                torch.Generator().set_state(rng)  # 只驗可還原型態，不抽樣或更新。
            assert saved['numpy_rng'] == 'not used by this trainer or data generator'
            auxiliary = saved['auxiliary']
            assert auxiliary['best'] == best and auxiliary['initial_parameter_sha256'] == training['initial_parameter_sha256']
            assert auxiliary['logs'] == [row for row in logs if row['update'] <= expected_step]
            if name == 'codec':
                assert auxiliary['code_usage'].shape == torch.Size([256])
                assert torch.isfinite(auxiliary['code_usage']).all() and (auxiliary['code_usage'] >= 0).all()
            else:
                assert auxiliary['code_usage'] is None
            if filename == 'best-continuation.pt':
                assert auxiliary['best_checkpoint_bytes'] is None
                assert all(torch.equal(state[k], weights[k]) for k in state)
            else:
                assert auxiliary['best_checkpoint_bytes'] == (directory / 'best-continuation.pt').read_bytes()
                assert state_sha(state) == training['latest_parameter_sha256']
            checkpoints[filename] = {'bytes': binding['bytes'], 'sha256': binding['sha256'], 'step': expected_step, 'model_tensors': len(state), 'optimizer_parameter_states': len(optimizer['state']), 'all_optimizer_steps': expected_step, 'python_rng_version': saved['python_rng'][0], 'torch_rng_bytes': saved['torch_rng'].numel(), 'torch_rng_sha256': digest_tensor(saved['torch_rng']), 'sampler_rng_bytes': saved['sampler_rng'].numel(), 'sampler_rng_sha256': digest_tensor(saved['sampler_rng']), 'metadata_exact': True, 'best_exported_weights_exact': filename == 'best-continuation.pt', 'embeds_exact_selected_best_bytes': filename == 'latest-continuation.pt'}
        assert trainer.parameter_sha(model) != training['initial_parameter_sha256']
        # 原 seed 與相同 constructor 可再現記錄的初值；只有初始化，沒有 forward。
        init_reproduced = None
        if recipes['jobs'][name]['kind'] in ('av', 'choice'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(configured['seed'])
                if recipes['jobs'][name]['kind'] == 'av':
                    initial_model = AVModel(AVConfig(**training['config']['model']))
                else:
                    initial_model = ChoiceModel(ChoiceConfig(**training['config']['model']))
                assert trainer.parameter_sha(initial_model) == training['initial_parameter_sha256']
                init_reproduced = True
        jobs[name] = {'kind': recipes['jobs'][name]['kind'], 'updates': training['updates'], 'complete': True, 'source_unchanged_during_training': True, 'optimizer_updates_observed': True, 'trainable_parameters': training['trainable_parameters'], 'initial_parameter_sha256': training['initial_parameter_sha256'], 'initial_sha_reproduced_without_forward': init_reproduced, 'best_update': best['update'], 'best_dev_score': best['dev_score'], 'best_dev_metrics': best.get('dev_metrics'), 'config': training['config'], 'job_origin_sha256': trainer.file_sha(CORE / name / 'job-origin.json'), 'job_result_sha256': trainer.file_sha(CORE / name / 'job-result.json'), 'training_result_sha256': trainer.file_sha(directory / 'training-result.json'), 'weights': training['files']['weights.safetensors'], 'checkpoint_checks': checkpoints}

    av_names = [name for name in recipes['job_order'] if recipes['jobs'][name]['kind'] == 'av']
    for seed in (11, 29, 47):
        left, right = jobs[f'av-additive-s{seed}'], jobs[f'av-time-rope-s{seed}']
        assert left['initial_parameter_sha256'] == right['initial_parameter_sha256']
        assert left['trainable_parameters'] == right['trainable_parameters'] == 119693
        assert {k: v for k, v in left['config'].items() if k not in ('representation', 'model')} == {k: v for k, v in right['config'].items() if k not in ('representation', 'model')}
        assert {k: v for k, v in left['config']['model'].items() if k != 'representation'} == {k: v for k, v in right['config']['model'].items() if k != 'representation'}
    av_ranking = sorted(av_names, key=lambda name: (-jobs[name]['best_dev_score'], jobs[name]['best_update'], av_names.index(name)))
    assert av_ranking[0] == selection['selected']['av'] == 'av-time-rope-s29'
    augmented, compact = jobs['c-augmented'], jobs['c-compact']
    assert augmented['initial_parameter_sha256'] == compact['initial_parameter_sha256']
    assert augmented['trainable_parameters'] == compact['trainable_parameters'] == 279170
    assert {k: v for k, v in augmented['config'].items() if k not in ('data_mode', 'train_questions')} == {k: v for k, v in compact['config'].items() if k not in ('data_mode', 'train_questions')}
    full_questions = trainer.choice_training_core('augmented')
    compact_questions = trainer.choice_training_core('compact')
    assert compact_questions == [row for index, row in enumerate(full_questions) if index % 12 < 4]
    route_count = len(training_routes())
    assert len(full_questions) == 10836 and len(compact_questions) == 3612 and route_count == 1031
    assert len(full_questions) + route_count == augmented['config']['train_questions']
    assert len(compact_questions) + route_count == compact['config']['train_questions']
    codec = jobs['codec']
    assert codec_selection['weights_sha256'] == codec['weights']['sha256']
    assert codec_selection['training_result_sha256'] == codec['training_result_sha256']
    assert codec_selection['model'] == codec['config']['model']
    assert codec_selection['dev_best'] == read(CORE / 'codec/training/training-result.json')['best']
    assert datetime.fromisoformat(read(CORE / 'codec/job-result.json')['finished_at']) <= datetime.fromisoformat(codec_selection['created_at']) < datetime.fromisoformat(read(CORE / 'talker/job-origin.json')['started_at'])
    assert read(CORE / 'talker/job-result.json')['codec_selection_sha256'] == trainer.file_sha(FROZEN / 'codec-selection.json')
    assert jobs['talker']['config']['codec_parameter_sha256'] == trainer.parameter_sha(loaded['codec'])
    assert jobs['talker']['config']['targets_families'] == len(data['sound']['train']) == 2560
    assert len(jobs['talker']['config']['train_targets_sha256']) == 64
    latest_finish = max(datetime.fromisoformat(read(CORE / name / 'job-result.json')['finished_at']) for name in recipes['jobs'])
    assert latest_finish < datetime.fromisoformat(selection['created_at'])
    data_time = datetime.fromisoformat(data['created_at'])
    metrics_time = datetime.fromisoformat(read(FROZEN / 'metrics.json')['created_at'])
    recipe_time = datetime.fromisoformat(recipes['created_at'])
    first_start = min(datetime.fromisoformat(read(CORE / name / 'job-origin.json')['started_at']) for name in recipes['jobs'])
    assert data_time < metrics_time < recipe_time < first_start

    rows_by_split = {}
    for filename, split, count in [('calibration.json', 'calibration', 80), ('policy-predictions.json', 'policy', 60)]:
        rows = read(EVIDENCE / filename)
        originals = {row['family']: row for row in data['C']['panels'][split]}
        assert len(rows) == count and len(originals) == count and {r['family'] for r in rows} == set(originals)
        for row in rows:
            original = originals[row['family']]
            for key in ('family', 'source', 'panel', 'world_supported', 'choice_information_sufficient'):
                assert row[key] == original[key]
            assert set(row['tasks']) == {'choice', 'info', 'rubric'}
            for kind, record in row['tasks'].items():
                verify_prediction(record, original['questions'][kind])
            verify_prediction(row['choice_info'], original['questions']['choice_info'], gold=False)
        rows_by_split[split] = rows
    assert not ({r['family'] for r in rows_by_split['calibration']} & {r['family'] for r in rows_by_split['policy']})
    stored_temperature = read(EVIDENCE / 'temperature.json')
    stored_policy = read(EVIDENCE / 'policy.json')
    # 真固定函式重放保存的 logits；沒有生成任何新 prediction。
    assert select_temperature(rows_by_split['calibration']) == stored_temperature
    assert select_policy(rows_by_split['policy'], stored_temperature['temperature']) == stored_policy
    temperature_grid = []
    for k in range(33):
        temperature = 0.25 * 2 ** (k / 8)
        losses = []
        for row in rows_by_split['calibration']:
            task_losses = []
            for record in row['tasks'].values():
                scaled = [x / temperature for x in record['logits']]
                top = max(scaled)
                task_losses.append(top + math.log(sum(math.exp(x - top) for x in scaled)) - scaled[record['ids'].index(record['gold'])])
            losses.append(sum(task_losses) / 3)
        temperature_grid.append({'temperature': temperature, 'family_equal_type_equal_nll': sum(losses) / 80})
    assert all(math.isclose(a['family_equal_type_equal_nll'], b['family_equal_type_equal_nll'], rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(temperature_grid, stored_temperature['grid'], strict=True))
    independent_t = min(temperature_grid, key=lambda row: (row['family_equal_type_equal_nll'], abs(row['temperature'] - 1), row['temperature']))['temperature']
    assert independent_t == selection['temperature'] == 0.6484197773255048
    policy_grid = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99):
        accepted = errors = 0
        for row in rows_by_split['policy']:
            choice, info = row['tasks']['choice'], row['choice_info']
            answer = choice['ids'][winner(choice)]
            accept = max(softmax(choice['logits'], independent_t)) >= threshold and answer != 'none' and info['ids'][winner(info)] == 'yes'
            error = accept and (answer != choice['gold'] or not row['world_supported'] or not row['choice_information_sufficient'])
            accepted += accept
            errors += error
        policy_grid.append({'threshold': threshold, 'families': 60, 'accepted': accepted, 'errors': errors, 'coverage': accepted / 60, 'empirical_risk': errors / accepted if accepted else None})
    assert policy_grid == [{k: v for k, v in row.items() if k != 'rows'} for row in stored_policy['grid']]
    feasible = [row for row in policy_grid if row['accepted'] and row['errors'] / row['accepted'] <= .05]
    independent_tau = max(feasible, key=lambda row: (row['coverage'], row['threshold']))['threshold'] if feasible else None
    assert independent_tau == selection['threshold'] == 0.6 and selection['policy_feasible'] is True
    assert not forward_attempts
    result = {'schema': 'omni-core-checkpoint-selection-independent-verification-v1', 'reviewer_task': '/root/omni_development_execution_review', 'started_at': started, 'finished_at': datetime.now(UTC).isoformat(), 'all_assertions_passed': True, 'runtime': {'torch': str(torch.__version__), 'device': 'cpu'}, 'common_checked_selection': {'actual_positive': True, 'function': 'scripts.omni_revision.core_final.checked_selection', 'real_loader_calls': load_calls, 'all10_real_safetensors_strict_loads': True, 'nn_module_forward_attempts': forward_attempts, 'final_opening_absent': True}, 'jobs': jobs, 'fairness': {'same_seed_three_av_pairs_initial_sha_and_parameter_counts_equal': True, 'all8_av_choice_initial_shas_reproduced_without_forward': True, 'av_fixed8_per_head_frequencies4_heads': True, 'c_same_initial_sha_model_seed_batch_updates_lr_routes': True, 'compact_exact_original_variant0_subset': True, 'augmented_core_questions': len(full_questions), 'compact_core_questions': len(compact_questions), 'shared_route_questions': route_count}, 'codec_talker': {'codec_success_precedes_immutable_selection_precedes_talker': True, 'same_selected_codec_weight_parameter_sha_in_talker_config': True, 'target_train_families': 2560, 'target_codes_recomputed': False, 'optimizer_state_contains_only_talker_parameters': True}, 'av_dev_ranking': [{'job': name, 'dev_score': jobs[name]['best_dev_score'], 'update': jobs[name]['best_update'], 'job_order': av_names.index(name)} for name in av_ranking], 'calibration_policy': {'calibration_families': 80, 'policy_families': 60, 'all_raw_records_match_original_frozen_metadata_ids_gold': True, 'all_stored_raw_predictions_probabilities_self_consistent': True, 'same_actual_functions_exact_replay': True, 'independent_stdlib_math_grid_replay': True, 'temperature': independent_t, 'temperature_grid': temperature_grid, 'threshold': independent_tau, 'policy_feasible': True, 'policy_grid': policy_grid}, 'selection_sha256': trainer.file_sha(FROZEN / 'model-selection.json'), 'codec_selection_sha256': trainer.file_sha(FROZEN / 'codec-selection.json'), 'current_final_audit_sha256': trainer.file_sha(core_final.AUDIT), 'current_final_audit57_source_match': True, 'limits': ['No training, optimizer update, checkpoint resume execution, NN forward, new prediction, final opening or paid dispatch was performed.', 'Continuation completeness is established by the real saved model/AdamW/RNG/config/source/best bytes structure and consistency. This is not a new full-training or cross-runtime exact-resume experiment.', 'Talker target-code fingerprint is bound to the selected formal codec and train-only 2560 families in the saved config. Codes were not regenerated because that would require codec NN inference.', 'Saved calibration/policy logits were checked against frozen family/gold/candidate metadata and replayed arithmetically; neural predictions were not recomputed, and no capability or final-panel verdict is issued.']}
    (HERE / 'checkpoint-selection-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_assertions_passed': True, 'real_models_loaded': len(load_calls), 'checkpoints_verified': 20, 'original_initial_shas_reproduced': 8, 'selected_av': av_ranking[0], 'temperature': independent_t, 'threshold': independent_tau, 'nn_forward_attempts': len(forward_attempts)}))
finally:
    guard.remove()
