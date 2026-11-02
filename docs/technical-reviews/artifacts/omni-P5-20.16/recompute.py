"""Portable independent recalculation of saved logits; never loads a model.

The original fixed T/tau grids are replayed to audit the preserved selection.
This script makes no new selection or change to any research artifact.
"""

import array
import contextlib
import hashlib
import io
import json
import math
import platform
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent


def prob(z, temperature):
    scaled = [v / temperature for v in z]
    weights = [math.exp(v - max(scaled)) for v in scaled]
    return [v / sum(weights) for v in weights]


def answer(task):
    return min(range(len(task['ids'])), key=lambda i: (-task['logits'][i], task['ids'][i]))


def nll(task, temperature):
    z = [v / temperature for v in task['logits']]
    return max(z) + math.log(sum(math.exp(v - max(z)) for v in z)) - z[task['ids'].index(task['gold'])]


def accepted(rows, temperature, threshold):
    values = []
    for row in rows:
        c, info = row['tasks']['choice'], row['choice_info']
        a = c['ids'][answer(c)]
        confidence = max(prob(c['logits'], temperature))
        take = confidence >= threshold and a != 'none' and info['ids'][answer(info)] == 'yes'
        error = take and (a != c['gold'] or not row['world_supported'] or not row['choice_information_sufficient'])
        values.append({'family': row['family'], 'accepted': take, 'error': bool(error), 'answer': a, 'p_max': confidence})
    count = sum(v['accepted'] for v in values)
    errors = sum(v['error'] for v in values)
    return {'families': len(rows), 'accepted': count, 'errors': errors, 'coverage': count / len(rows),
            'risk': errors / count if count else None}


def main():
    data = json.loads((HERE / 'minimal-inputs.json').read_bytes())
    rows = data['kev_rows']
    group = lambda s, l: [r for r in rows if r['split'] == s and r['language'] == l]
    calibration, policy = group('calibration', 'english'), group('policy', 'english')
    assert len(calibration) == 80 and len(policy) == 60
    panel_sets = {k: set(v) for k, v in data['C_family_panels'].items()}
    assert all(len(v) == len(data['C_family_panels'][k]) for k, v in panel_sets.items())
    assert all(not a & b for i, a in enumerate(panel_sets.values()) for b in list(panel_sets.values())[i + 1:])
    grid = []
    for k in range(33):
        t = 0.25 * 2 ** (k / 8)
        loss = sum(sum(nll(r['tasks'][kind], t) for kind in ['choice', 'info', 'rubric']) / 3 for r in calibration) / 80
        grid.append({'temperature': t, 'family_type_equal_NLL': loss})
    chosen = min(grid, key=lambda r: (r['family_type_equal_NLL'], abs(r['temperature'] - 1), r['temperature']))
    temperature = chosen['temperature']
    assert temperature == data['preserved_selection']['T']
    policy_grid = [{'tau': t, **accepted(policy, temperature, t)} for t in [.5, .6, .7, .8, .85, .9, .95, .975, .99]]
    feasible = [r for r in policy_grid if r['accepted'] and r['risk'] <= .05]
    tau = max(feasible, key=lambda r: (r['coverage'], r['tau']))['tau']
    assert tau == data['preserved_selection']['tau'] == .8
    final = {}
    for language, expected in [('english', [42, 59, 35]), ('chinese', [32, 49, 33])]:
        values = group('final', language)
        assert len(values) == 60 and {r['family'] for r in values} == panel_sets['final']
        scores = {}
        for kind, gold_count in zip(['choice', 'info', 'rubric'], expected):
            correct = sum(r['tasks'][kind]['ids'][answer(r['tasks'][kind])] == r['tasks'][kind]['gold'] for r in values)
            assert correct == gold_count
            scores[kind] = {'correct': correct, 'families': len(values),
                'raw_NLL': sum(nll(r['tasks'][kind], 1) for r in values) / 60,
                'calibrated_NLL': sum(nll(r['tasks'][kind], temperature) for r in values) / 60}
        scores['acceptance'] = accepted(values, temperature, tau)
        c = values[0]['tasks']['choice']
        scores['first_family_choice'] = {'family': values[0]['family'], 'ids': c['ids'], 'logits': c['logits'],
            'answer': c['ids'][answer(c)], 'gold': c['gold'], 'T_does_not_change_argmax':
            answer(c) == max(range(len(c['ids'])), key=lambda i: prob(c['logits'], temperature)[i])}
        final[language] = scores
    assert (final['english']['acceptance']['accepted'], final['english']['acceptance']['errors']) == (18, 1)
    assert (final['chinese']['acceptance']['accepted'], final['chinese']['acceptance']['errors']) == (1, 0)
    assert round(final['chinese']['choice']['raw_NLL'], 4) == 2.1189
    assert round(final['chinese']['choice']['calibrated_NLL'], 4) == 1.5782
    cache = data['cache']['rows']
    assert len(cache) == 8 and all(r['prefix_before'] == r['prefix_after'] for r in cache)
    assert max(r['uncached_cached_max_abs'] for r in cache) < 1e-5
    assert max(r['Q1_Q2_Q1_probability_max_abs'] for r in cache) == 0
    inline = (HERE / 'inline-example.py').read_text()
    demonstrations = []
    for threshold in [.85, .95]:
        stream = io.StringIO(); namespace = {}
        with contextlib.redirect_stdout(stream):
            exec(compile(inline.replace('T, tau = 1.25, 0.85', f'T, tau = 1.25, {threshold}'), 'inline-example.py', 'exec'), namespace)
        values = [{'family': f, 'variant': v, 'p_max': max(namespace['probabilities'](z)),
                   'gold': gold, 'prediction': max(range(len(z)), key=lambda i: z[i])} for f, v, z, gold in namespace['rows']]
        demonstrations.append({'tau': threshold, 'stdout': stream.getvalue(), 'row_checks': values})
    assert '逐題 4 / 6' in demonstrations[0]['stdout'] and 'accepted risk 0.4' in demonstrations[0]['stdout']
    assert '接受 0 / 6' in demonstrations[1]['stdout'] and 'accepted risk None' in demonstrations[1]['stdout']
    for case in data['qwen_AV_inputs']:
        assert len(case['frames']) == 8 and case['fps'] == 2 and case['duration_seconds'] == 4
        for i, frame in enumerate(case['frames']):
            bit = (case['pattern'] >> (7 - i)) & 1
            assert frame['size'] == [112, 112] and frame['mode'] == 'RGB' and frame['timestamp_ms'] == i * 500
            assert frame['uniform_rgb'] == ([220, 20, 20] if bit == 0 else [20, 20, 220])
        for sound in case['pair']:
            frame = case['frames'][int(sound['center_seconds'] * 2)]
            assert sound['gold'] == ('red' if frame['uniform_rgb'][0] == 220 else 'blue')
            begin = round((sound['center_seconds'] - .04) * 16000)
            values = [0] * 64000
            for i in range(1280):
                values[begin + i] = round(32767 * .6 * math.sin(2 * math.pi * 440 * i / 16000))
            stream = io.BytesIO()
            with wave.open(stream, 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(array.array('h', values).tobytes())
            assert hashlib.sha256(stream.getvalue()).hexdigest() == sound['sha256']
    assert len(data['qwen_audio_inputs']) == 24 and len(data['qwen_AV_inputs']) == 24 and len(data['qwen_speech_prompts']) == 8
    assert all(r['sample_rate'] == 16000 and r['channels'] == 1 and r['sample_width'] == 2 for r in data['qwen_audio_inputs'])
    runtime = data['qwen_runtime_receipt']
    from datetime import datetime as date_time
    startup_seconds = (date_time.fromisoformat(runtime['cleanup_started_at']) - date_time.fromisoformat(runtime['paid_call_at'])).total_seconds()
    assert 480 <= startup_seconds < 481 and runtime['phase'] == 'waiting_for_task_id'
    assert 'input_payload_uploaded' not in runtime and runtime['error_type'] == 'TimeoutError'
    assert data['qwen_final_stop']['compute_stopped'] and data['qwen_final_stop']['resources_verified']
    assert data['qwen_final_stop']['result_asset_names'] == ['input.zip']
    controls = {k: {'families': v['families'], 'remove_image_answer2': sum(x == 2 for x in v['remove_image_answers']),
                   'remove_audio_answer2': sum(x == 2 for x in v['remove_audio_answers'])} for k, v in data['own_A_remove_controls'].items()}
    output = {'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_p5_technical_20_16',
        'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=. uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/omni-P5-20.16/recompute.py',
        'environment': {'python': platform.python_version(), 'device': 'CPU', 'dependencies': 'Python standard library only; no torch/model imports'},
        'input_projection_sha256': hashlib.sha256((HERE / 'minimal-inputs.json').read_bytes()).hexdigest(),
        'result': 'All assertions passed. Independent arithmetic on preserved rows and exact inline CPU example; no neural inference.',
        'inline_demonstrations': demonstrations, 'temperature_grid': grid, 'selected_temperature': chosen,
        'policy_grid': policy_grid, 'selected_tau': tau, 'final': final,
        'family_counts': {k: len(v) for k, v in panel_sets.items()}, 'split_families_pairwise_disjoint': True,
        'cache_record_reduction': {'families': len(cache), 'max_probability_difference': max(r['uncached_cached_max_abs'] for r in cache),
            'max_Q1_Q2_Q1_difference': max(r['Q1_Q2_Q1_probability_max_abs'] for r in cache), 'all_preserved_fingerprints_unchanged': True,
            'scope': 'Reduction of original per-family diagnostic rows; cache tensors and NN forward were not recreated.'},
        'qwen_input_inspection': {'audio_pairs': 24, 'AV_pairs': 24, 'speech_prompts': 8,
            'all_48_AV_waveforms_reconstructed_digest_equal': True, 'frame_pixel_and_gold_checks': True,
            'synthetic_speech_duration_range': [min(r['duration_seconds'] for r in data['qwen_audio_inputs']), max(r['duration_seconds'] for r in data['qwen_audio_inputs'])]},
        'qwen_failure': {'startup_seconds': startup_seconds, 'phase': runtime['phase'], 'input_upload_reached': False,
            'no_result_zip_asset': True, 'official_app_stopped': data['qwen_app_after']['state'] == 'stopped'},
        'own_A_control_recount': controls}
    path = HERE / 'verification-v1.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'T': temperature, 'tau': tau, 'english': final['english']['acceptance'],
                      'chinese': final['chinese']['acceptance'], 'startup_seconds': startup_seconds}))


if __name__ == '__main__':
    main()
