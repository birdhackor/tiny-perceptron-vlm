"""本人19.24回核：只核immutable bytes與原始列算術，0神經推論。"""

import difflib
import hashlib
import itertools
import json
import math
import platform
import re
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tiny_perceptron.selftrained.tools import ToolCallError, parse_tool_call


ROOT = Path('/tmp/tpb').resolve()
HERE = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-24'
OLD = ROOT / 'docs/technical-reviews/artifacts/omni-P5-19.24'
RAW = Path('/workspace/work/omni-final-completed-41004229262/verified')
FINAL = RAW.parent / 'final-results.zip'
CORE = Path('/workspace/work/omni-core-completed-40988311100/core-checkpoints.zip')
STARTED = datetime.now(UTC).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def trim(value):
    if isinstance(value, dict):
        return {k: trim(v) for k, v in value.items() if k not in {'seconds', 'scope', 'wav_path'}}
    if isinstance(value, list):
        return [trim(v) for v in value]
    return value


def softmax(logits, temperature):
    values = [v / temperature for v in logits]
    exps = [math.exp(v - max(values)) for v in values]
    return [v / math.fsum(exps) for v in exps]


def analysis_ok(analysis, metrics):
    measured, expected = analysis['measured'], analysis['expected']
    return measured['legal'] and len(measured['events']) == len(expected) and all(
        abs(a['frequency'] - b['frequency']) <= metrics['frequency_relative_tolerance'] * b['frequency']
        and abs(a['onset_ms'] - b['onset_ms']) <= metrics['timing_tolerance_ms']
        and abs(a['offset_ms'] - b['offset_ms']) <= metrics['timing_tolerance_ms']
        for a, b in zip(measured['events'], expected, strict=True)
    )


prior_path = 'docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.24/technical-current-v1.json'
original_path = 'docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.24/technical-original-v1.json'
assert file_sha(ROOT / prior_path) == '603d8249d79eeb94e92705c142d27b53d1d32c2cf8d240839d8fe1d866d5953c'
assert file_sha(ROOT / original_path) == '1cefb37887387e48343ee83d472a2a25e4ad1c64b08407782bfff9ff66bfb65f'
prior = read(ROOT / prior_path)
preserved_artifacts = []
for item in prior['artifacts']:
    assert file_sha(ROOT / item['path']) == item['sha256']
    preserved_artifacts.append({'path': item['path'], 'sha256': item['sha256'], 'kind': item['kind']})
dependencies = []
for source in prior['sources']:
    if source['kind'] == 'repository_code':
        actual = file_sha(ROOT / source['path'])
        assert actual == source['sha256'], source['path']
        dependencies.append({'path': source['path'], 'sha256': actual, 'byte_identical': True})
assert len(dependencies) == 19
projection = read(OLD / 'measurement_projection_v1.json')
assert file_sha(OLD / 'measurement_projection_v1.json') == '4a7e20ba975181a8d02dd7119e637b4fa9af2d52f575a7bf47388f9191a088c3'
prov = projection['provenance']
assert file_sha(FINAL) == prov['final_archive_sha256'] == '50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3'
assert FINAL.stat().st_size == prov['final_archive_bytes'] == 1827422
assert file_sha(CORE) == prov['core_archive_sha256'] == '107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee'
assert CORE.stat().st_size == prov['core_archive_bytes'] == 54751619
raw_bindings = []
with zipfile.ZipFile(FINAL) as archive:
    for name, expected in prov['raw_members'].items():
        data = (RAW / name).read_bytes()
        assert len(data) == expected['bytes'] and sha(data) == expected['sha256']
        assert archive.read(name) == data
        raw_bindings.append({'member': name, **expected, 'local_and_zip_equal': True})

definition = read(RAW / 'frozen/data-definition.json')
selection = read(RAW / 'frozen/model-selection.json')
metrics = read(RAW / 'frozen/metrics.json')
integration = read(RAW / 'measurements/integration/results.json')
av_result = read(RAW / 'measurements/av/results.json')
sound_result = read(RAW / 'measurements/sound/results.json')
choice_result = read(RAW / 'measurements/choice/results.json')
origin = read(RAW / 'measurements/integration/part-origin.json')
completion = read(RAW / 'measurements/integration/part-completion.json')
opening = read(RAW / 'frozen/final-opening.json')
execution_sources = read(RAW / 'execution-sources.json')
assert execution_sources['revision'] == prov['source_revision'] == 'ee799d24fd1ea26108af709ec41e49db63d58e31'
assert completion['results_sha256'] == file_sha(RAW / 'measurements/integration/results.json')
assert completion['part_origin_sha256'] == file_sha(RAW / 'measurements/integration/part-origin.json')
for value in (origin, completion):
    assert value['model_selection_sha256'] == file_sha(RAW / 'frozen/model-selection.json')
    assert value['final_opening_sha256'] == file_sha(RAW / 'frozen/final-opening.json')
for name, expected in prov['current_source_bindings_equal_original_execution'].items():
    assert file_sha(ROOT / name) == expected == execution_sources['files'][name]
    if name in opening['evaluation_sources_sha256']:
        assert expected == opening['evaluation_sources_sha256'][name] == origin['evaluation_sources_sha256'][name]
assert origin['data_definition_sha256'] == file_sha(RAW / 'frozen/data-definition.json')
assert origin['metrics_sha256'] == file_sha(RAW / 'frozen/metrics.json')
assert opening['created_at'] <= origin['created_at'] <= completion['created_at']
for name in ('data-definition.json', 'metrics.json', 'model-selection.json'):
    assert file_sha(ROOT / 'docs/omni-products/v1' / name) == file_sha(RAW / 'frozen' / name)

# 真原結果與舊小投影逐欄一致；不以舊報告pass或summary作數據權威。
for projection_key, raw_value in (
    ('auto_rows', integration['auto']['rows']),
    ('old_A_rows', integration['old_A_retention']['rows']),
    ('B_rows', integration['B_retention']['rows']),
    ('B_ordered_rows', integration['B_retention']['ordered_diagnostic']['rows']),
):
    assert projection[projection_key] == trim(raw_value)
assert projection['counterfactual_rows'] == integration['old_A_retention']['counterfactual_diagnostics']['rows']
assert projection['B_expected_representatives'] == definition['B_retention']['values']
assert projection['C_core_final_parents'] == definition['C']['families']['final']
assert projection['C_auto_parents'] == definition['C']['families']['auto']
assert projection['C_primary_rows'] == choice_result['primary']['rows']
assert projection['AV_selected_records'] == av_result['models'][selection['selected']['av']]['records']
assert projection['metrics'] == metrics
assert projection['selection'] == {k: selection[k] for k in ('created_at', 'selected', 'temperature', 'threshold', 'policy_feasible')}

b_reps = integration['B_retention']['rows']
expected_reps = list(itertools.combinations_with_replacement(range(7), 3))
assert definition['B_retention']['values'] == [list(v) for v in expected_reps]
assert len(b_reps) == 84 and {tuple(r['parent']) for r in b_reps} == set(expected_reps)
b_counts = {'old': 0, 'wrapper': 0}
for row in b_reps:
    assert row['parent'] == sorted(row['parent']) and row['gold'] == sum(row['parent'])
    for name in b_counts:
        answer = row[name]
        assert answer['loops'] == 1
        correct = answer['answer'] == row['gold']
        assert row[name + '_correct'] == correct
        b_counts[name] += correct
assert b_counts == {'old': 60, 'wrapper': 60}
ordered = integration['B_retention']['ordered_diagnostic']['rows']
assert len(ordered) == 343 and {tuple(r['numbers']) for r in ordered} == set(itertools.product(range(7), repeat=3))
for row in ordered:
    assert row['parent'] == sorted(row['numbers']) and row['gold'] == sum(row['numbers'])
    assert row['actual']['loops'] == 1
    assert row['correct'] == (row['actual']['answer'] == row['gold'])
ordered_correct = sum(row['correct'] for row in ordered)
assert ordered_correct == 240

auto_defs = {row['family']: row for row in definition['Auto']['families']}
auto_rows = integration['auto']['rows']
assert len(auto_defs) == len(auto_rows) == 100
assert {row['family'] for row in auto_rows} == set(auto_defs)
core_parents = {tuple(v) for v in definition['C']['families']['final']}
auto_parents = {tuple(v) for v in definition['C']['families']['auto']}
assert len(core_parents) == 60 and len(auto_parents) == 30 and not core_parents & auto_parents
auto_table = {key: Counter() for key in ('A', 'B', 'C', 'none')}
composition = Counter()
missing = Counter()
disagreements = []
calculator_executed = 0
max_probability_delta = 0.0
b_index = {tuple(r['parent']): r for r in b_reps}
for row in auto_rows:
    fixed = auto_defs[row['family']]
    route = fixed['gold_route']
    assert route == row['gold_route']
    correct_route = row['actual_route'] == route
    assert correct_route == row['route_correct']
    prediction = row['route_logits']
    probabilities = softmax(prediction['logits'], selection['temperature'])
    chosen = min(range(len(probabilities)), key=lambda i: (-probabilities[i], prediction['ids'][i]))
    assert prediction['answer'] == prediction['ids'][chosen] == row['actual_route']
    max_probability_delta = max(max_probability_delta, *(abs(a - b) for a, b in zip(probabilities, prediction['probabilities'], strict=True)))
    for key in ('gold_route_result', 'real_route_result'):
        if key not in row:
            continue
        result = row[key]
        if route == 'B':
            assert result['correct'] == b_index[tuple(fixed['payload']['numbers'])]['wrapper_correct']
            missing['B_answer'] += not ({'answer', 'actual'} & set(result))
        elif route == 'C':
            assert result['correct'] == (result['actual'] == fixed['gold'])
            missing['C_service_logits'] += 'logits' not in result
            missing['C_choice_corresponding_info'] += not ({'information', 'choice_info'} & set(result))
        elif route == 'A':
            if row['family'].startswith('auto-A-calculator'):
                assert result['tool_trace']['executed'] is False and result['correct'] is False
                calculator_executed += result['tool_trace']['executed']
            elif 'video_audio_definition' in fixed['payload']:
                assert result['actual'] == fixed['gold']
                assert result['correct'] == analysis_ok(result['sound_analysis'], metrics['sound'])
            else:
                assert result['correct'] == (result['actual'] == fixed['gold'])
    assert row['gold_route_end_to_end_correct'] == row['gold_route_result']['correct']
    real = correct_route and not row['contract_error'] and row.get('real_route_result', {}).get('correct', False)
    assert real == row['real_route_end_to_end_correct']
    auto_table[route].update(n=1, route_correct=correct_route, gold_chain=row['gold_route_end_to_end_correct'], real_chain=real)
    if route == 'A':
        composition[fixed['payload'].get('old_A_input', {}).get('task', 'video_audio_definition')] += 1
    if not correct_route:
        disagreements.append({'family': row['family'], 'gold_route': route, 'actual_route': row['actual_route'], 'gold_success': row['gold_route_end_to_end_correct']})
assert max_probability_delta < 1e-12
assert [sum(t[k] for t in auto_table.values()) for k in ('n', 'route_correct', 'gold_chain', 'real_chain')] == [100, 89, 66, 55]
assert dict(composition) == {'tool_call': 15, 'vision_clothing': 1, 'vision_relation': 1, 'ocr': 1, 'voice_qa': 1, 'video_audio_definition': 1}
assert len(disagreements) == 11
assert sum(d['gold_route'] == 'A' and d['actual_route'] == 'none' and d['gold_success'] for d in disagreements) == 5
assert sum(d['gold_route'] == 'none' for d in disagreements) == 6
assert dict(missing) == {'C_service_logits': 60, 'C_choice_corresponding_info': 60, 'B_answer': 60}

old_defs = {(r['task'], r['family'], r['record']): r for r in definition['old_A_retention']}
old_rows = integration['old_A_retention']['rows']
assert len(old_rows) == len(old_defs) == 258
assert len({r['family'] for r in old_rows}) == 180
old_table = {}
for row in old_rows:
    fixed = old_defs[(row['task'], row['family'], row['source_record'])]
    counts = old_table.setdefault(row['task'], Counter())
    counts['observations'] += 1
    for name in ('old', 'wrapper'):
        actual = row[name]['actual']
        if row['task'] == 'tool_call':
            try:
                correct = parse_tool_call(actual).to_dict() == fixed['gold_tool_call']
            except ToolCallError:
                correct = False
        else:
            correct = actual == fixed['gold']
            if row['task'] == 'tool_reply':
                assert not correct  # 回答字串已錯；不把缺少的真executor trace補造為成功。
        assert correct == row[name + '_correct']
        counts[name + '_correct'] += correct
for counts in old_table.values():
    counts['difference_percentage_points'] = 100 * (counts['wrapper_correct'] - counts['old_correct']) / counts['observations']
    assert counts['difference_percentage_points'] == 0
cf = integration['old_A_retention']['counterfactual_diagnostics']['rows']
assert len(cf) == 24 and all(r['actual'] == r['gold'] and r['correct'] and not r['actual_executor_used'] for r in cf)

legacy = integration['legacy_binding']
assert legacy['old_B_manifest_sha256'] == definition['media_bindings']['old_B']['manifest_sha256'] == 'dd077fd7a0f11fa5d95bdcf76a73bf6e035819ab135fa3191ff94f4e3643a1a3'
assert legacy['old_A']['manifest_sha256'] == definition['media_bindings']['old_A']['manifest_sha256'] == 'f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
assert legacy['old_B_depth'] == integration['B_retention']['frozen_depth'] == 1
assert all(file_sha(ROOT / p) == v for p, v in prov['current_source_bindings_equal_original_execution'].items())

# 沿用原測量的逐題預測/事件分析/SNR；沒有重算WAV、重挑模型或再開final。
av = projection['AV_selected_records']
paired = sum(r['answers'] == r['gold'] for r in av)
panels = {str(i): {'n': sum(r['panel'] == i for r in av), 'correct': sum(r['panel'] == i and r['answers'] == r['gold'] for r in av)} for i in range(4)}
removed = {key: sum(r[key + '_answer'] == 2 for r in av) for key in ('remove_image', 'remove_audio')}
swapped = sum(r['waveform_swap_answers'] == r['waveform_swap_gold'] for r in av)
sound = sound_result['codec_and_gold_program']
snd_correct = sum(r['generation']['eos'] and not r['generation']['capped'] and analysis_ok(r['generation']['analysis'], metrics['sound']) for r in sound['rows'])
snr = math.fsum(r['active_reference_snr_db'] for r in sound['rows']) / len(sound['rows'])
chain = sound_result['actual_A_chain']['rows']
chain_correct = sum(r['answer'] == r['gold'] and r['legal'] and analysis_ok(r['analysis_against_gold_program'], metrics['sound']) for r in chain)
assert len(av) == len(sound['rows']) == len(chain) == 200
a_pass = paired >= 170 and all(p['n'] == 50 and p['correct'] / 50 >= .75 for p in panels.values()) and all(v >= 180 for v in removed.values()) and len(av_result['models']) == 6 and snr >= 10 and snd_correct >= 170 and chain_correct >= 160 and all(r['generation']['legal'] for r in sound['rows']) and all(r['legal'] for r in chain) and len(sound['chunk_controls']) == 10 and all(r['chunks_bit_exact'] and r['future_code_prefix_bit_exact'] for r in sound['chunk_controls']) and all(c['difference_percentage_points'] >= -5 for c in old_table.values()) and b_counts['old'] == b_counts['wrapper']
c_rows = choice_result['primary']['rows']
c_counts = {kind: sum(r['tasks'][kind]['answer'] == r['tasks'][kind]['gold'] for r in c_rows) for kind in ('choice', 'info', 'rubric')}
accepted = errors = 0
for row in c_rows:
    choice = row['tasks']['choice']
    accept = max(softmax(choice['logits'], selection['temperature'])) >= selection['threshold'] and choice['answer'] != 'none' and row['choice_info']['answer'] == 'yes'
    error = choice['answer'] != choice['gold'] or not row['world_supported'] or not row['choice_information_sufficient']
    accepted += accept
    errors += accept and error
c_pass = c_counts['choice'] >= 54 and c_counts['info'] >= 51 and c_counts['rubric'] >= 51 and accepted >= 30 and errors / accepted <= .05
assert a_pass and not c_pass and c_counts == {'choice': 39, 'info': 55, 'rubric': 35} and (accepted, errors) == (53, 19)

old_freeze = read(ROOT / prior['source_freeze']['path'])
assert file_sha(ROOT / prior['source_freeze']['path']) == prior['source_freeze']['sha256']
old_page = next(p for p in old_freeze['pages'] if p['page_id'] == '19.24')
old_body = (ROOT / old_page['snapshot']).read_text()
assert sha(old_body.encode()) == prior['source_sha256'] == old_page['source_sha256']
current_body = re.search(r'^## 19\.24\s.*?(?=^## |\Z)', (ROOT / 'course/chapters/19.md').read_text(), re.M | re.S).group(0)
old_blocks = re.findall(r'```python\n(.*?)```', old_body, re.S)
current_blocks = re.findall(r'```python\n(.*?)```', current_body, re.S)
assert old_blocks == current_blocks
table_lines = lambda body: [line for line in body.splitlines() if line.startswith('|')]
assert table_lines(old_body) == table_lines(current_body)
delta = ''.join(difflib.unified_diff(old_body.splitlines(True), current_body.splitlines(True), fromfile=old_page['snapshot'], tofile='source21:19.24'))

result = {
    'kind': 'source21 19.24 immutable retention plus original-row arithmetic; no new model measurement',
    'reviewer_task': '/root/c_v3_technical_coord/canonical_19_24',
    'started_at': STARTED,
    'completed_at': datetime.now(UTC).isoformat(),
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU', 'model_inference': 'none; stdlib arithmetic and strict tool-call parser only'},
    'prior_owner_and_scope': {'owner': prior['reviewer_task'], 'original_created_at': read(ROOT / original_path)['created_at'], 'current_created_at': prior['created_at'], 'projection_collected_at': projection['collected_at'], 'scope': '2026-10-30 fixed C-v2/A/B measurements at revision ee799d24fd1ea26108af709ec41e49db63d58e31. Prior pass not used as data authority; raw rows and producer bindings checked.'},
    'prior_reports': [{'path': p, 'sha256': file_sha(ROOT / p)} for p in (original_path, prior_path)],
    'preserved_artifacts': preserved_artifacts,
    'dependency_identities': dependencies,
    'raw_archive_identities': {'final': {'local_path': str(FINAL), 'url': prov['final_archive_url'], 'bytes': FINAL.stat().st_size, 'sha256': file_sha(FINAL)}, 'core': {'local_path': str(CORE), 'url': prov['core_archive_url'], 'bytes': CORE.stat().st_size, 'sha256': file_sha(CORE)}, 'scope': 'external original bytes read in place; not Git artifacts; no weights copied or loaded'},
    'raw_member_bindings': raw_bindings,
    'producer': {'revision': execution_sources['revision'], 'final_opened_at': opening['created_at'], 'integration_started_at': origin['created_at'], 'integration_completed_at': completion['created_at'], 'current_original_execution_dependencies_identical': len(prov['current_source_bindings_equal_original_execution']), 'projection_matches_original_result_rows': True},
    'source_delta': {'prior_sha256': prior['source_sha256'], 'current_sha256': sha(current_body.encode()), 'exact_two_code_blocks_unchanged': True, 'both_tables_byte_identical': True, 'unified_diff': delta, 'semantic_scope': 'Only C-v2 historical versus C-v3 typed-device boundary and concluding generation label changed; fresh source-first probe checks this delta separately.'},
    'Auto': {'table': auto_table, 'totals_n_route_gold_real': [100, 89, 66, 55], 'A_composition': composition, 'calculator_executed_in_preserved_trace': calculator_executed, 'route_disagreements': disagreements, 'max_recalculated_route_probability_delta': max_probability_delta, 'core_final_parents': len(core_parents), 'auto_C_parents': len(auto_parents), 'parent_overlap': 0, 'missing_outputs_in_recorded_results': missing, 'observed_main_row_keys': sorted(set().union(*(row.keys() for row in auto_rows))), 'observed_B_result_key_sets': sorted({tuple(sorted(row[k])) for row in auto_rows if row['gold_route'] == 'B' for k in ('gold_route_result', 'real_route_result') if k in row}), 'observed_C_result_key_sets': sorted({tuple(sorted(row[k])) for row in auto_rows if row['gold_route'] == 'C' for k in ('gold_route_result', 'real_route_result') if k in row}), 'scope': 'B84 comparison checks stored correctness consistency only; it does not reconstruct missing original Auto B answers. No service logits/info/adapter outputs fabricated.'},
    'old_A': {'observations': len(old_rows), 'parent_families': len({r['family'] for r in old_rows}), 'tasks': old_table, 'counterfactual_n': len(cf), 'counterfactual_correct': sum(r['correct'] for r in cf), 'counterfactual_actual_executor_used': False, 'counterfactual_example': cf[0]},
    'old_B': {'representatives': {'n': len(b_reps), **b_counts, 'loops': 1, 'predeclared_sorted_with_repetition': True}, 'ordered': {'n': len(ordered), 'correct': ordered_correct, 'parents': len({tuple(r['parent']) for r in ordered})}, 'manifest_sha256_from_actual_bound_preflight': legacy['old_B_manifest_sha256'], 'fixed_source': definition['media_bindings']['old_B'], 'scope': 'Historical binding retained from raw legacy_preflight receipt with byte-identical producer/check_manifest source, actual result loops=1, and original core/final archives. No old-B weights reloaded or newly hashed in this callback.'},
    'old_A_binding': {'manifest_sha256': legacy['old_A']['manifest_sha256'], 'files': legacy['old_A']['files'], 'fixed_source': definition['media_bindings']['old_A'], 'scope': 'retained actual preflight identity; no new A inference'},
    'gates': {'A': {'paired': paired, 'n': 200, 'panels': panels, 'deleted_cue_correct': removed, 'waveform_swap_paired': swapped, 'sound_gold': snd_correct, 'sound_n': 200, 'mean_preserved_snr_db': snr, 'actual_chain_correct': chain_correct, 'actual_chain_n': 200, 'passed': a_pass}, 'C_v2': {'correct_by_type': c_counts, 'families_per_correlated_type': len(c_rows), 'accepted': accepted, 'errors': errors, 'risk': errors / accepted, 'passed': c_pass}, 'fixed_selection': projection['selection'], 'scope': 'Retained original predictions/event analyses/SNR and frozen thresholds; no waveform DSP, quality experiment, calibration, seed selection, or final reopening'},
    'assertions_passed': True,
    'new_model_inference_calls': 0,
    'weights_loaded_or_copied': 0,
    'training_updates': 0,
    'new_final_openings': 0,
    'Modal_calls': 0,
    'network_requests': 0,
}
output = HERE / 'retention-execution.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(output), 'sha256': file_sha(output), 'assertions_passed': True, 'Auto': [100, 89, 66, 55], 'A_retention': [258, 180], 'B_representatives': [60, 84], 'B_ordered': [240, 343], 'A_gate': a_pass, 'C_v2_gate': c_pass, 'new_model_inference_calls': 0}, ensure_ascii=False))
