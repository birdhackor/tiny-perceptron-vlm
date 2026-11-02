"""Portable 19.23 review: saved-logit arithmetic and artificial CPU examples only."""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
import types
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.omni_products import data, questions
from tiny_perceptron.omni_products.training_runtime import choice_training_core


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def argmax(row):
    return min(range(len(row['ids'])), key=lambda i: (-row['logits'][i], row['ids'][i]))


def answer(row):
    return row['ids'][argmax(row)]


def probabilities(row, temperature=1.0):
    scaled = [v / temperature for v in row['logits']]
    peak = max(scaled)
    exponentials = [math.exp(v - peak) for v in scaled]
    denominator = sum(exponentials)
    return [v / denominator for v in exponentials]


def nll(row, temperature):
    scaled = [v / temperature for v in row['logits']]
    peak = max(scaled)
    return peak + math.log(sum(math.exp(v - peak) for v in scaled)) - scaled[row['ids'].index(row['gold'])]


def policy(rows, temperature, threshold):
    accepted, errors = 0, 0
    for row in rows:
        prediction = answer(row['tasks']['choice'])
        take = (max(probabilities(row['tasks']['choice'], temperature)) >= threshold
                and prediction != 'none' and answer(row['choice_info']) == 'yes')
        accepted += take
        errors += take and (prediction != row['tasks']['choice']['gold'] or not row['world_supported']
                           or not row['choice_information_sufficient'])
    return {'accepted': accepted, 'errors': errors, 'families': len(rows),
            'coverage': accepted / len(rows), 'risk': errors / accepted if accepted else None}


raw = json.loads((OUT / 'fixed-inputs.json').read_text())
T, tau = raw['temperature'], raw['threshold']
stats = {}
for name in ('primary', 'compact'):
    rows = raw[name]
    assert len(rows) == len({row['family'] for row in rows}) == 60
    stats[name] = {kind: {'correct': sum(answer(row['tasks'][kind]) == row['tasks'][kind]['gold'] for row in rows),
                         'n': 60} for kind in ('choice', 'info', 'rubric')}
    assert all(answer(record) == record['answer'] for row in rows for record in row['tasks'].values())
stats['primary_panels'] = {panel: {'correct': sum(answer(row['tasks']['choice']) == row['tasks']['choice']['gold']
                                                for row in raw['primary'] if row['panel'] == panel),
                                   'n': sum(row['panel'] == panel for row in raw['primary'])}
                           for panel in ('complete', 'removed', 'missing')}
stats['choice_NLL_raw'] = sum(nll(row['tasks']['choice'], 1) for row in raw['primary']) / 60
stats['choice_NLL_fixed_T'] = sum(nll(row['tasks']['choice'], T) for row in raw['primary']) / 60
assert abs(stats['choice_NLL_raw'] - 2.690098901766063) < 1e-12
assert abs(stats['choice_NLL_fixed_T'] - 3.962385753167719) < 1e-12
stats['final_acceptance'] = policy(raw['primary'], T, tau)
assert stats['final_acceptance']['accepted'] == 53 and stats['final_acceptance']['errors'] == 19
stats['candidate_diagnostics'] = {}
for count in range(2, 9):
    rows = [row for row in raw['diagnostic'] if row['total_choices_including_none'] == count]
    assert len(rows) == len({row['family'] for row in rows}) == 60
    assert all(len(row['ids']) == count and answer(row) == row['answer'] for row in rows)
    stats['candidate_diagnostics'][str(count)] = {'correct': sum(answer(row) == row['gold'] for row in rows), 'n': len(rows)}
assert [v['correct'] for v in stats['candidate_diagnostics'].values()] == [37, 22, 29, 37, 37, 32, 26]
temperature_grid = [0.25 * 2 ** (k / 8) for k in range(33)]
calibration_grid = [{'T': temperature, 'equal_family_equal_type_NLL':
                     sum(sum(nll(row['tasks'][kind], temperature) for kind in ('choice', 'info', 'rubric')) / 3
                         for row in raw['calibration']) / len(raw['calibration'])}
                    for temperature in temperature_grid]
recorded_grid_winner = min(calibration_grid, key=lambda row: (row['equal_family_equal_type_NLL'], abs(row['T'] - 1), row['T']))
assert len(raw['calibration']) == 80 and recorded_grid_winner['T'] == T
policy_grid = [{'tau': value, **policy(raw['policy'], T, value)} for value in raw['original_C_metrics']['threshold_grid']]
feasible = [row for row in policy_grid if row['accepted'] and row['risk'] <= .05]
recorded_policy_winner = max(feasible, key=lambda row: (row['coverage'], row['tau']))
assert len(raw['policy']) == 60 and recorded_policy_winner['tau'] == tau
stats['selection_arithmetic_only'] = {'T': T, 'tau': tau, 'calibration_grid': calibration_grid, 'policy_grid': policy_grid,
                                     'fixed_policy_winner': recorded_policy_winner,
                                     'no_new_parameter_chosen_from_final': True}
invariants = raw['development_invariants']['rows']
assert len(invariants) == 8
largest = max(value for row in invariants for key, value in row.items() if key.endswith('_max_abs'))
assert largest <= 1e-5 and all(row['permutation_answer_same'] for row in invariants)
stats['recorded_development_invariants_recomputed'] = {'n': 8, 'largest_recorded_error': largest,
                                                      'all_within_1e-5': True,
                                                      'trained_model_not_reexecuted': True}
stats['training_log_inspection'] = {}
for mode, bundle in raw['training'].items():
    result, log = bundle['result'], bundle['log']
    best = max(log, key=lambda row: (row['dev_score'], -row['update']))
    assert result['complete'] and result['updates'] == log[-1]['update'] == 6000
    assert best['update'] == result['best']['update'] == 5000
    assert result['trainable_parameters'] == 279170
    stats['training_log_inspection'][mode] = {'updates': result['updates'], 'trainable_parameters': result['trainable_parameters'],
                                            'seed': result['config']['seed'], 'best_update_from_12_dev_checkpoints': best['update'],
                                            'initial_parameter_sha256': result['initial_parameter_sha256']}
assert raw['training']['augmented']['result']['initial_parameter_sha256'] == raw['training']['compact']['result']['initial_parameter_sha256']

# Data generation, not neural prediction. Parse visible strings independently to verify all labels.
generated = questions.panel_c('final')
frozen_first = raw['first_final_input']
assert generated[0]['questions']['choice'].state == frozen_first['questions']['choice']['state']
assert [c.id for c in generated[0]['questions']['choice'].candidates] == [c['id'] for c in frozen_first['questions']['choice']['candidates']]
label_comparisons = 0
for generated_row, preserved_row, original_labels in zip(generated, raw['primary'], raw['original_final_labels']):
    assert generated_row['family'] == preserved_row['family']
    assert generated_row['family'] == original_labels['family']
    assert [c.id for c in generated_row['questions']['choice'].candidates] == original_labels['choice_ids']
    state = generated_row['questions']['choice'].state
    if generated_row['source'] == 'compact':
        facts = re.findall(r'[ABC]: task=(record|play|save); permission=(yes|no|unknown); ready=(yes|no|unknown)', state)
    elif generated_row['source'] == 'sentence':
        facts = re.findall(r'Device [ABC] requests (record|play|save)\. Its permission is (yes|no|unknown); its readiness is (yes|no|unknown)', state)
    elif generated_row['source'] == 'chinese':
        translate = {'是': 'yes', '否': 'no', '未知': 'unknown'}
        facts = [(task, translate[p], translate[r]) for task, p, r in
                 re.findall(r'裝置[ABC]：任務(record|play|save)；許可(是|否|未知)；就緒(是|否|未知)', state)]
    else:
        facts = [(task, p, r) for task, r, p in re.findall(
            r'Status for [ABC]: requested action (record|play|save), readiness (yes|no|unknown), permission (yes|no|unknown)', state)]
    assert len(facts) == 3
    task, permission, ready = facts[0]
    choice_info = 'no' if 'unknown' in (permission, ready) else 'yes'
    gold_choice = ('none' if choice_info == 'no' else 'stop' if permission == 'no' else 'wait' if ready == 'no' else task)
    if gold_choice not in [c.id for c in generated_row['questions']['choice'].candidates]:
        gold_choice = 'none'
    btask, bp, br = facts[1]
    required = (bp, br) if btask == 'record' else (bp,) if btask == 'play' else (br,)
    gold_info = 'no' if 'unknown' in required else 'yes'
    ctask, cp, cr = facts[2]
    p, r = cp == 'yes', cr == 'yes'
    grade = int(p) + int(r) if ctask == 'record' else (2 if p else int(r)) if ctask == 'play' else (2 if r else int(p))
    for kind, expected in [('choice', gold_choice), ('info', gold_info), ('rubric', str(grade)), ('choice_info', choice_info)]:
        assert generated_row['questions'][kind].gold == expected
        assert original_labels['golds'][kind] == expected
        if kind != 'choice_info':
            assert preserved_row['tasks'][kind]['gold'] == expected
        label_comparisons += 1
splits = data.c_splits()
assert {key: len(value) for key, value in splits.items()} == raw['original_frozen_C_family_counts']
visible_by_split = {}
for split, families in splits.items():
    seen = set()
    for index, parent in enumerate(families):
        panel = (('complete', 'removed', 'missing')[index % 3] if split == 'train' else
                 'missing' if split in ('policy', 'final') and index >= 54 else
                 'removed' if split in ('policy', 'final') and index >= 48 else 'complete')
        for variant in (range(3) if split == 'train' else (0,)):
            seen.add(data.fingerprint(data.visible_c_facts(parent, index + variant, panel)))
    visible_by_split[split] = seen
intersections = {f'{left}/{right}': len(visible_by_split[left] & visible_by_split[right])
                 for i, left in enumerate(splits) for right in list(splits)[i+1:]}
assert not any(intersections.values())
augmented, compact = choice_training_core('augmented'), choice_training_core('compact')
assert len(augmented) == 10836 and len(compact) == 3612
assert [q for i, q in enumerate(augmented) if i % 12 < 4] == compact
stats['data_inspection'] = {'family_counts': {key: len(value) for key, value in splits.items()},
                            'core_train_questions_augmented': len(augmented), 'compact_variant0': len(compact),
                            'cross_split_visible_fact_intersections': intersections,
                            'independent_visible_label_comparisons': label_comparisons,
                            'held_out_final_families': sum(row['source'] == 'held-out' for row in generated),
                            'first_final_golds': {k: q.gold for k, q in generated[0]['questions'].items()}}

# Execute the exact frozen teaching code. All weights below are newly random, never loaded.
lesson = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.23.md'
body = lesson.read_text()
blocks = re.findall(r'```python\n(.*?)\n```', body, flags=re.S)
assert len(blocks) == 3
module = types.ModuleType('review_19_23_inline')
sys.modules[module.__name__] = module
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for index, block in enumerate(blocks):
        exec(compile(block, f'frozen-19.23-python-block-{index+1}', 'exec'), module.__dict__)
formal_definitions = blocks[2].split('# 完整正式架構的隨機初始化核對')[0]
repo = ROOT / 'tiny_perceptron/omni_products/choice.py'
repo_tree = ast.parse(repo.read_text())
repo_tree.body = repo_tree.body[1:]  # module docstring only
assert ast.dump(repo_tree, include_attributes=False) == ast.dump(ast.parse(formal_definitions), include_attributes=False)
model = module.formal_c
assert sum(p.numel() for p in model.parameters()) == 279170
torch.set_num_threads(1)
with torch.no_grad():
    tiny_before = module.model.score(module.M, *module.choice)
    third = (module.choice[0], module.choice[1], module.choice[2] + [('bill', '查帳單明細')])
    tiny_after = module.model.score(module.M, *third)
    added_candidate_score_error = float((tiny_before['scores'][:2] - tiny_after['scores'][:2]).abs().max())
    assert added_candidate_score_error == 0
    q1, qi = module.example_choice, module.example_info
    qr = module.Question(q1.state + ' Extra.', 'rubric', 'Grade fields.',
                         tuple(module.Candidate(str(i), 'A has grade ' + str(i)) for i in range(3)))
    batched = model(module.collate_questions([q1, qi, qr]))['logits']
    batch_errors = []
    for index, q in enumerate((q1, qi, qr)):
        single = model(module.collate_questions([q]))['logits'][0]
        n = len(q.candidates) + int(q.kind == 'choice')
        batch_errors.append(float((single[:n] - batched[index, :n]).abs().max()))
        assert torch.isneginf(batched[index, n:]).all()
    assert max(batch_errors) <= 1e-5
    mem = model.state_memory(q1.state)
    prior = mem.values.clone()
    mem.values.add_(1)  # artificial returned copy only; internal cache is never written
    assert torch.equal(model.state_memory(q1.state).values, prior)
    assert model.memory_key(q1.state) != model.memory_key(q1.state + ' changed')
    other = module.ChoiceModel().eval().requires_grad_(False)
    assert other.memory_key(q1.state) != model.memory_key(q1.state)
    nonfinite = {}
    for name, value in [('nan', float('nan')), ('positive_infinity', float('inf'))]:
        try:
            prediction = model.predict(q1, temperature=value)
            nonfinite[name] = {'rejected': False, 'probabilities_finite': all(math.isfinite(p) for p in prediction['probabilities']),
                               'probabilities_repr': repr(prediction['probabilities'])}
        except ValueError as error:
            nonfinite[name] = {'rejected': True, 'error': str(error)}
tie_scores = torch.tensor([0.0, 1e-8], dtype=torch.float32)
tie_probabilities = tie_scores.softmax(0)
tie_selected = module.selected({'ids': ['a', 'b'], 'scores': tie_scores, 'p': tie_probabilities})
assert tie_scores[0] < tie_scores[1] and tie_probabilities[0] == tie_probabilities[1] and tie_selected == 'a'
inline = {
    'python_block_sha256': [sha(block.encode()) for block in blocks],
    'exact_frozen_teaching_code_stdout': captured.getvalue(),
    'formal_inline_AST_equals_repository_implementation_excluding_module_docstring': True,
    'formal_random_model_parameters': sum(p.numel() for p in model.parameters()),
    'third_candidate_original_tiny_scores_max_abs': added_candidate_score_error,
    'third_candidate_original_probabilities_before': tiny_before['p'][:2].tolist(),
    'third_candidate_original_probabilities_after': tiny_after['p'][:2].tolist(),
    'artificial_formal_mixed_batch_errors': batch_errors,
    'artificial_formal_padding_and_returned_cache_copy_checks': True,
    'tiny_softmax_false_tie_counterexample': {'scores': tie_scores.tolist(), 'p': tie_probabilities.tolist(),
                                             'selected_from_p': tie_selected, 'raw_score_winner': 'b',
                                             'scope': 'Only short TinyDecision selected(); full ChoiceModel correctly selects raw logits.'},
    'formal_predict_nonfinite_temperature': nonfinite,
    'utf8_Han171_bytes': len(('漢' * 171).encode('utf-8')),
    'no_trained_weights_loaded_no_gradient_or_parameter_update': True,
}
result = {'created_at': datetime.now(timezone.utc).isoformat(),
          'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'CPU',
                          'dtype': 'float32 artificial neural examples; Python float64 math for saved logits',
                          'platform': platform.platform(), 'threads': str(torch.get_num_threads())},
          'source_sha256': sha(lesson.read_bytes()), 'statistics': stats, 'inline': inline,
          'limits': ['No repeated formal final-family neural inference, training, T/tau retuning, Modal or workflow dispatch.',
                     'Preserved recorded dev invariants inspected and recomputed from values; original hidden tensors were not available.',
                     'No actual web/mobile rendering, network model download, human reader study, GPU/MPS/Windows or checkpoint continuation.']}
(OUT / 'recompute-output.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'primary': stats['primary'], 'compact': stats['compact'], 'final_acceptance': stats['final_acceptance'],
                  'choice_NLL': [stats['choice_NLL_raw'], stats['choice_NLL_fixed_T']],
                  'tiny_false_tie_selected': tie_selected, 'expected_raw_score_winner': 'b',
                  'formal_AST_equal': True, 'no_trained_network_executed': True}, ensure_ascii=False))
