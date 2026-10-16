"""19.17 independent bounded CPU audit; no training or formal evaluation entrypoint."""
import hashlib
import itertools
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.data import CANDIDATES, REQUIRED, answer, render
from tiny_perceptron.lesson_products.inference import Product, auto_solve, rule_depth

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).with_name('19.17-execution.json')

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def load(path):
    return json.loads((ROOT / path).read_text())

def rows(path):
    with (ROOT / path).open() as fp:
        for line in fp:
            yield json.loads(line)

torch.set_num_threads(2)
result = {
    'reviewer_task': '/root/backfill_tech_19_17',
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/19.17-probe.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'threads': '2'},
    'scope': 'New execution only recomputes existing raw predictions/data and bounded public CPU inference. It does not retrain, reselect settings, or invoke evaluate_frozen/select_settings. Formal test and timing figures remain existing primary receipts.',
}
freeze_page = next(x for x in load('docs/course-revision-20261016/freeze-14.json')['pages'] if x['page_id'] == '19.17')
body = lesson_slices((ROOT / 'course/chapters/19.md').read_text())['19.17']
assert body == (ROOT / freeze_page['snapshot']).read_text()
assert hashlib.sha256(body.encode()).hexdigest() == freeze_page['source_sha256']
result['frozen_section'] = freeze_page
manifest = load('docs/lesson-products/data-v2-manifest.json')
assert sha('outputs/lesson-products/data-v2/manifest.json') == sha('docs/lesson-products/data-v2-manifest.json')
for name, meta in manifest['files'].items():
    assert sha('outputs/lesson-products/data-v2/' + name) == meta['sha256']
settings = load('docs/lesson-products/results/settings-freeze-v1.json')
formal = load('docs/lesson-products/results/final-report-v1.json')
assert sha('docs/lesson-products/results/settings-freeze-v1.json') == formal['settings_freeze_sha256']
assert settings['data_manifest_sha256'] == sha('docs/lesson-products/data-v2-manifest.json')
for name, expected in settings['source_sha256s'].items():
    assert sha('tiny_perceptron/lesson_products/' + name) == expected
raw_path = 'outputs/lesson-products/final-report-v1.predictions.jsonl'
assert sha(raw_path) == formal['predictions']['sha256']
result['primary_identity'] = {
    'settings_freeze_sha256': formal['settings_freeze_sha256'],
    'data_manifest_sha256': settings['data_manifest_sha256'],
    'raw_predictions_path': raw_path,
    'raw_predictions_sha256': sha(raw_path),
    'raw_predictions_bytes': (ROOT / raw_path).stat().st_size,
    'matched_frozen_code_files': settings['source_sha256s'],
}
b_core = {}
family_splits = defaultdict(set)
b_counts = Counter()
for row in rows('outputs/lesson-products/data-v2/b.jsonl'):
    if row['split'] not in ('train', 'calibration', 'policy', 'test'):
        continue
    family_splits[row['family_id']].add(row['split'])
    b_counts[row['split'] + '/' + row['task']] += 1
    assert answer(row['world']) == row['gold']
    if row['split'] == 'test':
        b_core[row['id']] = row
c_counts, c_tasks = Counter(), Counter()
for row in rows('outputs/lesson-products/data-v2/c.jsonl'):
    if row['split'] not in ('train', 'calibration', 'policy', 'test'):
        continue
    family_splits[row['family_id']].add(row['split'])
    c_counts[row['split'] + '/' + row['question']] += 1
    c_tasks[row['split'] + '/' + row['task']] += 1
    if row['question'] == 'choice' and not row.get('missing_best'):
        assert tuple(row['candidates']) == CANDIDATES['choice']
        assert row['gold'] == row['task']
    if row['question'] == 'info':
        assert row['gold'] == ('no' if row['omitted'] else 'yes')
assert all(len(s) == 1 for s in family_splits.values())
result['data_contract_audit'] = {
    'b_core_counts': dict(sorted(b_counts.items())),
    'c_core_question_counts': dict(sorted(c_counts.items())),
    'c_core_task_counts': dict(sorted(c_tasks.items())),
    'joint_families': len(family_splits), 'joint_family_cross_split_overlap': 0,
    'checked': 'Every core B gold recomputed from world events; every C non-reduced Choice label/candidate set and info label checked against task/omitted fields. C generator labels use required-field contract, not solver outcomes.',
    'required': REQUIRED,
    'limitation': manifest['audit']['c']['scope'],
}
policy = load('docs/lesson-products/results/b-auto-policy.json')
grid = settings['auto_search']
expected_rules = [{'version': 1, 'low_max_events': low, 'medium_max_events': medium, 'sum_depth': d} for low, medium, d in itertools.product(range(5), range(5), (1, 3, 5)) if low <= medium]
assert [x['rule'] for x in grid] == expected_rules
feasible = [x for x in grid if x['quality']['mean_actual_b_loops_all_complete'] <= 3]
selected = min(feasible, key=lambda x: (-x['quality']['accuracy_end_to_end'], x['quality']['mean_actual_b_loops_all_complete'], json.dumps(x['rule'], sort_keys=True)))
assert all(policy[k] == v for k, v in selected['rule'].items())
assert policy['policy_quality'] == selected['quality']
result['policy_receipt_recalculation'] = {
    'grid_count': len(grid), 'feasible_count': len(feasible), 'selected': selected,
    'boxes_depth_by_event_count': {str(n): rule_depth(policy, 'boxes', n) for n in range(1, 7)},
    'sum_depth': rule_depth(policy, 'sum'),
    'scope': 'Recomputed grid enumeration and constrained ranking from existing frozen policy qualities; did not regenerate policy logits or rerun selection.',
}
records = []
long_shift = Counter()
for row in rows(raw_path):
    if row['product'] == 'b' and row['split'] == 'test':
        data = b_core[row['id']]
        assert row['gold'] == data['gold'] and row['family_id'] == data['family_id']
        for d, logits in row['scores_by_depth'].items():
            assert max(range(len(logits)), key=logits.__getitem__) == row['answers_by_depth'][d]
        records.append(row)
    if row['product'] == 'b' and row['split'] == 'long_shift':
        long_shift['count'] += 1
        long_shift['correct'] += max(range(len(row['raw_scores'])), key=row['raw_scores'].__getitem__) == row['gold']
assert len(records) == len(b_core) == 2998
recomputed = {}
for effort, fixed in [('low', 1), ('medium', 3), ('high', 5), ('auto', None)]:
    correct = total = covered = 0
    choices = Counter()
    for row in records:
        if row['c_choice'] not in ('boxes', 'sum') or row['c_info'] != 'yes':
            continue
        covered += 1
        data = b_core[row['id']]
        d = fixed or rule_depth(policy, row['c_choice'], len(data['world'].get('events', [])))
        total += d
        correct += row['answers_by_depth'][str(d)] == row['gold']
        choices[row['task'] + '/' + str(d)] += 1
    assert correct == formal['auto'][effort]['correct_end_to_end']
    assert total / len(records) == formal['auto'][effort]['mean_actual_b_loops_all_complete']
    recomputed[effort] = {'count': len(records), 'correct': correct, 'covered': covered, 'total_b_loops': total, 'mean_b_loops': total / len(records), 'routing_counts': dict(choices)}
changed = [row for row in records if row['task'] == 'sum']
result['test_raw_recalculation'] = {
    'efforts': recomputed,
    'changed_arithmetic_cases': len(changed),
    'medium_minus_auto_b_loops': recomputed['medium']['total_b_loops'] - recomputed['auto']['total_b_loops'],
    'medium_minus_auto_correct': recomputed['medium']['correct'] - recomputed['auto']['correct'],
    'arithmetic_correct_low1': sum(x['answers_by_depth']['1'] == x['gold'] for x in changed),
    'arithmetic_correct_medium3': sum(x['answers_by_depth']['3'] == x['gold'] for x in changed),
    'long_shift_depth5': dict(long_shift),
}
b, c = Product(ROOT / 'outputs/lesson-products/products/b', 'b'), Product(ROOT / 'outputs/lesson-products/products/c', 'c')
assert b.auto_policy == policy
states = {
    'complete_boxes': '兩盒追蹤;A=2;B=6;事件=交換;查詢=B',
    'missing_a': '兩盒追蹤;A=?;B=6;事件=交換;查詢=B',
    'missing_b': '兩盒追蹤;A=2;B=?;事件=交換;查詢=B',
    'missing_events': '兩盒追蹤;A=2;B=6;事件=?;查詢=B',
    'missing_query': '兩盒追蹤;A=2;B=6;事件=交換;查詢=?',
    'complete_sum': '三數加法;x=1;y=2;z=3',
    'missing_sum': '三數加法;x=1;y=?;z=3',
    'unsupported': '天氣預測',
}
probes = []
for name, effort, limit in [('complete_boxes','auto',5), ('complete_boxes','low',5), ('complete_boxes','medium',5), ('complete_boxes','high',5), ('complete_boxes','high',2), ('complete_sum','auto',5), ('missing_a','high',5), ('missing_b','auto',5), ('missing_events','auto',5), ('missing_query','auto',5), ('missing_sum','high',5), ('unsupported','high',5)]:
    value = auto_solve(b, c, states[name], effort, limit)
    expected = 'unsupported' if name == 'unsupported' else 'clarification_needed' if name.startswith('missing') else 'budget_limited' if limit == 2 else 'answered'
    assert value['status'] == expected, (name, value)
    assert value['b_loops'] <= limit
    probes.append({'case': name, 'state': states[name], **value})
invalids = []
for effort,limit in [('auto', 0),('auto',7),('invalid',5),('auto',True)]:
    try:
        auto_solve(b,c,states['complete_boxes'],effort,limit)
    except ValueError as e:
        invalids.append({'effort': effort, 'max_loops': limit, 'error': str(e)})
    else:
        raise AssertionError('bad effort/cap accepted')
result['bounded_public_cpu_inference'] = {'probes': probes, 'invalid_budget_probes': invalids, 'count': len(probes), 'scope': 'Current frozen actual B/C weights; individual finite grammar examples, not new accuracy estimate.'}
block_calls = []
hook = b.model.shared.register_forward_hook(lambda module, args, output: block_calls.append(1))
b.solve(states['complete_boxes'], loops=1)
b.solve(states['complete_boxes'], loops=5)
hook.remove()
assert len(block_calls) == 6
result['restart_work_probe'] = {'separate_depths': [1,5], 'actual_shared_block_calls': len(block_calls), 'interpretation': 'Each solve initializes its own hidden state; separate low then high executes 1+5 shared blocks.'}
latency = formal['latency']
assert len(latency['sample_ids']) == 24 and latency['warmup_per_effort'] == 3 and latency['repetitions'] == 3
assert all(v['requests'] == 72 for v in latency['measurements'].values())
result['existing_latency_receipt'] = latency
result['missing_fact_derivation'] = {
    'input': states['missing_a'],
    'completions': [{'a':a, 'b_final':answer({'task':'boxes','a':a,'b':6,'events':[('swap',0)],'query':'B'})} for a in [2,3]],
    'interpretation': 'Same visible input permits B_final=2 or 3. More computation cannot identify an unprovided initial a without an added assumption or clarification. Contract missing_b also asks clarification even where that particular answer is inferable.',
}
result['checked_repository_sha256s'] = {p: sha(p) for p in [
    'tiny_perceptron/lesson_products/inference.py', 'tiny_perceptron/lesson_products/data.py',
    'tiny_perceptron/lesson_products/evaluation.py', 'tiny_perceptron/lesson_products/model.py',
    'tiny_perceptron/lesson_products/train.py', 'tiny_perceptron/lesson_products/training.py',
    'docs/lesson-products/results/settings-freeze-v1.json', 'docs/lesson-products/results/pretest-freeze-record-v1.json',
    'docs/lesson-products/results/final-report-v1.json', 'docs/lesson-products/results/b-auto-policy.json',
    'docs/lesson-products/results/initial-c-execution.json', 'docs/lesson-products/data-v2-manifest.json',
    freeze_page['snapshot'],
]}
result['all_assertions_passed'] = True
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'output': str(OUT.relative_to(ROOT)), 'sha256':sha(str(OUT.relative_to(ROOT))), 'recalculated': result['test_raw_recalculation'], 'bounded_cpu_probes': len(probes), 'all_assertions_passed': True}, ensure_ascii=False))
