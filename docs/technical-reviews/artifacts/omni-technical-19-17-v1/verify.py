"""19.17 的有界 CPU 核驗；不訓練、不重開正式 final。"""

import ast
import contextlib
import gzip
import hashlib
import inspect
import io
import itertools
import json
import platform
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.lesson_products import data, evaluation, inference


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-19-17-v1')
C_CACHE = Path('/workspace/work/reference/omni-technical-19-16-v1/product-c')
COMMAND = 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-17-v1/verify.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


torch.set_num_threads(2)
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'command': COMMAND,
    'environment': {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device': 'cpu',
        'threads': str(torch.get_num_threads()),
        'platform': platform.platform(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    },
    'scope': 'Current CPU verification of embedded lesson code and small frozen published B/C inference, plus recomputation from historical raw predictions. No training, GPU, paid work, or new formal final evaluation.',
}
freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'
freeze = read(freeze_path)
page = next(p for p in freeze['pages'] if p['page_id'] == '19.17')
body = (ROOT / page['snapshot']).read_text()
assert sha(ROOT / page['snapshot']) == page['source_sha256']
current = (ROOT / page['source']).read_text()
section = re.search(r'^## 19\.17 .*?(?=^## |\Z)', current, re.M | re.S).group()
assert section == body
assert page['figures_sha256'] == {} and not re.search(r'\.svg(?:\)|\s|\")', body)
result['freeze'] = {'path': str(freeze_path.relative_to(ROOT)), 'sha256': sha(freeze_path), 'snapshot': page['snapshot'], 'source_sha256': page['source_sha256'], 'current_section_exact_match': True, 'figure_map': {}}
result['linked_context'] = []
for ident in ('7.23', '19.14', '1.15', '19.24'):
    p = next(p for p in freeze['pages'] if p['page_id'] == ident)
    assert sha(ROOT / p['snapshot']) == p['source_sha256']
    result['linked_context'].append({'page_id': ident, 'snapshot': p['snapshot'], 'source_sha256': p['source_sha256'], 'selector': p['selector'], 'figures_sha256': p['figures_sha256']})
blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(blocks) == 2
embedded = '\n'.join(blocks)
code_path = OUT / 'embedded-lesson-code.py'
if code_path.exists():
    assert code_path.read_text() == embedded
else:
    code_path.write_text(embedded)
namespace = {}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    for index, code in enumerate(blocks):
        exec(compile(code, page['snapshot'] + f':python-block-{index+1}', 'exec'), namespace)
func_ast = ast.parse(blocks[0]).body[0]
library_ast = ast.parse(inspect.getsource(inference.rule_depth)).body[0]
assert ast.dump(func_ast, include_attributes=False) == ast.dump(library_ast, include_attributes=False)
depths = [namespace['rule_depth'](namespace['policy'], 'boxes', n) for n in range(1, 7)]
assert depths == [3, 3, 3, 3, 5, 5]
assert namespace['rule_depth'](namespace['policy'], 'sum') == 1
assert inference.EFFORT == {'low': 1, 'medium': 3, 'high': 5}
assert min(inference.EFFORT['medium'], 2) == 2
changed = {**namespace['policy'], 'medium_max_events': 2}
changed_depths = [namespace['rule_depth'](changed, 'boxes', n) for n in range(1, 7)]
assert changed_depths == [3, 3, 5, 5, 5, 5]
rejections = []
for bad in ({**namespace['policy'], 'version': 2}, {**namespace['policy'], 'sum_depth': 2}):
    try:
        namespace['rule_depth'](bad, 'sum')
        raise AssertionError('invalid rule accepted')
    except ValueError as error:
        rejections.append(str(error))
result['embedded_code'] = {'path': str(code_path.relative_to(ROOT)), 'sha256': sha(code_path), 'blocks_executed_in_order': 2, 'stdout': stream.getvalue(), 'rule_ast_matches_library': True, 'base_boxes_depths_for_1_to_6': depths, 'sum_depth': 1, 'manual_effort': inference.EFFORT, 'medium_cap2': 2, 'exercise_depths_for_1_to_6': changed_depths, 'exercise_equal_weight_six_examples_mean': sum(changed_depths) / 6, 'invalid_rule_errors': rejections}

world = {'task': 'boxes', 'a': 2, 'b': 6, 'events': [('swap', 0)], 'query': 'B'}
assert data.answer(world) == 2
assert data.judgment(world, (), 'info', data.CANDIDATES['info']) == 'yes'
assert data.judgment(world, ('a',), 'info', data.CANDIDATES['info']) == 'no'
possible = sorted({data.answer({**world, 'a': a}) for a in range(8)})
assert possible == list(range(8))
result['field_contract'] = {'complete_state': data.render(world), 'complete_gold_answer': 2, 'complete_info': 'yes', 'missing_a_state': data.render(world, ('a',)), 'missing_a_info': 'no', 'compatible_B_final_values_if_A_unknown': possible, 'candidates': list(data.CANDIDATES['choice']), 'required': data.REQUIRED}

# 先用明標 fixture 核所有控制分支；這些回應不冒充模型能力。
class FakeB:
    auto_policy = namespace['policy']

    def __init__(self):
        self.calls = []

    def synchronize(self):
        pass

    def solve(self, state, loops):
        self.calls.append(loops)
        return {'answer': 2, 'loops': loops, 'seconds': 0.0}


class FakeC:
    def __init__(self, choice, info):
        self.values = {'choice': choice, 'info': info}
        self.calls = []

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {'selected': self.values[question], 'seconds': 0.0}


fixture_rows = []
for choice, info, effort, cap, status, expected_calls in [
    ('card', 'yes', 'high', 5, 'unsupported', []),
    ('boxes', 'no', 'high', 5, 'clarification_needed', []),
    ('boxes', 'yes', 'medium', 2, 'budget_limited', [2]),
    ('boxes', 'yes', 'low', 5, 'answered', [1]),
    ('boxes', 'yes', 'auto', 5, 'answered', [3]),
    ('sum', 'yes', 'auto', 5, 'answered', [1]),
]:
    b, c = FakeB(), FakeC(choice, info)
    state = data.render(world) if choice != 'sum' else '三數加法;x=1;y=2;z=3'
    reply = inference.auto_solve(b, c, state, effort, cap)
    assert reply['status'] == status and b.calls == expected_calls
    assert c.calls == ['choice', 'info']
    fixture_rows.append({'fixture': {'choice': choice, 'info': info}, 'effort': effort, 'max_loops': cap, 'status': reply['status'], 'b_solve_calls': b.calls, 'requested_loops': reply.get('requested_loops'), 'b_loops': reply['b_loops'], 'c_calls': c.calls})
result['control_flow_fixtures'] = fixture_rows

b = inference.Product(CACHE / 'b', 'b', 'cpu')
c = inference.Product(C_CACHE, 'c', 'cpu')
assert (CACHE / 'b/auto-policy.json').read_bytes() == (ROOT / 'docs/lesson-products/results/b-auto-policy.json').read_bytes()
assert (CACHE / 'b/artifact-manifest.json').read_bytes() == (ROOT / 'docs/lesson-products/results/b-artifact-manifest.json').read_bytes()
model_results = []
for state, effort, cap in [
    (data.render(world), 'auto', 5),
    (data.render(world), 'medium', 2),
    (data.render(world), 'low', 5),
    (data.render(world, ('a',)), 'high', 5),
    ('卡片進度;申請編號=123456', 'high', 5),
    ('三數加法;x=1;y=2;z=3', 'auto', 5),
]:
    reply = inference.auto_solve(b, c, state, effort, cap)
    model_results.append({'state': state, 'effort': effort, 'max_loops': cap, 'choice': reply['choice']['selected'], 'info': reply['info']['selected'], 'status': reply['status'], 'requested_loops': reply.get('requested_loops'), 'b_loops': reply['b_loops'], 'answer': reply.get('solution', {}).get('answer'), 'integrated_seconds': reply['seconds'], 'c_seconds': reply['c_seconds'], 'b_seconds': reply.get('solution', {}).get('seconds')})
assert [r['status'] for r in model_results] == ['answered', 'budget_limited', 'answered', 'clarification_needed', 'unsupported', 'answered']
assert [r['b_loops'] for r in model_results] == [3, 2, 1, 0, 0, 1]
result['published_small_models_current_cpu'] = {'scope': 'Six individual regression demonstrations; no accuracy or latency generalization. Public C-v1 finite-character model, not planned byte C-v2.', 'b_manifest': {'path': str(CACHE / 'b/artifact-manifest.json'), 'sha256': sha(CACHE / 'b/artifact-manifest.json')}, 'c_manifest': {'path': str(C_CACHE / 'artifact-manifest.json'), 'sha256': sha(C_CACHE / 'artifact-manifest.json')}, 'public_policy_sha256': sha(CACHE / 'b/auto-policy.json'), 'repo_policy_sha256': sha(ROOT / 'docs/lesson-products/results/b-auto-policy.json'), 'policy_byte_equal': True, 'cases': model_results}

settings_path = ROOT / 'docs/lesson-products/results/settings-freeze-v1.json'
settings = read(settings_path)
historic_path = ROOT / 'docs/lesson-products/results/final-report-v1.json'
historic = read(historic_path)
assert historic['settings_freeze_sha256'] == sha(settings_path)
source_hashes = {name: sha(ROOT / 'tiny_perceptron/lesson_products' / name) for name in settings['source_sha256s']}
assert source_hashes == settings['source_sha256s']
grid = settings['auto_search']
expected_grid = [
    {'version': 1, 'low_max_events': low, 'medium_max_events': medium, 'sum_depth': arithmetic}
    for low, medium, arithmetic in itertools.product(range(5), range(5), (1, 3, 5)) if low <= medium
]
assert len(grid) == 45 and [row['rule'] for row in grid] == expected_grid
feasible = [row for row in grid if row['quality']['mean_actual_b_loops_all_complete'] <= 3]
selected = min(feasible, key=lambda row: (-row['quality']['accuracy_end_to_end'], row['quality']['mean_actual_b_loops_all_complete'], json.dumps(row['rule'], sort_keys=True)))
assert selected['rule'] == namespace['policy']
assert selected['quality'] == b.auto_policy['policy_quality']

manifest_path = ROOT / 'docs/lesson-products/data-v2-manifest.json'
manifest = read(manifest_path)
assert sha(manifest_path) == historic['data_manifest_sha256'] == settings['data_manifest_sha256']
salt = manifest['config']['split_salt']
test_worlds = {}
for w in itertools.chain(data.box_worlds(), data.arithmetic_worlds()):
    if w['task'] == 'boxes' and len(w['events']) > 4:
        continue
    if data.split_of(w['family_id'], salt) != 'test':
        continue
    state = data.render(w)
    test_worlds[data.digest((w['family_id'], state))] = w
assert len(test_worlds) == 2998

raw_path = CACHE / 'raw/final-report-v1.predictions.jsonl.gz'
raw_manifest = read(CACHE / 'raw/predictions-manifest.json')
assert sha(raw_path) == raw_manifest['gzip_sha256']
raw_sha = hashlib.sha256()
raw_bytes, raw_rows = 0, 0
observed = {k: {'correct': 0, 'loops': 0, 'covered': 0} for k in ('low', 'medium', 'high', 'auto')}
lengths, tasks = Counter(), Counter()
seen = set()
changed_total = 0
stress = Counter()
for line in gzip.open(raw_path, 'rb'):
    raw_sha.update(line)
    raw_bytes += len(line)
    raw_rows += 1
    row = json.loads(line)
    if row['product'] == 'b' and row['split'] == 'test':
        seen.add(row['id'])
        w = test_worlds[row['id']]
        assert row['task'] == w['task'] and row['gold'] == data.answer(w) and row['family_id'] == w['family_id']
        tasks[row['task']] += 1
        if w['task'] == 'boxes':
            lengths[len(w['events'])] += 1
        accepted = row['c_choice'] in ('boxes', 'sum') and row['c_info'] == 'yes'
        assert row['c_choice'] == w['task'] and row['c_info'] == 'yes'
        if accepted:
            event_count = len(w['events']) if w['task'] == 'boxes' else 0
            for effort in observed:
                depth = inference.EFFORT[effort] if effort != 'auto' else namespace['rule_depth'](namespace['policy'], row['c_choice'], event_count)
                observed[effort]['covered'] += 1
                observed[effort]['loops'] += depth
                observed[effort]['correct'] += row['answers_by_depth'][str(depth)] == row['gold']
            changed_total += namespace['rule_depth'](changed, row['c_choice'], event_count)
    elif row['product'] == 'b' and row['split'] == 'long_shift':
        stress['count'] += 1
        stress['correct'] += max(range(len(row['raw_scores'])), key=lambda i: row['raw_scores'][i]) == row['gold']
assert seen == set(test_worlds)
assert raw_sha.hexdigest() == historic['predictions']['sha256'] == raw_manifest['sha256']
assert raw_bytes == historic['predictions']['bytes'] == raw_manifest['bytes']
for effort, count in observed.items():
    original = historic['auto'][effort]
    assert count['correct'] == original['correct_end_to_end']
    assert count['covered'] == original['covered']
    assert abs(count['loops'] / 2998 - original['mean_actual_b_loops_all_complete']) < 1e-12
assert observed['auto']['correct'] == 2926 and observed['medium']['correct'] == 2927
assert tasks == {'boxes': 2956, 'sum': 42}
assert observed['medium']['loops'] - observed['auto']['loops'] == 42 * 2 == 84
assert changed_total / 2998 > 3
assert stress == {'count': 3200, 'correct': 1245}
result['historical_recomputation'] = {'scope': 'Recomputed historical published 2026-10-16 raw scores/decisions; not a new run of formal test models. Generator reconstructed original fixed inputs only. All five frozen historical source file SHA values equal current files.', 'settings_path': str(settings_path.relative_to(ROOT)), 'settings_sha256': sha(settings_path), 'settings_recorded_at': settings['recorded_at'], 'test_start': read(ROOT / 'docs/lesson-products/results/final-report-v1.started.json'), 'final_report_path': str(historic_path.relative_to(ROOT)), 'final_report_sha256': sha(historic_path), 'final_recorded_at': historic['recorded_at'], 'historical_source_hashes': source_hashes, 'grid_count': len(grid), 'feasible_grid_count': len(feasible), 'selected': selected, 'data_manifest_path': str(manifest_path.relative_to(ROOT)), 'data_manifest_sha256': sha(manifest_path), 'split_salt': salt, 'raw_gzip_path': str(raw_path), 'raw_gzip_sha256': sha(raw_path), 'raw_uncompressed_sha256': raw_sha.hexdigest(), 'raw_uncompressed_bytes': raw_bytes, 'raw_rows': raw_rows, 'test_count': len(seen), 'test_families': len({w['family_id'] for w in test_worlds.values()}), 'test_tasks': dict(tasks), 'boxes_event_counts': dict(sorted(lengths.items())), 'effort': observed, 'loops_saved_auto_vs_medium': 84, 'correct_difference_auto_minus_medium': -1, 'changed_threshold2_test_mean_loops': changed_total / 2998, 'historical_long_shift_depth5': dict(stress), 'historical_latency_scope': {k: historic['latency'][k] for k in ('warmup_per_effort', 'repetitions', 'batch_size', 'model_reload_excluded', 'device', 'threads', 'python', 'torch')}, 'historical_latency_measurements': historic['latency']['measurements']}
result['result'] = 'All embedded output, field, control-flow, publication fingerprint, frozen-grid and historical raw-score assertions passed.'
output = OUT / 'verification-output.json'
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(output.relative_to(ROOT)), 'sha256': sha(output), 'result': result['result'], 'embedded_stdout': stream.getvalue(), 'historical_effort': observed, 'test_tasks': dict(tasks), 'changed_threshold2_test_mean_loops': changed_total / 2998}, ensure_ascii=False, indent=2))
