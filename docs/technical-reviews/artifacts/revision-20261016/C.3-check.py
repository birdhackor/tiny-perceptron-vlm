"""C.3: run authored toy code and recount saved historical candidates; no training/generation."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
from tiny_perceptron.data import ByteTokenizer
import torch

OUT = ROOT / 'docs/technical-reviews/artifacts/revision-20261016'
source = ROOT / 'course/chapters/0C.md'
body = lesson_slices(source.read_text())['C.3']
freeze = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == 'C.3')
assert hashlib.sha256(body.encode()).hexdigest() == page['source_sha256']
assert body == (ROOT / page['snapshot']).read_text()
code = re.findall(r'```python\n(.*?)```', body, flags=re.S)[0]
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, 'frozen-C.3-python-block', 'exec'), namespace)
expected = [.3, .51, .7599, .942352]
observed = [round(1 - (1 - .3) ** n, 6) for n in (1, 2, 4, 8)]
assert observed == expected
assert namespace['candidates'] == [3, 4, 3, 5]
assert 4 in namespace['candidates'] and namespace['candidates'][0] != 4
namespace['candidates'] = [3] * 4
assert 4 not in namespace['candidates'] and namespace['p'] == .3

# Exact rational enumeration, independently of the snippet's floating-point arithmetic.
exact = {str(n): str(1 - Fraction(7, 10) ** n) for n in (1, 2, 4, 8)}
pass_special = []
for k in (1, 2, 4, 8):
    for c in range(k + 1):
        estimate = 1 - Fraction(math.comb(k - c, k), math.comb(k, k))
        assert estimate == int(c > 0)
        pass_special.append({'n': k, 'k': k, 'c': c, 'estimate': int(estimate)})

historical_code = (OUT / 'C.3-historical-source.py').read_text()
verifier_ast = next(n for n in ast.parse(historical_code).body if isinstance(n, ast.FunctionDef) and n.name == '_verify_reasoning')
verifier_ns = {'re': re}
exec(compile(ast.Module(body=[verifier_ast], type_ignores=[]), 'historical-_verify_reasoning', 'exec'), verifier_ns)
verify = verifier_ns['_verify_reasoning']
report_path = ROOT / 'docs/course-experiments/results/reasoning.json'
report = json.loads(report_path.read_text())
result = report['results']
expected_counts = {'direct': [1, 1, 2, 1], 'steps': [11, 13, 13, 14]}
tokenizer = ByteTokenizer()
summary = {}
all_questions = set()
for mode, branch in result['comparison'].items():
    budgets = branch['budgets']
    assert [b['candidate_count'] for b in budgets] == [1, 2, 4, 8]
    branch_summary = []
    question_orders = []
    for budget, expected_count in zip(budgets, expected_counts[mode]):
        k = budget['candidate_count']
        samples = budget['samples']
        assert len(samples) == 24
        question_orders.append([s['question'] for s in samples])
        covered = 0
        for sample in samples:
            assert sample['truth'] == sample['a'] + sample['b'] + sample['c']
            assert sample['question'] == f"({sample['a']}+{sample['b']})+{sample['c']}=?"
            assert sample['family'] == ':'.join(map(str, sorted((sample['a'], sample['b'], sample['c']))))
            generation = sample['generation']
            assert generation['temperature'] == .7 and generation['candidate_count'] == k
            assert generation['max_new_tokens'] == (8 if mode == 'direct' else 48)
            assert len(sample['candidates']) == k
            for candidate in sample['candidates']:
                ids = candidate['generated_ids']
                assert ids and all(isinstance(i, int) and 0 <= i < 264 for i in ids)
                assert candidate['eos'] == (ids[-1] == tokenizer.eos_id)
                raw = ids[:-1] if candidate['eos'] else ids
                assert tokenizer.decode(raw) == candidate['generated']
                assert candidate['invalid_special_tokens'] == [i for i in raw if i < 8]
                actual = verify(candidate, sample, mode)
                for name, value in actual.items():
                    assert candidate[name] == value, (mode, k, sample['question'], name)
                assert candidate['final_correct'] == (candidate['final_answer'] == sample['truth'])
            correct = any(c['final_correct'] for c in sample['candidates'])
            assert sample['oracle_coverage'] == correct
            covered += correct
            all_questions.add(sample['question'])
        assert covered == expected_count == budget['oracle_coverage']['numerator']
        assert budget['oracle_coverage']['denominator'] == 24
        assert budget['oracle_coverage']['rate'] == covered / 24
        branch_summary.append({'k': k, 'coverage_numerator': covered, 'question_denominator': 24,
                               'candidate_count': sum(len(s['candidates']) for s in samples),
                               'temperature': .7, 'max_new_tokens': 8 if mode == 'direct' else 48})
    assert all(q == question_orders[0] for q in question_orders)
    assert len(set(question_orders[0])) == 24
    total = sum(b['candidate_count'] for b in branch_summary)
    assert total == 360
    adjacent_nonprefix = []
    for prev, nxt in zip(budgets, budgets[1:]):
        n = prev['candidate_count']
        different = sum([c['generated_ids'] for c in a['candidates']] != [c['generated_ids'] for c in b['candidates'][:n]]
                        for a, b in zip(prev['samples'], nxt['samples']))
        adjacent_nonprefix.append({'from_k': n, 'to_k': nxt['candidate_count'], 'questions_with_changed_prefix': different})
    assert branch['training']['steps'] == 900
    summary[mode] = {'budgets': branch_summary, 'total_candidates': total,
                     'different_questions': 24, 'test_families': len(set(s['family'] for s in budgets[0]['samples'])),
                     'base_state_sha256': branch['base_state_sha256'],
                     'training_steps': branch['training']['steps'],
                     'training_effective_answer_tokens': branch['training']['effective_tokens'],
                     'adjacent_budgets_are_not_nested': adjacent_nonprefix}
assert len(all_questions) == 24
assert summary['direct']['base_state_sha256'] == summary['steps']['base_state_sha256']
assert summary['steps']['training_effective_answer_tokens'] > summary['direct']['training_effective_answer_tokens']

# Exercise the historical budget loop with a recording stub, without a model or generation.
calls = []
def recording_sample(model, messages, ctx, count, tokens, temperature):
    calls.append({'model': str(model), 'question': messages[0]['content'],
                  'count': count, 'tokens': tokens, 'temperature': temperature})
    return {'samples': [{'generated': '0', 'invalid_special_tokens': [], 'generated_tokens': 2}
                        for _ in range(count)], 'generated_tokens': count * 2,
            'forward_input_tokens': count * 13, 'seconds': 0.0}

samples_ast = next(n for n in ast.parse(historical_code).body
                   if isinstance(n, ast.FunctionDef) and n.name == '_reasoning_samples')
loop_ns = {'_sample': recording_sample, '_verify_reasoning': verify,
           'Counter': Counter, '_rate': lambda n, d: {'numerator': n, 'denominator': d, 'rate': n/d if d else None}}
exec(compile(ast.Module(body=[samples_ast], type_ignores=[]), 'historical-budget-loop', 'exec'), loop_ns)
rows = result['comparison']['direct']['budgets'][0]['samples']
for mode in ('direct', 'steps'):
    traced = loop_ns['_reasoning_samples'](mode, rows, None, mode)
    assert [b['candidate_count'] for b in traced] == [1, 2, 4, 8]
assert len(calls) == 2 * 4 * 24
for call in calls:
    assert call['temperature'] == .7
    assert call['tokens'] == (8 if call['model'] == 'direct' else 48)
artifact = {
    'reviewer_task': '/root/tech_20261016_c_3', 'lesson_id': 'C.3',
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/C.3-check.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'platform': platform.platform()},
    'scope': 'Executed authored Python snippet, exact combinatorial checks, and saved-candidate recount only. No training, model generation, GPU benchmark or HumanEval run.',
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'historical_report': {'path': report_path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                          'code_revision': report['revision'], 'recorded_environment_not_reexecuted': {k: report[k] for k in ('python_version', 'torch_version', 'device', 'gpu', 'seed', 'step_scale', 'evidence_status')},
                          'split': result['split'], 'limits': result['limits']},
    'snippet_stdout': stdout.getvalue(), 'coverage_float_rounded_6': observed, 'coverage_exact': exact,
    'list_check': {'original': [3,4,3,5], 'contains_truth': True, 'first_choice_correct': False, 'all_3_contains_truth': False, 'p_after_replacement': namespace['p']},
    'pass_at_k_n_equals_k_cases': pass_special,
    'heterogeneous_question_example': {'p_per_question': [0,1], 'n': 2, 'actual_mean_coverage': .5, 'using_mean_p_instead': .75},
    'historical_recount': summary, 'all_assertions_passed': True,
    'historical_sampling_loop_recording_stub': {
        'model_generation': False, 'calls': len(calls), 'per_branch': 96,
        'fresh_call_per_question_and_k': True, 'k_values': [1, 2, 4, 8],
        'scope': 'Executed historical orchestration with a recording _sample stub. Supports fresh call schedule and forwarded arguments only; not stochastic independence or model accuracy.'},
}
(OUT / 'C.3-execution.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'all_assertions_passed': True, 'toy': observed, 'historical': summary}, ensure_ascii=False))
