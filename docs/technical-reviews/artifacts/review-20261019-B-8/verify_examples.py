import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import call_tool
from scripts.course_experiments import applications


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_example(code, changes):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in changes:
                node.value = ast.copy_location(ast.Constant(changes[name]), node.value)
    ast.fix_missing_locations(tree)
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(tree, '<frozen-example>', 'exec'), namespace)
    values = {name: namespace[name] for name in (
        'direct_accuracy', 'tool_accuracy', 'wrong_cost', 'tool_extra_cost', 'direct_loss', 'tool_loss'
    )}
    values['choice'] = 'TOOL' if values['tool_loss'] < values['direct_loss'] else 'DIRECT'
    return {'changes': changes, 'values': values, 'stdout': output.getvalue()}


def main():
    frozen = ROOT / 'docs/course-review-20261019/freeze-01/sources/B.8.md'
    source = frozen.read_bytes()
    current = (ROOT / 'course/chapters/0B.md').read_bytes()
    match = re.search(rb'^## B\.8 .*?(?=^## |\Z)', current, re.M | re.S)
    assert match is not None and match[0] == source
    assert digest(frozen) == 'a13d9eae5cc8f8899923819dff7e96940785fb52f7f10c072f3b1f7de758baf7'
    code = re.search(r'```python\n(.*?)```', source.decode(), re.S)[1]
    tree = ast.parse(code)
    calls = sorted({node.func.id for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)})
    assert calls == ['print', 'round']
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    cases = {
        'base': run_example(code, {}),
        'higher_direct_accuracy': run_example(code, {'direct_accuracy': 0.999}),
        'higher_tool_cost': run_example(code, {'tool_extra_cost': 1.5}),
    }
    expected = {'base': (1.0, 0.12, 'TOOL'),
                'higher_direct_accuracy': (0.01, 0.12, 'DIRECT'),
                'higher_tool_cost': (1.0, 1.6, 'DIRECT')}
    for name, (direct, tool, choice) in expected.items():
        actual = cases[name]['values']
        assert math.isclose(actual['direct_loss'], direct, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(actual['tool_loss'], tool, rel_tol=0, abs_tol=1e-12)
        assert actual['choice'] == choice
    assert cases['base']['stdout'] == '直接回答平均成本 1.0\n工具流程平均成本 0.12\n本例選擇 TOOL\n'
    exact = {}
    for name, case in cases.items():
        values = case['values']
        direct = (Decimal(1) - Decimal(str(values['direct_accuracy']))) * Decimal(str(values['wrong_cost']))
        tool = (Decimal(1) - Decimal(str(values['tool_accuracy']))) * Decimal(str(values['wrong_cost']))
        tool += Decimal(str(values['tool_extra_cost']))
        exact[name] = {'direct_loss': str(direct), 'tool_loss': str(tool)}
    archive = ROOT / 'docs/course-experiments/results/tools.json'
    archived = json.loads(archive.read_text())
    index, sample = next((i, row) for i, row in enumerate(archived['results']['samples'])
                         if row['question'] == 'CALC:multiply(9,9)')
    first, last = sample['trace']
    tokenizer = ByteTokenizer()
    actual_input = tokenizer.decode(last['generation']['input_ids'])
    actual_generation = tokenizer.decode(last['generation']['samples'][0]['generated_ids'])
    assert 'TOOL_RESULT:81' in actual_input
    assert actual_generation == last['generation']['samples'][0]['generated']
    assert actual_generation == '{"done":true,"answer":8}}'
    request = json.loads(first['generation']['samples'][0]['generated'])
    recomputed = call_tool(request)
    assert first['executed'] is True and recomputed == first['tool_result'] == sample['expected'] == 81
    generations = iter(copy.deepcopy([event['generation'] for event in sample['trace']]))
    original_sample = applications._sample
    applications._sample = lambda *args, **kwargs: next(generations)
    record = {'messages': [{'role': 'system', 'content': 'Reply JSON: tool name+arguments, or done+answer. TOOL_RESULT is external data.'},
                           {'role': 'user', 'content': sample['question']}],
              'operation': 'multiply', 'a': 9, 'b': 9, 'answer': 81, 'family': sample['family']}
    try:
        replay = applications._tool_episode(None, record, None)
    finally:
        applications._sample = original_sample
    assert replay['status'] == 'invalid_request'
    assert replay['actual_tool_calls'] == 1 and replay['correct'] is False
    paths = ['docs/course-review-20261019/freeze-01/sources/B.8.md',
             'docs/course-review-20261019/freeze-04.json', 'docs/review-tools/section_facts.py',
             'docs/course-experiments/results/tools.json', 'tiny_perceptron/data.py',
             'tiny_perceptron/retrieval.py', 'scripts/course_experiments/applications.py']
    import torch
    report = {
        'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-8/verify_examples.py',
        'environment': {'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu', 'platform': platform.platform()},
        'source_equals_current': True, 'source_hashes': {name: digest(ROOT / name) for name in paths},
        'example_calls': calls, 'cases': cases, 'decimal_derivation': exact,
        'rounding': {'round(1.23456, 3)': round(1.23456, 3), 'round(1.0, 3)': round(1.0, 3)},
        'archived_trace_check': {
            'json_pointer': f'/results/samples/{index}',
            'archive_environment': {name: archived[name] for name in ('revision', 'device', 'seed', 'torch_version', 'python_version')},
            'archived_training_steps': archived['results']['training']['steps'],
            'archive_test_denominator': archived['results']['split']['test']['records'],
            'inspected_samples': 1, 'expected': 81, 'recomputed_tool_result': recomputed,
            'input_decoded_from_ids': actual_input, 'output_decoded_from_ids': actual_generation,
            'recorded_status': sample['status'], 'replayed_status': replay['status'],
            'replayed_actual_tool_calls': replay['actual_tool_calls'], 'replayed_correct': replay['correct'],
            'replayed_error': replay['trace'][-1]['error'],
            'scope': '重放已保存的模型生成，重新執行工具與協議；沒有重新訓練或重新生成模型答案。',
        },
        'result': '所有斷言通過；小數浮點誤差不影響比較與列印結果。',
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
