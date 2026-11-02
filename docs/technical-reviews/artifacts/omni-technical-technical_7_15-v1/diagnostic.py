"""7.15 independent CPU audit: execute the exact lesson cell and audit existing results.

No training, checkpoint download, or model-quality rerun is performed.
"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer

ROOT = Path(__file__).resolve().parents[4]
RAW = Path('/workspace/work/reference/omni-technical-technical_7_15-v1')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_ref(path):
    raw = (ROOT / path).read_bytes()
    return {'path': path, 'sha256': sha(raw), 'bytes': len(raw)}


def capture(code):
    stream, variables = io.StringIO(), {}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, '7.15-exact-lesson-cell', 'exec'), variables)
    assert len(variables['measurements']) == 2
    assert all(v is None for row in variables['measurements'].values() for v in row.values())
    return {'stdout': stream.getvalue(), 'measurements': variables['measurements'],
            'a_questions': variables['a_questions'], 'b_questions': variables['b_questions']}


def main():
    output = {
        'created_at': datetime.now(UTC).isoformat(),
        'reviewer_task': '/root/omni_technical_coordinator/technical_7_15',
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_7_15-v1/diagnostic.py',
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'platform': platform.platform(), 'device': 'cpu',
                        'repository_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'scope': 'Exact lesson Python cell, stated exercise, deterministic split regeneration and independent recomputation of existing recorded metrics. No training or generation with a trained model.',
    }
    freeze_path = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json'
    freeze = json.loads((ROOT / freeze_path).read_bytes())
    page = next(p for p in freeze['pages'] if p['page_id'] == '7.15')
    raw = (ROOT / page['snapshot']).read_bytes()
    live = re.search(r'^## 7\.15 .*?(?=^## |\Z)', (ROOT / page['source']).read_text(), re.M | re.S).group(0).encode()
    assert raw == live and sha(raw) == page['source_sha256']
    assert page['figures_sha256'] == {} and not re.search(rb'!\[', raw)
    output['source_freeze'] = file_ref(freeze_path)
    output['lesson'] = {'snapshot': file_ref(page['snapshot']), 'live_slice_sha256': sha(live), 'figures': {}}
    cell = re.findall(r'```python\n(.*?)\n```', raw.decode(), re.S)
    assert len(cell) == 1
    output['lesson_cell'] = capture(cell[0])
    # Change the exact A-question assignment as requested by the exercise.
    line = 'a_questions = ["紅色圓形是什麼形狀？", "藍色方塊是什麼形狀？"]'
    assert cell[0].count(line) == 1
    exercise = cell[0].replace(line, line[:-1] + ', "紅色方塊是什麼形狀？"]')
    output['exercise'] = capture(exercise)
    assert len(output['exercise']['a_questions']) == 3
    assert output['exercise']['measurements'] == output['lesson_cell']['measurements']

    record_path = 'docs/course-experiments/results/sft_ablation.json'
    record = json.loads((ROOT / record_path).read_bytes())
    result = record['results']
    output['record'] = file_ref(record_path)
    output['recorded_environment'] = {k: record[k] for k in ('revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'elapsed_seconds', 'timing_scope', 'step_scale', 'evidence_status')}
    assert record['seed'] == 42 and record['step_scale'] == 1.0
    assert round(record['elapsed_seconds'], 1) == 23.5
    datasets = {'attributes': split_records(generate_records('attributes-sft'), seed=42),
                'arithmetic': split_records(arithmetic_records(), seed=42)}
    output['splits'] = {}
    for name, parts in datasets.items():
        families = {s: {r['family'] for r in rows} for s, rows in parts.items()}
        assert not (families['train'] & families['validation'] or families['train'] & families['test'] or families['validation'] & families['test'])
        output['splits'][name] = {}
        for split, rows in parts.items():
            encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
            observed = {'records': len(rows), 'families': len(families[split]), 'sha256': sha(encoded)}
            expected = result['data'][name][split]
            assert all(observed[k] == expected[k] for k in observed)
            output['splits'][name][split] = observed
    arithmetic = [row for rows in datasets['arithmetic'].values() for row in rows]
    assert len(arithmetic) == 64
    assert {r['a'] for r in arithmetic} == set(range(8))
    assert {r['b'] for r in arithmetic} == set(range(8))
    pair_splits = {(r['a'], r['b']): s for s, rows in datasets['arithmetic'].items() for r in rows}
    assert all(pair_splits[(a, b)] == pair_splits[(b, a)] for a in range(8) for b in range(8))
    output['commuted_pair_family'] = {'2+3': pair_splits[(2, 3)], '3+2': pair_splits[(3, 2)],
                                       'all_64_pairs_share_split_with_reverse': True}
    tok = ByteTokenizer()
    metrics = []
    for state, groups in [('before', result['before']), ('b-only', result['runs']['b-only'])]:
        for task, data_name in [('A_attributes', 'attributes'), ('B_arithmetic', 'arithmetic')]:
            for split in ['validation', 'test']:
                m, rows = groups[task][split], datasets[data_name][split]
                assert len(m['samples']) == m['records'] == m['examples'] == len(rows)
                count = sum(int((y != IGNORE).sum()) for _, y in text_examples(rows, 'sft'))
                assert count == m['effective_tokens']
                correct = ended = 0
                for sample, row in zip(m['samples'], rows, strict=True):
                    assert sample['messages'] == row['messages'][:-1]
                    assert sample['expected'] == row['messages'][-1]['content']
                    ids = sample['generated_ids']
                    eos = tok.eos_id in ids
                    content = ids[:ids.index(tok.eos_id)] if eos else ids
                    assert tok.decode(content) == sample['generated']
                    exact = content == tok.encode(sample['expected'])
                    assert exact == sample['exact'] and eos == sample['eos']
                    correct += exact
                    ended += eos
                assert correct == m['matches'] and correct / len(rows) == m['exact_match']
                assert ended / len(rows) == m['eos_rate']
                assert abs(m['nll_sum'] / count - m['nll']) < 1e-12
                metrics.append({'state': state, 'task': task, 'split': split, 'matches': correct,
                                'records': len(rows), 'effective_tokens': count, 'nll': m['nll'],
                                'nll_sum': m['nll_sum'], 'eos_count': ended})
    output['metrics_recomputed_from_recorded_samples_and_sums'] = metrics
    a_before = result['before']['A_attributes']['test']
    a_after = result['runs']['b-only']['A_attributes']['test']
    b_before = result['before']['B_arithmetic']['test']
    b_after = result['runs']['b-only']['B_arithmetic']['test']
    assert (a_before['matches'], a_after['matches'], b_before['matches'], b_after['matches']) == (5, 0, 0, 0)
    assert f"{b_before['nll']:.5f}" == '7.05820' and f"{b_after['nll']:.5f}" == '4.57756'
    sample = next(s for s in a_after['samples'] if s['expected'] == 'circle' and s['generated'] == '9')
    output['circle_to_9_sample'] = sample
    training = result['runs']['b-only']['training']
    assert training['steps'] == 500 and training['records'] == 49
    assert training['initial_loss'] > training['final_loss']
    assert result['runs']['b-only']['source_records'] == {'A': 0, 'B': 49}
    output['recorded_training'] = {k: training[k] for k in ('steps', 'records', 'initial_loss', 'final_loss', 'effective_tokens', 'parameters', 'trainable_parameters')}
    baseline_path = 'docs/course-experiments/results/sft.json'
    baseline = json.loads((ROOT / baseline_path).read_bytes())
    assert baseline['results']['after'] == result['before']['A_attributes']
    output['baseline'] = {**file_ref(baseline_path), 'sft_after_exactly_equals_ablation_A_before': True}
    output['implementation_versions'] = []
    for path in ['scripts/course_experiments/text.py', 'scripts/course_experiments/common.py',
                 'scripts/course_experiments/run.py', 'tiny_perceptron/data.py', 'tiny_perceptron/model.py']:
        original = subprocess.check_output(['git', 'show', record['revision'] + ':' + path], cwd=ROOT)
        assert sha(original) == record['code_sha256'][path]
        current = (ROOT / path).read_bytes()
        output['implementation_versions'].append({'path': path, 'recorded_sha256': sha(original),
                                                  'current_sha256': sha(current), 'equal': current == original})
    help_result = subprocess.run([sys.executable, '-m', 'scripts.course_experiments.run', '--help'], cwd=ROOT, text=True, capture_output=True)
    assert help_result.returncode == 0
    assert '--experiment' in help_result.stdout and '{cpu,cuda}' in help_result.stdout
    output['cli_help'] = {'command': sys.executable + ' -m scripts.course_experiments.run --help',
                          'exit_code': help_result.returncode, 'stdout': help_result.stdout, 'stderr': help_result.stderr}
    output['result'] = 'PASS: all assertions passed; no new training and no GPU execution'
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
