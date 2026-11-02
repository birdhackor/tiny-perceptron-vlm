"""13.9 獨立重算：只執行教材例子、讀既有結果、檢查遮罩；不訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.common import new_lm, split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat

CACHE = Path('/workspace/work/reference/omni-technical-technical_13_9-v1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def function_dump(path, name):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def preference_summary(report):
    rows = report['samples']
    assert len(rows) == report['records']
    for row in rows:
        policy = row['policy_chosen_logp'] - row['policy_rejected_logp']
        relative = policy - row['reference_margin']
        assert math.isclose(policy, row['policy_margin'], abs_tol=1e-10)
        assert math.isclose(relative, row['relative_margin'], abs_tol=1e-10)
        for side in ('chosen', 'rejected'):
            assert row[side + '_answer_tokens'] == len(row[side].encode('utf-8')) + 1
    relative_count = sum(row['relative_margin'] > 0 for row in rows)
    absolute_count = sum(row['policy_margin'] > 0 for row in rows)
    assert relative_count == report['relative_preference_improved']
    assert absolute_count == report['chosen_higher_absolute_probability']
    assert math.isclose(sum(row['relative_margin'] for row in rows) / len(rows), report['mean_relative_margin'], abs_tol=1e-10)
    return {
        'records': len(rows),
        'relative_positive': relative_count,
        'chosen_higher': absolute_count,
        'relative_positive_but_chosen_not_higher': sum(row['relative_margin'] > 0 and row['policy_margin'] <= 0 for row in rows),
        'all_score_and_length_fields_recomputed': True,
    }


def generation_summary(report):
    rows = report['samples']
    tok = ByteTokenizer()
    assert len(rows) == report['records']
    counted = 0
    for row in rows:
        ids = row['generated_ids']
        raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        exact = raw == tok.encode(row['expected'])
        assert exact == row['exact']
        assert tok.decode(raw) == row['generated']
        assert (tok.eos_id in ids) == row['eos']
        counted += exact
    assert counted == report['matches']
    assert report['exact_match'] == counted / len(rows)
    return {'records': len(rows), 'exact_matches': counted, 'all_raw_ids_checked': True}


def main():
    freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
    page = next(p for p in read_json(freeze_path)['pages'] if p['page_id'] == '13.9')
    snapshot = ROOT / page['snapshot']
    body = snapshot.read_text()
    current = re.search(r'^## 13\.9 .*?(?=^## |\Z)', (ROOT / page['source']).read_text(), re.M | re.S).group()
    assert current == body
    assert digest(snapshot) == page['source_sha256']
    assert page['figures_sha256'] == {} and '![' not in body

    code = re.findall(r'```python\n(.*?)```', body, re.S)
    assert len(code) == 1
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code[0], str(snapshot) + ':embedded', 'exec'), namespace)
    lines = output.getvalue().splitlines()
    assert sum(namespace['recipe'].values()) == 100
    assert len(namespace['metrics']) == 6
    assert sum(line.endswith('尚未量測') for line in lines) == 6
    assert all(value is None for value in namespace['metrics'].values())
    zero_print = '尚未量測' if 0 is None else 0
    assert zero_print == 0
    exercise = dict(namespace['recipe'])
    exercise['應拒絕題'] += exercise['正常題完成']
    exercise['正常題完成'] = 0
    assert exercise == {'風格': 30, '格式': 30, '應拒絕題': 40, '正常題完成': 0}
    assert sum(exercise.values()) == 100

    token_checks = []
    for target_count, expected in [(100, 6000), (10, 600)]:
        examples = [render_chat([{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'a' * (target_count - 1)}]) for _ in range(60)]
        x, labels, valid = pad_batch(examples)
        effective = int((labels != -100).sum())
        assert effective == expected
        assert int((labels == ByteTokenizer.eos_id).sum()) == 60
        logits = torch.zeros((*labels.shape, ByteTokenizer.vocab_size), dtype=torch.float64)
        scored = sequence_log_probability(logits, labels)
        error = float((scored + target_count * math.log(ByteTokenizer.vocab_size)).abs().max())
        assert error < 1e-11
        token_checks.append({'pairs': 30, 'sides': 2, 'effective_per_side': target_count, 'total_effective': effective, 'input_shape': list(x.shape), 'label_dtype': str(labels.dtype), 'logits_dtype': str(logits.dtype), 'uniform_score_max_error': error})
    mixed = [render_chat([{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': answer}]) for answer in ['a' * 99, 'a' * 9]]
    x, labels, valid = pad_batch(mixed)
    assert labels[~valid].eq(-100).all()
    assert (labels != -100).sum(-1).tolist() == [100, 10]

    dpo_path = ROOT / 'docs/course-experiments/results/dpo.json'
    style_path = ROOT / 'docs/course-experiments/results/style.json'
    dpo, style = read_json(dpo_path), read_json(style_path)
    results = dpo['results']
    summaries = {}
    for name, run in results['runs'].items():
        summaries[name] = {'beta': run['beta'], 'steps': run['training']['steps'], 'test_generation': generation_summary(run['arithmetic']['test']), 'preference': {split: preference_summary(v) for split, v in run['preference'].items()}}
        assert summaries[name]['test_generation'] == {'records': 7, 'exact_matches': 0, 'all_raw_ids_checked': True}
    baseline = generation_summary(style['results']['content_evaluation']['test'])
    assert baseline['records'] == 7 and baseline['exact_matches'] == 0
    formatted = results['format_only']
    format_summary = {'test_generation': generation_summary(formatted['arithmetic']['test']), 'preference': {split: preference_summary(v) for split, v in formatted['preference'].items()}}
    assert format_summary['preference']['test']['relative_positive'] == 7
    assert format_summary['test_generation']['exact_matches'] == 0
    expected_questions = [r['messages'][0]['content'] for r in split_records(arithmetic_records(), seed=42)['test']]
    for evaluation in [style['results']['content_evaluation']['test'], formatted['arithmetic']['test'], *[r['arithmetic']['test'] for r in results['runs'].values()]]:
        assert [r['messages'][0]['content'] for r in evaluation['samples']] == expected_questions

    pilot = results['ultrafeedback_pilot']
    pilot_summary = {split: preference_summary(v) for split, v in pilot['evaluation'].items()}
    assert pilot_summary['validation']['relative_positive'] == 6 and pilot_summary['validation']['chosen_higher'] == 7
    assert pilot_summary['test']['relative_positive'] == 8 and pilot_summary['test']['chosen_higher'] == 4
    assert pilot['sft_training']['steps'] == pilot['training']['steps'] == 80
    assert set(pilot['evaluation']) == {'validation', 'test'}
    assert all('generated' not in row for v in pilot['evaluation'].values() for row in v['samples'])
    assert pilot['source_records'] == 100
    assert [pilot['data'][s]['records'] for s in ['train', 'validation', 'test']] == [80, 10, 10]
    assert all(len(row[k].encode('utf-8')) <= 120 for v in pilot['evaluation'].values() for row in v['samples'] for k in ['prompt', 'chosen', 'rejected'])
    model = new_lm(SimpleNamespace(seed=42, device='cpu'), width=32, layers=1, max_length=256)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == pilot['sft_training']['parameters'] == 37728

    historical = {}
    current_code = ROOT / 'scripts/course_experiments/behavior.py'
    for label, report in [('dpo', dpo), ('style', style)]:
        previous = CACHE / (label + '-historical-behavior.py')
        assert digest(previous) == report['code_sha256']['scripts/course_experiments/behavior.py']
        names = ['_natural_dpo_pilot', '_preference_evaluate', '_dpo_train', 'run_dpo'] if label == 'dpo' else ['run_style']
        historical[label] = {'revision': report['revision'], 'historical_sha256': digest(previous), 'matching_recorded_code_sha256': True, 'functions_unchanged_as_ast': {n: function_dump(previous, n) == function_dump(current_code, n) for n in names}}
        assert all(historical[label]['functions_unchanged_as_ast'].values())

    result = {
        'created_at': datetime.now(UTC).isoformat(),
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_13_9-v1/verify_lesson.py',
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'cpu'},
        'scope': 'Embedded Python executed exactly. Existing saved GPU-run scores and generation IDs independently recalculated on CPU; no training, GPU inference, timing, or quality replication.',
        'freeze': {'path': str(freeze_path.relative_to(ROOT)), 'sha256': digest(freeze_path), 'snapshot': page['snapshot'], 'source_sha256': digest(snapshot), 'figures': {}, 'current_slice_identical': True},
        'embedded': {'code_sha256': hashlib.sha256(code[0].encode()).hexdigest(), 'stdout': output.getvalue(), 'none_count': 6, 'zero_branch_output': zero_print, 'exercise_recipe': exercise, 'exercise_total': sum(exercise.values()), 'exercise_metrics_unchanged': all(v is None for v in namespace['metrics'].values())},
        'token_checks': token_checks,
        'mixed_length_mask': {'input_shape': list(x.shape), 'effective_by_side': [100, 10], 'padding_labels_all_ignored': True, 'prompt_labels_ignored': True},
        'results_sha256': {'dpo.json': digest(dpo_path), 'style.json': digest(style_path)},
        'saved_experiment_environment': {'device': dpo['device'], 'gpu': dpo['gpu'], 'seed': dpo['seed'], 'step_scale': dpo['step_scale'], 'python': dpo['python_version'], 'torch': dpo['torch_version']},
        'baseline': baseline, 'dpo_runs': summaries, 'format_only': format_summary,
        'same_seven_questions_before_after': expected_questions,
        'pilot': {'architecture': {'width': model.config.width, 'layers': model.config.layers, 'parameters': parameters}, 'sft_steps': 80, 'dpo_steps': 80, 'split_records': [80, 10, 10], 'source_records': 100, 'evaluation': pilot_summary, 'no_generation_evaluation': True, 'read_fragment_limit_bytes': 120},
        'historical_code_checks': historical,
        'status': 'all_assertions_passed',
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
