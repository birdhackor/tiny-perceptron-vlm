"""9.10 frozen inline code and archived-logit audit; no training or model downloads."""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import re
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.course_experiments.modalities import _audio_records, _vision_records
from tiny_perceptron.alignment import reliability_bins as repository_bins


ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-9-10-v1')
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def run_inline(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, 'frozen-9.10-inline', 'exec'), namespace)
    return namespace, stdout.getvalue()


def independent_metrics(logits, labels, temperature):
    """Python float64 math and explicit intervals, independent of Torch metric code."""
    rows = []
    for z, label in zip(logits, labels, strict=True):
        top = max(z)
        exps = [math.exp((value - top) / temperature) for value in z]
        total = math.fsum(exps)
        p = [value / total for value in exps]
        predicted = z.index(top)
        rows.append({
            'confidence': max(p), 'correct': int(predicted == label),
            'nll': math.log(total) - (z[label] - top) / temperature,
            'brier': math.fsum((v - int(i == label)) ** 2 for i, v in enumerate(p)),
        })
    ece = 0.0
    for i in range(5):
        chosen = [r for r in rows if i / 5 <= r['confidence'] and
                  (r['confidence'] <= (i + 1) / 5 if i == 4 else r['confidence'] < (i + 1) / 5)]
        if chosen:
            ece += abs(math.fsum(r['correct'] - r['confidence'] for r in chosen)) / len(rows)
    return {'ece': ece, **{key: math.fsum(r[key] for r in rows) / len(rows)
                           for key in ['confidence', 'nll', 'brier']}}


def main():
    freeze = json.loads(FREEZE.read_text())
    page = next(p for p in freeze['pages'] if p['page_id'] == '9.10')
    source = (ROOT / page['snapshot']).read_text()
    current = re.search(r'^## 9\.10 .*?(?=^## |\Z)',
                        (ROOT / page['source']).read_text(), re.M | re.S).group()
    assert digest_text(source) == page['source_sha256'] == digest_text(current)
    assert all(sha(ROOT / path) == value for path, value in page['figures_sha256'].items())
    blocks = re.findall(r'```python\n(.*?)```', source, re.S)
    assert len(blocks) == 2
    original, stdout = run_inline(blocks[0])
    assert len(original['bins']) == 1
    assert {k: original['bins'][0][k] for k in ['left', 'right', 'count', 'accuracy']} == {
        'left': .5, 'right': 1., 'count': 4, 'accuracy': .5}
    assert abs(original['bins'][0]['confidence'] - .75) < 1e-7
    assert abs(original['ece'] - .25) < 1e-7
    embedded = {'torch': torch}
    exec(compile(blocks[1], 'frozen-9.10-embedded-definition', 'exec'), embedded)
    inline_bins = embedded['reliability_bins']
    ast_equal = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(repository_bins)))) == ast.dump(ast.parse(blocks[1]))
    assert ast_equal
    assert inline_bins(original['confidence'], original['correct'], 2) == original['bins']
    assert len(inline_bins(original['confidence'], original['correct'])) == 3
    modified = blocks[0].replace('[0.55, 0.65, 0.85, 0.95]', '[0.55, 0.65, 0.85, 0.95, 0.25]').replace('[1, 0, 1, 0]', '[1, 0, 1, 0, 1]')
    exercise, exercise_stdout = run_inline(modified)
    assert len(exercise['bins']) == 2 and exercise['bins'][0]['count'] == 1
    assert abs(exercise['ece'] - .35) < 1e-7
    assert inline_bins(exercise['confidence'], exercise['correct'], 2) == exercise['bins']
    boundary = inline_bins(torch.tensor([0., .5, 1.]), torch.tensor([1, 0, 1]), 2)
    assert [r['count'] for r in boundary] == [1, 2]
    masks = [(original['confidence'] >= i / 2) &
             (original['confidence'] <= (i + 1) / 2 if i == 1 else original['confidence'] < (i + 1) / 2)
             for i in range(2)]
    bin_sensitivity = {str(n): sum(r['count'] / 4 * abs(r['accuracy'] - r['confidence'])
                                 for r in inline_bins(original['confidence'], original['correct'], n))
                       for n in [1, 2, 5]}

    experiment_path = ROOT / 'docs/course-experiments/results/encoders.json'
    experiment = json.loads(experiment_path.read_text())
    implementation = ROOT / 'scripts/course_experiments/modalities.py'
    assert sha(implementation) == experiment['code_sha256']['scripts/course_experiments/modalities.py']
    historical = subprocess.run(['git', 'show', experiment['revision'] + ':scripts/course_experiments/modalities.py'],
                                cwd=ROOT, check=True, capture_output=True).stdout
    assert hashlib.sha256(historical).hexdigest() == sha(implementation)
    tree = ast.parse(implementation.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_encoders')
    metric_node = next(n for n in ast.walk(run) if isinstance(n, ast.FunctionDef) and n.name == 'confidence_metrics')
    replays = {}
    for modality, make_records in [('vision', _vision_records), ('audio', _audio_records)]:
        recorded = experiment['results'][modality]
        records = make_records()
        assert all(records[s] == recorded['data']['splits'][s]['records'] for s in records)
        families = {s: set(row['family'] for row in rows) for s, rows in records.items()}
        assert not any(families[a] & families[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
        namespace = {'torch': torch, 'F': F, 'classes': recorded['classes']}
        exec(compile(ast.Module(body=[metric_node], type_ignores=[]), 'actual-confidence-metrics', 'exec'), namespace)
        actual_metrics = namespace['confidence_metrics']
        c = recorded['calibration']
        validation_logits = torch.tensor(c['validation']['logits'], dtype=torch.float32)
        validation_labels = torch.tensor(c['validation']['labels'], dtype=torch.long)
        selection = [{'temperature': t, 'nll': float(F.cross_entropy(validation_logits / t, validation_labels))}
                     for t in c['temperature_grid']]
        assert all(abs(x['nll'] - y['validation_nll']) < 1e-7 for x, y in zip(selection, c['selection'], strict=True))
        chosen = min(selection, key=lambda row: row['nll'])['temperature']
        assert chosen == c['chosen_temperature'] == .5
        replay = {'classes': recorded['classes'], 'selection': selection, 'chosen_temperature': chosen,
                  'training': {k: recorded['training'][k] for k in ['steps', 'effective_targets', 'weights_changed', 'nonzero_gradient_seen', 'cpu_smoke']},
                  'splits': {s: {'count': len(rows), 'families': sorted(families[s])} for s, rows in records.items()}, 'metrics': {}}
        for split in ['validation', 'test']:
            v = c[split]
            logits = torch.tensor(v['logits'], dtype=torch.float32)
            labels = torch.tensor(v['labels'], dtype=torch.long)
            assert v['families'] == [r['family'] for r in records[split]]
            assert v['labels'] == [recorded['classes'].index(r['answer']) for r in records[split]]
            for state, temperature in [('original', 1.), ('calibrated', chosen)]:
                observed = actual_metrics(logits, labels, temperature)
                expected = v[state]
                numeric_fields = ['ece', 'brier', 'nll', 'mean_confidence', 'accuracy']
                assert all(abs(observed[k] - expected[k]) < 1e-7 for k in numeric_fields)
                assert observed['predicted_labels'] == expected['predicted_labels']
                assert sum(b['count'] for b in observed['bins']) == len(labels)
                independent = independent_metrics(v['logits'], v['labels'], temperature)
                difference = max(abs(observed[k] - independent['confidence' if k == 'mean_confidence' else k])
                                 for k in ['ece', 'brier', 'nll', 'mean_confidence'])
                assert difference < 2e-7
                replay['metrics'][split + '.' + state] = {k: observed[k] for k in ['count', 'correct', *numeric_fields]}
                replay['metrics'][split + '.' + state]['independent_float64_max_abs_difference'] = difference
            assert torch.equal(logits.argmax(-1), (logits / chosen).argmax(-1))
        original_losses = F.cross_entropy(validation_logits, validation_labels, reduction='none')
        scaled_losses = F.cross_entropy(validation_logits / chosen, validation_labels, reduction='none')
        correct = validation_logits.argmax(-1) == validation_labels
        replay['validation_nll_change_sum'] = {
            'correct_examples': float((scaled_losses - original_losses)[correct].sum()),
            'incorrect_examples': float((scaled_losses - original_losses)[~correct].sum())}
        replay['test_wrong_families'] = [r['family'] for r, p, y in zip(records['test'], c['test']['original']['predicted_labels'], c['test']['labels'], strict=True) if p != y]
        replays[modality] = replay

    prerequisites = []
    for lesson, path, semantics in [
        ('9.2', 'course/chapters/09.md', 'A if 條件 else B is the conditional expression, explained at target-generation paragraph'),
        ('9.6', 'course/chapters/09.md', 'Boolean indexing followed by float().mean() uses the selected subgroup denominator'),
        ('1.7', 'course/chapters/01.md', 'Softmax exponentiates candidates and normalizes each example along its candidate axis'),
        ('1.8', 'course/chapters/01.md', 'Mean cross entropy uses negative natural log of the true-label probability'),
        ('T.6', 'course/training.md', 'Folded fixed-experiment commands include --experiment encoders; CPU marked interface smoke')]:
        p = next(p for p in freeze['pages'] if p['page_id'] == ('training' if lesson == 'T.6' else lesson))
        pattern = '^## ' + re.escape(lesson) + r' .*?(?=^## |\Z)'
        snapshot = re.search(pattern, (ROOT / p['snapshot']).read_text(), re.M | re.S).group()
        now = re.search(pattern, (ROOT / path).read_text(), re.M | re.S).group()
        assert snapshot == now
        prerequisites.append({'lesson_id': lesson, 'source': path, 'snapshot': p['snapshot'],
                              'slice_sha256': digest_text(snapshot), 'matches_current_slice': True,
                              'inspected_semantics': semantics})
    result = {
        'created_at': datetime.now(UTC).isoformat(),
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-10-v1/verify.py',
        'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'torch_git_version': torch.version.git_version, 'device': 'cpu'},
        'freeze': {'path': str(FREEZE.relative_to(ROOT)), 'sha256': sha(FREEZE), 'source_sha256': page['source_sha256'], 'figure_sha256': page['figures_sha256']},
        'inline': {'block_sha256': [digest_text(b) for b in blocks], 'stdout': stdout, 'bins': original['bins'], 'ece': original['ece'],
                   'embedded_function_matches_repository_ast': ast_equal, 'embedded_definition_executed': True,
                   'masks': [m.tolist() for m in masks], 'boundary_0_half_1': boundary, 'default_bins': 5,
                   'exercise_stdout': exercise_stdout, 'exercise_bins': exercise['bins'], 'exercise_ece': exercise['ece'],
                   'ece_by_bin_count': bin_sensitivity},
        'small_numeric_examples': {'negative_log_0_8': -math.log(.8), 'negative_log_0_2': -math.log(.2), 'brier_0_8_0_2_target_high': .8 ** 2 + (.2 - 1) ** 2},
        'archived_empirical_replay': {'input_path': str(experiment_path.relative_to(ROOT)), 'input_sha256': sha(experiment_path),
            'original_run': {k: experiment[k] for k in ['revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'evidence_status']},
            'implementation_path': str(implementation.relative_to(ROOT)), 'implementation_sha256': sha(implementation),
            'matches_recorded_run_code_and_git_revision': True, 'fresh_training_executed': False, 'results': replays},
        'prerequisites': prerequisites,
        'result': 'All assertions passed; exact frozen inline programs, exercise, interval endpoints, and archived metrics agree. No training, GPU invocation or paid API.'}
    target = DEST / 'verification-output.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'result': result['result'], 'path': str(target.relative_to(ROOT)), 'sha256': sha(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
