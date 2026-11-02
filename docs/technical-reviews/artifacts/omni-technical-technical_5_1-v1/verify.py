"""5.1 reviewer-owned checks: exact CPU lesson and frozen public weight evaluation.

The only optimization is the existing 40-step CPU lesson. The historical L4
weights are evaluated without gradients; the 600-step training is not rerun.
Large/raw inputs and third-party originals remain outside Git.
"""

import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import random
import re
import runpy
import subprocess
import sys

import torch
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-technical_5_1-v1')
torch.set_num_threads(1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section(path, identifier):
    raw = Path(path).read_text(encoding='utf-8')
    match = re.search(r'(?m)^## ' + re.escape(identifier) + r' .*?(?=^## |\Z)', raw, re.S | re.M)
    return match[0]


def check_lesson():
    packet_path = ROOT / 'docs/course-revision-20261029-omni/reviews/technical-stage/dispatch-briefs-v10/technical_5_1.json'
    packet = json.loads(packet_path.read_text())
    freeze = json.loads((ROOT / packet['source_freeze']).read_text())
    page = next(p for p in freeze['pages'] if p['page_id'] == '5.1')
    snapshot = ROOT / page['snapshot']
    raw = snapshot.read_text()
    assert digest(snapshot) == page['source_sha256'] == packet['page']['source_sha256']
    assert hashlib.sha256(section(ROOT / page['source'], '5.1').encode()).hexdigest() == page['source_sha256']
    assert digest(ROOT / page['notebook']) == page['notebook_sha256']
    assert page['figures_sha256'] == {} and '![' not in raw
    snippets = re.findall(r'```python\n(.*?)```', raw, re.S)
    assert len(snippets) == 1
    code = snippets[0]
    notebook = json.loads((ROOT / page['notebook']).read_text())
    notebook_codes = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assert any(code.strip() == c.strip() for c in notebook_codes)
    lines = code.splitlines()
    before_line = next(i + 1 for i, x in enumerate(lines) if x.startswith('before ='))
    step_line = next(i + 1 for i, x in enumerate(lines) if x.strip() == 'optimizer.step()')
    loss_line = next(i + 1 for i, x in enumerate(lines) if x.startswith('    loss ='))
    filename = '<exact-frozen-5.1-code>'
    observations = {}
    initial = {}
    zero_checks = []

    def observe(frame, event, arg):
        if frame.f_code.co_filename == filename and event == 'line':
            ns = frame.f_locals
            if frame.f_lineno == before_line:
                initial.update({n: p.detach().clone() for n, p in ns['model'].named_parameters()})
                mp = {id(p) for p in ns['model'].parameters()}
                op = {id(p) for g in ns['optimizer'].param_groups for p in g['params']}
                observations['optimizer_has_all_parameters'] = mp == op
            if frame.f_lineno == loss_line and len(zero_checks) < 2:
                zero_checks.append(all(p.grad is None for p in ns['model'].parameters()))
            if frame.f_lineno == step_line and ns['step'] == 0:
                g = ns['model'].embedding.weight.grad
                observations['first_embedding_grad_shape'] = list(g.shape)
                observations['first_embedding_grad_norm'] = g.norm().item()
                observations['first_embedding_grad_sqrt_sum_squares'] = g.square().sum().sqrt().item()
                observations['first_embedding_rows_with_gradient'] = (g.abs().sum(-1) > 0).tolist()
                observations['all_first_parameter_gradients_finite'] = all(
                    p.grad is not None and torch.isfinite(p.grad).all().item() for p in ns['model'].parameters()
                )
        return observe

    ns = {'__name__': '__review_exact_lesson__'}
    output = io.StringIO()
    sys.settrace(observe)
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(code, filename, 'exec'), ns)
    finally:
        sys.settrace(None)
    model = ns['model']
    deltas = {n: float((p.detach() - initial[n]).abs().max()) for n, p in model.named_parameters()}
    states = sorted({int(s['step'].item()) for s in ns['optimizer'].state.values()})
    all_ignored = code.replace('y = torch.tensor([[2, 3, 4]])', 'y = torch.tensor([[-100, -100, -100]])')
    exercise_output = io.StringIO()
    exercise = {'labels': [[-100, -100, -100]]}
    try:
        with contextlib.redirect_stdout(exercise_output):
            exec(compile(all_ignored, '<5.1-all-ignored-exercise>', 'exec'), {})
    except ValueError as error:
        exercise.update(exception=type(error).__name__, message=str(error))
    else:
        raise AssertionError('all-ignore lesson did not raise')
    exercise['stdout'] = exercise_output.getvalue()
    assert '所有 labels 都被忽略' in exercise['message'] and exercise['stdout'] == ''
    x, y = ns['x'], ns['y']
    with torch.no_grad():
        logits = model(x)['logits']
        total, count = loss_sum(logits, y)
        mean = masked_loss(logits, y)
    assert mean.item() == ns['after'] and count.item() == 3
    assert states == [40] and all(zero_checks)
    assert observations['first_embedding_grad_norm'] > 0
    assert max(deltas.values()) > 0
    assert ns['after'] < ns['before']
    return {
        'packet': str(packet_path.relative_to(ROOT)), 'packet_sha256': digest(packet_path),
        'source_freeze': packet['source_freeze'], 'source_freeze_sha256': digest(ROOT / packet['source_freeze']),
        'source_sha256': page['source_sha256'], 'figures_sha256': {},
        'notebook_code_matches_exact_snapshot': True,
        'exact_snippet_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'stdout': output.getvalue(), 'config': model.config.__dict__,
        'x': x.tolist(), 'y': y.tolist(), 'x_dtype': str(x.dtype), 'y_dtype': str(y.dtype),
        'logits_shape': list(logits.shape), 'logits_dtype': str(logits.dtype),
        'before': ns['before'], 'after': ns['after'], 'effective_targets': count.item(),
        'final_loss_sum': total.item(), 'loss_sum_divided_by_count': total.item() / count.item(),
        'embedding_weight_shape': list(model.embedding.weight.shape),
        'embedding_weight_change_max': deltas['embedding.weight'],
        'changed_parameter_tensors': sum(v > 0 for v in deltas.values()),
        'parameter_tensors': len(deltas), 'optimizer_step_values': states,
        'post_zero_grad_first_two_iterations_all_none': zero_checks,
        'observations': observations, 'all_ignored_exercise': exercise,
    }


def check_history():
    result_path = ROOT / 'docs/course-experiments/results/text_foundation.json'
    result = json.loads(result_path.read_text())
    historical = {}
    for path, expected in result['code_sha256'].items():
        if path in {'tiny_perceptron/model.py', 'tiny_perceptron/data.py', 'tiny_perceptron/attention.py',
                    'tiny_perceptron/modern.py', 'tiny_perceptron/training.py',
                    'scripts/course_experiments/text.py', 'scripts/course_experiments/common.py',
                    'scripts/course_experiments/run.py'}:
            stored = CACHE / ('historical-' + path.replace('/', '__'))
            assert digest(stored) == expected
            historical[path] = {'recorded_sha256': expected, 'read_cache': str(stored),
                                'current_same': digest(ROOT / path) == expected}
    prep = CACHE / 'historical-scripts__prepare_data.py'
    assert digest(prep) == digest(ROOT / 'scripts/prepare_data.py')
    old = runpy.run_path(str(CACHE / 'historical-scripts__course_experiments__common.py'))
    prepare = runpy.run_path(str(prep))
    records = prepare['generate_records']('toy-text')
    parts = old['split_records'](records, seed=result['seed'])
    splits = {}
    rawdir = CACHE / 'regenerated-raw-splits'
    rawdir.mkdir(exist_ok=True)
    examples = {}
    for name, rows in parts.items():
        raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        path = rawdir / (name + '.jsonl')
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
        assert digest(path) == result['results']['data'][name]['sha256']
        examples[name] = old['text_examples'](rows, mode='text', max_length=128)
        lens = [int((y != IGNORE).sum()) for _, y in examples[name]]
        splits[name] = {'raw_cache': str(path), 'sha256': digest(path), 'records': len(rows),
                        'families': len({r['family'] for r in rows}), 'effective_target_lengths': lens,
                        'effective_targets': sum(lens), 'utf8_bytes': sum(len(r['text'].encode()) for r in rows),
                        'texts': [r['text'] for r in rows]}
    assert not (set(r['family'] for r in parts['train']) & set(r['family'] for r in parts['validation']))
    sample_rng = random.Random(result['seed'])
    exposure = sum(sum(len(y) for _, y in sample_rng.choices(examples['train'], k=16)) for _ in range(600))
    assert exposure == result['results']['training']['effective_tokens']
    manifest_path = ROOT / 'docs/course-experiments/public-releases/text_foundation.json'
    public = json.loads(manifest_path.read_text())['release']['public_manifest']
    filemap = {item['output']: item for item in public['files']}
    payloads, models, weights = {}, {}, {}
    evaluations = {}
    for name, label in [('start.pt', 'before'), ('model.pt', 'after')]:
        path = ROOT / 'checkpoints/course/text_foundation' / name
        entry = filemap[name]
        assert digest(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = TinyLM(ModelConfig(**payload['config']))
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        payloads[label], models[label] = payload, model
        weights[label] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size,
                          'url': f"https://huggingface.co/{public['repo']}/resolve/{public['revision']}/{entry['path']}",
                          'config': payload['config'], 'metadata': payload['metadata']}
        evaluations[label] = {}
        for split in ['train', 'validation']:
            x, y, valid = pad_batch(examples[split])
            with torch.no_grad():
                logits = model(x, valid=valid)['logits']
                total, count = loss_sum(logits, y)
                logp = logits.double().log_softmax(-1)
                manual_sum = -logp[y != IGNORE].gather(-1, y[y != IGNORE, None]).sum().item()
            original = result['results']['training']['initial_loss' if label == 'before' else 'final_loss'] \
                if split == 'train' else result['results'][label]['validation']['nll']
            nll = float(total) / int(count)
            assert abs(nll - original) < 1e-5
            assert abs(manual_sum / int(count) - nll) < 1e-6
            evaluations[label][split] = {'shape': list(logits.shape), 'valid_input_positions': int(valid.sum()),
                                        'ignored_labels': int((y == IGNORE).sum()), 'effective_targets': int(count),
                                        'nll_sum_float32': float(total), 'nll': nll,
                                        'manual_float64_logsoftmax_nll': manual_sum / int(count),
                                        'recorded_nll': original, 'absolute_difference': abs(nll - original)}
    changed = [name for name, start in payloads['before']['model'].items()
               if not torch.equal(start, payloads['after']['model'][name])]
    max_delta = max((start - payloads['after']['model'][name]).abs().max().item()
                    for name, start in payloads['before']['model'].items())
    assert changed
    # Inspect the actual T.4 generator's default split membership without saving data or training.
    families = sorted({r['family'] for r in records})
    random.Random(42).shuffle(families)
    a, b = int(.8 * len(families)), int(.9 * len(families))
    default_families = {'train': families[:a], 'validation': families[a:b], 'test': families[b:]}
    assert all(set(default_families[k]) == {r['family'] for r in parts[k]} for k in parts)
    return {'result_path': str(result_path.relative_to(ROOT)), 'result_sha256': digest(result_path),
            'historical_revision': result['revision'], 'historical_code': historical,
            'recorded_environment': {k: result[k] for k in ['device', 'gpu', 'torch_version', 'python_version', 'seed', 'step_scale']},
            'recorded_steps': result['results']['training']['steps'], 'batch_size_from_original_fit_default': 16,
            'reconstructed_600_step_effective_tokens': exposure,
            'splits': splits, 't4_default_split_membership_same': True,
            'public_manifest_path': str(manifest_path.relative_to(ROOT)), 'public_manifest_sha256': digest(manifest_path),
            'public_weight_inputs': weights, 'evaluations_without_gradients': evaluations,
            'changed_state_tensors': len(changed), 'total_state_tensors': len(payloads['before']['model']),
            'maximum_parameter_change': max_delta,
            'limits': 'Public inference-only exports independently establish changed weights and recomputed NLL. '
                      'Their optimizer state/step counter was removed. The private full step-600 checkpoint returned HTTP 401; '
                      '600 steps and NVIDIA L4 are recorded run metadata, not freshly measured GPU facts.'}


def main():
    assert Path.cwd().resolve() == ROOT
    current = {}
    for path, sec in [('course/chapters/04.md', '4.6'), ('course/chapters/04.md', '4.7'),
                      ('course/chapters/01.md', '1.12'), ('course/chapters/01.md', '1.13'),
                      ('course/chapters/02.md', '2.1'), ('course/chapters/05.md', '5.2'),
                      ('course/training.md', 'T.4')]:
        value = section(ROOT / path, sec)
        current[path + '#' + sec] = {'sha256': hashlib.sha256(value.encode()).hexdigest(),
                                     'inspection': 'Current exact section read; only textual/API prerequisites used; no prerequisite figure claim assessed.'}
    source_files = {}
    for name in ['torch-_tensor.py', 'torch-functional.py', 'torch-optim__optimizer.py', 'torch-optim__adamw.py',
                 'torch-optim__adam.py', 'torch-nn__modules__sparse.py', 'torch-nn__modules__module.py', 'adamw-v3.pdf']:
        source_files[name] = {'path': str(CACHE / name), 'sha256': digest(CACHE / name), 'bytes': (CACHE / name).stat().st_size}
    answer = {'reviewer_task': '/root/omni_technical_coordinator/technical_5_1',
              'checked_at': datetime.datetime.now(datetime.UTC).isoformat(),
              'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                              'torch_git_version': torch.version.git_version, 'device': 'cpu',
                              'default_dtype': str(torch.get_default_dtype()), 'threads': str(torch.get_num_threads())},
              'prerequisite_section_receipts': current, 'original_source_receipts': source_files,
              'lesson': check_lesson(), 'history': check_history()}
    print(json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
