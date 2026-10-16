"""Offline T.10 contract audit; no optimizer update or formal experiment run."""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments import compression, modalities, run
from scripts.course_experiments.common import Context, split_records
from scripts.prepare_data import generate_records
from scripts.reading_time import lesson_slices
from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    task_work = ROOT / 'work/T.10'
    task_work.mkdir(parents=True, exist_ok=True)
    result = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'device': 'cpu', 'cuda_available': str(torch.cuda.is_available())},
              'scope': 'Own offline contract probes and arithmetic; historical training only inspected, never rerun.'}
    body = lesson_slices((ROOT / 'course/training.md').read_bytes().decode('utf-8'))['T.10']
    freeze = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
    training_page = next(p for p in freeze['pages'] if p['page_id'] == 'training')
    frozen = lesson_slices((ROOT / training_page['snapshot']).read_bytes().decode('utf-8'))['T.10']
    assert body == frozen
    result['section'] = {'frozen_exact': True, 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                         'freeze_snapshot': training_page['snapshot'], 'figures': []}

    # Actual argparse dispatch, with the expensive execution boundary intercepted.
    dispatches = []
    for name, device in [('distillation', 'cpu'), ('joint', 'cuda'), ('multimodal_distillation', 'cuda')]:
        fake = {'experiment_id': name, 'elapsed_seconds': 0, 'evidence_status': 'intercepted', 'artifacts': []}
        with patch.object(sys, 'argv', ['run', '--experiment', name, '--device', device]), \
             patch.object(run, 'execute', return_value=fake) as call, contextlib.redirect_stdout(io.StringIO()):
            run.main()
        args = call.call_args.args
        expected_root = ROOT / 'outputs/course-experiments/course-v1'
        assert args[:2] == (name, device) and args[2] == expected_root / name and args[3] == expected_root
        dispatches.append({'experiment': name, 'device': device, 'output': str(args[2].relative_to(ROOT)),
                           'dependency_root': str(args[3].relative_to(ROOT)), 'dependencies': run.experiment_spec(name)['dependencies'],
                           'training_executed': False})
    result['cli_dispatch'] = dispatches
    command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.course_experiments.run', '--list-assets', 'distillation']
    listed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    assert listed.stdout.strip() == 'assets/training/gsm8k-v1.tar.gz'
    result['asset_cli'] = {'command': command, 'stdout': listed.stdout.strip(), 'exit_code': listed.returncode,
                           'fetch_scope': 'Read fetch_training_assets.py. Local archive is an LFS pointer; no LFS pull performed.'}
    fetch_command = [str(ROOT / '.venv/bin/python'), 'scripts/fetch_training_assets.py', '--list']
    fetch_list = subprocess.run(fetch_command, cwd=ROOT, text=True, capture_output=True, check=True)
    gsm = next(json.loads(line) for line in fetch_list.stdout.splitlines() if json.loads(line)['id'] == 'gsm8k')
    result['asset_cli']['fetch_list'] = {'command': fetch_command, 'selected_record': gsm, 'exit_code': fetch_list.returncode}

    seed_everything(42)
    teacher = TinyLM(ModelConfig(width=64, layers=2, max_length=128))
    teacher_path = task_work / 'dependencies/sft/model.pt'
    save_checkpoint(teacher_path, teacher, metadata={'fixture': 'untrained; no optimizer'})
    parts = split_records(generate_records('attributes-sft'), seed=42)
    (teacher_path.parent / 'dataset.json').write_text(json.dumps(parts))
    ctx = Context('cpu', task_work / 'output', task_work / 'dependencies', ROOT / 'assets/training')
    ctx.output.mkdir(parents=True, exist_ok=True)
    loaded, provenance = compression._teacher(ctx, 'sft')
    loaded_parts, data = compression._dataset(ctx, 'sft')
    assert loaded.config.width == 64 and loaded_parts == parts and data['family_intersections'] == 0
    missing = ''
    try:
        ctx.dependency('style')
    except FileNotFoundError as e:
        missing = str(e)
    assert 'style/model.pt' in missing
    result['teacher_loader_fixture'] = {'width': loaded.config.width, 'data_counts': data['counts'],
                                         'family_intersections': data['family_intersections'], 'missing_dependency_error': missing,
                                         'fixture_is_trained': False}
    records = parts['train']
    cache, _ = compression._cache_text(loaded, records, 'cpu')
    examples = compression._examples(records, 128)
    assert len(cache) == len(examples)
    for targets, (_, labels) in zip(cache, examples, strict=True):
        assert targets.shape == (int((labels != IGNORE).sum()), 264)
    result['teacher_cache_fixture'] = {'sequences': len(cache), 'vocab': 264,
                                       'all_cached_rows_match_valid_answer_labels': True,
                                       'teacher_frozen': all(not p.requires_grad for p in loaded.parameters())}

    seed_everything(42)
    student = TinyLM(ModelConfig(width=32, layers=1, heads=2, max_length=128))
    twin = copy.deepcopy(student)
    assert compression._parameter_hash(student) == compression._parameter_hash(twin)
    hidden_shape = []
    hook = student.embedding.register_forward_hook(lambda module, args, output: hidden_shape.append(list(output.shape)))
    with torch.no_grad():
        logits = student(torch.tensor([[1, 3, 4]]))['logits']
    hook.remove()
    assert hidden_shape == [[1, 3, 32]] and logits.shape == (1, 3, 264)
    result['student_width'] = {'width': student.config.width, 'embedding_output_shape': hidden_shape[0],
                               'parameters': sum(p.numel() for p in student.parameters()),
                               'teacher_parameters': sum(p.numel() for p in loaded.parameters()),
                               'twin_initialization_equal': True}

    s = torch.zeros(1, 3, 2, requires_grad=True)
    p = torch.tensor([[[.8, .2]] * 3])
    t = p.log().requires_grad_()
    labels = torch.tensor([[-100, 0, 1]])
    kl = distillation_kl(s, t, labels, temperature=1)
    manual = (p[0, 1] * (p[0, 1].log() - torch.tensor(.5).log())).sum()
    assert torch.allclose(kl, manual, atol=1e-7)
    kl.backward()
    assert t.grad is None and torch.equal(s.grad[:, 0], torch.zeros_like(s.grad[:, 0]))
    assert torch.allclose(distillation_loss(s.detach(), t, labels, alpha=0, temperature=1),
                          torch.tensor(2.0).log(), atol=1e-7)
    shape_error = ''
    try:
        distillation_kl(s[:, :2], t, labels, temperature=1)
    except ValueError as e:
        shape_error = str(e)
    assert shape_error
    result['kl_probe'] = {'kl': float(kl.detach()), 'manual': float(manual), 'atol': 1e-7,
                          'student_gradient': s.grad.tolist(), 'teacher_gradient': None, 'shape_error': shape_error}
    # Aligned answer rows remain valid despite unequal prefix lengths.
    tl = torch.zeros(1, 19, 2); sl = torch.zeros(1, 7, 2)
    tr, sr = [15, 16, 17], [3, 4, 5]
    tl[:, tr] = torch.tensor([.8, .2]).log()
    aligned = distillation_kl(sl[:, sr], tl[:, tr], torch.zeros(1, 3, dtype=torch.long), temperature=1)
    assert torch.allclose(aligned, kl.detach(), atol=1e-7)
    result['modal_row_probe'] = {'teacher_rows': tr, 'student_rows': sr, 'selected_shape': [1, 3, 2],
                                 'kl': float(aligned), 'scope': 'Artificial logits only; no visual quality measurement.'}
    prompt = 'color=red;shape=square;pitch=high;joint?'
    expected = [r['messages'][-1]['content'] for r in generate_records('attributes-sft') if r['messages'][0]['content'] == prompt]
    assert expected == ['square,high']
    fixture_student = task_work / 'untrained-w32.pt'
    save_checkpoint(fixture_student, student, metadata={'fixture': 'untrained; no optimizer'})
    command = [str(ROOT / '.venv/bin/python'), 'scripts/infer.py', str(fixture_student), '--chat', '--prompt', prompt,
               '--tokens', '24', '--device', 'cpu', '--json']
    inference = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    parsed = json.loads(inference.stdout)
    result['infer_cli_fixture'] = {'command': command, 'exit_code': inference.returncode, 'json_keys': list(parsed),
                                    'expected_by_data_rule': expected[0],
                                    'scope': 'Untrained fixture validates CLI/checkpoint/tokenizer contract, not student capability.'}

    report_path = ROOT / 'docs/course-experiments/results/distillation.json'
    historical = json.loads(report_path.read_text())
    h = {'path': str(report_path.relative_to(ROOT)), 'sha256': sha(report_path),
         'environment': {k: historical[k] for k in ['revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'evidence_status']},
         'scope': 'Existing complete run inspected; not rerun by this reviewer.', 'tasks': {}}
    for name, task in historical['results']['tasks'].items():
        a, b = (task['runs'][f'w32_{method}'] for method in ['ce', 'ce_kl'])
        matching = ['initialization_sha256', 'batch_plan_sha256', 'steps', 'training_examples', 'effective_supervised_tokens']
        assert all(a['training'][k] == b['training'][k] for k in matching)
        pairs = list(zip(a['test']['generated_samples'], b['test']['generated_samples'], strict=True))
        assert all(x['question'] == y['question'] and x['expected'] == y['expected'] for x, y in pairs)
        h['tasks'][name] = {'data': task['data'], 'matched_training': {k: a['training'][k] for k in matching},
                            'test_examples': a['test']['examples'], 'test_supervised_tokens': a['test']['supervised_tokens'],
                            'ce_correct': a['test']['correct'], 'ce_kl_correct': b['test']['correct'],
                            'item_improvements': sum(not x['exact'] and y['exact'] for x, y in pairs),
                            'item_regressions': sum(x['exact'] and not y['exact'] for x, y in pairs)}
        if name == 'attributes':
            selection = task['out_of_domain_gsm8k']['selection']
            h['gsm8k_selection'] = {k: selection[k] for k in ['source_rows', 'eligible_rows', 'selected_rows', 'skipped_rows', 'max_length', 'reserved_generation_tokens']}
            h['prompt_pair'] = [{'method': method, **{k: row[k] for k in ['question', 'expected', 'generated']}}
                                for method, branch in [('ce', a), ('ce_kl', b)] for row in branch['test']['generated_samples'] if row['question'] == prompt]
    result['historical_audit'] = h
    # Transparent objective/work distinction, not a delivered recurrent-training result.
    result['depth_recipe_derivation'] = {'fixed3': 'L3', 'sample135_uniform_expectation': '(L1+L3+L5)/3',
                                         'deep12345': '(L1+L2+L3+L4+L5)/5',
                                         'fixed3_core_calls': 3, 'sample135_mean_core_calls': (1+3+5)/3,
                                         'scope': 'Equal means here do not equalize variance, heads/losses, backward policy or measured time.'}
    result['official_torch_kl_source'] = {'version': str(torch.__version__), 'source_file_sha256': sha(Path(inspect.getsourcefile(F.kl_div))),
                                         'function': 'torch.nn.functional.kl_div', 'source_read': True,
                                         'checked': 'input log-probabilities; target same shape; reduction=none avoids implicit support-size averaging.'}
    output = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/T.10-contract-execution.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'artifact': str(output.relative_to(ROOT)), 'all_assertions': 'passed',
                      'environment': result['environment'], 'historical_tasks': h['tasks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
