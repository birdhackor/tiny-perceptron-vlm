"""T.3 reviewer-owned CPU mechanism inspection; no formal training or final inference."""

import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import train_simple
from scripts.course_release import inference_payload
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, split_documents, toy_documents
from tiny_perceptron.model import generate
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-p6-t3-fresh-20261030')
torch.set_num_threads(2)
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'reviewer_task': '/root/omni_p6_technical_t3',
    'scope': 'CPU synthetic mechanism only: default controlled documents, at most one optimizer step per model; deterministic stub generation; no real trained model forward, no formal run.',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git_version': torch.version.git_version, 'device': 'cpu',
                    'platform': platform.platform()},
}

docs = toy_documents()
parts = split_documents(docs, 42)
vocab = {c: i + 2 for i, c in enumerate(sorted(set(''.join(parts['train']))))}
result['data'] = {
    'documents': docs, 'unique_count': len(set(docs)),
    'split_counts': {k: len(v) for k, v in parts.items()},
    'split_documents': parts,
    'duplicate_removal_preserves_split': split_documents(docs + docs[:2], 42) == parts,
    'vocabulary': vocab, 'vocab_size_with_boundary_and_unk': len(vocab) + 2,
    'validation_only_characters': sorted(set(''.join(parts['validation'])) - set(vocab)),
}
assert len(docs) == len(set(docs)) == 12
assert [len(parts[s]) for s in ('train', 'validation', 'test')] == [9, 1, 2]
assert result['data']['duplicate_removal_preserves_split']
result['examples'] = []
for context in (1, 3, 5):
    x, y = train_simple.examples(parts['train'], vocab, context)
    expected_targets = sum(len(d) + 1 for d in parts['train'])
    assert tuple(x.shape) == (expected_targets, context)
    assert y.numel() == expected_targets and x.dtype == y.dtype == torch.int64
    offset = 0
    boundaries = []
    for doc in parts['train']:
        assert x[offset].tolist() == [0] * context
        assert y[offset + len(doc)].item() == 0
        boundaries.append({'offset': offset, 'eos_target': offset + len(doc)})
        offset += len(doc) + 1
    result['examples'].append({'context': context, 'input_shape': list(x.shape),
                               'target_count': y.numel(), 'dtype': str(x.dtype),
                               'boundaries': boundaries})

V, W = len(vocab) + 2, 16
result['models'] = []
for name, context in (('bigram', 1), ('mlp1', 1), ('mlp3', 3), ('mlp5', 5)):
    model = BigramLM(V) if name == 'bigram' else ContextMLP(V, context, W)
    count = sum(p.numel() for p in model.parameters())
    expected = V * V if name == 'bigram' else V * W + context * W * W + W + W * V + V
    output = model(torch.zeros((2, context), dtype=torch.long))
    assert count == expected and list(output.shape) == [2, V]
    result['models'].append({'name': name, 'context': context, 'width': W,
                            'parameters': count, 'formula_parameters': expected,
                            'output_shape': list(output.shape),
                            'state_shapes': {k: list(v.shape) for k, v in model.state_dict().items()}})

OriginalAdamW = torch.optim.AdamW
observers = []

class SpyAdamW(OriginalAdamW):
    def __init__(self, parameters, **kwargs):
        parameters = list(parameters)
        super().__init__(parameters, **kwargs)
        self.inspected_parameters = parameters
        self.initial = [p.detach().clone() for p in parameters]
        self.observed_steps = 0
        observers.append(self)

    def step(self, *args, **kwargs):
        self.observed_steps += 1
        return super().step(*args, **kwargs)

result['train_simple_one_step'] = []
with tempfile.TemporaryDirectory(prefix='t3-mechanism-', dir=CACHE) as temp:
    for name in ('bigram', 'mlp'):
        for trained in (False, True):
            target = Path(temp) / f'{name}-{trained}.pt'
            argv = ['scripts/train_simple.py', '--model', name, '--context', '3', '--width', '16',
                    '--seed', '42', '--device', 'cpu', '--steps', '1', '--output', str(target)]
            if trained:
                argv += ['--train']
            buffer = io.StringIO()
            with patch.object(sys, 'argv', argv), patch.object(torch.optim, 'AdamW', SpyAdamW), contextlib.redirect_stdout(buffer):
                train_simple.main()
            report = json.loads(buffer.getvalue())
            observer = observers[-1]
            difference = max(float((p.detach() - before).abs().max()) for p, before in
                             zip(observer.inspected_parameters, observer.initial, strict=True))
            assert observer.observed_steps == int(trained)
            assert (difference > 0) == trained
            item = {'in_process_argv': argv, 'report': report, 'optimizer_steps': observer.observed_steps,
                    'maximum_parameter_change': difference, 'checkpoint_exists': target.is_file(),
                    'parameters_with_gradients': sum(p.grad is not None for p in observer.inspected_parameters)}
            assert target.is_file() == trained
            if trained:
                saved = torch.load(target, map_location='cpu', weights_only=True)
                context = saved['context']
                restored = BigramLM(V) if name == 'bigram' else ContextMLP(V, context, 16)
                restored.load_state_dict(saved['model'], strict=True)
                recomputed = {}
                for split in ('train', 'validation'):
                    x, y = train_simple.examples(parts[split], saved['vocabulary'], context)
                    with torch.no_grad():
                        logits = restored(x)
                        recomputed[split] = float(F.cross_entropy(logits, y))
                        explicit_mean = float(-logits.log_softmax(-1).gather(1, y[:, None]).mean())
                    assert abs(explicit_mean - recomputed[split]) < 1e-6
                    key = 'train_loss' if split == 'train' else 'validation_loss'
                    assert abs(recomputed[split] - report[key]) < 1e-7
                item['recomputed_post_update_losses'] = recomputed
                item['checkpoint_keys'] = sorted(saved)
                try:
                    load_checkpoint(target, 'cpu')
                except ValueError as error:
                    item['transformer_loader_rejection'] = str(error)
                else:
                    raise AssertionError('Simple checkpoint accepted by native Transformer loader')
            result['train_simple_one_step'].append(item)
    for name in ('bigram', 'mlp'):
        before, after = [x for x in result['train_simple_one_step'] if x['report']['model'] == name]
        assert before['report']['train_loss'] == after['report']['last_batch_loss_before_update']

    shell = subprocess.run(['bash', '-c', 'mkdir -p outputs checkpoints\nprintf old > outputs/report.json\nprintf \'{"mode":"mechanism"}\\n\' > outputs/report.json\n'], cwd=temp, text=True, capture_output=True, check=True)
    overwritten = (Path(temp) / 'outputs/report.json').read_text()
    assert json.loads(overwritten) == {'mode': 'mechanism'} and 'old' not in overwritten
    result['bash_redirection'] = {'command': shell.args, 'returncode': shell.returncode,
                                  'stdout': shell.stdout, 'result_file': overwritten,
                                  'directories_created': [(Path(temp) / x).is_dir() for x in ('outputs', 'checkpoints')]}

class ScriptedModel(torch.nn.Module):
    def __init__(self, answers, max_length=100):
        super().__init__()
        self.answers, self.calls = answers, 0
        self.config = SimpleNamespace(max_length=max_length)

    def forward(self, ids, cache=None):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        logits = torch.full((ids.shape[0], ids.shape[1], 264), -100.0)
        logits[:, -1, answer] = 100.0
        return {'logits': logits, 'cache': None}

tokenizer = ByteTokenizer()
assert tokenizer.encode('A') == [73] and tokenizer.eos_id == 2
result['generation'] = []
for label, answers, budget, maximum in (('eos', [73, 2], 32, 100), ('token_limit', [73], 32, 100), ('context_limit', [73], 32, 2)):
    stub = ScriptedModel(answers, maximum)
    output = generate(stub, torch.tensor([[tokenizer.bos_id]]), max_new_tokens=budget, eos_id=2)
    report = generation_report(tokenizer, output[0, 1:].tolist())
    result['generation'].append({'case': label, 'budget': budget, 'max_context': maximum,
                                 'forward_calls': stub.calls, 'report': report})
assert result['generation'][0]['report'] == {'answer': 'A', 'generated_ids': [73, 2], 'eos': True, 'invalid_special_tokens': [], 'valid_answer_tokens': True, 'generation_status': 'eos'}
assert len(result['generation'][1]['report']['generated_ids']) == 32
assert len(result['generation'][2]['report']['generated_ids']) == 1
same_text_limit = generation_report(tokenizer, [73])
assert same_text_limit['answer'] == result['generation'][0]['report']['answer']
assert same_text_limit['eos'] is False
result['same_visible_answer_different_stopping'] = same_text_limit

result['cli'] = []
commands = [['scripts/train_simple.py', '--help'], ['scripts/fetch_course_models.py', '--help'],
            ['scripts/infer.py', '--help'], ['-m', 'scripts.course_experiments.run', '--help'],
            ['scripts/fetch_course_models.py', '--list']]
for args in commands:
    completed = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    if args[-1] == '--list':
        selected = [json.loads(s) for s in completed.stdout.splitlines() if json.loads(s)['id'] in ('simple_models', 'text_foundation')]
        stdout = selected
    else:
        stdout = completed.stdout
    result['cli'].append({'command': completed.args, 'returncode': completed.returncode,
                          'stdout': stdout, 'stderr': completed.stderr})

raw = json.loads((ROOT / 'docs/course-experiments/results/simple_models.json').read_text())
result['preserved_result_recomputation'] = {
    'input_path': 'docs/course-experiments/results/simple_models.json',
    'input_sha256': hashlib.sha256((ROOT / 'docs/course-experiments/results/simple_models.json').read_bytes()).hexdigest(),
    'elapsed_seconds': raw['elapsed_seconds'], 'rounded_one_decimal': round(raw['elapsed_seconds'], 1),
    'timing_scope': raw['timing_scope'], 'device': raw['device'], 'seed': raw['seed'],
    'python_version': raw['python_version'], 'torch_version': raw['torch_version'],
    'modal': raw['modal'], 'revision': raw['revision'], 'step_scale': raw['step_scale'],
    'variants': {k: {'context': v['context'], 'parameters': v['parameters'], 'steps': v['steps'],
                      'last_recorded_step': v['history'][-1]['step']} for k, v in raw['results']['runs'].items()},
    'data_counts': {k: raw['results']['data'][k]['records'] for k in ('train', 'validation', 'test')},
}
assert round(raw['elapsed_seconds'], 1) == 4.3 and raw['device'] == 'cpu'
assert set(raw['results']['runs']) == {'bigram', 'mlp1', 'mlp3', 'mlp5'}
assert all(v['steps'] == v['history'][-1]['step'] == 200 for v in raw['results']['runs'].values())

foundation = generate_records('toy-text')
result['text_foundation_data'] = {'count': len(foundation), 'first_records': foundation[:2]}
simple_synthetic = {'format_version': 'simple-v1', 'model': {}, 'vocabulary': {'A': 2}, 'kind': 'bigram', 'context': 1,
                    'width': 16, 'optimizer': 'DO_NOT_EXPORT', 'torch_rng': 'DO_NOT_EXPORT', 'metadata': {}}
native_synthetic = {'format_version': 1, 'config': {}, 'model': {}, 'optimizer': 'DO_NOT_EXPORT',
                    'torch_rng': 'DO_NOT_EXPORT', 'python_rng': 'DO_NOT_EXPORT', 'training_state': 'DO_NOT_EXPORT', 'metadata': {}}
result['inference_export'] = {}
for label, payload in (('simple', simple_synthetic), ('native', native_synthetic)):
    clean = inference_payload(payload, {'reviewer_probe': 'synthetic'})
    assert not {'optimizer', 'torch_rng', 'python_rng', 'training_state'} & set(clean)
    result['inference_export'][label] = {'input_keys': sorted(payload), 'output_keys': sorted(clean)}

result['completed_at'] = datetime.now(UTC).isoformat()
print(json.dumps(result, ensure_ascii=False, indent=2))
