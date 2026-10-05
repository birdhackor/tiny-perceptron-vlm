"""Bounded CPU checks: no training, model/data downloads, or checkpoint loads."""
import ast
import hashlib
import json
import random
import sys
import time
from pathlib import Path

A = Path(__file__).resolve().parents[1]
ROOT = A.parents[3]
sys.path.insert(0, str(ROOT))
import torch
import tiny_perceptron.attention as attention
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, shifted
from tiny_perceptron.model import ModelConfig, TinyLM

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8))
result = {'environment': {'python': sys.version, 'torch': torch.__version__,
           'torch_git_version': torch.version.git_version, 'device': 'cpu',
           'num_threads': torch.get_num_threads()}, 'checks': {}}
params = [{'name': n, 'shape': list(p.shape), 'numel': p.numel(),
           'element_size': p.element_size(), 'dtype': str(p.dtype)} for n, p in model.named_parameters()]
result['parameters'] = params
count = sum(p['numel'] for p in params)
raw_bytes = sum(p['numel'] * p['element_size'] for p in params)
assert count == model.description()['parameters'] == 6104
assert raw_bytes == 24416 and all(p['dtype'] == 'torch.float32' for p in params)
result['checks']['parameter_storage'] = {'count': count, 'fp32_bytes': raw_bytes}

# Observe this repository's real attention weights without modifying its source.
original_attention = attention.manual_attention
shapes = []
def inspected_attention(*args, **kwargs):
    output, weights = original_attention(*args, **kwargs)
    shapes.append(list(weights.shape))
    return output, weights
attention.manual_attention = inspected_attention
model.eval()
out_grad = model(torch.tensor([[1, 2, 3]]))['logits']
assert out_grad.requires_grad and model.training is False
measurements = {}
with torch.no_grad():
    for length, ids in [(3, [[1, 2, 3]]), (6, [[1, 2, 3, 1, 2, 3]])]:
        x = torch.tensor(ids)
        before = sum(p.numel() for p in model.parameters())
        model(x)  # unchanged warmup policy
        repetitions = []
        for repeat in range(5):
            began = time.perf_counter()
            for iteration in range(100):
                output = model(x)
            repetitions.append((time.perf_counter() - began) / 100)
        assert not output['logits'].requires_grad
        assert list(output['logits'].shape) == [1, length, 264]
        assert sum(p.numel() for p in model.parameters()) == before == 6104
        measurements[str(length)] = {'batch_size': 1, 'length': length,
            'attention_weight_shape': shapes[-1], 'parameters': before,
            'fp32_bytes': raw_bytes, 'seconds_per_forward_100_call_repeats': repetitions}
attention.manual_attention = original_attention
assert measurements['3']['attention_weight_shape'] == [1, 1, 3, 3]
assert measurements['6']['attention_weight_shape'] == [1, 1, 6, 6]
result['checks']['six_position_variation'] = measurements
result['checks']['eval_and_no_grad'] = {'eval_logits_requires_grad': bool(out_grad.requires_grad),
    'no_grad_logits_requires_grad': bool(output['logits'].requires_grad),
    'parameters_have_grad': any(p.grad is not None for p in model.parameters()),
    'parameter_updates': 0}

# Use unchanged functions from the recorded source revision; AST extraction
# avoids importing the historical experiment runner or invoking training.
namespace = {'hashlib': hashlib, 'json': json, 'random': random,
             'ByteTokenizer': ByteTokenizer, 'shifted': shifted}
selected = {
    'scripts/prepare_data.py': ['generate_records'],
    'scripts/course_experiments/common.py': ['split_records', 'records_sha256', 'text_examples'],
}
provenance = []
for relative, names in selected.items():
    path = A / 'historical' / relative
    tree = ast.parse(path.read_bytes(), filename=str(path))
    for name in names:
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, str(path), 'exec'), namespace)
        provenance.append({'source': str(path.relative_to(ROOT)), 'function': name,
                           'lines': [node.lineno, node.end_lineno],
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

recorded = json.loads((A / 'inputs/docs/course-experiments/results/text_foundation.json').read_text())
parts = namespace['split_records'](namespace['generate_records']('toy-text'), seed=42)
split_info = {}
for split, rows in parts.items():
    raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    h = hashlib.sha256(raw).hexdigest()
    assert h == recorded['results']['data'][split]['sha256']
    assert len(rows) == recorded['results']['data'][split]['records']
    split_info[split] = {'records': len(rows), 'jsonl_sha256': h}
assert namespace['records_sha256'](parts['train']) == recorded['results']['training']['records_sha256']
examples = namespace['text_examples'](parts['train'], mode='text', max_length=128)
assert len(examples) == len(parts['train']) == 9
sampler = random.Random(42)
tokens, exposures = 0, 0
per_step = []
for step in range(600):
    batch = sampler.choices(examples, k=16)
    x, y, valid = pad_batch(batch)
    observed = int((y != IGNORE).sum())
    tokens += observed
    exposures += len(batch)
    per_step.append(observed)
assert tokens == recorded['results']['training']['effective_tokens'] == 342462
assert exposures == 9600
historical_model = TinyLM(ModelConfig(width=64, layers=2, max_length=128))
historic_params = sum(p.numel() for p in historical_model.parameters())
historic_bytes = sum(p.numel() * p.element_size() for p in historical_model.parameters())
assert historic_params == recorded['results']['training']['parameters'] == 141568
assert historic_bytes == 566272
assert round(recorded['results']['training']['seconds'], 4) == 6.1885
assert round(recorded['elapsed_seconds'], 4) == 16.0354
assert recorded['gpu'] == 'NVIDIA L4'
result['checks']['original_json_audit'] = {
    'original_result_revision': recorded['revision'], 'gpu': recorded['gpu'],
    'original_torch_version': recorded['torch_version'],
    'original_python_version': recorded['python_version'], 'seed': recorded['seed'],
    'split_info': split_info, 'ast_function_provenance': provenance,
    'train_example_target_lengths': [len(y) for _, y in examples],
    'unique_training_records': len(parts['train']), 'optimizer_updates_recorded': 600,
    'batch_size': 16, 'record_exposures_reconstructed_without_training': exposures,
    'effective_targets_reconstructed_without_training': tokens,
    'per_step_target_counts': per_step,
    'parameters_reconstructed': historic_params, 'fp32_bytes': historic_bytes,
    'training_seconds_recorded': recorded['results']['training']['seconds'],
    'experiment_seconds_recorded': recorded['elapsed_seconds'],
    'training_seconds_rounded_4dp': round(recorded['results']['training']['seconds'], 4),
    'experiment_seconds_rounded_4dp': round(recorded['elapsed_seconds'], 4),
    'timing_scope_recorded': recorded['timing_scope'],
    'training_reexecuted': False, 'weights_loaded': False}
path = A / 'bounded-check-results.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({k:v for k,v in result['checks'].items() if k != 'original_json_audit'}, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in result['checks']['original_json_audit'].items()
                 if k not in ['per_step_target_counts', 'ast_function_provenance']}, ensure_ascii=False, indent=2))
