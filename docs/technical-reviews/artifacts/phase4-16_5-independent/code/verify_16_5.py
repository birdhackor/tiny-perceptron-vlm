"""Independent, bounded CPU verification; no training/evaluation of saved models."""
import ast
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import ByteTokenizer, IGNORE, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from scripts.course_experiments.common import text_examples, records_sha256

OUT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.manual_seed(101)
print('ENV', json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
                        'device': 'cpu', 'threads': str(torch.get_num_threads())}))
raw_path = OUT / 'input/efficiency-original.json'
raw = json.loads(raw_path.read_text())
data_path = OUT / 'input/dataset-original.json'
data = json.loads(data_path.read_text())
print('RAW_SHA256', hashlib.sha256(raw_path.read_bytes()).hexdigest())
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == next(
    a['sha256'] for a in raw['artifacts'] if a['path'] == 'dataset.json')
for split in ('train', 'validation', 'test'):
    assert records_sha256(data[split]) == raw['results']['dataset'][split]['sha256']
    assert len(data[split]) == raw['results']['dataset'][split]['records']
print('DATASET', {s: len(data[s]) for s in ('train', 'validation', 'test')})
sets = [{json.dumps(r, sort_keys=True, ensure_ascii=False) for r in data[s]}
        for s in ('train', 'validation', 'test')]
assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
print('DATASET_RECORD_OVERLAP', 0)

# Only named original measurements/provenance are selected; notes are never displayed.
p = raw['results']['packing']
measurements = {k: p[k] for k in (
    'document_prefix_lengths', 'padded_positions', 'effective_input_positions',
    'packed_positions', 'positions_reset_per_document', 'cross_document_targets_added',
    'logit_max_error', 'gradient_max_error',
    'second_document_error_after_first_document_change_with_isolation',
    'second_document_error_without_isolation')}
print('RAW_PACKING_MEASUREMENTS', json.dumps(measurements, sort_keys=True))
assert measurements['document_prefix_lengths'] == [7, 7]
assert measurements['padded_positions'] == measurements['packed_positions'] == 14
assert measurements['effective_input_positions'] == 14
for key, printed, tolerance in [('logit_max_error', 1.90735e-6, 5e-12),
                               ('gradient_max_error', 1.78814e-7, 5e-13),
                               ('second_document_error_without_isolation', 5.42937, 5e-6)]:
    assert abs(measurements[key] - printed) <= tolerance
assert measurements['second_document_error_after_first_document_change_with_isolation'] == 0

max_length = raw['results']['models']['mha']['model']['config']['max_length']
assistant = [{'text': next(m['content'] for m in reversed(r['messages'])
                          if m['role'] == 'assistant')} for r in data['train']]
examples = text_examples(assistant, max_length=max_length)
probe = [(x[:n], y[:n]) for (x, y), n in zip(
    examples[:2], [min(19, max_length // 3), min(31, max_length // 3)], strict=True)]
assert [len(x) for x, _ in probe] == [7, 7]
print('RECONSTRUCTED_PROBE', [{'input_ids': x.tolist(), 'target_ids': y.tolist()}
                              for x, y in probe])
prefixes = [(x[:max(2, max_length // 3)], y[:max(2, max_length // 3)])
            for x, y in examples]
sampler = random.Random(raw['seed'])
count = sum(int((y != IGNORE).sum()) for _ in range(40)
            for _, y in sampler.choices(prefixes, k=3))
assert count == 845
print('RECONSTRUCTED_EFFECTIVE_TARGETS', count)
u = p['actual_updates']
for name in ('padded', 'packed'):
    v = u[name]
    assert v['requested_steps'] == v['steps'] == v['optimizer_updates'] == 40
    assert v['effective_tokens'] == count and v['skipped_updates'] == 0
    print('RAW_UPDATE', name, {k: v[k] for k in ('requested_steps', 'steps',
        'optimizer_updates', 'effective_tokens', 'warm_step_median_seconds')},
        'warm_step_median_ms', v['warm_step_median_seconds'] * 1000)
assert abs(u['weight_max_error_after_updates'] - 5.97909e-6) <= 5e-12
print('RAW_WEIGHT_MAX_ERROR', u['weight_max_error_after_updates'])
assert round(u['padded']['warm_step_median_seconds'] * 1000, 3) == 10.799
assert round(u['packed']['warm_step_median_seconds'] * 1000, 3) == 11.191

tok = ByteTokenizer()
sx, sy = render_chat([{'role': 'user', 'content': 'Q'},
                      {'role': 'assistant', 'content': 'A'}], tok)
assert sx.tolist() == [tok.bos_id, tok.user_id, *tok.encode('Q'), tok.eos_id,
                       tok.assistant_id, *tok.encode('A')]
assert sy.tolist() == [IGNORE, IGNORE, IGNORE, IGNORE, *tok.encode('A'), tok.eos_id]
print('CPU_SFT_ROLE_TARGETS', {'input_ids': sx.tolist(), 'labels': sy.tolist(),
    'effective_targets': int((sy != IGNORE).sum()), 'optimizer_updates': 0})
for prefix, heldout in [('mha_baseline', raw['results']['models']['mha']['heldout'])] + [
        (name, u[name]['heldout']) for name in ('padded', 'packed')]:
    for split in ('validation', 'test'):
        h = heldout[split]
        samples = h['samples']
        assert len(samples) == h['records'] == len(data[split])
        matches = 0
        for i, s in enumerate(samples):
            ids = s['generated_ids']
            ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact = ids == tok.encode(s['expected'])
            assert exact == s['exact']
            matches += exact
            assert s['expected'] == data[split][i]['messages'][-1]['content']
            assert s['messages'] == data[split][i]['messages'][:-1]
            if split == 'test' and any('shape?' in m['content'] for m in s['messages']):
                print('RAW_SHAPE_SAMPLE', prefix, i, s)
                if prefix in ('padded', 'packed') and s['expected'] == 'circle':
                    assert s['generated'] == 'low'
        assert matches == h['matches']
        assert matches / len(samples) == h['exact_match']
        print('RECOMPUTED_EXACT_MATCH', prefix, split, f'{matches}/{len(samples)}')

# Original AST-selected computation functions only, with their original source bytes retained.
source = OUT / 'code/architecture-at-raw-revision.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('_packing_batch', '_gradients', '_gradient_error')]
namespace = {'torch': torch, 'pad_batch': pad_batch, 'loss_sum': loss_sum}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
ctx = SimpleNamespace(device='cpu')
small = [(torch.tensor([1, 20, 21]), torch.tensor([20, 21, 2])),
         (torch.tensor([1, 30]), torch.tensor([30, 2]))]
padded, packed = namespace['_packing_batch'](small, ctx)
assert padded[0].numel() == 6 and int(padded[2].sum()) == packed[0].numel() == 5
assert packed[1].tolist() == [[20, 21, 2, 30, 2]]
assert packed[3].tolist() == [[0, 1, 2, 0, 1]]
assert packed[4].tolist() == [[0, 0, 0, 1, 1]]
model = TinyLM(ModelConfig(width=8, heads=2, max_length=8)).eval()
a, ga, la = namespace['_gradients'](model, *padded)
b, gb, lb = namespace['_gradients'](model, *packed[:3], positions=packed[3], segments=packed[4])
flat = torch.cat([a[i, :len(x)] for i, (x, _) in enumerate(small)])
logit_error = float((flat - b[0]).abs().max())
gradient_error = namespace['_gradient_error'](ga, gb)
assert logit_error < 1e-6 and gradient_error < 1e-6 and abs(la-lb) < 1e-6
print('CPU_PACKING_FORWARD_BACKWARD', {'padded_positions': 6, 'packed_positions': 5,
    'valid_targets': 5, 'logit_max_error': logit_error, 'gradient_max_error': gradient_error,
    'loss_error': abs(la-lb), 'optimizer_updates': 0})
positions = torch.arange(4)
segments = torch.tensor([[0, 0, 1, 1]])
allowed = attention_mask(positions, positions, segments=segments)
assert allowed.shape == (1, 1, 4, 4)
assert allowed[0, 0].int().tolist() == [[1,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,1,1]]
assert attention_mask(positions, positions)[0,0,2].int().tolist() == [1,1,1,0]
print('EXERCISE_NO_SEGMENTS_ROW_2', [1,1,1,0])
assert attention_mask(torch.tensor([0,1,0,1]), torch.tensor([0,1,0,1]), segments=segments).equal(allowed)
print('RESET_POSITIONS_WITH_SEGMENTS_MASK', allowed[0,0].int().tolist())
ids = torch.tensor([[20, tok.eos_id, 30, 31]])
changed = ids.clone(); changed[0,0] = 40
with torch.no_grad():
    isolated_a = model(ids, segments=segments)['logits'][:,2:]
    isolated_b = model(changed, segments=segments)['logits'][:,2:]
    causal_a = model(ids)['logits'][:,2:]
    causal_b = model(changed)['logits'][:,2:]
isolated_error = float((isolated_a-isolated_b).abs().max())
causal_error = float((causal_a-causal_b).abs().max())
assert isolated_error == 0 and causal_error > 1e-6
print('CPU_EOS_PRESERVED_PERTURBATION', {'eos_id': tok.eos_id, 'eos_input_position': 1,
    'isolated_second_document_max_error': isolated_error, 'causal_second_document_max_error': causal_error})
print('TOY_PADDING_ARITHMETIC', {'padded_positions': 2*4, 'content_positions': 2*2,
    'pad_positions': 2*4-2*2, 'pad_fraction': (2*4-2*2)/(2*4), 'packed_positions': 4})
print('ALL_ASSERTIONS_PASSED')
