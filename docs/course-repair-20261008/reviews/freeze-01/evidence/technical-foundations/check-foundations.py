"""Independent bounded checks for frozen 2.5/4.4; no training or writes to inputs."""

import ast
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-repair-20261008/reviews/freeze-01'
OUT = BASE / 'evidence/technical-foundations'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((BASE / 'manifest.json').read_text())
technical = json.loads((BASE / 'checks/technical-inputs-manifest.json').read_text())
checks = {}
for name in ('tiny_perceptron/__init__.py', 'tiny_perceptron/simple.py',
             'tiny_perceptron/data.py', 'tiny_perceptron/modern.py',
             'scripts/course_experiments/text.py', 'uv.lock'):
    values = {
        'manifest_sha256': manifest['implementation_sha256'][name],
        'frozen_sha256': sha(BASE / 'freeze/implementation' / name),
        'live_sha256': sha(ROOT / name),
    }
    values['match'] = len(set(values.values())) == 1
    assert values['match'], name
    checks[name] = values

sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.modern import DenseFFN
from tiny_perceptron.simple import ContextMLP

torch.set_num_threads(1)
result_path = BASE / 'freeze/technical-data/docs/course-experiments/results/simple_models.json'
assert sha(result_path) == technical['files_sha256']['docs/course-experiments/results/simple_models.json']
historical = json.loads(result_path.read_text())
data_root = ROOT / 'outputs/course-experiments/course-v1/simple_models'
public_root = ROOT / 'checkpoints/course/simple_models'
export = json.loads((public_root / 'export-manifest.json').read_text())
download = json.loads((public_root / 'download-manifest.json').read_text())
inputs = {}
for name in ('data/train.jsonl', 'data/validation.jsonl', 'data/test.jsonl', 'vocabulary.json'):
    path = data_root / name
    expected = next(a['sha256'] for a in historical['artifacts'] if a['path'] == name)
    assert sha(path) == expected
    inputs[str(path)] = {'sha256': sha(path), 'historical_artifact_match': True}
for name in ('export-manifest.json', 'download-manifest.json'):
    path = public_root / name
    inputs[str(path)] = {'sha256': sha(path)}
    if name == 'export-manifest.json':
        assert sha(path) == next(f['sha256'] for f in download['files'] if f['output'] == name)

parts = {split: [json.loads(line)['text'] for line in (data_root / f'data/{split}.jsonl').read_text().splitlines()]
         for split in ('train', 'validation', 'test')}
assert split_documents(toy_documents(), 42) == parts
assert not (set(parts['train']) & set(parts['validation']))
assert not (set(parts['train']) & set(parts['test']))
assert not (set(parts['validation']) & set(parts['test']))
vocabulary = json.loads((data_root / 'vocabulary.json').read_text())
assert vocabulary == {char: index + 2 for index, char in enumerate(sorted(set(''.join(parts['train']))))}

# Compile only the original frozen helper, avoiding unrelated experiment imports.
tree = ast.parse((BASE / 'freeze/implementation/scripts/course_experiments/text.py').read_text())
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_simple_examples')
scope = {'torch': torch}
exec(compile(ast.Module(body=[helper], type_ignores=[]), '<frozen:_simple_examples>', 'exec'), scope)
simple_examples = scope['_simple_examples']
recomputed = {}
for name, context in (('mlp1', 1), ('mlp3', 3), ('mlp5', 5)):
    path = public_root / f'{name}.pt'
    entry = next(a for a in export['files'] if a['output'] == path.name)
    expected_original = next(a['sha256'] for a in historical['artifacts'] if a['path'] == path.name)
    assert sha(path) == entry['sha256']
    assert entry['source_sha256'] == expected_original
    assert sha(path) == next(a['sha256'] for a in download['files'] if a['output'] == path.name)
    inputs[str(path)] = {'sha256': sha(path), 'original_checkpoint_sha256': expected_original,
                        'provenance': 'matching export manifest connects exported weights to historical checkpoint'}
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['context'] == context and saved['width'] == 16 and saved['kind'] == 'mlp'
    assert saved['vocabulary'] == vocabulary
    model = ContextMLP(len(vocabulary) + 2, context=context, width=16)
    model.load_state_dict(saved['model'])
    model.eval()
    values = {'parameters': sum(p.numel() for p in model.parameters()), 'splits': {}}
    assert values['parameters'] == historical['results']['runs'][name]['parameters']
    with torch.no_grad():
        for split, docs in parts.items():
            x, y = simple_examples(docs, vocabulary, context)
            nll_sum = float(F.cross_entropy(model(x), y, reduction='sum'))
            mean = float(F.cross_entropy(model(x), y))
            reference = historical['results']['runs'][name]['after_nll_same_post_update_time'][split]
            assert abs(mean - reference) < 1e-6
            values['splits'][split] = {
                'documents': len(docs), 'unicode_characters': sum(map(len, docs)),
                'eos_targets': len(docs), 'effective_targets': int(y.numel()),
                'left_context_zero_padding_is_not_scored_as_padding': True,
                'nll_sum': nll_sum, 'mean_nll': mean, 'historical_mean_nll': reference,
                'absolute_difference': abs(mean - reference),
                'unknown_target_characters': sum(char not in vocabulary for doc in docs for char in doc),
            }
    recomputed[name] = values

windows = {length: [(prefix[-length:], answer) for prefix, answer in
                    [('紅色物體是', '圓'), ('藍色物體是', '方')]] for length in (1, 3, 4, 5)}
extended = {length: [(prefix[-length:], answer) for prefix, answer in
                     [('紅色的小小物體是', '圓'), ('藍色的小小物體是', '方')]] for length in (7, 8)}
assert all(pair[0][0] == pair[1][0] for length, pair in windows.items() if length < 5)
assert windows[5][0][0] != windows[5][1][0]
assert extended[7][0][0] == extended[7][1][0]
assert extended[8][0][0] != extended[8][1][0]

torch.manual_seed(42)
layer = DenseFFN(4)
x = torch.ones(1, 2, 4)
y = layer(x)
assert tuple(layer.up.weight.shape) == (16, 4)
assert tuple(layer.down.weight.shape) == (4, 16)
assert tuple(y.shape) == (1, 2, 4)
assert torch.allclose(y[:, 0], y[:, 1])
original = y.detach().clone()
x[0, 1, 0] = 2
changed = layer(x)
assert torch.equal(changed[:, 0], original[:, 0])
assert not torch.allclose(changed[:, 1], original[:, 1])
gelu_x = torch.tensor([-3., -1., -0.5, 0., 0.5, 1., 3.])
gelu_y = F.gelu(gelu_x)
gelu_formula = gelu_x * 0.5 * (1 + torch.erf(gelu_x / (2 ** 0.5)))
gelu_difference = float((gelu_y - gelu_formula).abs().max())
assert torch.allclose(gelu_y, gelu_formula, atol=1e-6)
gelu_doc = F.gelu.__doc__
linear_doc = torch.nn.Linear.__doc__
documentation = {'torch.nn.functional.gelu': gelu_doc, 'torch.nn.Linear': linear_doc}
(OUT / 'installed-pytorch-docstrings.json').write_text(json.dumps(documentation, ensure_ascii=False, indent=2) + '\n')

evidence = {
    'reviewer': '/root/repair_tech_foundations',
    'at': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                    'device': 'cpu', 'threads': 1, 'grad_backward_executed': False,
                    'optimizer_or_training_executed': False},
    'implementation_hash_checks': checks,
    'historical_result_sha256': sha(result_path),
    'additional_local_inputs': inputs,
    'reconstructed_seed42_splits_match_historical_jsonls': True,
    'vocabulary': {'characters': len(vocabulary), 'including_two_special_ids': len(vocabulary) + 2},
    'window_literal_program': windows,
    'extended_exercise': {'prefix_lengths': [len('紅色的小小物體是'), len('藍色的小小物體是')], 'windows': extended},
    'table_recomputed_with_exported_weights_no_training': recomputed,
    'parameter_formula': 'V*D + (C*D)*H + H + H*V + V; V=17, D=H=16; 577+256*C',
    'dense_ffn': {'up_weight': list(layer.up.weight.shape), 'down_weight': list(layer.down.weight.shape),
                  'up_bias': list(layer.up.bias.shape), 'down_bias': list(layer.down.bias.shape),
                  'original_output': original.tolist(), 'changed_output': changed.detach().tolist(),
                  'identical_positions_allclose': True, 'unchanged_position_bitwise_equal': True,
                  'changed_position_differs': True, 'parameters': sum(p.numel() for p in layer.parameters()),
                  'gelu_input': gelu_x.tolist(), 'gelu_output': gelu_y.tolist(),
                  'gelu_x_times_normal_cdf_allclose': True,
                  'formula_absolute_tolerance': 1e-6,
                  'formula_max_absolute_difference': gelu_difference,
                  'check_development_note': 'First attempted atol=2e-7 failed at x=-3: built-in GELU and separately evaluated erf formula differed by 5.364418029785156e-7 in float32. Observed difference is retained; check uses atol=1e-6, without changing implementation.'},
    'official_installed_documentation': {
        'version': torch.__version__, 'file': str(OUT / 'installed-pytorch-docstrings.json'),
        'sha256': sha(OUT / 'installed-pytorch-docstrings.json'),
        'scope': 'Official installed PyTorch API docstrings; no external webpage or paper fetched'},
    'limits': ['No historical training was repeated.',
               'Checkpoint export provenance is supported by manifests and matching loss recomputation; the private original checkpoints were not loaded.',
               'No GPU/MPS/Colab execution or full-page browser verification.'],
}
(OUT / 'offline-checks.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(OUT / 'offline-checks.json'), 'sha256': sha(OUT / 'offline-checks.json'),
                  'environment': evidence['environment'], 'recomputed': recomputed,
                  'dense_ffn': evidence['dense_ffn']}, ensure_ascii=False, indent=2))
