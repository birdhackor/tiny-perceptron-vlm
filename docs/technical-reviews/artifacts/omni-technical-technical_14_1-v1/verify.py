"""14.1 technical review: execute frozen inline code; no training or GPU use."""

import ast
import contextlib
import hashlib
import io
import json
import math
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, IGNORE, SPECIALS, load_jsonl
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import rope as actual_rope
from scripts.course_experiments.common import records_sha256, split_records, text_examples

torch.set_num_threads(1)
torch.manual_seed(141)
CACHE = Path('/workspace/work/reference/omni-technical-technical_14_1-v1')
OUT = Path(__file__).with_name('verification-output.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


result = {
    'created_at': datetime.now(UTC).isoformat(),
    'environment': {'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu',
                    'torch_git_version': str(torch.version.git_version), 'threads': '1'},
    'scope': 'Frozen inline examples and CPU mechanism checks; arithmetic/source audit of existing GPU record; no training.',
}
freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == '14.1')
snapshot = ROOT / page['snapshot']
assert sha(snapshot) == page['source_sha256']
body = snapshot.read_text()
blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(blocks) == 3
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    for index, code in enumerate(blocks):
        exec(compile(code, f'{snapshot}:python_block_{index + 1}', 'exec'), namespace)
lines = stdout.getvalue().splitlines()
assert lines == ['-0.7071', '-0.7071', '0.0', '2對5 3.9298 12對15 3.9298',
                 '3對5 5.1274', '與函式庫版本相同： True'], lines
score, inline_rope = namespace['score'], namespace['rope']
result['inline_blocks'] = {
    'source': str(snapshot.relative_to(ROOT)), 'source_sha256': sha(snapshot),
    'block_sha256': [hashlib.sha256(code.encode()).hexdigest() for code in blocks],
    'stdout': lines, 'random_seed_for_unseeded_check': 141,
    'random_shape': list(namespace['check'].shape), 'random_equal': bool(namespace['same']),
    'r_shape': list(namespace['r'].shape),
}
frequencies = 10000 ** (-torch.arange(0, 8, 2) / 8)
assert torch.allclose(frequencies, torch.tensor([1.0, 0.1, 0.01, 0.001]), atol=1e-8, rtol=0)
stack = torch.stack((torch.tensor([10, 11]), torch.tensor([20, 21])), -1)
assert stack.tolist() == [[10, 20], [11, 21]]
assert stack.flatten(-2).tolist() == [10, 20, 11, 21]
result['trigonometry_and_layout'] = {
    'unit_vectors': {str(deg): [math.cos(math.radians(deg)), math.sin(math.radians(deg))]
                     for deg in [45, 90, 135, 180]},
    'frequencies_d8': frequencies.tolist(), 'fastest_degrees_per_position': math.degrees(1),
    'angle_table_shape': list((torch.arange(16)[:, None] * frequencies[None, :]).shape),
    'broadcast_shape': list((torch.arange(16)[:, None] * frequencies[None, :])[None, None].shape),
    'stack': stack.tolist(), 'flatten': stack.flatten(-2).tolist(),
    'three_position_reference': 2 * sum(math.cos(a) for a in [3, .3, .03, .003]),
    'two_position_reference': 2 * sum(math.cos(a) for a in [2, .2, .02, .002]),
}
assert round(result['trigonometry_and_layout']['three_position_reference'], 4) == 3.9298
assert round(result['trigonometry_and_layout']['two_position_reference'], 4) == 5.1274
exercise_r = inline_rope(torch.ones(1, 1, 16, 2), torch.arange(16))[0, 0]
result['exercises'] = {'score_22_25': round(score(22, 25), 4),
                       'score_22_26': round(score(22, 26), 4),
                       'd2_2_dot_5': float(exercise_r[2] @ exercise_r[5]),
                       'd2_analytic': 2 * math.cos(3)}
assert result['exercises']['score_22_25'] == -0.7071
assert result['exercises']['score_22_26'] == -1.0
assert abs(result['exercises']['d2_2_dot_5'] - 2 * math.cos(3)) < 1e-6

# Independent block-matrix reference; also use nonidentical coordinates and nonidentical Q/K.
xx = torch.randn(2, 4, 16, 8)
theta = torch.arange(16)[:, None] * frequencies[None, :]
reference = torch.empty_like(xx)
for position in range(16):
    for pair in range(4):
        c, s = math.cos(float(theta[position, pair])), math.sin(float(theta[position, pair]))
        rot = torch.tensor([[c, -s], [s, c]])
        reference[:, :, position, 2 * pair:2 * pair + 2] = xx[:, :, position, 2 * pair:2 * pair + 2] @ rot.T
actual = inline_rope(xx, torch.arange(16))
max_matrix_error = float((actual - reference).abs().max())
assert max_matrix_error < 5e-7
qq, kk = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
dot_before = float((actual_rope(qq, torch.tensor([2])) * actual_rope(kk, torch.tensor([5]))).sum())
dot_after = float((actual_rope(qq, torch.tensor([12])) * actual_rope(kk, torch.tensor([15]))).sum())
assert abs(dot_before - dot_after) < 2e-6
norm_error = float((actual.square().sum(-1) - xx.square().sum(-1)).abs().max())
assert norm_error < 5e-6
batched_positions = torch.stack((torch.arange(16), torch.arange(16) + 2))
batched = actual_rope(xx, batched_positions)
batched_reference = torch.cat([actual_rope(xx[i:i+1], batched_positions[i]) for i in range(2)])
assert torch.equal(batched, batched_reference)
try:
    inline_rope(torch.ones(1, 1, 2, 3), torch.arange(2))
except ValueError as error:
    odd_error = str(error)
else:
    raise AssertionError('odd feature dimension not rejected')
result['rotation_reference_checks'] = {
    'independent_matrix_max_abs_error': max_matrix_error,
    'common_shift_nonidentical_qk_dot': [dot_before, dot_after],
    'norm_squared_max_abs_error_float32': norm_error,
    'batched_positions_equal_per_row': bool(torch.equal(batched, batched_reference)),
    'odd_feature_error': odd_error,
    'limits': 'Mathematical identities hold over real arithmetic. Float32 checks use absolute tolerance, not bitwise shift invariance.'
}

# Apply the two edits literally to the necessary 4.6 predecessor snippet.
prereq = next(p for p in freeze['pages'] if p['page_id'] == '4.6')
snippet = re.findall(r'```python\n(.*?)```', (ROOT / prereq['snapshot']).read_text(), re.S)[0]
first_edit = snippet.replace('self.position = nn.Embedding(config.max_length, config.width)',
                             'self.position = None if config.rotary else nn.Embedding(config.max_length, config.width)')
both_edits = first_edit.replace('x = self.embedding(ids) + self.position(positions)',
                               'x = self.embedding(ids)\n        if self.position is not None:\n            x = x + self.position(positions)')
assert first_edit != snippet and both_edits != first_edit
ns1, ns2 = {}, {}
exec(first_edit, ns1)
exec(both_edits, ns2)
ids = torch.tensor([[1, 2, 3]])
try:
    ns1['TinyLM'](ModelConfig(vocab_size=20, width=8, rotary=True))(ids)
except TypeError as error:
    type_error = str(error)
else:
    raise AssertionError('one-edit version did not fail')
patched_matches = {}
for rotary in [False, True]:
    config = ModelConfig(vocab_size=20, width=8, rotary=rotary)
    local, library = ns2['TinyLM'](config), TinyLM(config)
    library.load_state_dict(local.state_dict())
    err = float((local(ids)['logits'] - library(ids)['logits']).detach().abs().max())
    assert err == 0
    patched_matches[str(rotary)] = err
model = TinyLM(ModelConfig(vocab_size=20, width=8, heads=2, rotary=True))
calls = []
def traced_rope(x, positions, base=10000):
    calls.append({'shape': list(x.shape), 'positions': positions.tolist()})
    return actual_rope(x, positions, base)
attention = model.blocks[0].attention
with patch('tiny_perceptron.modern.rope', traced_rope):
    features = model.blocks[0].norm1(model.embedding(ids))
    _, cache = attention(features)
raw_v = attention.v(features).view(1, 3, 2, 4).transpose(1, 2)
assert len(calls) == 2 and torch.equal(cache[1], raw_v)
result['model_integration'] = {
    'one_edit_error_class': 'TypeError', 'one_edit_error': type_error,
    'both_edits_max_abs_logits_difference_from_library': patched_matches,
    'rotary_model_position_table_is_none': model.position is None,
    'attention_rotation_calls': calls, 'cached_v_equals_unrotated_v_projection': bool(torch.equal(cache[1], raw_v)),
}

# Read original stored result, recompute all quoted arithmetic and dataset denominators.
run_path = ROOT / 'docs/course-experiments/results/modern.json'
run = json.loads(run_path.read_text())
variants = run['results']['variants']
data_path = ROOT / 'data/training/text-initial/tinystories-train-512.jsonl'
asset_entry = next(item for item in run['assets'][0]['files'] if item['path'].endswith('tinystories-train-512.jsonl'))
assert sha(data_path) == asset_entry['sha256']
records = load_jsonl(data_path)
for record in records:
    record['family'] = record.get('text_sha256', records_sha256([{'text': record['text']}]))
dataset = split_records(records, run['seed'])
tok = ByteTokenizer()
data_audit = {}
for split, rows in dataset.items():
    examples = text_examples(rows, max_length=128)
    effective = sum(int((labels != IGNORE).sum()) for _, labels in examples)
    independent = sum(len(row['text'].encode('utf-8')) + 1 for row in rows)
    assert independent == effective
    assert len(rows) == run['results']['dataset'][split]['records']
    assert records_sha256(rows) == run['results']['dataset'][split]['sha256']
    if split in ['validation', 'test']:
        for name in ['baseline', 'rope']:
            assert effective == variants[name]['heldout'][split]['effective_tokens']
    data_audit[split] = {'records': len(rows), 'sha256': records_sha256(rows),
                         'examples': len(examples), 'effective_targets': effective,
                         'independent_byte_plus_eos_count': independent}
train_examples = text_examples(dataset['train'], max_length=128)
sampler = random.Random(run['seed'])
training_tokens = sum(sum(int((y != IGNORE).sum()) for _, y in sampler.choices(train_examples, k=16))
                      for _ in range(240))
variant_audit = {}
for name, record in variants.items():
    config = ModelConfig(**record['model']['config'])
    nparams = sum(p.numel() for p in TinyLM(config).parameters())
    assert nparams == record['model']['parameters']
    train = record['training']
    assert train['steps'] == train['optimizer_updates'] == train['requested_steps'] == 240
    assert train['skipped_updates'] == 0 and train['effective_tokens'] == training_tokens
    assert config.width == 64 and config.layers == 2 and config.heads == 4
    for split in ['validation', 'test']:
        held = record['heldout'][split]
        assert abs(held['nll_sum'] / held['effective_tokens'] - held['nll']) < 1e-12
    variant_audit[name] = {'changes': record['changes'], 'parameters': nparams,
                          'updates_recorded': train['optimizer_updates'], 'training_effective_tokens': training_tokens}
metric_audit = {name: {split: {'nll': variants[name]['heldout'][split]['nll'],
                                    'nll_recomputed': variants[name]['heldout'][split]['nll_sum'] / variants[name]['heldout'][split]['effective_tokens'],
                                    'rounded_5': format(variants[name]['heldout'][split]['nll'], '.5f'),
                                    'effective_tokens': variants[name]['heldout'][split]['effective_tokens']}
                              for split in ['validation', 'test']} for name in ['baseline', 'rope']}
assert [metric_audit[n][s]['rounded_5'] for n in ['baseline','rope'] for s in ['validation','test']] == [
    '2.26331','2.29163','1.74158','1.77812']
sample = variants['rope']['heldout']['test']['samples'][0]
assert sample['prompt'] == 'Once upon a time, there '
assert tok.decode(sample['generated_ids']) == sample['generated'] == 'was a loked there was a bough a '
parameter_difference = variants['baseline']['model']['parameters'] - variants['rope']['model']['parameters']
assert parameter_difference == 128 * 64 == 8192
code_audit = {}
for path in ['tiny_perceptron/modern.py','tiny_perceptron/model.py','tiny_perceptron/attention.py',
             'tiny_perceptron/data.py','scripts/course_experiments/architecture.py','scripts/course_experiments/common.py']:
    historical = subprocess.run(['git', 'show', f"{run['revision']}:{path}"], capture_output=True, check=True).stdout
    historical_hash = hashlib.sha256(historical).hexdigest()
    assert historical_hash == run['code_sha256'][path]
    code_audit[path] = {'recorded_sha256': run['code_sha256'][path], 'git_object_sha256': historical_hash,
                        'current_sha256': sha(ROOT / path), 'full_file_equal': historical == (ROOT / path).read_bytes()}
    if path.endswith('architecture.py'):
        old_nodes = {node.name: node for node in ast.parse(historical).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        new_nodes = {node.name: node for node in ast.parse((ROOT / path).read_text()).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        relevant = ['_copy_matching', '_clone_config', '_text_dataset', '_steps', '_nll', '_train', '_heldout', 'run_modern']
        equal = {name: ast.dump(old_nodes[name], include_attributes=False) == ast.dump(new_nodes[name], include_attributes=False)
                 for name in relevant}
        assert all(equal.values())
        code_audit[path]['modern_relevant_ast_equal'] = equal
result['stored_experiment_audit'] = {
    'result_source': str(run_path.relative_to(ROOT)), 'result_sha256': sha(run_path),
    'revision': run['revision'], 'recorded_environment': {k: run[k] for k in ['device','gpu','python_version','torch_version','seed']},
    'asset_source': str(data_path.relative_to(ROOT)), 'asset_sha256': sha(data_path),
    'dataset': data_audit, 'variants': variant_audit, 'metrics': metric_audit,
    'parameter_difference': parameter_difference, 'rope_test_first': {'prompt': sample['prompt'], 'generated': sample['generated']},
    'code_reconciliation': code_audit,
    'limits': 'Audit of stored one-seed 240-update L4 run and source/data provenance. Original GPU run and checkpoint inference not rerun. No claim of general quality, speed, or >128-position extrapolation.'
}
result['tokenizer'] = {'vocab': tok.vocab_size, 'specials': len(SPECIALS), 'eos': tok.eos_id,
                        'A_raw_utf8': list('A'.encode('utf-8')), 'cat_raw_utf8': list('貓'.encode('utf-8')),
                        'A_token_ids': tok.encode('A'), 'cat_token_ids': tok.encode('貓')}

# SVG coordinate arithmetic is separate from the actual human-visible render inspection receipt.
svg_audit = {}
ns = {'s': 'http://www.w3.org/2000/svg'}
for path, digest in page['figures_sha256'].items():
    assert sha(ROOT / path) == digest
    tree = ET.parse(ROOT / path)
    lines_svg = tree.findall('s:line', ns)
    arrows = [line for line in lines_svg if line.get('stroke') in ['#107b6e','#ac5016','#245ba8']]
    vectors = [[float(line.get('x2')) - float(line.get('x1')), float(line.get('y1')) - float(line.get('y2'))]
               for line in arrows]
    svg_audit[path] = {'sha256': digest, 'arrow_vectors_math_y_up': vectors}
    if 'shared' in path:
        angles = []
        for i in [0, 2]:
            qv, kv = vectors[i:i+2]
            normq, normk = math.hypot(*qv), math.hypot(*kv)
            cosine = sum(a*b for a,b in zip(qv,kv)) / normq / normk
            angle = math.degrees(math.acos(cosine))
            assert abs(normq - 80) < 1e-12 and abs(normk - 80) < 1e-12
            assert abs(angle - 135) < 1e-12
            angles.append({'q_length_svg':normq,'k_length_svg':normk,'angle_degrees':angle,'unit_dot':cosine})
        svg_audit[path]['panels'] = angles
    else:
        for i, angle in [(0,45),(3,135)]:
            x,y = vectors[i]
            assert abs(x / 93 - math.cos(math.radians(angle))) < 1e-12
            assert abs(y / 93 - math.sin(math.radians(angle))) < 1e-12
        svg_audit[path]['both_unit_radius_svg'] = 93
result['svg_coordinate_audit'] = svg_audit
result['status'] = 'passed'
with OUT.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'status': 'passed', 'output': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
                  'inline_stdout': lines, 'no_training': True}, ensure_ascii=False))
