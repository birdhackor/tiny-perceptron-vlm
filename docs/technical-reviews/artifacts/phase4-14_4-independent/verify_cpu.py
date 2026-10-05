"""Bounded CPU checks; no optimizer, training, weights, model evaluation or downloads."""
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
RESULT = Path('docs/course-experiments/results/modern.json')
raw = RESULT.read_bytes()
data = json.loads(raw)
pointers = {}
def take(pointer):
    value = data
    for key in pointer.strip('/').split('/'):
        value = value[int(key)] if isinstance(value, list) else value[key]
    pointers[pointer] = value
    return value

for pointer in ['/revision', '/seed', '/device', '/torch_version', '/python_version',
                '/gpu', '/step_scale', '/results/runtime', '/results/dataset']:
    take(pointer)

rows = {}
for name in ['baseline', 'relu2']:
    root = '/results/variants/' + name
    take(root + '/model'); take(root + '/changes'); take(root + '/copied_initial_tables')
    for field in ['initial/examples', 'requested_steps', 'steps', 'optimizer_updates',
                  'skipped_updates', 'effective_tokens', 'batch_size', 'learning_rate',
                  'warm_step_median_seconds', 'budget_exhausted']:
        take(root + '/training/' + field)
    values = {}
    for side in ['validation', 'test']:
        base = root + '/heldout/' + side
        selected = {key: take(base + '/' + key) for key in
                    ['nll', 'nll_sum', 'effective_tokens', 'examples', 'records']}
        derived = selected['nll_sum'] / selected['effective_tokens']
        assert abs(derived - selected['nll']) < 1e-12
        values[side] = {'derived_nll': derived, 'rounded_5': round(derived, 5),
                        'denominators': {k: selected[k] for k in ['effective_tokens', 'examples', 'records']}}
    values['median_ms'] = take(root + '/training/warm_step_median_seconds') * 1000
    rows[name] = values
sample = take('/results/variants/relu2/heldout/test/samples/0')
assert sample['generated'] == 'was a was the the the the ar and'
# ByteTokenizer uses UTF-8 bytes offset by eight reserved tokens.
assert bytes(value - 8 for value in sample['generated_ids']).decode('utf-8') == sample['generated']
for field in ['requested_steps', 'steps', 'optimizer_updates']:
    assert all(take(f'/results/variants/{n}/training/{field}') == 240 for n in rows)
assert all(take(f'/results/variants/{n}/training/effective_tokens') == 452102 for n in rows)
assert all(take(f'/results/variants/{n}/model/parameters') == 141568 for n in rows)

# Re-derive structural parameter count, including biases, without a model forward pass.
V, D, H, L, T = 264, 64, 256, 2, 128
parameter_count = V*D + T*D + L*(4*D*D + 4*D + D*H+H + H*D+D) + 2*D + V*D
assert parameter_count == 141568
count = take('/results/variants/baseline/training/initial/examples')
assert count == take('/results/variants/relu2/training/initial/examples')
def draw():
    rng = random.Random(take('/seed'))
    return [rng.choices(range(count), k=16) for _ in range(240)]
assert draw() == draw()

x = torch.tensor([-2., -1., 0., 1., 2.], dtype=torch.float64)
reference = torch.tensor([v*0.5*math.erfc(-v/math.sqrt(2)) for v in x.tolist()], dtype=torch.float64)
gelu = F.gelu(x)
assert torch.allclose(gelu, reference, atol=1e-14, rtol=0)
assert F.relu(x).square().tolist() == [0., 0., 0., 1., 4.]
exercise = torch.tensor([-4., -2., 0., 2., 4.])
assert F.relu(exercise).square()[-1].item() == 16
assert abs(F.gelu(exercise)[-1].item() - 4.) < 0.0002
assert F.gelu(exercise)[0].item() < 0
integer = torch.tensor([-4, -2, 0, 2, 4])
integer_failure = None
try:
    F.gelu(integer)
except (RuntimeError, NotImplementedError) as error:
    integer_failure = {'dtype': str(integer.dtype), 'type': type(error).__name__, 'message': str(error)}
assert integer_failure is not None
shaped = torch.tensor([[-1., 0., 2.], [2., 0., -1.]])
assert torch.equal(F.gelu(shaped).reshape(-1), F.gelu(shaped.reshape(-1)))
assert F.gelu(shaped).shape == shaped.shape
scalar = {'linear': [3*(2*v) for v in [1,2]], 'with_square': [3*(2*v)**2 for v in [1,2]]}
assert scalar == {'linear': [6,12], 'with_square': [12,48]}
A = torch.tensor([[1.,2.], [3.,4.]], dtype=torch.float64)
B = torch.tensor([[2.,0.], [0.,3.]], dtype=torch.float64)
z = torch.tensor([1.,2.], dtype=torch.float64)
assert torch.equal(B @ (A @ z), (B @ A) @ z)

output = {'environment': {'python': sys.version, 'torch': str(torch.__version__),
                          'torch_git_version': torch.version.git_version, 'device': 'cpu',
                          'platform': platform.platform()},
          'raw_result': {'path': str(RESULT), 'sha256': hashlib.sha256(raw).hexdigest(),
                         'selected_pointers': pointers},
          'checks': {'gelu_float64_reference': reference.tolist(), 'gelu_float64_actual': gelu.tolist(),
                     'max_abs_error': float((gelu-reference).abs().max()),
                     'exercise_gelu': F.gelu(exercise).tolist(),
                     'exercise_gelu_round4': F.gelu(exercise).round(decimals=4).tolist(),
                     'exercise_relu2': F.relu(exercise).square().tolist(),
                     'integer_gelu_failure': integer_failure,
                     'elementwise_and_shape': True, 'scalar_composition': scalar,
                     'matrix_composition': True, 'parameter_count': parameter_count,
                     'sampler_sequences_equal_from_identical_seed_and_population': True,
                     'sampler_draws_per_variant': 3840,
                     'existing_report_recomputed': rows,
                     'generation_token_decode_matches': True}}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
