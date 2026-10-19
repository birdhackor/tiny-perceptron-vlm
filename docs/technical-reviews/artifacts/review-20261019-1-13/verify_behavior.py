"""核對平方算例、梯度儲存與最佳化器的實際行為。"""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import re

import torch
from torch import nn

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
torch.set_num_threads(1)
torch.set_default_device('cpu')

frozen = ROOT / 'docs/course-review-20261019/freeze-01/sources/1.13.md'
source = frozen.read_bytes()
code = re.search(r'```python\n(.*?)```', source.decode(), re.S).group(1)
assert code.count('w.grad = None\n') == 1


def run_example(program):
    namespace = {}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(program, str(frozen), 'exec'), namespace)
    return {
        'stdout': output.getvalue(),
        'first': namespace['first'].item(),
        'second': namespace['second'].item(),
        'third': namespace['w'].grad.item(),
        'w': namespace['w'].item(),
        'assertion_passed': True,
    }


original = run_example(code)
exercise_code = code.replace('w.grad = None\n', '# w.grad = None\n')
exercise = run_example(exercise_code)
assert original['stdout'] == '4.0 8.0 4.0 2.0\n'
assert exercise['stdout'] == '4.0 8.0 12.0 2.0\n'

w = nn.Parameter(torch.tensor(2.0))
plain = torch.tensor(2.0, requires_grad=True)
parameter_properties = {
    'type': type(w).__name__,
    'requires_grad': w.requires_grad,
    'is_leaf': w.is_leaf,
    'grad_initially_none': w.grad is None,
    'shape': list(w.shape),
    'dtype': str(w.dtype),
    'plain_requires_grad': plain.requires_grad,
    'repr_prefix': repr(w).splitlines()[0],
}
assert w.requires_grad and w.is_leaf and plain.requires_grad
independent_derivative = torch.autograd.grad(w.square(), w)[0].item()
assert independent_derivative == 4.0
trace = []
snapshots = []
for clear in (False, False, True):
    if clear:
        w.grad = None
    before = None if w.grad is None else w.grad.item()
    loss = w.square()
    loss.backward()
    trace.append({
        'cleared_before': clear,
        'prior_grad': before,
        'loss': loss.item(),
        'new_contribution': w.grad.item() - (before or 0),
        'accumulated_grad': w.grad.item(),
        'w': w.item(),
    })
    snapshots.append(w.grad.clone())
assert [row['new_contribution'] for row in trace] == [4.0, 4.0, 4.0]
assert [row['accumulated_grad'] for row in trace] == [4.0, 8.0, 4.0]
assert all(row['w'] == 2.0 and row['loss'] == 4.0 for row in trace)
clone_properties = {
    'snapshot_values_after_all_backward_calls': [value.item() for value in snapshots],
    'snapshot_storage_independent': all(value.data_ptr() != w.grad.data_ptr() for value in snapshots),
}
assert clone_properties['snapshot_values_after_all_backward_calls'] == [4.0, 8.0, 4.0]
assert clone_properties['snapshot_storage_independent']

embedding = nn.Embedding(3, 3)
with torch.no_grad():
    embedding.weight.zero_()
    embedding.weight[0, 1] = 2.0
embedding_properties = {
    'weight_is_parameter': isinstance(embedding.weight, nn.Parameter),
    'requires_grad': embedding.weight.requires_grad,
    'registered_parameter_names': list(dict(embedding.named_parameters())),
    'repr_prefix': repr(embedding.weight).splitlines()[0],
    'shape': list(embedding.weight.shape),
}
assert embedding_properties['weight_is_parameter']
assert embedding_properties['repr_prefix'] == 'Parameter containing:'
assert parameter_properties['repr_prefix'] == 'Parameter containing:'

fresh = nn.Parameter(torch.tensor(2.0))
loss_a = fresh.square()
loss_b = fresh.square()
fresh_graphs = {
    'distinct_loss_objects': loss_a is not loss_b,
    'distinct_grad_functions': loss_a.grad_fn is not loss_b.grad_fn,
}
loss_a.backward()
loss_b.backward()
fresh_graphs['grad_after_both_backward_calls'] = fresh.grad.item()
assert all(fresh_graphs[key] for key in ('distinct_loss_objects', 'distinct_grad_functions'))
assert fresh.grad.item() == 8.0

same = nn.Parameter(torch.tensor(2.0))
same_loss = same.square()
same_loss.backward()
try:
    same_loss.backward()
except RuntimeError as error:
    same_graph_error = str(error)
else:
    raise AssertionError('平方代價重用已釋放的計算圖時應失敗')
assert 'Trying to backward through the graph a second time' in same_graph_error

p = nn.Parameter(torch.tensor(2.0))
optimizer = torch.optim.SGD([p], lr=0.25)
optimizer.zero_grad()
optimizer_trace = {'after_initial_zero_grad_is_none': p.grad is None}
p.square().backward()
optimizer_trace.update(before_step=p.item(), grad_before_step=p.grad.item())
optimizer.step()
optimizer_trace.update(after_step=p.item(), grad_after_step=p.grad.item())
optimizer.zero_grad()
optimizer_trace['after_final_zero_grad_is_none'] = p.grad is None
assert optimizer_trace == {
    'after_initial_zero_grad_is_none': True,
    'before_step': 2.0,
    'grad_before_step': 4.0,
    'after_step': 1.0,
    'grad_after_step': 4.0,
    'after_final_zero_grad_is_none': True,
}

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
whole = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
split = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
(whole * x).square().mean().backward()
for part in (x[:1], x[1:]):
    ((split * part).square().sum() / len(x)).backward()
accumulation = {
    'input': x.tolist(),
    'parameter': 1.0,
    'microbatch_sizes': [1, 2],
    'common_denominator': len(x),
    'whole_gradient': whole.grad.item(),
    'accumulated_gradient': split.grad.item(),
    'absolute_difference': abs(whole.grad.item() - split.grad.item()),
    'parameter_unchanged': whole.item() == split.item() == 1.0,
}
assert abs(split.grad.item() - 28 / 3) < 1e-12
assert accumulation['absolute_difference'] < 1e-12

result = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-13/verify_behavior.py',
    'result': '全部斷言通過',
    'environment': {
        'python': platform.python_version(),
        'torch': str(torch.__version__),
        'torch_git_version': str(torch.version.git_version),
        'platform': platform.platform(),
        'device': 'cpu',
        'default_dtype': str(torch.get_default_dtype()),
        'threads': str(torch.get_num_threads()),
    },
    'source_sha256': hashlib.sha256(source).hexdigest(),
    'verification_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'original': original,
    'exercise': exercise,
    'independent_derivative': independent_derivative,
    'parameter_properties': parameter_properties,
    'trace': trace,
    'clone_properties': clone_properties,
    'embedding_properties': embedding_properties,
    'fresh_graphs': fresh_graphs,
    'same_graph_error': same_graph_error,
    'optimizer_trace': optimizer_trace,
    'accumulation': accumulation,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
