"""Independent CPU checks of the frozen 14.13 arithmetic, without training."""
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

body = lesson_slices((ROOT / 'course/chapters/14.md').read_text())['14.13']
code = re.findall(r'```python\n(.*?)```', body, re.S)[0]
ns = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, 'course/chapters/14.md#14.13', 'exec'), ns)
assert stdout.getvalue() == 'state 0.6 loss 0.16 grad -2.4\n'
assert abs(ns['w'].item() - .2) < 1e-6
actual = {'state': ns['state'].item(), 'loss': ns['loss'].item(), 'grad': ns['w'].grad.item(), 'w_after_backward': ns['w'].item()}
for key, expected in [('state', .6), ('loss', .16), ('grad', -2.4)]:
    assert abs(actual[key] - expected) < 1e-6
state = torch.tensor(0.)
for _ in range(2):
    state = state + torch.tensor(.2)
two = {'state': state.item(), 'loss': (state - 1).square().item()}
assert abs(two['state'] - .4) < 1e-6
assert abs(two['loss'] - .36) < 1e-6

# Retain the three intermediate state gradients, then sever only the first two
# applications for a separate truncation comparison. No optimizer is created.
w = torch.tensor(.2, requires_grad=True)
state = torch.tensor(0.)
states = []
for _ in range(3):
    state = state + w
    state.retain_grad()
    states.append(state)
(state - 1).square().backward()
full = {'state': state.item(), 'w_grad': w.grad.item(), 'state_grads': [s.grad.item() for s in states]}
w = torch.tensor(.2, requires_grad=True)
state = torch.tensor(0.)
with torch.no_grad():
    for _ in range(2):
        state = state + w
state = state + w
(state - 1).square().backward()
truncated = {'state': state.item(), 'w_grad': w.grad.item()}
assert abs(full['w_grad'] + 2.4) < 1e-6
assert abs(truncated['w_grad'] + .8) < 1e-6
assert all(abs(g + .8) < 1e-6 for g in full['state_grads'])
result = {
    'schema_version': 1,
    'reviewer_task': '/root/tech_20261016_14_13',
    'command': 'uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/14.13-check.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'dtype': str(torch.get_default_dtype())},
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
    'exact_snippet_stdout': stdout.getvalue(),
    'exact_snippet_raw_values': actual,
    'two_steps': two,
    'full_three_steps': full,
    'last_step_only_backward': truncated,
    'result': 'All assertions passed at absolute tolerance 1e-6. The exact snippet changes gradients but leaves w unchanged. Truncation leaves the forward state unchanged but changes the gradient.',
    'scope': 'CPU scalar float32 mechanism checks only; no optimizer update, Transformer, training, memory benchmark, GPU or model quality result.'
}
output = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/14.13-execution.json'
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
