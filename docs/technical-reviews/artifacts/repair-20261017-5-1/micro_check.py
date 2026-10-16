"""Independent 5.1 bounded CPU audit; no checkpoint reads or long training."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.set_num_threads(1)
root = Path(__file__).resolve().parents[4]
frozen = Path('/workspace/work/textbook-repair-20261017/input-freeze02/pages/5.1.md')
body = frozen.read_text(encoding='utf-8')
current = (root / 'course/chapters/05.md').read_text(encoding='utf-8')
current_body = re.search(r'^## 5\.1 .*?(?=^## |\Z)', current, re.M | re.S).group()
assert current_body == body

# Run the exact frozen fenced Python block, without changing the update count.
code = re.search(r'```python\n(.*?)```', body, re.S).group(1)
ns = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(code, 'frozen-5.1-fenced-code', 'exec'), ns)
assert ns['after'] < ns['before']

# A fresh copy makes the first backward/update boundary directly observable.
torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
parameters = dict(model.named_parameters())
initial = {name: p.detach().clone() for name, p in parameters.items()}
assert {id(p) for p in model.parameters()} == {id(p) for g in optimizer.param_groups for p in g['params']}
optimizer.zero_grad()
logits = model(x)['logits']
total, count = loss_sum(logits, y)
loss = masked_loss(logits, y)
manual_nll = sum(-math.log(float(logits[0, i].detach().softmax(-1)[target])) for i, target in enumerate(y[0])) / 3
assert abs(loss.item() - manual_nll) < 3e-7
loss.backward()
assert all(torch.equal(p, initial[name]) for name, p in parameters.items())
gradient = model.embedding.weight.grad.detach().clone()
norm_by_definition = math.sqrt(sum(v * v for v in gradient.flatten().tolist()))
assert abs(gradient.norm().item() - norm_by_definition) < 1e-7
optimizer.step()
first_delta = {name: float((p.detach() - initial[name]).abs().max()) for name, p in parameters.items()}
assert first_delta['embedding.weight'] > 0
row_changed = (model.embedding.weight.detach() - initial['embedding.weight']).abs().amax(1).tolist()
assert row_changed[0] == row_changed[4] == 0.0
first_after = masked_loss(model(x)['logits'], y).item()
optimizer.zero_grad()
assert all(p.grad is None for p in model.parameters())

# The exercise must fail before any optimizer update; partial masks average valid targets only.
empty_error = None
try:
    masked_loss(logits.detach(), torch.full_like(y, -100))
except ValueError as error:
    empty_error = str(error)
assert empty_error is not None and '所有 labels 都被忽略' in empty_error
partial_y = torch.tensor([[2, -100, 4]])
partial_sum, partial_count = loss_sum(logits.detach(), partial_y)
partial_loss = masked_loss(logits.detach(), partial_y)
manual_partial = F.cross_entropy(logits.detach()[0, [0, 2]], torch.tensor([2, 4])).item()
assert partial_count.item() == 2 and abs(partial_loss.item() - manual_partial) < 1e-7
length_error = None
try:
    model(torch.ones(1, 9, dtype=torch.long))
except ValueError as error:
    length_error = str(error)
assert length_error is not None

# Two scalar steps check the AdamW state/history claim against its published recurrence.
p = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
adam = torch.optim.AdamW([p], lr=0.03, weight_decay=0.0)
manual_p, m, v = 1.0, 0.0, 0.0
adam_steps = []
for t, g in enumerate((2.0, -1.0), 1):
    adam.zero_grad()
    p.grad = torch.tensor(g, dtype=torch.float64)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    expected_delta = -0.03 * (m / (1 - 0.9**t)) / (math.sqrt(v / (1 - 0.999**t)) + 1e-8)
    manual_p += expected_delta
    adam.step()
    assert abs(p.item() - manual_p) < 1e-14
    adam_steps.append({'step': t, 'gradient': g, 'm': m, 'v': v, 'expected_parameter': manual_p, 'observed_parameter': p.item(), 'adam_delta': expected_delta, 'sgd_delta': -0.03 * g})

report = {
    'reviewer_task': '/root/repair_tech_5_1',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git_version': torch.version.git_version, 'device': 'cpu', 'dtype': 'float32 except explicit float64 scalar AdamW', 'num_threads': str(torch.get_num_threads())},
    'scope': 'Exact 40-step width8 fixed-example CPU run; one independently observed first step; two scalar AdamW steps. No historical model/checkpoint loaded or historical training repeated.',
    'source_sha256': digest(frozen),
    'source_matches_current_section': current_body == body,
    'exact_frozen_code': {'stdout': printed.getvalue(), 'before': ns['before'], 'after': ns['after'], 'steps': 40, 'seed': 42, 'input': x.tolist(), 'labels': y.tolist(), 'effective_targets_per_step': 3, 'optimizer_defaults': {k: str(v) for k, v in ns['optimizer'].defaults.items()}},
    'boundary_audit': {'config': vars(model.config), 'embedding_shape': list(model.embedding.weight.shape), 'logits_shape': list(logits.shape), 'input_dtype': str(x.dtype), 'labels_dtype': str(y.dtype), 'all_registered_parameters_in_optimizer': True, 'parameter_tensors': len(parameters), 'parameters': sum(p.numel() for p in parameters.values()), 'all_require_grad': all(p.requires_grad for p in parameters.values()), 'loss_sum': total.item(), 'effective_targets': count.item(), 'loss': loss.item(), 'manual_softmax_nll': manual_nll, 'backward_alone_preserved_all_parameters': True, 'embedding_first_gradient': gradient.tolist(), 'embedding_grad_norm': gradient.norm().item(), 'norm_by_definition': norm_by_definition, 'first_update_max_delta_by_parameter': first_delta, 'embedding_max_delta_by_row': row_changed, 'first_update_loss': first_after, 'zero_grad_default_clears_to_none': True, 'overlength_error': length_error},
    'mask_audit': {'ignore_constant': IGNORE, 'empty_error': empty_error, 'empty_optimizer_steps': 0, 'partial_labels': partial_y.tolist(), 'partial_effective_targets': partial_count.item(), 'partial_sum': partial_sum.item(), 'partial_mean': partial_loss.item(), 'manual_valid_only_mean': manual_partial},
    'scalar_adamw_audit': {'learning_rate': 0.03, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'eps': 1e-8, 'steps': adam_steps, 'tolerance': 1e-14},
    'assertions': 'all passed',
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
