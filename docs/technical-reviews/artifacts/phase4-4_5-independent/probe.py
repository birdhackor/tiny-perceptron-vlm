"""Bounded CPU check of the current 4.5 Block; no optimizer or training data."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import Block, ModelConfig

torch.set_num_threads(1)
assert torch.version.cuda is None
torch.set_default_device('cpu')
torch.manual_seed(42)
block = Block(ModelConfig(width=8))
x = torch.randn(2, 3, 8)
before = {name: p.detach().clone() for name, p in block.named_parameters()}
events = []
values = {}
def capture(name):
    def hook(module, inputs, output):
        events.append(name)
        values[name] = (inputs[0].detach().clone(),
                        (output[0] if isinstance(output, tuple) else output).detach().clone())
    return hook
handles = [getattr(block, name).register_forward_hook(capture(name))
           for name in ('norm1', 'attention', 'norm2', 'ffn')]
y, cache, auxiliary = block(x)
scalar = auxiliary.item()
for handle in handles:
    handle.remove()
attended, expected_cache = block.attention(block.norm1(x))
u = x + attended
expected = u + block.ffn(block.norm2(u))
assert events == ['norm1', 'attention', 'norm2', 'ffn']
assert torch.equal(values['norm1'][0], x)
assert torch.equal(values['attention'][0], values['norm1'][1])
assert torch.equal(values['norm2'][0], u)
assert torch.equal(values['ffn'][0], values['norm2'][1])
assert torch.equal(y, expected)
assert y.shape == x.shape and scalar == 0.0 and auxiliary.shape == ()
assert all(torch.equal(before[name], p) and p.grad is None for name, p in block.named_parameters())
normed = block.norm1(x)
expected_k = block.attention.k(normed).view(2, 3, 1, 8).transpose(1, 2)
expected_v = block.attention.v(normed).view(2, 3, 1, 8).transpose(1, 2)
assert torch.equal(cache[0], expected_k) and torch.equal(cache[1], expected_v)
result = {
    'environment': {'python': sys.version, 'torch': torch.__version__,
                    'torch_git_version': torch.version.git_version, 'cuda_build': str(torch.version.cuda),
                    'device': str(x.device), 'threads': torch.get_num_threads()},
    'seed': 42, 'default_config': vars(ModelConfig(width=8)),
    'original': {'x_shape': list(x.shape), 'y_shape': list(y.shape), 'elements': x.numel(),
                 'hook_order': events, 'formula_max_abs_error': float((y-expected).detach().abs().max()),
                 'y_minus_x_max_abs': float((y-x).detach().abs().max()),
                 'cache_shapes': [list(c.shape) for c in cache],
                 'cache_exact_k_v_projection': True, 'aux_shape': list(auxiliary.shape),
                 'aux_item': scalar, 'aux_item_type': type(scalar).__name__,
                 'aux_requires_grad': auxiliary.requires_grad, 'y_requires_grad': y.requires_grad,
                 'parameter_count': sum(p.numel() for p in block.parameters()),
                 'parameter_tensor_count': len(before), 'parameters_unchanged_after_forward_item': True,
                 'parameter_grads_all_none_after_forward_item': True},
}
variants = []
for shape in [(1, 3, 8), (2, 4, 8)]:
    changed_y, _, changed_aux = block(torch.randn(*shape))
    assert tuple(changed_y.shape) == shape and changed_aux.item() == 0
    variants.append({'input_shape': list(shape), 'output_shape': list(changed_y.shape),
                     'elements': changed_y.numel(), 'auxiliary': changed_aux.item()})
wider = Block(ModelConfig(width=12))
wider_y, _, _ = wider(torch.randn(2, 3, 12))
assert wider_y.shape == (2, 3, 12)
variants.append({'input_shape': [2, 3, 12], 'config_width': 12, 'output_shape': list(wider_y.shape),
                 'elements': wider_y.numel()})
try:
    block(torch.randn(2, 3, 12))
except RuntimeError as error:
    width_error = str(error)
else:
    raise AssertionError('width mismatch must fail')
result['variants'] = variants
result['width_mismatch_error'] = width_error
future = x.clone()
future[:, 2, 0] += 3
future_y, _, _ = block(future)
prefix_delta = (future_y[:, :2] - y[:, :2]).detach().abs().max().item()
last_delta = (future_y[:, 2] - y[:, 2]).detach().abs().max().item()
assert prefix_delta == 0 and last_delta > 0
past = x.clone()
past[:, 0, 0] += 3
past_y, _, _ = block(past)
later_delta = (past_y[:, 1:] - y[:, 1:]).detach().abs().max().item()
assert later_delta > 0
local_before = block.ffn(block.norm2(u))
local_u = u.clone()
local_u[:, 2, 0] += 3
local_after = block.ffn(block.norm2(local_u))
assert torch.equal(local_before[:, :2], local_after[:, :2])
assert not torch.equal(local_before[:, 2], local_after[:, 2])
result['data_flow'] = {'perturbation': '+3 in feature 0, both batch rows',
                       'future_token_prefix_max_abs_delta': prefix_delta,
                       'future_token_last_max_abs_delta': last_delta,
                       'past_token_later_max_abs_delta': later_delta,
                       'LN_FFN_unchanged_other_positions': True}
# A separate synthetic scalar verifies differentiability. It is not a language-model loss.
gx = x.detach().clone().requires_grad_(True)
gy, _, _ = block(gx)
gy.square().sum().backward()
grads = {name: {'finite': bool(torch.isfinite(p.grad).all()), 'abs_sum': float(p.grad.abs().sum())}
         for name, p in block.named_parameters()}
assert all(g['finite'] and g['abs_sum'] > 0 for g in grads.values())
assert torch.isfinite(gx.grad).all()
assert all(torch.equal(before[name], p) for name, p in block.named_parameters())
result['backward_probe'] = {'scalar': 'sum(y**2), synthetic diagnostic only',
                            'input_grad_shape': list(gx.grad.shape),
                            'input_grad_abs_sum': float(gx.grad.abs().sum()),
                            'parameter_grads': grads, 'parameters_unchanged_without_optimizer': True}
# With both branch outputs exactly zero, the residual route is the identity, including its Jacobian.
identity = copy.deepcopy(block)
with torch.no_grad():
    identity.attention.out.weight.zero_()
    identity.ffn.down.weight.zero_()
    identity.ffn.down.bias.zero_()
ix = x.detach().clone().requires_grad_(True)
iy, _, _ = identity(ix)
iy.sum().backward()
assert torch.equal(iy, ix) and torch.equal(ix.grad, torch.ones_like(ix))
result['zero_branches'] = {'output_exact_identity': True, 'input_gradient_exact_ones': True}
result['repository_hashes'] = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ('tiny_perceptron/model.py', 'tiny_perceptron/attention.py', 'tiny_perceptron/modern.py')
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
