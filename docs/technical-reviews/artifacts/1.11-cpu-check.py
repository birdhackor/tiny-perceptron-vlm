"""1.11 原例、教材練習及有界 CPU API 檢查；沒有參數訓練。"""
from pathlib import Path
import re
import sys
import warnings
import torch

print('environment', {'python': sys.version, 'torch': str(torch.__version__),
      'torch_git_version': str(torch.version.git_version), 'device': 'cpu',
      'default_dtype': str(torch.get_default_dtype()), 'optimization': str(sys.flags.optimize),
      'num_threads': str(torch.get_num_threads())})
assert torch.version.cuda is None
text = Path('course/chapters/01.md').read_text()
section = re.search(r'(?ms)^## 1\.11 .*?(?=^## )', text).group()
code = re.search(r'```python\n(.*?)```', section, re.S).group(1)
print('\n[原封不動的原例]')
ns = {'__name__': '__main__'}
exec(compile(code, 'course/chapters/01.md#1.11:original', 'exec'), ns)
w = ns['w']
first_cost, second_cost, loss = (ns[x] for x in ('first_cost', 'second_cost', 'loss'))
print('parameter_shape', tuple(w.shape), 'gradient_shape', tuple(w.grad.shape))
print('indices', w[0].item(), w[1].item(), 'requires_grad', w.requires_grad)
print('leaf_flags', {'w': w.is_leaf, 'first_cost': first_cost.is_leaf,
                     'second_cost': second_cost.is_leaf, 'loss': loss.is_leaf})
print('grad_fn', {'w': str(w.grad_fn), 'first_cost': type(first_cost.grad_fn).__name__,
                  'second_cost': type(second_cost.grad_fn).__name__, 'loss': type(loss.grad_fn).__name__})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    print('default_nonleaf_grad', first_cost.grad, second_cost.grad, loss.grad)
    print('nonleaf_warning_count', len(caught))
print('detach', w.detach().tolist(), w.detach().requires_grad)
print('item_types', type(first_cost.item()).__name__, type(second_cost.item()).__name__, type(loss.item()).__name__)

print('\n[教材指定練習：只改係數及預期第二格]')
exercise = code.replace('second_cost = 0.5 * w[1].square()', 'second_cost = 2 * w[1].square()')
exercise = exercise.replace('torch.tensor([-4.0, 2.0])', 'torch.tensor([-4.0, 8.0])')
ens = {'__name__': '__main__'}
exec(compile(exercise, 'course/chapters/01.md#1.11:exercise', 'exec'), ens)
print('gradient_ratio', (ens['w'].grad / w.grad).tolist())

print('\n[比較與 assert 的成功、容忍及失敗行為]')
ref = torch.tensor([-4.0, 2.0])
near = ref + torch.tensor([1e-6, 1e-6])
wrong = ref + torch.tensor([0.0, 0.1])
print('allclose', {'exact': torch.allclose(w.grad, ref), 'near': torch.allclose(w.grad, near),
                   'wrong': torch.allclose(w.grad, wrong)})
try:
    assert torch.allclose(w.grad, wrong)
except AssertionError as e:
    print('intentional_assertion_failure', type(e).__name__)
print('parameter_after_checks', w.detach().tolist())

print('\n[一萬格 dense 葉參數的梯度形狀]')
large = torch.ones(10000, requires_grad=True)
large.square().sum().backward()
print('large_parameter_shape', tuple(large.shape), 'large_gradient_shape', tuple(large.grad.shape),
      'all_gradients_two', torch.equal(large.grad, torch.full((10000,), 2.0)))

print('\n[本例兩條獨立路徑的各項偏導]')
q = torch.tensor([1.0, 2.0], requires_grad=True)
a, b = (q[0] - 3).square(), 0.5 * q[1].square()
print('first_term_partials', torch.autograd.grad(a, q, retain_graph=True)[0].tolist())
print('second_term_partials', torch.autograd.grad(b, q)[0].tolist())

print('\n[在当前位置沿各坐標微調]')
def cost(x, y):
    return (x - 3) ** 2 + 0.5 * y ** 2
print('h=0.001', {'base': cost(1.0, 2.0), 'increase_first': cost(1.001, 2.0),
                  'decrease_second': cost(1.0, 1.999)})
print('completed')
