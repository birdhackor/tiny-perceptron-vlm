from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None
torch.set_default_device('cpu')
env = {'python': sys.version, 'python_executable': sys.executable,
       'platform': platform.platform(), 'torch': str(torch.__version__),
       'torch_git_version': torch.version.git_version, 'device': 'cpu',
       'cuda_build': str(torch.version.cuda), 'threads': str(torch.get_num_threads())}
(BASE / 'runs/environment.json').write_text(json.dumps(env, indent=2) + '\n')

fence = BASE / 'code/fence-original.py'
run = subprocess.run([sys.executable, str(fence)], capture_output=True, timeout=30, check=False)
(BASE / 'runs/original.stdout.txt').write_bytes(run.stdout)
(BASE / 'runs/original.stderr.txt').write_bytes(run.stderr)
assert run.returncode == 0, run.stderr.decode()
expected = '教師梯度 None\n教師最大改動 0.0\n學生有改動 True\n輸入可微時教師輸出記圖 True\n'
assert run.stdout.decode() == expected
print('ORIGINAL FENCE stdout exact:', run.stdout.decode().strip())

raw = fence.read_bytes()
replacement = b'with torch.no_grad():\n    print("' + '輸入可微時教師輸出記圖'.encode() + b'", teacher(probe).requires_grad)'
old = b'print("' + '輸入可微時教師輸出記圖'.encode() + b'", teacher(probe).requires_grad)'
assert raw.count(old) == 1
variant = raw.replace(old, replacement)
(BASE / 'code/fence-no-grad-variant.py').write_bytes(variant)
result = subprocess.run([sys.executable, str(BASE / 'code/fence-no-grad-variant.py')], capture_output=True, timeout=30, check=False)
(BASE / 'runs/no-grad-variant.stdout.txt').write_bytes(result.stdout)
(BASE / 'runs/no-grad-variant.stderr.txt').write_bytes(result.stderr)
assert result.returncode == 0
assert result.stdout.decode() == expected.replace('記圖 True', '記圖 False')
print('EXERCISE only final teacher forward in no_grad: last flag=False; original update flags retained')

torch.manual_seed(0)
teacher = nn.Linear(2, 3, dtype=torch.float64).eval().requires_grad_(False)
student = nn.Linear(2, 3, dtype=torch.float64)
x = torch.tensor([[1., 2.]], dtype=torch.float64)
before = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
student_before = {k: v.detach().clone() for k, v in student.state_dict().items()}
reference = student.weight
shared_detached = student.weight.detach()
cloned_snapshot = student.weight.detach().clone()
with torch.no_grad():
    target = teacher(x)
output = student(x)
residual = (output - target).detach()
loss = (output - target).square().mean()
assert tuple(output.shape) == (1, 3)
assert torch.equal(loss.detach(), residual.square().sum() / 3)
loss.backward()
weight_gradient = 2 / 3 * residual.T @ x
bias_gradient = 2 / 3 * residual.squeeze(0)
assert torch.allclose(student.weight.grad, weight_gradient, atol=1e-14, rtol=1e-14)
assert torch.allclose(student.bias.grad, bias_gradient, atol=1e-14, rtol=1e-14)
optimizer = torch.optim.SGD(student.parameters(), lr=.1)
assert not ({id(p) for p in teacher.parameters()} & {id(p) for group in optimizer.param_groups for p in group['params']})
optimizer.step()
assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())
assert all(p.grad is None for p in teacher.parameters())
assert torch.allclose(student.weight, student_before['weight'] - .1 * weight_gradient, atol=1e-14, rtol=1e-14)
assert torch.allclose(student.bias, student_before['bias'] - .1 * bias_gradient, atol=1e-14, rtol=1e-14)
assert reference is student.weight
assert torch.equal(shared_detached, student.weight)
assert torch.equal(cloned_snapshot, student_before['weight'])
assert not torch.equal(cloned_snapshot, student.weight)
print('MSE shape=(batch=1, output_features=3); denominator=3; dW=(2/3)*residual.T@x; db=(2/3)*residual; SGD parameter_new=parameter_old-.1*gradient')
print('Full teacher weight+bias exactly unchanged; both student weight+bias changed; clone snapshot independent, reference/detach share updated storage')

probe = x.clone().requires_grad_()
y = teacher(probe)
assert y.requires_grad
y.sum().backward()
assert torch.allclose(probe.grad, teacher.weight.sum(dim=0, keepdim=True), atol=1e-14, rtol=1e-14)
assert all(p.grad is None for p in teacher.parameters())
with torch.no_grad():
    assert not teacher(probe).requires_grad
drop = nn.Dropout(p=1.)
drop.eval()
assert torch.equal(drop(x), x)
assert drop(x.clone().requires_grad_()).requires_grad
drop.train()
assert torch.equal(drop(x), torch.zeros_like(x))
print('eval changes Dropout behavior while retaining ordinary autograd; frozen Linear propagates input gradient; no_grad blocks this reverse graph')

# Isolate the optimizer parameter list from the teacher gradient settings.
teacher2 = nn.Linear(2, 3, dtype=torch.float64).eval()
student2 = nn.Linear(2, 3, dtype=torch.float64)
teacher2_before = {k: v.detach().clone() for k, v in teacher2.state_dict().items()}
optimizer2 = torch.optim.SGD(student2.parameters(), lr=.1)
((student2(x) - teacher2(x)).square().mean()).backward()
assert teacher2.weight.grad is not None
optimizer2.step()
assert all(torch.equal(teacher2_before[k], v) for k, v in teacher2.state_dict().items())
print('Unfrozen teacher computes gradients, but a student-only optimizer still leaves all teacher parameters unchanged')
print(json.dumps({'checks': 7, 'status': 'pass', 'mse': loss.item(),
                  'original_fence_sha256': hashlib.sha256(raw).hexdigest(),
                  'variant_sha256': hashlib.sha256(variant).hexdigest()}, ensure_ascii=False))
