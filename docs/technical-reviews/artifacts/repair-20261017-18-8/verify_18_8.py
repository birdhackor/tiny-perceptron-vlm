"""Bounded CPU checks for frozen lesson 18.8; no training or historical rerun."""

import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[4]
section_path = Path('/workspace/work/textbook-repair-20261017/input-freeze02/pages/18.8.md')
body = section_path.read_text()
code = re.findall(r'```python\n(.*?)```', body, flags=re.S)[0]
result = {
    'environment': {'python': sys.version, 'torch': torch.__version__,
                    'device': 'cpu', 'platform': platform.platform(), 'threads': '1'},
    'scope': 'Literal lesson code, one-label variant and deterministic arithmetic/gradient probes only; no optimizer steps, no training, no historical experiment rerun.',
    'source_sha256': hashlib.sha256(section_path.read_bytes()).hexdigest(),
}

def run_literal(source):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, 'frozen-18.8-example', 'exec'), namespace)
    return {'stdout': output.getvalue(), 'loss': namespace['loss'].item(),
            'student_gradient': namespace['student'].grad.tolist(),
            'teacher_gradient_is_none': namespace['teacher'].grad is None,
            'student_shape': list(namespace['student'].shape),
            'teacher_shape': list(namespace['teacher'].shape)}

result['literal_example'] = run_literal(code)
result['exercise_all_three'] = run_literal(code.replace('[[-100, 0, 1]]', '[[0, 0, 1]]'))
result['labels_same_mask_swapped_classes'] = run_literal(code.replace('[[-100, 0, 1]]', '[[-100, 1, 0]]'))

p = [0.8, 0.2]
q = [0.5, 0.5]
terms = [a * (math.log(a) - math.log(b)) for a, b in zip(p, q, strict=True)]
entropy = -sum(a * math.log(a) for a in p)
ce = -sum(a * math.log(b) for a, b in zip(p, q, strict=True))
reverse = sum(b * (math.log(b) - math.log(a)) for a, b in zip(p, q, strict=True))
result['double_arithmetic'] = {'forward_terms': terms, 'forward_kl': sum(terms),
    'reverse_kl': reverse, 'teacher_entropy': entropy, 'soft_target_ce': ce,
    'ce_minus_entropy': ce - entropy, 'kl_identity_residual': sum(terms) - (ce - entropy),
    'top_one_teacher': max(range(2), key=p.__getitem__),
    'top_one_student_tie_first': max(range(2), key=q.__getitem__)}

temperature_checks = []
for temperature in [1.0, 2.0, 4.0]:
    student = torch.tensor([[[0.2, -0.3], [0.7, 0.1], [-0.4, 0.8]]], requires_grad=True)
    teacher = torch.tensor([[[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]]).log().requires_grad_()
    labels = torch.tensor([[-100, 0, 1]])
    value = distillation_kl(student, teacher, labels, temperature)
    grad = torch.autograd.grad(value, student, retain_graph=True)[0]
    target = (teacher.detach() / temperature).softmax(-1)
    log_q = (student / temperature).log_softmax(-1)
    manual = ((target * (target.log() - log_q)).sum(-1)[labels != -100].mean() * temperature**2)
    manual_grad = torch.autograd.grad(manual, student)[0]
    temperature_checks.append({'temperature': temperature, 'helper_loss': value.item(),
        'formula_loss': manual.item(), 'absolute_loss_error': abs(value.item() - manual.item()),
        'max_absolute_gradient_error': (grad - manual_grad).abs().max().item(),
        'teacher_distribution': target.tolist()})
result['temperature_formula_checks'] = temperature_checks

student = torch.zeros(1, 3, 2, requires_grad=True)
teacher = torch.tensor([[p, p, p]]).log().requires_grad_()
labels = torch.tensor([[-100, 0, 1]])
kl = distillation_kl(student, teacher, labels, 1.0)
target = teacher.detach().softmax(-1)
ce_tensor = (-(target * student.log_softmax(-1)).sum(-1))[labels != -100].mean()
g_kl = torch.autograd.grad(kl, student)[0]
g_ce = torch.autograd.grad(ce_tensor, student)[0]
result['ce_kl_student_gradient'] = {'kl': kl.item(), 'ce': ce_tensor.item(),
    'max_absolute_gradient_error': (g_kl - g_ce).abs().max().item(),
    'student_gradient': g_kl.tolist(),
    'softmax_log_p_max_error': (teacher.detach().softmax(-1) - torch.tensor([[p, p, p]])).abs().max().item()}

torch.manual_seed(7)
model = TinyLM(ModelConfig(vocab_size=2, width=4, layers=1, heads=1, max_length=3))
embeddings = torch.randn(1, 3, 4, requires_grad=True)
logits = model(embeddings=embeddings)['logits']
logits.retain_grad()
loss = distillation_kl(logits, torch.tensor([[p, p, p]]).log(), labels, 1.0)
loss.backward()
result['ignored_prefix_indirect_gradient'] = {
    'seed': 7, 'model_config': model.description()['config'],
    'labels': labels.tolist(), 'loss': loss.item(),
    'ignored_position_direct_logits_gradient': logits.grad[0, 0].tolist(),
    'ignored_prefix_input_gradient': embeddings.grad[0, 0].tolist(),
    'ignored_prefix_input_gradient_l1': embeddings.grad[0, 0].abs().sum().item(),
    'note': 'The first embedding remains a causal attention input. Only its direct logits loss is excluded; this does not apply to masked PAD keys.'}

x, y, valid = pad_batch([(torch.tensor([1, 2, 3]), labels[0]),
                         (torch.tensor([1, 2]), torch.tensor([-100, 1]))])
result['padding'] = {'inputs': x.tolist(), 'labels': y.tolist(), 'attention_valid': valid.tolist(),
                     'padded_label_is_ignore': y[1, 2].item() == -100}

assert result['literal_example']['stdout'].splitlines() == [
    'KL 0.1927',
    '學生分數梯度 [[[0.0, 0.0], [-0.15000000596046448, 0.15000000596046448], [-0.15000000596046448, 0.15000000596046448]]]',
    '教師梯度 None',
]
assert abs(result['literal_example']['loss'] - sum(terms)) < 1e-6
assert result['literal_example']['teacher_gradient_is_none']
assert result['labels_same_mask_swapped_classes']['student_gradient'] == result['literal_example']['student_gradient']
assert abs(result['exercise_all_three']['loss'] - sum(terms)) < 1e-6
assert torch.allclose(torch.tensor(result['exercise_all_three']['student_gradient']),
                      torch.tensor([[[-0.1, 0.1]] * 3]), atol=1e-7, rtol=0)
assert all(r['absolute_loss_error'] < 1e-6 and r['max_absolute_gradient_error'] < 1e-6 for r in temperature_checks)
assert result['ce_kl_student_gradient']['max_absolute_gradient_error'] == 0
assert result['ignored_prefix_indirect_gradient']['ignored_position_direct_logits_gradient'] == [0.0, 0.0]
assert result['ignored_prefix_indirect_gradient']['ignored_prefix_input_gradient_l1'] > 1e-8
assert result['padding']['padded_label_is_ignore']
result['assertions'] = 'passed'
print(json.dumps(result, ensure_ascii=False, indent=2))
