"""Bounded CPU checks for the claims of 18.8; no model training or optimizer."""
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
from tiny_perceptron.alignment import distillation_kl

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device('cpu')
environment = {
    'python': platform.python_version(), 'torch': str(torch.__version__),
    'torch_git_version': str(torch.version.git_version), 'device': 'cpu',
    'cuda_build': str(torch.version.cuda), 'cuda_available': str(torch.cuda.is_available()),
    'threads': str(torch.get_num_threads()), 'optimizer_steps': '0',
}
(BASE/'execution/variants-environment.json').write_text(json.dumps(environment, indent=2)+'\n')

def run(labels, temperature=1., student_values=None, probabilities=None):
    student = (torch.zeros(1, 3, 2) if student_values is None else torch.tensor(student_values)).requires_grad_()
    teacher = torch.tensor(probabilities or [[[.8, .2]] * 3]).log().requires_grad_()
    student_before = student.detach().clone()
    loss = distillation_kl(student, teacher, torch.tensor([labels]), temperature=temperature)
    loss.backward()
    assert teacher.grad is None
    assert torch.equal(student, student_before)
    return loss.item(), student.grad.detach().clone(), teacher.grad

terms = [p*(math.log(p)-math.log(q)) for p,q in zip([.8,.2],[.5,.5], strict=True)]
kl = sum(terms)
reverse = sum(q*(math.log(q)-math.log(p)) for p,q in zip([.8,.2],[.5,.5], strict=True))
entropy = -sum(p*math.log(p) for p in [.8,.2])
ce = -sum(p*math.log(q) for p,q in zip([.8,.2],[.5,.5], strict=True))
assert abs(kl-(ce-entropy)) < 1e-15
print(json.dumps({'numeric': {'kl_terms_nats': terms, 'kl_nats': kl, 'reverse_kl_nats': reverse, 'entropy_nats': entropy, 'cross_entropy_nats': ce}}, ensure_ascii=False))

base, gradient, _ = run([-100,0,1])
assert abs(base-kl) < 1e-7
assert torch.allclose(gradient, torch.tensor([[[0.,0.],[-.15,.15],[-.15,.15]]]),atol=1e-7)
allvalid, allgrad, _ = run([0,0,1])
assert abs(allvalid-base) < 1e-7
assert torch.allclose(allgrad, torch.tensor([[[-.1,.1],[-.1,.1],[-.1,.1]]]),atol=1e-7)
swap, swapgrad, _ = run([-100,1,0])
assert swap == base and torch.equal(swapgrad,gradient)
changed_masked, changedgrad, _ = run([-100,0,1],student_values=[[[4.,-4.],[0.,0.],[0.,0.]]])
assert changed_masked == base and torch.equal(changedgrad,gradient)
print(json.dumps({'masks': {'valid2_loss':base,'valid2_gradient':gradient.tolist(),'valid3_loss':allvalid,'valid3_gradient':allgrad.tolist(),'swapped_label_ids_identical':True,'changed_masked_logits_identical':True,'teacher_grad':None,'no_parameters_updated':True}}))

T=2.
lt, gt, _ = run([-100,0,1],temperature=T)
pT = torch.tensor([.8,.2],dtype=torch.float64).log().div(T).softmax(-1)
qT = torch.tensor([.5,.5],dtype=torch.float64)
manual_T = T*T * (pT*(pT.log()-qT.log())).sum().item()
manual_gradient = (T*(qT-pT)/2).to(torch.float32)
assert abs(lt-manual_T)<1e-6
assert torch.allclose(gt[0,1],manual_gradient,atol=1e-7)
print(json.dumps({'temperature2': {'teacher_distribution':pT.tolist(),'manual_T_squared_once_loss':manual_T,'observed_loss':lt,'gradient':gt.tolist(),'expected_valid_position_gradient':manual_gradient.tolist()}}))

# The entropy constant must not affect student gradients when the teacher is fixed.
s = torch.tensor([.3,-.4],requires_grad=True)
p = torch.tensor([.8,.2])
logq = s.log_softmax(-1)
loss_kl=(p*(p.log()-logq)).sum()
loss_ce=-(p*logq).sum()
g_kl=torch.autograd.grad(loss_kl,s,retain_graph=True)[0]
g_ce=torch.autograd.grad(loss_ce,s)[0]
assert torch.equal(g_kl,g_ce)
print(json.dumps({'ce_kl': {'student_logits':[.3,-.4], 'ce':loss_ce.item(),'kl':loss_kl.item(),'ce_gradient':g_ce.tolist(),'kl_gradient':g_kl.tolist(),'gradient_equal':True}}))

# A masked position can still supply an earlier feature to later causal scores.
x=torch.zeros(1,3,requires_grad=True)
prefix=x.cumsum(dim=1)
logits=torch.stack([prefix,-prefix],dim=-1)
logits.retain_grad()
loss=distillation_kl(logits,torch.tensor([[[.8,.2]]*3]).log(),torch.tensor([[-100,0,1]]),temperature=1.)
loss.backward()
assert torch.equal(logits.grad[0,0],torch.zeros(2))
assert abs(x.grad[0,0].item()+.6)<1e-7
print(json.dumps({'indirect_prefix': {'masked_position_direct_logits_grad':logits.grad[0,0].tolist(),'earlier_feature_gradient':x.grad.tolist(),'construction':'causal cumulative feature; ordinary context, no padding attention mask'}}))

# Primitive API and denominator checks used by the raw fence/helper.
assert torch.tensor([.5,.5]).argmax().item()==0
assert torch.tensor([[[.8,.2]]*3]).shape==(1,3,2)
assert torch.allclose(torch.tensor([.8,.2]).log().softmax(-1),torch.tensor([.8,.2]),atol=1e-7)
assert torch.equal(torch.zeros(2).softmax(-1),torch.tensor([.5,.5]))
assert torch.equal(torch.zeros(2).log_softmax(-1).exp(),torch.tensor([.5,.5]))
pointwise=torch.nn.functional.kl_div(torch.tensor([.5,.5]).log(),torch.tensor([.8,.2]),reduction='none')
assert abs(pointwise.sum().item()-kl)<1e-7
assert abs(pointwise.mean().item()-kl/2)<1e-7
print(json.dumps({'api':{'argmax_tie_index':0,'teacher_shape':[1,3,2],'pointwise_kl':pointwise.tolist(),'vocabulary_sum':pointwise.sum().item(),'vocabulary_mean_wrong_denominator':pointwise.mean().item()}}))
print('All bounded CPU assertions passed; no optimizer updates, no empirical model score produced.')
