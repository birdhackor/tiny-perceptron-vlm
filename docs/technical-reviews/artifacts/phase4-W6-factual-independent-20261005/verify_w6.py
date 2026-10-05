"""Bounded independent W.6 arithmetic, original-fence variants, and autograd checks."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import platform
import sys
import torch

BASE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()

def loss(w):
    return (w - 3) ** 2

arithmetic = {
    'L(1)': str(loss(F(1))),
    'L(1.1)': str(loss(F(11, 10))),
    'decrease_1_to_1.1': str(loss(F(1)) - loss(F(11, 10))),
    'difference_quotients': [],
    'updates': [],
}
assert loss(F(1)) == 4 and loss(F(11, 10)) == F(361, 100)
assert loss(F(1)) - loss(F(11, 10)) == F(39, 100)
for h in [F(1,10), F(1,100), F(1,1000), F(-1,1000)]:
    quotient = (loss(1+h) - loss(F(1))) / h
    assert quotient == -4 + h
    arithmetic['difference_quotients'].append({'h':str(h),'quotient':str(quotient)})
for rate, expected_w, expected_L in [(F(1,10),F(7,5),F(64,25)),(F(3,5),F(17,5),F(4,25)),(F(11,10),F(27,5),F(144,25))]:
    updated = 1 - rate * (-4)
    assert updated == expected_w and loss(updated) == expected_L
    arithmetic['updates'].append({'learning_rate':str(rate),'new_w':str(updated),'new_L':str(loss(updated)),'new_derivative':str(2*(updated-3))})

fence = (BASE / 'original-fence-run/fence-1.py').read_bytes()
variants = []
for rate, expected_w, expected_L in [(0.1,1.4,2.56),(0.6,3.4,0.16),(1.1,5.4,5.76)]:
    # Preserve every original statement except the one learning-rate literal.
    code = fence.replace(b'w -= 0.1 * w.grad', f'w -= {rate} * w.grad'.encode())
    assert code.count(f'w -= {rate} * w.grad'.encode()) == 1
    path = BASE / f'variant-lr-{rate}.py'
    path.write_bytes(code)
    namespace = {'__name__':'__main__'}
    print(f'original-fence variant lr={rate}')
    exec(compile(code,str(path),'exec'),namespace)
    w = namespace['w']
    observed = {'learning_rate':rate,'w':w.item(),'loss':loss(w).item(),'stored_grad':w.grad.item(),'requires_grad':w.requires_grad,'is_leaf':w.is_leaf}
    assert abs(observed['w']-expected_w)<1e-6
    assert abs(observed['loss']-expected_L)<1e-6
    assert observed['stored_grad']==-4.0 and w.is_leaf and w.requires_grad
    variants.append(observed)

w=torch.tensor(1.,requires_grad=True)
initial_loss=(w-3).square()
initial_loss.backward()
assert w.item()==1. and w.grad.item()==-4.
with torch.no_grad():
    w -= .1*w.grad
assert w.is_leaf and w.grad_fn is None and w.requires_grad
new_loss=(w-3).square()
assert new_loss.requires_grad and new_loss.grad_fn is not None
new_loss.backward()
accumulated=w.grad.item()
assert abs(accumulated-(-7.2))<1e-6
w.grad=None
(w-3).square().backward()
fresh=w.grad.item()
assert abs(fresh-(-3.2))<1e-6

# A genuine two-variable partial-derivative calculation, holding the other fixed.
v=torch.tensor([1.,2.],requires_grad=True)
cost=(v[0]-3).square() + 2*(v[1]-1).square()
cost.backward()
assert v.grad.tolist()==[-4.,4.]

result={
    'arithmetic':arithmetic,
    'fence_variants':variants,
    'autograd':{'backward_leaves_w_unchanged':True,'no_grad_update_preserves_leaf':True,'new_loss_rebuilds_graph':True,'uncleared_second_backward_grad':accumulated,'cleared_second_backward_grad':fresh,'two_variable_gradient':v.grad.tolist()},
    'tolerance':{'exact_arithmetic':'Fractions; exact equality','float32':'absolute error < 1e-6'},
    'environment':{'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'torch':str(torch.__version__),'torch_git':str(torch.version.git_version),'device':'CPU','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available())},
    'original_fence_sha256':hashlib.sha256(fence).hexdigest(),
    'conclusion_scope':'Scalar squared-error calculation and one-step parameter updates only; no training or model capability evaluation.'
}
(BASE/'verification-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
