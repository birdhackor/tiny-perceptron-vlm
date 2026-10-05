"""Independent bounded CPU checks for 16.10; no datasets/models downloaded."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum

OUT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
print(json.dumps({"python": sys.version, "torch": str(torch.__version__),
                  "torch_git": torch.version.git_version, "device": "cpu",
                  "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}))
assert torch.version.cuda is None and not torch.cuda.is_available()

# Execute the extracted original bytes, without rewriting the original fence.
fence = (OUT / "code/fence-1.py").read_bytes()
namespace = {}
exec(compile(fence, "course/chapters/16.md#16.10:source-line-463", "exec"), namespace)
assert tuple(namespace["y_plain"].shape) == (2, 4)
assert torch.equal(namespace["y_plain"], namespace["y_recompute"])
assert torch.equal(namespace["x_plain"].grad, namespace["x_recompute"].grad)
shared = namespace["layer"]
reference = copy.deepcopy(shared)
reference.zero_grad(set_to_none=True)
x = namespace["base"].clone().requires_grad_()
reference(x).square().sum().backward()
accumulation_error = max(float((p.grad - 2*q.grad).abs().max())
                         for p, q in zip(shared.parameters(), reference.parameters(), strict=True))
assert accumulation_error < 1e-6
print("original_shared_weight_gradient_vs_twice_single", accumulation_error)

# Exercise width 16 and compare input and all parameter gradients with fresh copies.
for width in (8, 16):
    torch.manual_seed(0)
    plain = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, 4))
    recompute = copy.deepcopy(plain)
    base = torch.randn(2, 4)
    x1, x2 = base.clone().requires_grad_(), base.clone().requires_grad_()
    y1, y2 = plain(x1), checkpoint(recompute, x2, use_reentrant=False)
    y1.square().sum().backward(); y2.square().sum().backward()
    output_error = float((y1-y2).abs().max().detach())
    input_error = float((x1.grad-x2.grad).abs().max())
    parameter_error = max(float((p.grad-q.grad).abs().max()) for p,q in zip(plain.parameters(),recompute.parameters(),strict=True))
    assert y1.shape == (2,4) and max(output_error,input_error,parameter_error) < 1e-6
    opt1, opt2 = torch.optim.AdamW(plain.parameters(),lr=.003), torch.optim.AdamW(recompute.parameters(),lr=.003)
    before = [p.detach().clone() for p in plain.parameters()]
    opt1.step(); opt2.step()
    weight_error = max(float((p-q).abs().max().detach()) for p,q in zip(plain.parameters(),recompute.parameters(),strict=True))
    weight_change = max(float((p-q).abs().max().detach()) for p,q in zip(plain.parameters(),before,strict=True))
    assert weight_change > 0 and weight_error < 1e-6
    print("mlp", json.dumps({"hidden_width":width,"shape":list(y1.shape),"output_error":output_error,
                              "input_gradient_error":input_error,"parameter_gradient_error":parameter_error,
                              "one_update_weight_error":weight_error,"weight_change":weight_change}))

# CPU dropout: demonstrate default RNG preservation, and a controlled negative case.
drop = nn.Sequential(nn.Dropout(.5), nn.Linear(4, 3))
base = torch.linspace(.1, 4., 40).reshape(10,4)
def dropout_trial(use_checkpoint, preserve=True):
    torch.manual_seed(99)
    model = copy.deepcopy(drop)
    x = base.clone().requires_grad_()
    y = checkpoint(model, x, use_reentrant=False, preserve_rng_state=preserve) if use_checkpoint else model(x)
    y.square().sum().backward()
    return y.detach(), x.grad, [p.grad for p in model.parameters()]
a,ga,pa=dropout_trial(False)
b,gb,pb=dropout_trial(True)
c,gc,pc=dropout_trial(True,False)
assert torch.equal(a,b) and torch.equal(ga,gb)
assert all(torch.equal(p,q) for p,q in zip(pa,pb,strict=True))
negative_error=float((ga-gc).abs().max())
assert negative_error > 0 and torch.equal(a,c)
print("dropout",json.dumps({"preserved_output_error":float((a-b).abs().max()),
                           "preserved_input_gradient_error":float((ga-gb).abs().max()),
                           "no_preservation_input_gradient_error":negative_error}))

# Execute exact original experiment helper definitions from the recorded revision.
source = OUT / "code/original-revision/scripts/course_experiments/architecture.py"
tree=ast.parse(source.read_bytes())
names={"_checkpoint_forward","_gradients","_gradient_error"}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
helpers={"torch":torch,"checkpoint":checkpoint,"loss_sum":loss_sum}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),helpers)
torch.manual_seed(42)
model=TinyLM(ModelConfig(width=8,layers=2,heads=2,max_length=64))
other=copy.deepcopy(model)
records=[[{"role":"user","content":"Q"},{"role":"assistant","content":answer}] for answer in ["A","OK","答","答案"]]
examples=[render_chat(r) for r in records]
ids,labels,valid=pad_batch(examples)
effective=int((labels!=IGNORE).sum());expected=sum(len(ByteTokenizer().encode(r[-1]["content"]))+1 for r in records)
assert effective==expected==16
forward=lambda *args,**kwargs:helpers["_checkpoint_forward"](other,*args,**kwargs)
a,ga,la=helpers["_gradients"](model,ids,labels,valid)
b,gb,lb=helpers["_gradients"](other,ids,labels,valid,forward=forward)
logit_error=float((a-b).abs().max());gradient_error=helpers["_gradient_error"](ga,gb)
assert logit_error < 1e-6 and gradient_error < 1e-6
oa,ob=torch.optim.AdamW(model.parameters(),lr=.003),torch.optim.AdamW(other.parameters(),lr=.003)
oa.step();ob.step()
update_error=max(float((p-q).abs().max().detach()) for p,q in zip(model.parameters(),other.parameters(),strict=True))
assert update_error < 1e-6
print("original_tinylm_helpers",json.dumps({"layers":2,"ids_shape":list(ids.shape),"effective_answer_plus_eos_tokens":effective,
                                        "logit_error":logit_error,"weight_gradient_error":gradient_error,
                                        "one_update_weight_error":update_error,"losses":[la,lb]}))
print("all bounded CPU checks passed")
