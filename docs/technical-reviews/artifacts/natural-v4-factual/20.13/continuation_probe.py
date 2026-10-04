"""Small CPU state-continuation demonstration; not natural model replication."""
import copy
import json
import tempfile
from pathlib import Path
import torch

def update(model, optimizer):
    x=torch.rand(5,2);y=x[:,0:1]-2*x[:,1:2]
    optimizer.zero_grad();loss=((model(x)-y)**2).mean();loss.backward();optimizer.step()
    return float(loss.detach())

torch.manual_seed(42)
m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=0.01)
update(m,o);update(m,o)
saved={"model":copy.deepcopy(m.state_dict()),"optimizer":copy.deepcopy(o.state_dict()),
       "rng":torch.get_rng_state(),"step":2}
expected=update(m,o);expected_weights=copy.deepcopy(m.state_dict())
with tempfile.TemporaryDirectory() as temp:
    path=Path(temp)/'state.pt';torch.save(saved,path);state=torch.load(path,weights_only=True)
    restored=torch.nn.Linear(2,1);opt=torch.optim.AdamW(restored.parameters(),lr=0.01)
    restored.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
    observed=update(restored,opt)
    full=all(torch.equal(expected_weights[k],restored.state_dict()[k]) for k in expected_weights)
    only=torch.nn.Linear(2,1);fresh=torch.optim.AdamW(only.parameters(),lr=0.01)
    only.load_state_dict(state['model']);torch.set_rng_state(state['rng']);update(only,fresh)
    weights_only_equal=all(torch.equal(expected_weights[k],only.state_dict()[k]) for k in expected_weights)
assert expected==observed and full and not weights_only_equal
print(json.dumps({'python_device':'cpu','torch':torch.__version__,'saved_step':2,'next_step':3,
                  'loss_next_uninterrupted':expected,'loss_next_restored':observed,
                  'full_state_next_weights_equal':full,'weights_only_with_fresh_optimizer_equal':weights_only_equal,
                  'scope':'One tiny same-runtime AdamW example. Confirms why weight-only state cannot promise exact continuation; no GPU or complete natural model resume test.'},indent=2))
