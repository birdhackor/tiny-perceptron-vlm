"""Independent short CPU arithmetic/gradient checks; no training or data download."""
from pathlib import Path
import hashlib
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.posttraining import ppo_clipped_objective

torch.set_default_device('cpu')
torch.set_num_threads(1)
DEST=Path(__file__).resolve().parent
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_printoptions(precision=12)
old=torch.tensor([0.2,0.2],dtype=torch.float64)
new=torch.tensor([0.4,0.1],dtype=torch.float64)
ratios=(new.log()-old.log().detach()).exp()
assert torch.allclose(ratios,torch.tensor([2.,0.5],dtype=torch.float64),atol=1e-12,rtol=0)
exercise=(torch.tensor([0.3,0.1],dtype=torch.float64).log()-old.log()).exp()
assert torch.allclose(exercise,torch.tensor([1.5,0.5],dtype=torch.float64),atol=1e-12,rtol=0)
wrong=(new.log()-new.log()).exp()
assert torch.equal(wrong,torch.ones(2,dtype=torch.float64))
old_distribution=torch.stack([old,1-old],dim=1)
new_distribution=torch.stack([new,1-new],dim=1)
assert torch.equal(old_distribution.sum(1),torch.ones(2,dtype=torch.float64))
assert torch.equal(new_distribution.sum(1),torch.ones(2,dtype=torch.float64))
selected_action=torch.tensor([0,0])
selected_new=new_distribution.gather(1,selected_action[:,None]).squeeze(1)
selected_old=old_distribution.gather(1,selected_action[:,None]).squeeze(1)
assert torch.allclose(selected_new/selected_old,ratios,atol=1e-12,rtol=0)
advantage=torch.tensor([0.6,-0.4],dtype=torch.float64)
product=ratios*advantage
assert abs(product[0].item()-1.2)<1e-12

old_log=old.log().requires_grad_(True)
new_log=new.log().requires_grad_(True)
terms=ppo_clipped_objective(new_log,old_log,advantage)
assert torch.allclose(terms['ratio'],ratios,atol=1e-12,rtol=0)
terms['unclipped'].sum().backward()
assert old_log.grad is None
assert torch.allclose(new_log.grad,product,atol=1e-12,rtol=0)

# detach severs autograd, while a separately saved clone preserves historical values.
live=torch.tensor([-1.,-2.],requires_grad=True)
detached=live.detach()
historical=live.detach().clone()
with torch.no_grad(): live.add_(0.25)
assert torch.equal(detached,live)
assert torch.equal(historical,torch.tensor([-1.,-2.]))

# A zero-probability denominator is outside the selected-action example's contract.
# Log probabilities can retain valid numbers even when exp underflows in float64.
log_old=torch.tensor([-1000.],dtype=torch.float64)
log_new=torch.tensor([-999.],dtype=torch.float64)
robust=(log_new-log_old).exp()
direct=log_new.exp()/log_old.exp()
assert torch.isfinite(robust).all() and abs(robust.item()-2.718281828459045)<1e-12
assert torch.isnan(direct).all()

env={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
     'torch_git_version':str(torch.version.git_version),'device':'cpu',
     'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),
     'threads':str(torch.get_num_threads()),'dtype':'float64 (alias check: float32)'}
result={'ratio':ratios.tolist(),'exercise_ratio':exercise.tolist(),
        'wrong_denominator_ratio':wrong.tolist(),'ratio_times_advantage':product.tolist(),
        'old_distribution_batch_action':old_distribution.tolist(),
        'new_distribution_batch_action':new_distribution.tolist(),
        'per_context_probability_sums':new_distribution.sum(1).tolist(),
        'old_log_gradient':None,'new_log_gradient':new_log.grad.tolist(),
        'detached_shares_storage_observed':detached.tolist(),
        'saved_clone_retained_history':historical.tolist(),
        'small_log_ratio':robust.tolist(),'direct_probability_division':'nan (0/0 underflow)',
        'contract':'two independent selected actions, not the full categorical action axis',
        'numerical_tolerance':'1e-12 float64; original float32 rounded output checked separately',
        'updates_performed':0,'training_executed':False}
(DEST/'bounded-cpu.environment.json').write_text(json.dumps(env,indent=2)+'\n')
(DEST/'bounded-cpu.result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
