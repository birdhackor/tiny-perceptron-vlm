import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from tiny_perceptron.posttraining import (
    FiniteResponsePolicy, FiniteValueModel, bandit_advantage, ppo_clipped_objective,
)

BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(1311)
assert torch.version.cuda is None and not torch.cuda.is_available()
records={
 'environment':{'python':sys.version,'torch':torch.__version__,'torch_git_version':torch.version.git_version,
   'device':'cpu','platform':platform.platform(),'seed':1311,'threads':1},
 'meaning':'Independent bounded arithmetic/autograd/one-step optimizer checks; no existing model evaluation or full training.'}
rewards=torch.tensor([1.,0.],requires_grad=True)
old=torch.tensor([.4,.4],requires_grad=True)
a=bandit_advantage(rewards,old)
torch.testing.assert_close(a,torch.tensor([.6,-.4]),rtol=0,atol=1e-7)
assert not a.requires_grad and a.grad_fn is None
records['base']={'rewards':rewards.tolist(),'old_values':old.tolist(),'advantage':a.tolist(),
 'requires_grad':a.requires_grad,'grad_fn':str(a.grad_fn),'round_2':[round(v,2) for v in a.tolist()]}
changed=bandit_advantage(rewards,torch.tensor([.9,.9]))
torch.testing.assert_close(changed,torch.tensor([.1,-.9]),rtol=0,atol=1e-7)
records['exercise']={'old_values':[.9,.9],'advantage':changed.tolist(),'round_2':[round(v,2) for v in changed.tolist()]}
negative_correct=bandit_advantage(torch.tensor([1.]),torch.tensor([1.2]))
positive_incorrect=bandit_advantage(torch.tensor([0.]),torch.tensor([-.2]))
assert negative_correct.item()<0 and positive_incorrect.item()>0
records['sign_is_relative']={'reward_1_baseline_1_2':negative_correct.tolist(),
 'reward_0_baseline_minus_0_2':positive_incorrect.tolist()}
errors=[]
for r,v in [(torch.ones(2),torch.ones(1)),(torch.empty(0),torch.empty(0))]:
 try: bandit_advantage(r,v)
 except ValueError as e: errors.append(str(e))
 else: raise AssertionError('Invalid shape/empty input accepted')
records['contract_errors']=errors
new_log=torch.tensor([-.6,-.8],requires_grad=True)
old_log=torch.tensor([-.7,-.7],requires_grad=True)
raw_adv=torch.tensor([.6,-.4],requires_grad=True)
terms=ppo_clipped_objective(new_log,old_log,raw_adv)
terms['policy_loss'].backward()
assert new_log.grad is not None and old_log.grad is None and raw_adv.grad is None
records['ppo_detach_contract']={'ratio':terms['ratio'].detach().tolist(),
 'policy_loss':terms['policy_loss'].item(),'new_grad':new_log.grad.tolist(),
 'old_grad':None,'advantage_grad':None}
policy=FiniteResponsePolicy()
value=FiniteValueModel()
x=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]])
targets=torch.tensor([1.,0.])
old_values=value(x)
fixed=bandit_advantage(targets,old_values)
actor_optimizer=torch.optim.SGD(policy.parameters(),lr=.05)
critic_optimizer=torch.optim.SGD(value.parameters(),lr=.05)
critic_before=[p.detach().clone() for p in value.parameters()]
selected=policy(x).log_softmax(-1)[:,0]
actor_optimizer.zero_grad()
(-selected*fixed).mean().backward()
assert all(p.grad is None for p in value.parameters())
actor_optimizer.step()
assert all(torch.equal(a,b) for a,b in zip(critic_before,value.parameters()))
before_loss=torch.nn.functional.mse_loss(value(x),targets)
critic_optimizer.zero_grad()
before_loss.backward()
assert all(p.grad is not None for p in value.parameters())
critic_optimizer.step()
after_loss=torch.nn.functional.mse_loss(value(x),targets)
assert any(not torch.equal(a,b) for a,b in zip(critic_before,value.parameters()))
assert after_loss.item()<before_loss.item()
records['separate_value_update']={'actor_step_kept_value_parameters':True,
 'critic_step_changed_value_parameters':True,'value_loss_before':before_loss.item(),
 'value_loss_after':after_loss.item(),'samples':2,'actor_steps':1,'critic_steps':1}
records['source_file_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(BASE/'cpu-variants-result.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
