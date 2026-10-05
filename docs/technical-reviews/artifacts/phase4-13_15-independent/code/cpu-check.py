import copy, json, hashlib, ast, math, os, sys, platform
from pathlib import Path
import torch
import torch.nn.functional as F
from tiny_perceptron.posttraining import FiniteResponsePolicy,FiniteRewardModel,FiniteValueModel,ppo_clipped_objective,exact_kl,bandit_advantage
from scripts.course_experiments import posttraining as E
A=Path(__file__).resolve().parents[1]; torch.set_num_threads(1)
env={'python':platform.python_version(),'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_revision':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'threads':str(torch.get_num_threads())}
assert torch.version.cuda is None and not torch.cuda.is_available()
(A/'execution/environment-cpu.json').write_text(json.dumps(env,indent=2)+'\n')
print('Original fence follows:')
ns={'__name__':'__main__'}; code=(A/'code/fence-1.py').read_text(); exec(compile(code,str(A/'code/fence-1.py'),'exec'),ns)
assert ns['action'].item()==0 and ns['terms']['ratio'].item()==1
assert not torch.equal(ns['before'],next(ns['policy'].parameters()))
assert not any(p.grad is not None for p in ns['reward_model'].parameters())
assert not any(p.grad is not None for p in ns['reference'].parameters())
checks={}
for label,change in [('base',False),('explain',True)]:
 namespace={'__name__':'__main__'}
 instrument=code.replace('new_logits = policy(context)','snapshots = {k: [p.detach().clone() for p in v.parameters()] for k,v in {"policy":policy,"critic":critic,"reward":reward_model,"reference":reference}.items()}\nnew_logits = policy(context)')
 if change: instrument=instrument.replace('[[0.1, 0.2, 0.0, 0.0]]','[[0.1, 0.2, 1.0, 0.0]]')
 exec(compile(instrument,label,'exec'),namespace)
 changes={k:any(not torch.equal(old,p) for old,p in zip(namespace['snapshots'][k],v.parameters())) for k,v in {'policy':namespace['policy'],'critic':namespace['critic'],'reward':namespace['reward_model'],'reference':namespace['reference']}.items()}
 post=namespace['policy'](namespace['context']); ratio_after=(torch.distributions.Categorical(logits=post).log_prob(namespace['action'])-namespace['old_log_probability']).exp()
 checks[label]={'changes':changes,'action':namespace['action'].item(),'context_shape':list(namespace['context'].shape),'logits_shape':list(namespace['new_logits'].shape),'ratio_before':namespace['terms']['ratio'].item(),'ratio_after':ratio_after.item(),'fixed_record_requires_grad':{k:namespace[k].requires_grad for k in ['old_log_probability','reward','advantage']},'optimizer_parameter_ids':len({id(p) for g in namespace['optimizer'].param_groups for p in g['params']})}
 assert changes=={'policy':True,'critic':True,'reward':False,'reference':False}
 assert not any(checks[label]['fixed_record_requires_grad'].values())
checks['counts']={name:sum(p.numel() for p in cls().parameters()) for name,cls in [('policy',FiniteResponsePolicy),('reward',FiniteRewardModel),('critic',FiniteValueModel)]}
assert checks['counts']=={'policy':148,'reward':241,'critic':97}
x=torch.tensor([[0.4,-0.1,1.2,-0.7]],dtype=torch.float64,requires_grad=True); target=torch.tensor([0]); smooth=F.cross_entropy(x,target,label_smoothing=0.2); distribution=torch.tensor([[.85,.05,.05,.05]],dtype=torch.float64); explicit=-(distribution*x.log_softmax(-1)).sum(); assert torch.allclose(smooth,explicit,atol=1e-12,rtol=0)
grad=torch.autograd.grad(smooth,x)[0]; assert torch.allclose(grad,x.softmax(-1)-distribution,atol=1e-12,rtol=0)
checks['label_smoothing']={'target':distribution.tolist()[0],'smooth_loss':smooth.item(),'explicit_soft_target_loss':explicit.item(),'gradient':grad.tolist(),'tolerance':1e-12}
records=E.build_records(); splits=E.split_records(records); families={k:{r['family'] for r in rows} for k,rows in splits.items()}
assert all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
checks['data_contract']={'families':{k:len(v) for k,v in families.items()},'contexts':{k:len(v) for k,v in splits.items()},'all_families':len(set.union(*families.values())),'unique_train_preference_pairs':len(E._pairs(splits['train'])),'number_demonstrations':sum(r['mode']=='number' for r in splits['train']),'reserved_1_plus_2_family':{k:'pair:1:2' in v for k,v in families.items()},'example_number':next({k:r[k] for k in ['features','candidates','expected_action']} for r in splits['test'] if r['family']=='pair:1:2' and r['mode']=='number'),'example_explain':next({k:r[k] for k in ['features','expected_action']} for r in splits['test'] if r['family']=='pair:1:2' and r['mode']=='explain')}
assert checks['data_contract']['contexts']=={'train':132,'validation':15,'test':18}; assert checks['data_contract']['unique_train_preference_pairs']==660
checks['draw_arithmetic']={'reward_pair_draws':300*64,'ppo_sampled_actions':120*64,'policy_updates':120*3,'critic_updates':120*3,'reused_action_draws':120*64*3,'sft_draws':60*32,'effective_tokens':0}
(A/'execution/cpu-check.json').write_text(json.dumps(checks,indent=2)+'\n'); print('CPU checks:',json.dumps(checks,indent=2)); print('ENV:',json.dumps(env))
